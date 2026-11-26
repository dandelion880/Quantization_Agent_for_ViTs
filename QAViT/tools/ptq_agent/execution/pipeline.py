"""v3 可组合 PTQ 执行。预处理始终先于量化，W/A 阶段顺序实际参与校准。

所有变换发生在 FP 副本；每个 trial 从原模型重建。临时校准缓存逐位点释放，
导出保存量化参数和重参数化状态，加载时不重复拟合。
"""

from copy import deepcopy
import gc
import os
from pathlib import Path
import tempfile
import torch
from torch import nn
from torch.nn import functional as F

from ..schemas import require, fingerprint
from ..storage import atomic_json, read_json
from ..backends.server import file_sha256
from ..profiling.evidence import seal
from .model import forward, tensor_digest
from .contracts import RAW_FLOATS
from .codecs import DTYPES, fit as fit_rtn, qdq as rtn_qdq
from .plugin_contracts import calibration, site_id, WEIGHT_KINDS, ATTENTION_KINDS
from . import plugin_math as math_plugin
from .methods import get_target_method, plugin_versions
from .calibration import capture, combined_input, gram, reconstruction_score
from .sites import install_attention, semantic_edge_checks
from .metrics import OutputErrors
from . import grids, codecs
from .coordinate import operator_id, fit_operator
from .operators import FLOW_KINDS, install_flows, install_operators, bind_operators, audit_operators, bind_calibration_operators

PLUGIN_VERSION = "composable_reference_v2"


class ActivationQuantizer(nn.Module):
    def __init__(self, layer, execution, params=None):
        super().__init__()
        self.layer, self.execution = deepcopy(layer), deepcopy(execution)
        self.ptq_site_id = site_id(layer)
        self.enabled, self.gradient_probe, self.calls = False, False, 0
        self.parameter_names = []
        self.set_params(params or {})

    def set_params(self, params):
        for name in self.parameter_names:
            delattr(self, "q_" + name)
        self.parameter_names = list(params)
        for key, value in params.items():
            self.register_buffer("q_" + key, value.detach().clone())

    def params(self):
        return {k: getattr(self, "q_" + k) for k in self.parameter_names}

    def quantize(self, x):
        if not self.enabled or self.layer["activation"] is None:
            return x
        if hasattr(self,"_focus_bounds"):
            from .rank_search import focus
            return focus(x,self._focus_bounds,self.layer["activation"]["bits"])
        if hasattr(self,"_continuous_width"):
            from .progressive_bits import continuous
            return x if self._continuous_width is None else continuous(x,self.layer["activation"],self.params(),self._continuous_width)
        if hasattr(self,"_fitting"):
            from .reconstruction import reconstruct_activation
            return reconstruct_activation(x,self)
        if getattr(self,"ptq_window_member",False):
            from .window_routes import branch,low_grid
            bits=branch.get()
            if bits is not None:return low_grid(self,x,bits)
        p = self.params() if self.execution["activation_calibration"] == "static" else None
        q = get_target_method("activation", self.layer["activation_method"]).apply(
            x, self.layer["activation"], params=p, options=self.layer.get("options", {}))
        # 只在梯度测量上下文传递梯度；推理始终使用同一 QDQ 数值，无训练路径。
        return x + (q - x).detach() if self.gradient_probe else q

    def forward(self, x, rhs=None):
        self.calls += 1
        self.last_input_shape = list(x.shape)
        q = self.quantize(x)
        return q if rhs is None else q @ rhs


class PipelineLinear(ActivationQuantizer):
    def __init__(self, original, layer, execution, preprocess=None, record=None, params=None):
        super().__init__(layer, execution, params)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.register_buffer("bias", None if original.bias is None else original.bias.detach().clone())
        self.register_buffer("bias_mask", None if not hasattr(original, "bias_mask") else original.bias_mask.detach().clone())
        device = original.weight.device
        prep = preprocess or {"scale": torch.ones(self.in_features), "shift": torch.zeros(self.in_features)}
        self.register_buffer("input_scale", prep["scale"].to(device))
        self.register_buffer("input_shift", prep["shift"].to(device))
        self.register_buffer("reference_weight", original.weight.detach().clone() * self.input_scale, persistent=False)
        self.register_buffer("decoded_weight", self.reference_weight.clone(), persistent=False)
        self.weight_record = record
        self.bias_uses_decoded_weight = False
        self.weight_enabled = False
        if record is not None:
            self.decoded_weight = math_plugin.decode_weight(record, device)

    def calibration_input(self, x):
        return self.quantize(x / self.input_scale - self.input_shift)

    def effective_bias(self, weight=None):
        bias = self.bias if self.bias_mask is None or self.bias is None else self.bias * self.bias_mask
        # AdaLog 的最终补偿必须依赖实际解码/舍入权重；不缓存过期的补偿。
        if self.bias_uses_decoded_weight:
            source = weight if weight is not None else (self.decoded_weight if self.weight_enabled else self.reference_weight)
        else:
            source = self.reference_weight
        correction = source @ self.input_shift
        if self.weight_enabled and self.weight_record and "bias_delta" in self.weight_record["params"]:
            correction=correction+self.weight_record["params"]["bias_delta"].to(correction)
        if hasattr(self,"fitted_input_bias"):
            correction=correction+self.fitted_input_bias
        # DINO 的 K bias 掩码只作用于原 bias；仿射补偿不能再次被掩掉。
        return correction if bias is None else bias + correction

    def forward(self, x):
        self.calls += 1
        self.last_input_shape = list(x.shape)
        dtype = DTYPES[self.execution["compute_dtype"]]
        w = self.decoded_weight if self.weight_enabled else self.reference_weight
        return F.linear(self.calibration_input(x).to(dtype), w.to(dtype), self.effective_bias().to(dtype)).to(x.dtype)


class PipelineConv2d(PipelineLinear):
    """普通 Conv2d 复用相同的 W/A 插件；不改变卷积几何或 bias。"""
    def __init__(self, original, layer, execution, record=None, params=None, preprocess=None):
        ActivationQuantizer.__init__(self, layer, execution, params)
        self.stride, self.padding, self.dilation, self.groups = original.stride, original.padding, original.dilation, original.groups
        self.register_buffer("bias", None if original.bias is None else original.bias.detach().clone())
        self.transformed=preprocess is not None
        self.register_buffer("input_scale", (preprocess["scale"].to(original.weight) if preprocess else original.weight.new_tensor(1.)))
        self.register_buffer("input_shift", (preprocess["shift"].to(original.weight) if preprocess else original.weight.new_tensor(0.)))
        weight=original.weight.detach().clone()
        if preprocess:
            if self.groups==original.in_channels==original.out_channels: weight=weight*self.input_scale[:,None,None,None]
            else:
                require(self.groups==1 and original.kernel_size==(1,1) and original.padding==(0,0),"pointwise bias correction requires dense padding-free 1x1")
                weight=weight*self.input_scale[None,:,None,None]
        self.register_buffer("reference_weight", weight, persistent=False)
        self.register_buffer("decoded_weight", self.reference_weight.clone(), persistent=False)
        self.weight_record, self.weight_enabled = record, False
        if record is not None:
            self.decoded_weight = math_plugin.decode_weight(record, original.weight.device)

    def calibration_input(self, x):
        if self.transformed: x=x/self.input_scale[None,:,None,None]-self.input_shift[None,:,None,None]
        return self.quantize(x)

    def effective_bias(self, weight=None):
        if self.transformed and self.groups==1:
            w=self.decoded_weight if self.weight_enabled else self.reference_weight
            correction=(w if weight is None else weight)[:,:,0,0]@self.input_shift
            return correction if self.bias is None else self.bias+correction
        return self.bias

    def forward(self, x):
        self.calls += 1
        self.last_input_shape = list(x.shape)
        dtype = DTYPES[self.execution["compute_dtype"]]
        w = self.decoded_weight if self.weight_enabled else self.reference_weight
        bias = self.effective_bias()
        bias = None if bias is None else bias.to(dtype)
        return F.conv2d(self.calibration_input(x).to(dtype), w.to(dtype), bias,
                        self.stride, self.padding, self.dilation, self.groups).to(x.dtype)


def bind_weight_origins(model,plan):
    # A later grid change must start from the deployed fitted W when a continuous pre-rounding
    # parameter no longer exists. Do not recreate the original FP teacher in an optimized parent.
    updated={r["module"] for r in plan["layers"] if r.get("options",{}).get("qep")}
    updated.update(n for g in plan.get("activation_first",{}).get("groups",[]) for n in g["weight_modules"])
    updated.update(n for s in plan.get("structure",[]) for n in s.get("fit",{}).get("weight_modules",[]))
    for s in plan.get("postfit",[]):
        cfg=s.get("settings",{})
        if cfg.get("slice"):
            updated.update(cfg["slice"]["sites"])
        if s["kind"]=="graph_reconstruction":
            updated.update(n for g in cfg["groups"] for n in g["weight_modules"])
        if (s["kind"] in ("domain_affine_fit","merged_weight_fit","bit_shrinking") or
                s["kind"]=="block_reconstruction" and cfg.get("update")!="scale"):
            updated.update(r["module"] for r in plan["layers"] if r.get("weight") and r["module"].startswith(s["module"]+"."))
    qmap=quantizers(model)
    for n in updated:qmap[n].grid_edit_from_packed_state=True


def build_model(backend, plan, preprocess=None, weights=None, activations=None):
    model = deepcopy(backend.model).eval().requires_grad_(False)
    preprocess, weights, activations = preprocess or {}, weights or {}, activations or {}
    attention_targets = {l["module"] for l in plan["layers"] if l["kind"] in (*ATTENTION_KINDS, "attention_logits")}
    attention_targets.update(s["module"].rsplit(".",1)[0] for s in plan.get("postfit",()) if s["module"].endswith(("._ptq_qk","._ptq_probability")))
    install_attention(model, attention_targets)
    install_flows(model, plan["layers"])
    # 先替换有权重算子，再绑定输出 hook；配置数组的排列不能让 hook 丢失。
    for layer in sorted(plan["layers"], key=lambda l: l["kind"] not in WEIGHT_KINDS):
        name, sid = layer["module"], site_id(layer)
        original = model.get_submodule(name)
        if layer["kind"] in WEIGHT_KINDS:
            q = (PipelineLinear(original, layer, plan["execution"], preprocess.get(name), weights.get(name), activations.get(sid))
                 if layer["kind"] == "linear" else PipelineConv2d(original, layer, plan["execution"], weights.get(name), activations.get(sid),preprocess.get(name)))
            if layer["kind"] == "linear":
                q.bias_uses_decoded_weight = any(p["module"] == name and p["method"] in ("adalog_bias","noisy_shift") for p in plan["preprocess"])
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, q)
        else:
            q = ActivationQuantizer(layer, plan["execution"], activations.get(sid)).to(backend.device)
            if layer["kind"] in ATTENTION_KINDS:
                op = original._ptq_qk if layer["kind"] in ("attention_query", "attention_key") else original._ptq_probability
                attr = "quant_b" if layer["kind"] in ("attention_key", "attention_value") else "quant_a"
                require(not hasattr(op, attr), "MatMul 操作数量化器重复安装")
                op.add_module(attr, q)
            elif layer["kind"] in (*FLOW_KINDS, "attention_logits"):
                from .sites import observation_module
                observation_module(model, layer).add_module("quantizer", q)
            else:
                # 同一 LayerNorm 可有输入和输出两条独立边；不可共用一个属性。
                # hook 从其宿主读取模块，deepcopy 后也不会指回旧量化器对象。
                if layer["kind"] == "pre_layernorm":
                    require(not hasattr(original, "_ptq_pre_activation"), "输入边量化器重复安装")
                    original.add_module("_ptq_pre_activation", q)
                    original.register_forward_pre_hook(
                        lambda m, args: (m._ptq_pre_activation(args[0]), *args[1:]))
                else:
                    require(not hasattr(original, "_ptq_post_activation"), "输出边量化器重复安装")
                    original.add_module("_ptq_post_activation", q)
                    original.register_forward_hook(lambda m, args, output: m._ptq_post_activation(output))
    install_operators(model, plan)
    require(set(quantizers(model)) == {site_id(l) for l in plan["layers"]},
            "安装的量化器与配置不符", code="coverage_mismatch")
    # Replacing modules after deepcopy introduces fresh training=True children.
    # The frozen model contract applies recursively, including these operators.
    return model.eval().requires_grad_(False)


def quantizers(model):
    return {m.ptq_site_id: m for m in model.modules() if isinstance(m, ActivationQuantizer)}


def plan_uses_calibration(plan):
    """Whether this native plan fits any state from image activations."""
    if plan.get("structure") or plan.get("preprocess") or plan.get("postfit") or any(l.get("options",{}).get("qep") or l.get("weight_method") == "erq" for l in plan.get("layers",[])):
        return True
    for layer in plan.get("layers", []):
        for target in ("weight", "activation"):
            spec = layer.get(target)
            if not spec:
                continue
            if calibration(layer, target)["strategy"] == "coordinate":
                return True
            if target == "activation":
                if (plan["execution"]["activation_calibration"] == "static"
                        and spec["format"] not in RAW_FLOATS):
                    return True
            elif (layer.get("weight_method") in ("gptq", "gptaq", "adaround")
                  or calibration(layer, target)["strategy"] in ("search", "coordinate")):
                return True
    return False


def equivalence(backend, model, data):
    count, errors = 0, OutputErrors()
    for x, _ in data.batches("calibration"):
        errors.add(forward(backend, backend.model, x), forward(backend, model, x))
        count += 1
    result = errors.result()
    relative = result["relative_mse"]
    require(count == len(data.ids("calibration")) and relative is not None and relative <= 1e-8,
            f"FP 重参数化/attention 等价性不通过：relative MSE={relative}", code="fidelity_gate")
    return {"samples": count, **result, "maximum_relative_mse": 1e-8,
            "max_abs_error": max(v["max_abs_error"] or 0.0 for v in result["outputs"].values())}


class QuantizationExecutor:
    def __init__(self, backend, config, data, *, diagnostic=False):
        # 仅内部 calibration 诊断使用；不能导出模型，也不能进入正式候选。
        self.diagnostic = diagnostic
        self.backend, self.config, self.data = backend, config, data
        self.input_shapes, self.report = {}, {}
        self.cache_mib = config["loop"].get("calibration_cache_mib", 256)

    def preprocess(self, plan):
        values, reports = {}, []
        self.repq_activation_params = {}
        self.preview_state = None
        if any(p.get("observation") == "quantized_prefix" for p in plan["preprocess"]):
            # 一次性校准探针只生成 observer 状态；重参数化仍作用于原 FP 权重。
            # 不能拿已 QDQ 的权重去缩放，也不能重复拟合非目标层的 observer。
            preview = deepcopy(plan)
            preview["preprocess"] = []
            preps = {p["module"]: p for p in plan["preprocess"]}
            for layer in preview["layers"]:
                if layer["module"] in preps:
                    shape = self.input_shapes.get(site_id(layer))
                    require(shape is not None, "preview observer 缺少实际输入形状")
                    layer["activation"].update(granularity="per_channel", axis=len(shape) - 1, group_size=None)
                    layer["activation_calibration"] = deepcopy(preps[layer["module"]]["calibration"])
            probe = build_model(self.backend, preview)
            weights, params, report = self.fit_model(probe, preview)
            self.preview_state = weights, params
            reports.append({"observation": "quantized_prefix", "scope": "disposable calibration preview; original weights unchanged",
                            "calibration": report})
            del probe
            gc.collect()
            self.backend.reset()
        layer_map = {l["module"]: l for l in plan["layers"] if l["kind"] in WEIGHT_KINDS}
        for prep in plan["preprocess"]:
            name, layer = prep["module"], layer_map[prep["module"]]
            # 原 FP 只读 hook，逐边测量；不把全模型激活留在内存。
            norm_value, checked = [], [0]
            handles = []
            if prep["norm"] is not None:
                norm = self.backend.model.get_submodule(prep["norm"])
                handles.append(norm.register_forward_hook(lambda m, a, y: norm_value.append(y)))
                def check_edge(m, args):
                    require(norm_value and args[0].shape == norm_value[-1].shape and torch.equal(args[0], norm_value[-1]),
                            "指定 LN 输出不是该 Linear 的直接输入；不能强行重参数化")
                    norm_value.clear()
                    checked[0] += 1
                handles.append(self.backend.model.get_submodule(name).register_forward_pre_hook(check_edge))
            try:
                records, report = capture(self.backend, self.backend.model, layer, self.data, self.cache_mib)
            finally:
                for h in handles:
                    h.remove()
                norm_value.clear()
            x = combined_input(records).float()
            w = self.backend.model.get_submodule(name).weight.detach().cpu().float()
            options = ({"alpha": prep.get("alpha", 0.5)} if prep["method"] == "smoothquant" else
                       {"bits": layer["activation"]["bits"], "observer": prep.get("calibration")} if prep["method"] == "repq_ln" else {})
            if prep["method"]=="repq_ln":
                options.update(prep.get("settings",{}))
            if prep["method"]=="noisy_shift":
                options={"settings":prep["settings"],"spec":layer["activation"],"method":layer["activation_method"]}
            if prep["method"] in ("domain_affine","centered_smoothing","trio_dw_scale","trio_pw_shift"):
                options={"settings":prep["settings"]}
            if self.preview_state is not None:
                options["channel_params"] = self.preview_state[1][name]
            if prep["method"]=="repq_ln" and "learned_clip" in options:
                from .edge_transforms import bilateral_clip
                options["channel_params"],report["learned_clip"]=bilateral_clip(x,options["bits"],options.pop("learned_clip"))
            affine, activation = get_target_method("preprocess", prep["method"]).fit(x, w, **options)
            scale, shift = affine["scale"], affine["shift"]
            if activation is not None:
                self.repq_activation_params[name] = activation
            require(bool(torch.isfinite(scale).all()) and bool((scale > 0).all()) and bool(torch.isfinite(shift).all()), "重参数化参数非法")
            values[name] = {"scale": scale, "shift": shift}
            reports.append({"module": name, "method": prep["method"], "implementation": "explicit_edge_affine",
                            "norm_edge_calls_verified": checked[0], **report})
            del records, x, w
            gc.collect()
        return values, reports

    def fit_target(self, model, layer, target):
        sid, spec = site_id(layer), layer[target]
        module = model.get_submodule(layer["module"])
        c = calibration(layer, target)
        method = layer[target + "_method"]
        options = layer.get("options", {})
        need_data = target == "activation" or method in ("gptq", "gptaq", "adaround", "erq") or options.get("qep") or c["strategy"] == "search"
        need_data=need_data or (c["strategy"]=="weight_pot" and c["error_domain"]=="linear_output")
        records, report = [], {"samples": 0, "calls": 0, "cache_bytes": 0, "gradient_loss": None}
        autoscale=options.get("encoding",{}) if target=="activation" and method=="three_region" else {}
        gradient_loss=autoscale.get("gradient_loss") or (c["loss"] if c["metric"]=="hessian" else None)
        if need_data:
            records, report = capture(self.backend, model, layer, self.data, self.cache_mib,
                                      gradient_loss,
                                      observer_batch_size=getattr(self, "observer_batch_size", None))
            self.input_shapes[sid] = list(records[0]["x"].shape)
        teachers = None
        if method in ("gptaq", "adaround", "erq") or target == "weight" and options.get("qep"):
            teachers, teacher_report = capture(self.backend, self.backend.model, layer, self.data, self.cache_mib)
            require(report["cache_bytes"] + teacher_report["cache_bytes"] <= self.cache_mib * 1024 ** 2,
                    "成对 FP/量化缓存超过 calibration_cache_mib；不会减少样本", code="memory_budget")
            require(report["record_sample_ids"] == teacher_report["record_sample_ids"], "FP teacher/量化输入样本或调用顺序不一致")
            report["teacher"] = "original FP model module output; unchanged checkpoint/head"
            report["teacher_cache_bytes"] = teacher_report["cache_bytes"]
        if target == "weight" and (method == "erq" or options.get("qep")):
            from .paired_weights import correct_weight, erq
            qx = combined_input(records).to(module.reference_weight)
            fx = combined_input(teachers).to(qx) / module.input_scale
            if module.bias_uses_decoded_weight:
                qx = qx + module.input_shift
            else:
                fx = fx - module.input_shift
            if method == "erq":
                record, stages = erq(module.reference_weight,fx,qx,spec,options,module.bias)
                return record, {**report, **stages}
            settings = options["qep"]
            h = gram(records,self.backend.device)
            damping = settings["relative_damping"] * qx.double().reshape(-1,qx.shape[-1]).square().sum(0).mean()
            w, _ = correct_weight(module.reference_weight,fx,qx,alpha=settings["alpha"],damping=float(damping),normalization="sum")
            if method == "gptq":
                record, _ = math_plugin.gptq(w,h,spec,options)
            else:
                require(method == "rtn", "QEP adapter supports RTN and GPTQ")
                record = codecs.encode_weight(w,spec)
            return record, {**report,"correction":"qep","qep":settings,"actual_damping":float(damping)}
        if method == "gptaq":
            from .gptaq import statistics
            # 使用同一原始图像的两条真实输入流。AdaLog 的 bias 随 W 改变时，
            # H/dX 必须针对去掉平移后的有效输入，避免重复计算 bias 补偿。
            for row, teacher in zip(records, teachers):
                teacher["x"] = teacher["x"] / module.input_scale.cpu()
                if module.bias_uses_decoded_weight:
                    row["x"] = row["x"] + module.input_shift.cpu()
                else:
                    teacher["x"] = teacher["x"] - module.input_shift.cpu()
            errors = OutputErrors()
            with torch.no_grad():
                bias = module.bias
                if module.bias_mask is not None and bias is not None:
                    bias = bias * module.bias_mask
                if not module.bias_uses_decoded_weight:
                    correction = module.reference_weight @ module.input_shift
                    bias = correction if bias is None else bias + correction
                for teacher in teachers:
                    aligned = F.linear(teacher["x"].to(self.backend.device), module.reference_weight, bias)
                    errors.add(teacher["y"], aligned.cpu())
            report["original_fp_output_alignment"] = errors.result()
            require(report["original_fp_output_alignment"]["relative_mse"] <= 1e-8,
                    "GPTAQ 的 FP 输入/权重/bias 不能重构原始 FP teacher 输出", code="fidelity_gate")
            with math_plugin.fp32_statistics(self.backend.device):
                h, dx = statistics(records, teachers, self.backend.device)
                record, decoded = get_target_method("weight", method).fit(module.reference_weight, h, dx, spec, options)
                report["input_difference_frobenius"] = float(dx.norm())
            report.update(alpha=0.25, compensation="triangular original-FP input difference update",
                          statistics_precision="FP32; TF32 disabled", calibrated_input="actual quantized path")
            return record, report
        if method=="hc_rows":
            from .weight_codebook import fit
            return fit(module.reference_weight,spec,options["row_codebook"])
        if method == "gptq":
            with math_plugin.fp32_statistics(self.backend.device):
                h = gram(records, self.backend.device)
                record, decoded = get_target_method("weight", method).fit(module.reference_weight, h, spec, options)
            report["statistics_precision"] = "FP32; TF32 disabled for Gram/error updates only"
            del h, decoded, records
            return record, report
        flexible=None
        if target=="activation" and method in grids.METHODS and self.data.identity.get("task_calibration"):
            if spec["granularity"]=="per_tensor":flexible="scalar"
            elif spec["granularity"]=="per_channel":flexible=spec["axis"]
        x = module.reference_weight.detach().cpu() if target == "weight" else combined_input(records,flexible_axis=flexible)
        if target == "activation" and method == "rtn" and spec["granularity"] in ("per_token","per_group"):
            from .fixed_grids import fit
            if c["strategy"]=="first_batch_rows":
                from .fixed_grids import first_batch_rows
                ids=list(dict.fromkeys(report["record_sample_ids"]))
                count=c["first_batch_images"]
                require(count<=len(ids),"first calibration batch exceeds bound data")
                chosen=set(ids[:count])
                first=combined_input([r for r,i in zip(records,report["record_sample_ids"]) if i in chosen])
                return first_batch_rows(first,spec,c["percentiles"]),{**report,"first_batch_ids":ids[:count],"percentiles":c["percentiles"]}
            require(c["strategy"] == "minmax", "fixed RTN grids require a minmax calibration")
            return fit(x,spec), {**report,"fixed_feature_axis":spec["axis"],"position_shape":list(x.shape[1:])}
        if method == "ptf" and getattr(self, "observer_batch_size", None):
            options = dict(options, score_input=combined_input(records[-report["last_batch_records"]:]))
        grid_method = options["rounding_grid"] if method == "adaround" else method
        if method == "adalog":
            # 独立激活插件可保留显式 shift；启用 bias 组件时输入已经平移。
            from .sites import site_catalog
            semantic = site_catalog(self.backend)[sid]["semantic"]
            options = dict(options, _shift=(math_plugin.GELU_NEGATIVE_BOUND if semantic == "post_gelu"
                                           and not getattr(module, "bias_uses_decoded_weight", False) else 0.0))
        if autoscale.get("gradient_loss"):
            from . import region_grids
            def score_region(p):
                return reconstruction_score(layer,module,records,lambda v:region_grids.apply(v,spec,method,p,autoscale),"hessian")
            params=region_grids.fit(x,spec,method,autoscale,candidate_loss=score_region)
            report["autoscale_objective"]={"metric":"squared FC output deviation times squared observed task gradient","loss":gradient_loss,
                                           "selected_score":score_region(params),"reference":"current prefix and W at calibration"}
        else:
            params = (grids.fit(x, spec, grid_method) if grid_method in grids.METHODS else fit_rtn(x, spec)) if target == "weight" else get_target_method("activation", method).fit(x, spec, options=options)
        if target == "activation" and getattr(self, "repq_activation_params", {}).get(layer["module"]) is not None:
            params = {k: v.clone() for k, v in self.repq_activation_params[layer["module"]].items()}
        trials = []
        if target == 'activation' and options.get('token_partition') and c['strategy'] == 'percentile':
            from .grids_partition import fit
            params = fit(x, spec, options['token_partition'], c['percentiles'])
        elif c["strategy"]=="weight_pot":
            inputs=combined_input(records) if c["error_domain"]=="linear_output" else None
            params,report["power_search"]=grids.nearest_power_grid(x,spec,method,params,inputs,neighbors=c.get("neighbors",1))
        elif c["strategy"] == "percentile":
            params = (grids.fit(x, spec, method, percentiles=c["percentiles"]) if method == "uniform_affine" else
                      grids.percentile_log(x, spec, lambda value, shape, params: get_target_method("activation", method).apply(
                          value, shape, params=params, options=options), c["percentiles"]))
        elif c["strategy"]=="fpcs":
            from .research_search import fpcs
            from .adalog import parameters
            def make(q,factor):
                return parameters(params["scale"]*factor,spec["bits"],q,params["shift"].item())
            def measure(q,factor):
                trial=make(q,factor)
                return reconstruction_score(layer,module,records,lambda v:get_target_method("activation",method).apply(
                    v,spec,params=trial,options=options),c["metric"],weight=False)
            selected,report["fpcs"]=fpcs([c["base_values"],c["factors"]],measure,beam=c["beam"],
                refinements=c["refinements"],subdivisions=c["subdivisions"],integer_axes=(0,))
            params=make(*selected)
        elif c["strategy"] == "manual":
            params = math_plugin.adjusted_params(params, 1.0, method, manual=c["scale_values"], zero=c.get("zero_values"))
        elif c["strategy"] == "search":
            factors = c["factors"]
            if method == "twin_softmax" and not {"factors", "factor_grid"}.intersection(layer.get(target + "_calibration", {})):
                factors = [2.0 ** (-i) / params["split"].item() for i in range(20)]
            best, best_params = float("inf"), None
            bases = options.get("log_base_candidates", [options.get("log_base_q", 37)]) if method == "adalog" else [None]
            for base, factor in ((base, factor) for base in bases for factor in factors):
                trial = math_plugin.adjusted_params(params, factor, method)
                if method == "adalog":
                    from .adalog import parameters
                    trial = parameters(trial["scale"], spec["bits"], base, params["shift"].item())
                if target == "weight":
                    qw = grids.apply(x, spec, grid_method, trial) if grid_method in grids.METHODS else rtn_qdq(x, spec, trial)
                    transform = lambda _, w=qw: w
                else:
                    transform = lambda a, p=trial: get_target_method("activation", method).apply(a, spec, params=p, options=options)
                loss = reconstruction_score(layer, module, records, transform, c["metric"], weight=target == "weight")
                trials.append({"factor": factor, "score": loss, **({"base_q": base} if base is not None else {})})
                if loss < best:
                    best, best_params = loss, trial
            require(best_params is not None, "scale 搜索没有有限结果")
            params = best_params
        if method == "adaround":
            result, optimization = get_target_method("weight", method).fit(
                module.reference_weight, spec, params, records, teachers,
                lambda a, w: F.linear(a, w, module.effective_bias(w)), options)
            report.update(strategy=c["strategy"], metric=c["metric"], candidates=trials, rounding=optimization)
            return result, report
        if target == "activation" and "repq_affine" in getattr(self, "repq_activation_params", {}).get(layer["module"], {}):
            # percentile 可重新拟合变换后的共享尺度，但不能丢失仿射执行语义。
            # 只补标记，不修改拟合的 scale/zero；原 quantized_prefix 路径复用既有参数。
            params["repq_affine"] = torch.tensor(True)
        result = get_target_method("weight", method).fit(x, spec, params) if target == "weight" else {k: v.cpu() for k, v in params.items()}
        from .calibration import parameter_evidence
        report.update(strategy=c["strategy"], metric=c["metric"], candidates=trials,
                      selected_candidate=min(trials, key=lambda row: row["score"]) if trials else None,
                      selected_parameters=parameter_evidence(params))
        return result, report

    def fit_model(self, model, plan, *, initial_weights=None, initial_params=None, reset_weights=()):
        """通用校准调度。顺序模式用真实 forward 顺序，不能把 JSON 顺序当拓扑。"""
        weights, params = dict(initial_weights or {}), dict(initial_params or {})
        self.observer_batch_size = plan["execution"].get("observer_batch_size")
        reports, qmap = [], quantizers(model)
        groups = {}
        for layer in plan["layers"]:
            groups.setdefault(operator_id(layer), []).append(layer)
        if plan_uses_calibration(plan):
            seen = []
            handles = [q.register_forward_pre_hook(lambda m, a: seen.append(m.ptq_site_id)) for q in qmap.values()]
            try:
                x, _ = next(iter(self.data.batches("calibration")))
                forward(self.backend, model, x)
            finally:
                for h in handles:
                    h.remove()
            positions = {sid: seen.index(sid) for sid in qmap if sid in seen}
            require(set(positions) == set(qmap), "校准拓扑未覆盖所有量化位点")
            groups = sorted(groups.values(), key=lambda rows: min(positions[site_id(l)] for l in rows))
        else:
            # Independent weight fitting (or a pure FP control) has no activation
            # dependence whose topological order requires a calibration forward.
            groups = list(groups.values())
        sequential = plan["execution"].get("within_stage", "parallel") == "sequential"
        if plan["execution"].get("layer_order", "forward") == "reverse":
            groups.reverse()
        order = plan["execution"]["order"]
        blocks = plan["execution"].get("block_modules", [])
        units = []
        for rows in groups:
            matches = {b for b in blocks for l in rows if l["module"] == b or l["module"].startswith(b + ".")}
            require(len(matches) <= 1, "同一算子跨越不同 block")
            key = next(iter(matches), None)
            if key is not None:
                model.get_submodule(key)  # reject nonexistent block paths before fitting
            if key is not None and units and units[-1][0] == key:
                units[-1][1].append(rows)
            else:
                units.append((key, [rows]))
        require(all(sum(k == b for k, _ in units) == 1 for b in blocks),
                "block 位点在真实前向拓扑中必须连续，不能把命名分组伪称 block 调度")
        pass_index = 0
        stages = ("activation", "weight") if order == "a_then_w" else ("weight", "activation")

        installed_params = {}

        def install():
            for layer in plan["layers"]:
                sid, q = site_id(layer), qmap[site_id(layer)]
                if layer["module"] in weights and isinstance(q, PipelineLinear):
                    record = weights[layer["module"]]
                    # 已拟合记录不可原地修改；只解码新记录，避免逐位点反复解码全模型。
                    if q.weight_record is not record:
                        q.weight_record = record
                        q.decoded_weight = math_plugin.decode_weight(record, self.backend.device)
                if sid in params and installed_params.get(sid) is not params[sid]:
                    q.set_params(params[sid])
                    installed_params[sid] = params[sid]

        def fit(layer, target):
            sid = site_id(layer)
            if layer[target] is None:
                return
            if pass_index == 0 and target == "weight" and layer["module"] in weights and layer["module"] not in reset_weights:
                return
            if pass_index == 0 and target == "activation" and sid in params:
                return
            print(f"calibration {sid} {target}: {layer[target + '_method']}", flush=True)
            if target == "activation" and (plan["execution"]["activation_calibration"] == "dynamic" or layer[target]["format"] in RAW_FLOATS):
                result, report = {}, {"samples": 0, "strategy": "runtime_minmax_or_raw_cast"}
            else:
                flags = [(q, q.enabled, getattr(q, "weight_enabled", None)) for q in qmap.values()]
                try:
                    # Re-estimation observes the current peers, but never QDQ's the
                    # target twice or reconstructs its already quantized output.
                    if plan["execution"].get("calibration_context", "current") == "float":
                        for q, _, w in flags:
                            q.enabled = False
                            if w is not None:
                                q.weight_enabled = False
                    q = qmap[sid]
                    if target == "activation":
                        q.enabled = False
                    else:
                        q.weight_enabled = False
                    result, report = self.fit_target(model, layer, target)
                finally:
                    for q, enabled, weight_enabled in flags:
                        q.enabled = enabled
                        if weight_enabled is not None:
                            q.weight_enabled = weight_enabled
            if target == "weight":
                weights[layer["module"]] = result
            elif result:
                params[sid] = result
            reports.append({"site": sid, "target": target, "method": layer[target + "_method"],
                            "pass": pass_index + 1, "calibration_unit": plan["execution"].get("calibration_unit", "operator"),
                            "block": next((b for b in blocks if layer["module"] == b or layer["module"].startswith(b + ".")), None),
                            "layer_order": plan["execution"].get("layer_order", "forward"),
                            "context": plan["execution"].get("calibration_context", "current"),
                            "reference": "target disabled on actual observed input; peers follow context", **report})
            install()
            bind_calibration_operators(model, plan, params)
            gc.collect()
            self.backend.reset()

        def enable(rows, targets):
            for layer in rows:
                q = qmap[site_id(layer)]
                for target in targets:
                    if target == "activation":
                        q.enabled = layer[target] is not None
                    elif isinstance(q, PipelineLinear):
                        q.weight_enabled = layer[target] is not None

        def run_groups(selected_groups):
            if order == "joint":
                for rows in selected_groups:
                    if any(calibration(l, t)["strategy"] == "coordinate" for l in rows for t in stages if l[t]):
                        flags = [(q, q.enabled, getattr(q, "weight_enabled", None)) for q in qmap.values()]
                        try:
                            if plan["execution"].get("calibration_context", "current") == "float":
                                for q, _, w in flags:
                                    q.enabled = False
                                    if w is not None:
                                        q.weight_enabled = False
                            for layer in rows:
                                q = qmap[site_id(layer)]
                                q.enabled = False
                                if isinstance(q, PipelineLinear):
                                    q.weight_enabled = False
                            w, a, report = fit_operator(self, model, rows)
                        finally:
                            for q, enabled, weight_enabled in flags:
                                q.enabled = enabled
                                if weight_enabled is not None:
                                    q.weight_enabled = weight_enabled
                        weights.update(w)
                        params.update(a)
                        install()
                        reports.append({"sites": [site_id(l) for l in rows], "pass": pass_index + 1,
                                        "context": plan["execution"].get("calibration_context", "current"), **report})
                    else:
                        for layer in rows:
                            for target in stages:
                                fit(layer, target)
                    if sequential:
                        enable(rows, stages)
            else:
                for target in stages:
                    for rows in selected_groups:
                        for layer in rows:
                            fit(layer, target)
                        if sequential:
                            enable(rows, (target,))
                    enable([l for rows in selected_groups for l in rows], (target,))

        install()
        for pass_index in range(plan["execution"].get("reestimate_passes", 1)):
            if plan["execution"].get("calibration_unit", "operator") == "block":
                for _, unit_groups in units:
                    run_groups(unit_groups)
                    enable([l for rows in unit_groups for l in rows], stages)
            else:
                run_groups(groups)
            enable(plan["layers"], stages)
        return weights, params, reports

    def prepare(self, plan, shapes):
        if plan.get("activation_first"):
            from .graph_fit import activation_phase,restore_source
            source=self.backend
            proxy,receipt=activation_phase(source,self.config,self.data,plan)
            reduced={k:v for k,v in plan.items() if k!="activation_first"}
            try:
                self.backend=proxy
                self.reference_backend=source
                model,weights,params,_=self.prepare(reduced,shapes)
            finally:
                self.backend=source
                self.reference_backend=None
            restore_source(model,source)
            bind_weight_origins(model,plan)
            self.report["activation_first"]=receipt
            return model,weights,params,self.audit(model,plan)
        if not plan.get("structure"):
            return self._prepare(plan,shapes)
        from copy import copy
        from . import structure
        from ..backends.server import state_sha256
        original=self.backend
        proxy=copy(original)
        proxy.model=deepcopy(original.model)
        states,receipts={},[]
        for stage in plan["structure"]:
            if stage["kind"]=="calibrated_operator":
                from .nonlinear_calibration import fit
                state,ranking=fit(proxy,self.data,stage,self.cache_mib)
                states[stage["module"]]=state
                receipts.append({"module":stage["module"],"operator_selection":ranking,"images":list(self.data.ids("calibration"))})
            elif stage["kind"]=="integer_function":
                from .integer_functions import fit
                state,ranking=fit(proxy,self.data,stage,self.cache_mib)
                states[stage["module"]]=state
                receipts.append({"module":stage["module"],"operator_selection":ranking,"images":list(self.data.ids("calibration"))})
            elif stage["kind"]=="cached_kv":
                state,rows=structure.token_bank(proxy,self.data,stage)
                states[stage["module"]]=state
                receipts.append({"module":stage["module"],"candidate_tokens":rows,"images":list(self.data.ids("calibration"))})
            structure.configure(proxy.model,[stage],states)
            if stage.get("fit"):
                from .curvature_fit import mlp_replacement
                receipts.append(mlp_replacement(original,proxy,self.data,stage))
        proxy.original_state=state_sha256(proxy.model)
        try:
            self.backend=proxy
            result=self._prepare(plan,shapes)
            self.report["structure"]={"fit_before_observers":True,"stages":receipts,"reference":"structurally modified FP model"}
        finally:self.backend=original
        if any(s.get("fit") for s in plan["structure"]):
            from .graph_fit import restore_source
            restore_source(result[0],original)
        return result

    def _prepare(self, plan, shapes):
        from .capabilities import validate_plan, detect
        from .contracts import normalize_plan
        plan = normalize_plan(plan)
        validate_plan(self.backend, plan, self.config, detect(self.backend), shapes,
                      diagnostic=self.diagnostic, allow_unknown_shapes=True)
        # 只读真实形状探针；preview 需要按实际 feature 轴拟合，不猜 token 维。
        if any(p.get("observation") == "quantized_prefix" for p in plan["preprocess"]):
            probe_plan = deepcopy(plan)
            probe_plan["preprocess"] = []
            probe = build_model(self.backend, probe_plan)
            equivalence(self.backend, probe, self.data)
            self.input_shapes = {sid: q.last_input_shape for sid, q in quantizers(probe).items()}
            del probe
        preprocess, prep_report = self.preprocess(plan)
        model = build_model(self.backend, plan, preprocess)
        # 等价性 gate 使用 FP32，随后恢复用户声明的 compute dtype。该检查不包含量化误差。
        for q in quantizers(model).values():
            q.execution["compute_dtype"] = "FP32"
        if plan_uses_calibration(plan):
            with semantic_edge_checks(self.backend, model, plan) as edge_calls:
                fidelity = equivalence(self.backend, model, self.data)
            fidelity["semantic_edge_calls"] = edge_calls
        else:
            fidelity = {"performed": False, "reason": "No image-dependent fitting; state audit and evaluation remain applicable."}
            if any(l["activation"] and l["activation"]["granularity"] != "per_tensor" for l in plan["layers"]):
                allowed = getattr(self.data, "shape_observation_splits", ())
                require(allowed and all(s in ("calibration", "search") for s in allowed),
                        "宿主未声明合法 shape 观察数据视图；不能猜 shape 或读取 final", code="invalid_data")
                if self.diagnostic:
                    require(all(s == "calibration" for s in allowed),
                            "插件诊断 shape 观察不得读取 search/final", code="invalid_data")
                split = next((s for s in allowed if self.data.ids(s)), None)
                require(split is not None, "shape 观察的宿主允许数据视图为空", code="invalid_data")
                sample = next(iter(self.data.batches(split)), None)
                require(sample is not None, "形状准备需要真实输入样本", code="invalid_data")
                forward(self.backend, model, sample[0])
                fidelity["shape_observation"] = {"split": split, "sample_ids": [sample[1]],
                                                 "sample_count": 1, "purpose": "shape_validation_only",
                                                 "calibration_fitting": False}
        self.input_shapes = {sid: q.last_input_shape for sid, q in quantizers(model).items() if hasattr(q, "last_input_shape")}
        for q in quantizers(model).values():
            q.execution = deepcopy(plan["execution"])
        validate_plan(self.backend, plan, self.config, detect(self.backend), self.input_shapes, diagnostic=self.diagnostic)
        initial_weights, initial_params = self.preview_state if self.preview_state is not None else ({}, {})
        if self.preview_state is not None:
            initial_params.update(self.repq_activation_params)
        weights, params, reports = self.fit_model(model, plan, initial_weights=initial_weights,
                                                 initial_params=initial_params, reset_weights=preprocess)
        bind_operators(model, plan, quantizers(model))
        from .window_routes import activate
        activate(model,plan.get("structure",[]))
        self.report = {"order": plan["execution"]["order"], "within_stage": plan["execution"].get("within_stage", "parallel"),
                       "preprocessing": prep_report, "fidelity": fidelity, "calibration": reports}
        if plan.get("postfit"):
            from . import output_correction
            self.report["postfit"] = []
            for stage in plan["postfit"]:
                if stage["kind"]=="graph_reconstruction":
                    from .graph_fit import optimize
                    _,trace=optimize(getattr(self,"reference_backend",None) or self.backend,model,self.data,stage["settings"])
                    receipt={"kind":stage["kind"],"groups":trace}
                elif stage["kind"] in output_correction.CORRECTIONS:
                    receipt = output_correction.fit(self.backend, model, self.data, stage)
                else:
                    from .reconstruction import fit_stage
                    receipt = fit_stage(self.backend, model, self.data, stage, self.cache_mib)
                self.report["postfit"].append(receipt)
        bind_weight_origins(model,plan)
        self.preprocess_state = preprocess
        validate_plan(self.backend, plan, self.config, detect(self.backend), self.input_shapes, diagnostic=self.diagnostic)
        audit = self.audit(model, plan)
        return model, weights, params, audit

    def audit(self, model, plan):
        from .contracts import normalize_plan
        plan = normalize_plan(plan)
        qmap = quantizers(model)
        from . import output_correction
        output_correction.audit(model, plan)
        from .structure import fitted_state
        fitted_state(model,plan.get("structure",[]))
        audit_operators(model, plan, qmap)
        require(set(qmap) == {site_id(l) for l in plan["layers"]}, "插件实际位点与配置不符", code="coverage_mismatch")
        selected = {l["module"] + ".weight" for l in plan["layers"] if l["kind"] in WEIGHT_KINDS}
        current = model.state_dict()
        for name, tensor in self.backend.model.state_dict().items():
            if name not in selected:
                require(name in current and tensor_digest(tensor) == tensor_digest(current[name]), f"非目标状态改变：{name}", code="protection_failed")
        for layer in plan["layers"]:
            q = qmap[site_id(layer)]
            require(q.layer == layer and q.execution == plan["execution"], "运行规格与提交配置不同")
            require(q.enabled == (layer["activation"] is not None), "激活量化状态不符")
            if isinstance(q, PipelineLinear):
                require(q.weight_enabled == (layer["weight"] is not None), "权重量化状态不符")
                source = self.backend.model.get_submodule(layer["module"]).weight.detach()
                scale=q.input_scale
                if layer["kind"]=="conv2d" and scale.ndim:
                    scale = scale[:,None,None,None] if q.groups!=1 else scale[None,:,None,None]
                require(torch.equal(q.reference_weight, source * scale), "预处理参考权重被改变", code="protection_failed")
                require(bool((q.input_scale > 0).all()) and bool(torch.isfinite(q.input_shift).all()), "预处理参数非法")
                if layer["kind"] == "linear":
                    decoded_bias = any(p["module"] == layer["module"] and p["method"] in ("adalog_bias","noisy_shift") for p in plan["preprocess"])
                    require(q.bias_uses_decoded_weight == decoded_bias, "AdaLog bias 权重依赖与配置不符")
                    if any(p["module"] == layer["module"] and p["method"]=="adalog_bias" for p in plan["preprocess"]):
                        require(bool((q.input_scale == 1).all()) and bool((q.input_shift == -math_plugin.GELU_NEGATIVE_BOUND).all()),
                                "AdaLog bias 输入仿射不符")
                if not any(p["module"] == layer["module"] for p in plan["preprocess"]):
                    require(bool((q.input_scale == 1).all()) and bool((q.input_shift == 0).all()), "未声明的重参数化")
                if layer["weight"]:
                    require(q.weight_record.get("method", "rtn") == layer["weight_method"], "权重算法与实际编码不符")
                    if layer["weight_method"] == "adaround":
                        require(q.weight_record["grid_method"] == layer["options"]["rounding_grid"], "AdaRound 依赖网格不符")
                    math_plugin.validate_weight(q.weight_record, layer["weight"], list(q.reference_weight.shape))
                    require(torch.equal(q.decoded_weight, math_plugin.decode_weight(q.weight_record, q.decoded_weight.device)), "解码权重与编码不同")
            if layer["activation"] and layer["activation"]["format"] not in RAW_FLOATS and plan["execution"]["activation_calibration"] == "static":
                math_plugin.validate_activation(q.params(), layer, self.input_shapes[site_id(layer)])
        require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()), "候选模型未冻结")
        return {"configuration_hash": fingerprint(plan), "coverage_exact": True,
                "unselected_state_unchanged": True, "protected_head_neck_unchanged": True,
                "official_classifier_quantized": any(l["module"] == "head" for l in plan["layers"]),
                "all_parameters_frozen": True, "modules": list(qmap), "execution": deepcopy(plan["execution"])}

    def export(self, model, plan, records, params, directory, audit):
        from .contracts import normalize_plan
        plan = normalize_plan(plan)
        require(not self.diagnostic, "局部诊断模型不得导出为正式量化模型")
        require(audit == self.audit(model, plan), "导出前审计不一致")
        path = Path(directory)
        require(not path.exists(), "产物目录已存在，拒绝覆盖", code="run_exists")
        path.mkdir(parents=True)
        # Selection can replace a grid or weight record after prepare. Serialize the
        # audited live model, including affine transforms, even with old-style callers.
        qmap = quantizers(model)
        records = {row["module"]: qmap[site_id(row)].weight_record for row in plan["layers"] if row["weight"]}
        params = {site_id(row): qmap[site_id(row)].params() for row in plan["layers"] if row["activation"]
                  and row["activation"]["format"] not in RAW_FLOATS and plan["execution"]["activation_calibration"] == "static"}
        preprocess = {p["module"]: {"scale": qmap[p["module"]].input_scale.detach().cpu(),
                                     "shift": qmap[p["module"]].input_shift.detach().cpu()} for p in plan["preprocess"]}
        removed = {name + ".weight" for name in records}
        state = {k: v.detach().cpu().clone() for k, v in self.backend.model.state_dict().items() if k not in removed}
        payload = {"state": state, "weights": records, "activations": params, "preprocess": preprocess}
        if plan.get("activation_first") or any(s.get("fit") for s in plan.get("structure",())) or any(s.get("settings",{}).get("mean_bias_sites") for s in plan.get("postfit",())):
            payload["fitted_input_bias"]={n:q.fitted_input_bias.cpu() for n,q in qmap.items() if hasattr(q,"fitted_input_bias")}
        if plan.get("structure"):
            from .structure import fitted_state
            payload["structure"]=fitted_state(model,plan["structure"])
        if plan.get("postfit"):
            from . import output_correction
            payload["corrections"] = output_correction.audit(model,plan)
        handle, tmp = tempfile.mkstemp(dir=path, suffix=".tmp")
        os.close(handle)
        try:
            torch.save(payload, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, path / "weights.pt")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        def tensor_bytes(value):
            if torch.is_tensor(value):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(v) for v in value.values())
            return 0
        manifest = seal({
            "schema_version": 2, "status": "exported_pending_validation", "identity": self.backend.identity(),
            "plan": deepcopy(plan), "configuration_hash": fingerprint(plan), "implementation": PLUGIN_VERSION,
            "plugin_versions": plugin_versions(),
            "input_shapes": self.input_shapes,
            "runtime_config": {k: deepcopy(self.config[k]) for k in ("backend", "constraints", "loop")},
            "weights_sha256": file_sha256(path / "weights.pt"), "tensor_payload_bytes": tensor_bytes(payload),
            "file_bytes": (path / "weights.pt").stat().st_size,
            "calibration_ids": self.data.ids("calibration") if plan_uses_calibration(plan) else [],
            "calibration_report": self.report, "audit": audit,
            "representation": "reference_qdq; packed weight codes; explicit activation grids and affine preprocessing; no native speedup claim",
        })
        atomic_json(path / "manifest.json", manifest)
        return manifest

    def load(self, directory):
        from .capabilities import validate_plan, detect
        from .plugin_math import validate_activation
        path = Path(directory)
        m = read_json(path / "manifest.json")
        require(m.get("content_sha256") == fingerprint({k: v for k, v in m.items() if k != "content_sha256"}), "产物 manifest 损坏")
        expected, actual = deepcopy(m["identity"]), self.backend.identity()
        if "config" in expected:
            expected["config"]["device"] = actual["config"]["device"]
        require(expected == actual, "产物模型/数据/源码身份不匹配", code="identity_mismatch")
        require(m["schema_version"] == 2 and m["implementation"] == PLUGIN_VERSION, "插件实现版本不匹配")
        versions = plugin_versions()
        require(all(versions.get(target, {}).get(name) == version
                    for target, plugins in m["plugin_versions"].items() for name,version in plugins.items()),
                "数值插件版本不匹配")
        for layer in m["plan"]["layers"]:
            for target in ("activation","weight"):
                if layer[target]:
                    require(layer[target+"_method"] in m["plugin_versions"][target], "missing fitted plugin version")
        require(m["weights_sha256"] == file_sha256(path / "weights.pt"), "产物 tensor hash 不匹配")
        plan = m["plan"]
        require(m["configuration_hash"] == fingerprint(plan), "产物配置 hash 不符")
        validate_plan(self.backend, plan, self.config, detect(self.backend), m["input_shapes"])
        payload = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        fields = {"state", "weights", "activations", "preprocess"} | ({"corrections"} if plan.get("postfit") else set()) | ({"structure"} if plan.get("structure") else set())
        if plan.get("activation_first") or any(s.get("fit") for s in plan.get("structure",())) or any(s.get("settings",{}).get("mean_bias_sites") for s in plan.get("postfit",())):fields.add("fitted_input_bias")
        require(set(payload) == fields, "产物字段不符")
        weight_names = {l["module"] for l in plan["layers"] if l["weight"]}
        static = {site_id(l): l for l in plan["layers"] if l["activation"] and l["activation"]["format"] not in RAW_FLOATS and plan["execution"]["activation_calibration"] == "static"}
        require(set(payload["weights"]) == weight_names and set(payload["activations"]) == set(static), "量化状态覆盖不符")
        base = self.backend.model.state_dict()
        require(set(payload["state"]) == set(base) - {n + ".weight" for n in weight_names}, "原状态覆盖不符")
        for name, value in payload["state"].items():
            require(tensor_digest(value) == tensor_digest(base[name]), f"非量化状态被改变：{name}")
        require(set(payload["preprocess"]) == {p["module"] for p in plan["preprocess"]}, "重参数化覆盖不符")
        for name, prep in payload["preprocess"].items():
            original = self.backend.model.get_submodule(name)
            count = original.in_channels if isinstance(original,nn.Conv2d) else original.in_features
            require(set(prep) == {"scale", "shift"}, "重参数化字段不符")
            require(all(v.shape == (count,) and bool(torch.isfinite(v).all()) for v in prep.values()), "重参数化形状/有限性不符")
            require(bool((prep["scale"] > 0).all()), "重参数化 scale 非正")
        for layer in plan["layers"]:
            if layer["weight"]:
                record = payload["weights"][layer["module"]]
                require(record.get("method", "rtn") == layer["weight_method"], "权重算法编码不符")
                math_plugin.validate_weight(record, layer["weight"], list(self.backend.model.get_submodule(layer["module"]).weight.shape))
        for sid, layer in static.items():
            is_repq = any(p["module"] == layer["module"] and p["method"] == "repq_ln" for p in plan["preprocess"])
            require(("repq_affine" in payload["activations"][sid]) == is_repq, "RepQ 激活网格与预处理方案不一致")
            validate_activation(payload["activations"][sid], layer, m["input_shapes"][sid])
        model = build_model(self.backend, plan, payload["preprocess"], payload["weights"], payload["activations"])
        if "fitted_input_bias" in payload:
            permitted={n for group in plan.get("activation_first",{}).get("groups",[]) for n in group["weight_modules"]}
            permitted.update(n for s in plan.get("structure",[]) if s.get("fit") for n in s["fit"]["weight_modules"])
            permitted.update(n for s in plan.get("postfit",[]) for n in s.get("settings",{}).get("mean_bias_sites",[]))
            for n,bias in payload["fitted_input_bias"].items():
                q=quantizers(model)[n]
                require(n in permitted and bias.shape==(q.out_features,) and bool(torch.isfinite(bias).all()),"invalid folded calibration bias")
                q.register_buffer("fitted_input_bias",bias.to(q.reference_weight))
        if plan.get("structure"):
            from .structure import configure
            require(set(payload["structure"])=={s["module"] for s in plan["structure"] if s["kind"] in ("cached_kv","integer_function","calibrated_operator")},"KV structures differ")
            configure(model,plan["structure"],payload["structure"])
            from .window_routes import activate
            activate(model,plan["structure"])
        for q in quantizers(model).values():
            q.enabled = q.layer["activation"] is not None
            if isinstance(q, PipelineLinear):
                q.weight_enabled = q.layer["weight"] is not None
        bind_operators(model, plan, quantizers(model))
        if plan.get("postfit"):
            from . import output_correction
            corrections = {s["module"]:s["kind"] for s in plan["postfit"] if s["kind"] in output_correction.CORRECTIONS}
            require(set(payload["corrections"]) == set(corrections), "correction weights missing")
            for name, kind in corrections.items():
                state = payload["corrections"][name]
                fields = {"bias","weight"} if kind == "qwt" else {"bias","alpha"} if kind == "channel_affine" else {"bias"}
                if any(s["module"]==name and s.get("settings",{}).get("integer_fold") for s in plan["postfit"]):
                    from .integer_affine import FIELDS
                    fields=FIELDS
                require(set(state) == fields and all(bool(torch.isfinite(v).all()) for v in state.values()), "invalid correction state")
                output_correction.install(model,name,kind,state)
        self.input_shapes = m["input_shapes"]
        bind_weight_origins(model,plan)
        self.audit(model, plan)
        return model, m
