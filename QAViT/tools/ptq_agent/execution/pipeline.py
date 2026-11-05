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
from . import grids
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
    def __init__(self, original, layer, execution, record=None, params=None):
        ActivationQuantizer.__init__(self, layer, execution, params)
        self.stride, self.padding, self.dilation, self.groups = original.stride, original.padding, original.dilation, original.groups
        self.register_buffer("bias", None if original.bias is None else original.bias.detach().clone())
        self.register_buffer("input_scale", original.weight.new_tensor(1.))
        self.register_buffer("input_shift", original.weight.new_tensor(0.))
        self.register_buffer("reference_weight", original.weight.detach().clone(), persistent=False)
        self.register_buffer("decoded_weight", self.reference_weight.clone(), persistent=False)
        self.weight_record, self.weight_enabled = record, False
        if record is not None:
            self.decoded_weight = math_plugin.decode_weight(record, original.weight.device)

    def calibration_input(self, x):
        return self.quantize(x)

    def effective_bias(self, weight=None):
        return self.bias

    def forward(self, x):
        self.calls += 1
        self.last_input_shape = list(x.shape)
        dtype = DTYPES[self.execution["compute_dtype"]]
        w = self.decoded_weight if self.weight_enabled else self.reference_weight
        bias = None if self.bias is None else self.bias.to(dtype)
        return F.conv2d(self.calibration_input(x).to(dtype), w.to(dtype), bias,
                        self.stride, self.padding, self.dilation, self.groups).to(x.dtype)


def build_model(backend, plan, preprocess=None, weights=None, activations=None):
    model = deepcopy(backend.model).eval().requires_grad_(False)
    preprocess, weights, activations = preprocess or {}, weights or {}, activations or {}
    install_attention(model, {l["module"] for l in plan["layers"] if l["kind"] in (*ATTENTION_KINDS, "attention_logits")})
    install_flows(model, plan["layers"])
    # 先替换有权重算子，再绑定输出 hook；配置数组的排列不能让 hook 丢失。
    for layer in sorted(plan["layers"], key=lambda l: l["kind"] not in WEIGHT_KINDS):
        name, sid = layer["module"], site_id(layer)
        original = model.get_submodule(name)
        if layer["kind"] in WEIGHT_KINDS:
            q = (PipelineLinear(original, layer, plan["execution"], preprocess.get(name), weights.get(name), activations.get(sid))
                 if layer["kind"] == "linear" else PipelineConv2d(original, layer, plan["execution"], weights.get(name), activations.get(sid)))
            if layer["kind"] == "linear":
                q.bias_uses_decoded_weight = any(p["module"] == name and p["method"] == "adalog_bias" for p in plan["preprocess"])
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
    return model


def quantizers(model):
    return {m.ptq_site_id: m for m in model.modules() if isinstance(m, ActivationQuantizer)}


def plan_uses_calibration(plan):
    """Whether this native plan fits any state from image activations."""
    if plan.get("preprocess"):
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
        layer_map = {l["module"]: l for l in plan["layers"] if l["kind"] == "linear"}
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
            if self.preview_state is not None:
                options["channel_params"] = self.preview_state[1][name]
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
        need_data = target == "activation" or method in ("gptq", "gptaq", "adaround") or c["strategy"] == "search"
        records, report = [], {"samples": 0, "calls": 0, "cache_bytes": 0, "gradient_loss": None}
        if need_data:
            records, report = capture(self.backend, model, layer, self.data, self.cache_mib,
                                      c["loss"] if c["metric"] == "hessian" else None,
                                      observer_batch_size=getattr(self, "observer_batch_size", None))
            self.input_shapes[sid] = list(records[0]["x"].shape)
        teachers = None
        if method in ("gptaq", "adaround"):
            teachers, teacher_report = capture(self.backend, self.backend.model, layer, self.data, self.cache_mib)
            require(report["cache_bytes"] + teacher_report["cache_bytes"] <= self.cache_mib * 1024 ** 2,
                    "成对 FP/量化缓存超过 calibration_cache_mib；不会减少样本", code="memory_budget")
            require(report["record_sample_ids"] == teacher_report["record_sample_ids"], "FP teacher/量化输入样本或调用顺序不一致")
            report["teacher"] = "original FP model module output; unchanged checkpoint/head"
            report["teacher_cache_bytes"] = teacher_report["cache_bytes"]
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
        if method == "gptq":
            with math_plugin.fp32_statistics(self.backend.device):
                h = gram(records, self.backend.device)
                record, decoded = get_target_method("weight", method).fit(module.reference_weight, h, spec, options)
            report["statistics_precision"] = "FP32; TF32 disabled for Gram/error updates only"
            del h, decoded, records
            return record, report
        x = module.reference_weight.detach().cpu() if target == "weight" else combined_input(records)
        if method == "ptf" and getattr(self, "observer_batch_size", None):
            options = dict(options, score_input=combined_input(records[-report["last_batch_records"]:]))
        grid_method = options["rounding_grid"] if method == "adaround" else method
        if method == "adalog":
            # 独立激活插件可保留显式 shift；启用 bias 组件时输入已经平移。
            from .sites import site_catalog
            semantic = site_catalog(self.backend)[sid]["semantic"]
            options = dict(options, _shift=(math_plugin.GELU_NEGATIVE_BOUND if semantic == "post_gelu"
                                           and not getattr(module, "bias_uses_decoded_weight", False) else 0.0))
        params = (grids.fit(x, spec, grid_method) if grid_method in grids.METHODS else fit_rtn(x, spec)) if target == "weight" else get_target_method("activation", method).fit(x, spec, options=options)
        if target == "activation" and getattr(self, "repq_activation_params", {}).get(layer["module"]) is not None:
            params = {k: v.clone() for k, v in self.repq_activation_params[layer["module"]].items()}
        trials = []
        if c["strategy"] == "percentile":
            params = (grids.fit(x, spec, method, percentiles=c["percentiles"]) if method == "uniform_affine" else
                      grids.percentile_log(x, spec, get_target_method("activation", method).apply, c["percentiles"]))
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
        if target == "activation" and layer["module"] in getattr(self, "repq_activation_params", {}):
            # percentile 可重新拟合变换后的共享尺度，但不能丢失仿射执行语义。
            # 只补标记，不修改拟合的 scale/zero；原 quantized_prefix 路径复用既有参数。
            params["repq_affine"] = torch.tensor(True)
        result = get_target_method("weight", method).fit(x, spec, params) if target == "weight" else {k: v.cpu() for k, v in params.items()}
        report.update(strategy=c["strategy"], metric=c["metric"], candidates=trials)
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
        order = plan["execution"]["order"]
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
            if target == "weight" and layer["module"] in weights and layer["module"] not in reset_weights:
                return
            if target == "activation" and sid in params:
                return
            print(f"calibration {sid} {target}: {layer[target + '_method']}", flush=True)
            if target == "activation" and (plan["execution"]["activation_calibration"] == "dynamic" or layer[target]["format"] in RAW_FLOATS):
                result, report = {}, {"samples": 0, "strategy": "runtime_minmax_or_raw_cast"}
            else:
                result, report = self.fit_target(model, layer, target)
            if target == "weight":
                weights[layer["module"]] = result
            elif result:
                params[sid] = result
            reports.append({"site": sid, "target": target, "method": layer[target + "_method"], **report})
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

        install()
        if order == "joint":
            for rows in groups:
                if any(calibration(l, t)["strategy"] == "coordinate" for l in rows for t in stages if l[t]):
                    print(f"calibration coordinate: {[site_id(l) for l in rows]}", flush=True)
                    w, a, report = fit_operator(self, model, rows)
                    weights.update(w)
                    params.update(a)
                    install()
                    reports.append({"sites": [site_id(l) for l in rows], **report})
                else:
                    for layer in rows:
                        for target in stages:
                            fit(layer, target)
                if sequential:
                    enable(rows, stages)
        else:
            for target in stages:
                for rows in groups:
                    for layer in rows:
                        fit(layer, target)
                    if sequential:
                        enable(rows, (target,))
                enable(plan["layers"], (target,))
        enable(plan["layers"], stages)
        return weights, params, reports

    def prepare(self, plan, shapes):
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
                # Observe a real search input without fitting a calibration grid.
                sample = next(iter(self.data.batches("search")), None)
                require(sample is not None, "形状准备需要已有 search 数据样本", code="invalid_data")
                forward(self.backend, model, sample[0])
                fidelity["shape_observation"] = {"split": "search", "sample_ids": [sample[1]],
                                                 "purpose": "shape_validation_only"}
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
        self.report = {"order": plan["execution"]["order"], "within_stage": plan["execution"].get("within_stage", "parallel"),
                       "preprocessing": prep_report, "fidelity": fidelity, "calibration": reports}
        self.preprocess_state = preprocess
        validate_plan(self.backend, plan, self.config, detect(self.backend), self.input_shapes, diagnostic=self.diagnostic)
        audit = self.audit(model, plan)
        return model, weights, params, audit

    def audit(self, model, plan):
        from .contracts import normalize_plan
        plan = normalize_plan(plan)
        qmap = quantizers(model)
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
                require(torch.equal(q.reference_weight, source * q.input_scale), "预处理参考权重被改变", code="protection_failed")
                require(bool((q.input_scale > 0).all()) and bool(torch.isfinite(q.input_shift).all()), "预处理参数非法")
                if layer["kind"] == "linear":
                    decoded_bias = any(p["module"] == layer["module"] and p["method"] == "adalog_bias" for p in plan["preprocess"])
                    require(q.bias_uses_decoded_weight == decoded_bias, "AdaLog bias 权重依赖与配置不符")
                    if decoded_bias:
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
        require(not model.training and all(not p.requires_grad for p in model.parameters()), "候选模型未冻结")
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
        removed = {name + ".weight" for name in records}
        state = {k: v.detach().cpu().clone() for k, v in self.backend.model.state_dict().items() if k not in removed}
        payload = {"state": state, "weights": records, "activations": params, "preprocess": self.preprocess_state}
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
        require(m["plugin_versions"] == plugin_versions(), "数值插件版本不匹配")
        require(m["weights_sha256"] == file_sha256(path / "weights.pt"), "产物 tensor hash 不匹配")
        plan = m["plan"]
        require(m["configuration_hash"] == fingerprint(plan), "产物配置 hash 不符")
        validate_plan(self.backend, plan, self.config, detect(self.backend), m["input_shapes"])
        payload = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        require(set(payload) == {"state", "weights", "activations", "preprocess"}, "产物字段不符")
        weight_names = {l["module"] for l in plan["layers"] if l["weight"]}
        static = {site_id(l): l for l in plan["layers"] if l["activation"] and l["activation"]["format"] not in RAW_FLOATS and plan["execution"]["activation_calibration"] == "static"}
        require(set(payload["weights"]) == weight_names and set(payload["activations"]) == set(static), "量化状态覆盖不符")
        base = self.backend.model.state_dict()
        require(set(payload["state"]) == set(base) - {n + ".weight" for n in weight_names}, "原状态覆盖不符")
        for name, value in payload["state"].items():
            require(tensor_digest(value) == tensor_digest(base[name]), f"非量化状态被改变：{name}")
        require(set(payload["preprocess"]) == {p["module"] for p in plan["preprocess"]}, "重参数化覆盖不符")
        for name, prep in payload["preprocess"].items():
            count = self.backend.model.get_submodule(name).in_features
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
        for q in quantizers(model).values():
            q.enabled = q.layer["activation"] is not None
            if isinstance(q, PipelineLinear):
                q.weight_enabled = q.layer["weight"] is not None
        bind_operators(model, plan, quantizers(model))
        self.input_shapes = m["input_shapes"]
        self.audit(model, plan)
        return model, m
