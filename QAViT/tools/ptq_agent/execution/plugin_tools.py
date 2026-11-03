"""供 Profiling LLM 按需调用的插件程序；只读取 calibration 池，证据由程序生成。"""

from copy import deepcopy
from dataclasses import replace
import gc
import numpy as np
import torch

from ..schemas import STR, arr, obj, require
from ..registry import ToolDefinition
from .contracts import PLUGIN_PLAN_SCHEMA
from .plugin_contracts import KINDS, ATTENTION_KINDS, site_id
from .sites import site_catalog, install_attention
from .pipeline import QuantizationExecutor
from .model import forward
from .calibration import capture, combined_input
from .evaluation import VerifiedManifestData
from .metrics import ERROR_REDUCTION, OutputErrors
from .operators import install_flows


def require_regular_modules(backend, modules, tool):
    """保持普通模块与语义位点边界；拒绝时提供可直接修正的调用方式。"""
    unknown = sorted(set(modules) - {m["name"] for m in backend.modules()})
    if unknown:
        catalog = site_catalog(backend)
        sites = [{"module": r["name"], "kind": r["kind"]} for sid, r in catalog.items()
                 if sid in unknown or r["name"] in unknown]
        hint = f"profile_plugin_sites(sites={sites[:8]},count=...,seed=...,offset=...)" if sites else "query_structure 获取实际模块名"
        require(False, f"{tool} 不接受 {unknown}；分布请调用 {hint}；特殊方法误差请用 profile_ptq_candidates。不要重复原调用。",
                code="unsupported_site")


def plugin_capabilities(backend):
    from .objectives import NATIVE_KERNELS
    return {
        "weight": {"rtn": "existing finite-grid QDQ + scale selection", "gptq": "official Cholesky error compensation; static groups; optional act-order",
                   "gptaq": "original-FP aligned input-difference compensation; a_then_w/sequential; per-channel",
                   "adaround": "frozen grid; optimize alpha only; explicit iteration budget; hard freeze before inference",
                   "uniform_full": "full signed grid, amax/(qmax-.5); output row blocks supported", "uniform_affine": "unsigned affine grid, minmax/percentile observer", "uniform_minmax": "FQ-ViT signed/unsigned minmax grid"},
        "activation": {"rtn": "registered formats", "uniform_full": "full signed grid; tensor/channel/head scales",
                       "uniform_affine": "unsigned affine grid, independent percentile observer", "uniform_minmax": "FQ-ViT minmax grid with clamped zero point", "twin_softmax": "PTQ4ViT dual uniform grid",
                       "log2": "FQ-ViT log2 QDQ, no integer softmax claim", "log_sqrt2": "RepQ log sqrt(2)",
                       "adalog": "searchable logarithmic base and official inference LUT; softmax/verified GELU",
                       "twin_gelu": "PTQ4ViT fixed negative bound + positive scale",
                       "asymmetric_gelu": "separate observed positive/negative scales (adaptation)",
                       "ptf": "shared base scale + channel power-of-two; pre-LN original placement / post-LN adapted placement"},
        "operator": {"integer_layernorm": "independent FQ integer LN reference; uses frozen input/output scales",
                     "integer_softmax": "independent FQ LIS reference; uses frozen logit scale and output bits; floating simulation"},
        "preprocess": {"smoothquant": "X/s, W*s; alpha configurable", "repq_ln": "LN output scale/shift reparameterization; explicit edge affine",
                       "adalog_bias": "GELU input shift; bias recomputed from actual decoded weight, independent from activation method"},
        "order": ["preprocess → W → A", "preprocess → A → W", "preprocess → W/A fitted on the same FP reference"],
        "conflicts": ["no preprocessing after any W/A quantization", "one preprocessing transform per edge",
                      "PTF and RepQ are plan-wide mutually exclusive; PTF and SmoothQuant conflict on the same LN chain",
                      "one activation quantizer per tensor edge; twin/log/RTN cannot overlap",
                      "post-GELU twin grid cannot follow an input affine transform"],
        "scale_selection": {"strategies": ["minmax", "search", "manual", "percentile", "coordinate"], "metrics": ["mse", "cosine", "hessian"],
                            "hessian_requires": "real classification head and actual CE or teacher-KL gradients; no training",
                            "gptq": "official minmax grid; other scale search is explicitly rejected",
                            "block_float": "NVFP4/MXFP4 keep their fixed scaling recipes"},
        "native_kernels": {k: v.version for k, v in NATIVE_KERNELS.items()},
        "native_available_for_v3": False,
        "site_count": len(site_catalog(backend)),
        "sites_tool": "query_ptq_sites with prefix/offset/limit; use actual module and kind",
        "activation_axis_rules": {
            "rtn": "per_channel uses the catalog activation_channel_axis resolved against the measured rank: "
                   "last feature axis for Linear/Q/V/LN, second-last for transposed K, axis1 for NCHW",
            "uniform_full/affine/minmax": "per_channel may use an observed non-batch axis, including an attention head axis",
            "preflight": "non-per_tensor activation plans require measured shapes before submission; never guess an axis",
        },
        "feature_error_reduction": ERROR_REDUCTION,
        "method_documents": "query_method_library / read_method_document: independent components, no whole-model recipe",
        "validation_limits": ["operator/tiny-graph parity does not establish whole-model paper accuracy",
                              "registered plugins must also pass model-specific semantics and user constraints",
                              "other projects' ERQ/IGQ remain unregistered; new independent components are not whole-paper recipes"],
    }


class CalibrationSubset:
    def __init__(self, backend, config, count, seed, offset):
        self.full = VerifiedManifestData(backend, config)
        rows = self.full.rows["calibration"]
        require(count > 0 and offset >= 0 and offset + count <= len(rows), "诊断请求超出固定 calibration 池")
        order = np.random.default_rng(seed).permutation(len(rows))
        self.rows = [rows[int(i)] for i in order[offset:offset+count]]
        self.backend, self.labels = backend, self.full.labels

    def ids(self, split):
        require(split == "calibration", "插件诊断不得读取 search/final")
        return [r["sample_id"] for r in self.rows]

    def batches(self, split):
        self.ids(split)
        for row in self.rows:
            yield self.backend.data_adapter.load_image(row).to(self.backend.device), row["sample_id"]


def probability_statistics(x):
    """概率均值约为1/N是归一化结果；用熵和最大质量描述集中程度。"""
    rows = x.reshape(-1, x.shape[-1])
    entropy, top1, near_zero, row_min, row_max = 0.0, 0.0, 0, float("inf"), 0.0
    # 限制临时 log 张量大小，不再复制整个 attention map。
    for batch in rows.split(2048):
        sums = batch.sum(-1)
        row_min, row_max = min(row_min, sums.min().item()), max(row_max, sums.max().item())
        entropy += -(batch * batch.clamp_min(torch.finfo(batch.dtype).tiny).log()).sum().item()
        top1 += batch.amax(-1).sum().item()
        near_zero += (batch < 1e-3).sum().item()
    return {"row_sum_min": row_min, "row_sum_max": row_max,
            "entropy_nats_mean": entropy / len(rows),
            "top1_mass_mean": top1 / len(rows),
            "below_1e_minus3_fraction": near_zero / x.numel(),
            "note": "mean≈1/key_tokens不代表稀疏；结合熵、最大概率及小概率占比判断。"}


def register_plugin_tools(session):
    registry, backend = session.registry, session.backend
    free = {"type": "object", "additionalProperties": True}
    # 仅扩展可执行会话；旧 v1 profiling 工具和数值定义保持原样。
    ordinary = registry.definitions.get("profile_distribution")
    if ordinary is not None:
        def distribution(**arguments):
            require_regular_modules(backend, arguments["modules"], "profile_distribution")
            return ordinary.handler(**arguments)
        registry.definitions[ordinary.name] = replace(ordinary, handler=distribution,
            description=ordinary.description + "；仅实际参数模块输入/权重，post-softmax/GELU/LN明确位点用 profile_plugin_sites")
    registry.register(ToolDefinition("query_ptq_plugins", "W/A/预处理/scale 插件、关系与真实可执行激活位点", obj({}), free,
                                     ("profiling",), lambda: plugin_capabilities(backend)))
    # 原文整模型配方只保留开发者离线验证入口，不交给自主搜索 LLM。
    from .method_library import register_method_library
    register_method_library(session)
    def query_sites(prefix, offset, limit, include_outside_scope=False):
        scope = session.config["loop"]["scope_modules"]
        shapes = session.shapes()
        rows = [{"site_id": sid, **r} for sid, r in site_catalog(backend, session.config).items()
                if include_outside_scope or not scope or sid in scope or r["name"] in scope]
        # 查询复用实际测量，不能在已经 profile 后继续声称 shape 未知。
        for row in rows:
            if row["site_id"] in shapes:
                row["activation_shape"] = shapes[row["site_id"]]
                row["activation_shape_source"] = "verified_profile_or_search_artifact"
        available = rows
        rows = [r for r in rows if r["site_id"].startswith(prefix)]
        require(offset == 0 or offset < len(rows), f"分页 offset 越界：此 prefix 共 {len(rows)} 个位点，切换 prefix 应从 0 开始")
        result = {"total": len(rows), "sites": rows[offset:offset+limit], "offset": offset, "limit": limit,
                  "scope_applied": bool(scope) and not include_outside_scope}
        if not rows:
            # :: 是精确语义分隔符，不能代替模块层级的点；空查询也应可诊断。
            base = prefix.split("::", 1)[0]
            nearby = [r for r in available if r["site_id"].startswith(base)]
            result.update(note="prefix 是字面前缀。查询子模块用模块前缀（如 blocks.0.mlp），"
                          "只有精确语义位点使用 ::。本次无匹配，不表示这些模块不存在；"
                          "请检查 nearby_sites，或用空 prefix 从offset=0分页，不要逐block重复同类空查询。",
                          nearby_sites=[{"site_id": r["site_id"], "kind": r["kind"]} for r in nearby[:8]])
        return result
    registry.register(ToolDefinition("query_ptq_sites", "分页查询真实位点及 site_id；默认限本任务量化范围，需要调查范围外前驱可显式 include_outside_scope=true",
        obj({"prefix": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 32},
             "include_outside_scope": {"type": "boolean"}}, ["prefix", "offset", "limit"]),
        free, ("profiling",), query_sites))
    request = {"count": {"type": "integer", "minimum": 1, "maximum": 256},
               "seed": {"type": "integer", "minimum": 0}, "offset": {"type": "integer", "minimum": 0}}

    def distributions(sites, count, seed, offset):
        args = {"sites": sites, "count": count, "seed": seed, "offset": offset}
        def compute():
            catalog = site_catalog(backend, session.config)
            # 先核对整份请求，避免前面位点已做昂贵统计才发现后面位点拼错。
            for site in sites:
                sid = site_id(site)
                require(sid in catalog and not catalog[sid]["protected"],
                        f"诊断位点不支持或受保护：{sid}；先用 query_ptq_sites 查询真实 module/kind")
            data = CalibrationSubset(backend, session.config, count, seed, offset)
            rows, shapes = [], {}
            for site in sites:
                sid = site_id(site)
                model = deepcopy(backend.model).eval().requires_grad_(False)
                install_attention(model, [site["module"]] if site["kind"] in (*ATTENTION_KINDS, "attention_logits") else [])
                install_flows(model, [site])
                records, report = capture(backend, model, site, data, session.config["loop"].get("calibration_cache_mib", 256))
                x = combined_input(records).float()
                # 统计与执行使用同一布局约定；转置K的末轴是token，不能冒充feature。
                axis = catalog[sid]["activation_channel_axis"] % x.ndim
                channels = x.movedim(axis, -1).reshape(-1, x.shape[axis])
                ranges = channels.amax(0) - channels.amin(0)
                # NumPy 的精确分位数不受 torch.quantile 的 2^24 元素上限约束。
                quantiles = np.quantile(x.cpu().numpy(), [.001, .01, .5, .99, .999])
                shapes[sid] = list(records[0]["x"].shape)
                rows.append({"site": sid, "semantic": catalog[sid]["semantic"], "shape": shapes[sid],
                             "min": x.min().item(), "max": x.max().item(), "mean": x.mean().item(), "std": x.std(unbiased=False).item(),
                             "negative_fraction": (x < 0).float().mean().item(), "zero_fraction": (x == 0).float().mean().item(),
                             "quantiles": dict(zip(("p001", "p01", "p50", "p99", "p999"), quantiles.tolist())),
                             "channel_range_max": ranges.max().item(), "channel_range_median": ranges.median().item(),
                             "channel_range_ratio": ranges.max().item() / max(ranges.median().item(), 1e-12),
                             "channel_axis": axis,
                             **({"probability_statistics": probability_statistics(x)}
                                if site["kind"] == "post_softmax" else {}),
                             "note": "last axis denotes key-token positions, not embedding channels"
                             if site["kind"] in ("post_softmax", "attention_logits")
                             else "channel axis from the verified site layout",
                             **report})
                del records, x, channels, model
                gc.collect()
                backend.reset()
            return {"rows": rows, "shapes": shapes, "sample_ids": data.ids("calibration"), "scope": "FP calibration-only activation distributions"}
        detail, ref, cached = registry._cached_measurement("plugin_distributions", args, count * len(sites), compute, adjustable_request=True)
        return {"summary": detail["result"], "shapes": detail["result"]["shapes"], "evidence": ref, "cached": cached, "source": detail["source"]}

    registry.register(ToolDefinition("profile_plugin_sites", "显式激活位点（含Q/K/V、softmax/GELU/LN）分布；按实际布局统计通道差异",
        obj({**request, "sites": arr(obj({"module": STR, "kind": {"enum": list(KINDS)}}), minItems=1, maxItems=32)}), free, ("profiling",), distributions))

    from .planning import GROUPED_PLAN, expand_groups, session_trial, apply_plan_patch
    from .local_comparison import MUTABLE_TARGETS, TRIAL_VARIANTS, validate_scope, compare_trial

    def candidates(count, seed, offset, plans=None, grouped_plans=None, trial_id=None, variants=None,
                   mutable_targets=None, mode=None):
        require(sum(v is not None for v in (plans, grouped_plans, trial_id)) == 1,
                "plans/grouped_plans/trial_id 必须且只能提供一种")
        trial = None
        if trial_id is not None:
            require(mode == "refit_targets" and variants and mutable_targets, "trial背景比较需显式refit_targets与修改范围")
            trial = session_trial(session, trial_id)
            plans = [apply_plan_patch(trial["candidate"], **variant) for variant in variants]
            for plan in plans:
                validate_scope(trial["candidate"], plan, mutable_targets)
        else:
            require(variants is None and mutable_targets is None and mode is None, "背景比较字段不可混入FP局部诊断")
        if grouped_plans is not None:
            catalog = site_catalog(backend, session.config)
            plans = [expand_groups(p["plan_fields"], p["groups"], catalog, session.config,
                                   p.get("preprocess_groups", [])) for p in grouped_plans]
        args = {"plans": plans, "count": count, "seed": seed, "offset": offset}
        if trial is not None:
            args.update(trial_id=trial_id, variants=variants, mutable_targets=mutable_targets, mode=mode,
                        baseline_configuration_hash=trial["artifact"]["configuration_hash"],
                        baseline_weights_sha256=trial["artifact"]["weights_sha256"])
        # 校准次数与候选搜索也计费；复杂诊断不得只按一次工具调用计成本。
        from .plugin_contracts import calibration
        def target_cost(layer, target):
            c = calibration(layer, target)
            searches = 0
            if c["strategy"] in ("search", "coordinate"):
                searches = (20 if layer[target + "_method"] == "twin_softmax" and
                            not {"factors", "factor_grid"}.intersection(layer.get(target + "_calibration", {})) else len(c["factors"]))
            if c["strategy"] == "percentile":
                searches = len(c["percentiles"])
            searches *= len(layer.get("options", {}).get("log_base_candidates", [None])) if target == "activation" else 1
            return (2 if layer[target + "_method"] in ("gptaq", "adaround") else 1) + searches * c.get("rounds", 1)
        scope = {row["site"]: set(row["targets"]) for row in mutable_targets or []}
        units = count * sum(2 + (len(p["preprocess"]) if trial is None else
                                len(validate_scope(trial["candidate"], p, mutable_targets)[1]))
                            + sum(target_cost(l, t) for l in p["layers"]
                            for t in ("weight", "activation") if l[t] and (trial is None or t in scope.get(site_id(l), set()))) for p in plans)
        # 舍入优化的局部前后向亦计入预算，不以一次工具调用隐藏数千步训练成本。
        units += sum(l["options"]["rounding_iterations"] * min(count, l["options"].get("rounding_batch_size", 32))
                     for p in plans for l in p["layers"] if l["weight_method"] == "adaround"
                     and (trial is None or "weight" in scope.get(site_id(l), set())))
        if trial is not None:
            units += 2 * count  # 同数据实测FP和冻结基准，不复用不匹配样本的历史指标。
        def compute():
            data = CalibrationSubset(backend, session.config, count, seed, offset)
            if trial is not None:
                return compare_trial(session, trial, data, variants, mutable_targets)
            rows, shapes = [], {}
            for plan in plans:
                executor = QuantizationExecutor(backend, session.config, data, diagnostic=True)
                model, weights, params, audit = executor.prepare(plan, {})
                errors, observed = OutputErrors(), []
                classification = {"baseline_correct1": 0, "candidate_correct1": 0,
                                  "argmax_agreements": 0, "teacher_kl_sum": 0.0}
                from .diagnostics import ActivationDiagnostics
                probe = ActivationDiagnostics(model)
                try:
                    for x, sample_id in data.batches("calibration"):
                        reference, candidate = errors.add(forward(backend, backend.model, x), forward(backend, model, x))
                        if data.labels:
                            require(list(reference) == ["output"] and reference["output"].shape == (1, 1000),
                                    "分类诊断需要真实分类 logits，不能把特征当作类别")
                            ref, value = reference["output"].double(), candidate["output"].double()
                            a, b = ref.argmax(-1).item(), value.argmax(-1).item()
                            classification["baseline_correct1"] += a == data.labels[sample_id]
                            classification["candidate_correct1"] += b == data.labels[sample_id]
                            classification["argmax_agreements"] += a == b
                            classification["teacher_kl_sum"] += torch.nn.functional.kl_div(
                                value.log_softmax(-1), ref.softmax(-1), reduction="batchmean").item()
                        observed.append(sample_id)
                finally:
                    probe.close()
                require(observed == data.ids("calibration"), "诊断样本未完整结束")
                task = None
                if data.labels:
                    task = {**classification, "samples": len(observed),
                            "baseline_top1": 100 * classification["baseline_correct1"] / len(observed),
                            "top1": 100 * classification["candidate_correct1"] / len(observed),
                            "teacher_kl": classification["teacher_kl_sum"] / len(observed),
                            "scope": "calibration-only; same fitting samples, not search/final task accuracy"}
                rows.append({"plan": plan, **errors.result(), "audit": audit, "calibration": executor.report,
                             "classification_diagnostic": task,
                             "activation_errors": probe.result()})
                shapes.update(executor.input_shapes)
                del model, weights, params, executor
                gc.collect()
                backend.reset()
            return {"rows": rows, "shapes": shapes, "sample_ids": data.ids("calibration"),
                    "measurement_context": {"mode": "fresh_FP_partial_plan", "teacher_output_source": "original FP model",
                                            "fitting_and_test_split": "same calibration subset; never search/final",
                                            "same_inputs_all_variants": True, "independent_from_same_baseline": True},
                    "scope": "calibration-only diagnostic; not independent task accuracy; no final data, model export or native speedup"}
        detail, ref, cached = registry._cached_measurement("plugin_candidates", args, units, compute, adjustable_request=True)
        return {"summary": detail["result"], "shapes": detail["result"]["shapes"], "evidence": ref, "cached": cached, "source": detail["source"]}

    registry.register(ToolDefinition("profile_ptq_candidates", "比较calibration子集上1至4个独立方案。旧plans/grouped_plans从FP局部方案出发；或trial_id+mode=refit_targets+mutable_targets+variants从已完成search模型独立出发，只重拟合明确W/A范围，非目标保持冻结。variants使用增量updates，plan_fields仅允许显式预处理联动；超界拒绝，无隐式全模型重校准。返回同样本基准、配置差异、带fact_id的指标/差值。只读旁路用profile_trial_ablation。局部排名不能替代完整候选独立search评价",
        {"anyOf": [obj({**request, "plans": arr(PLUGIN_PLAN_SCHEMA, minItems=1, maxItems=4)}),
                   obj({**request, "grouped_plans": arr(GROUPED_PLAN, minItems=1, maxItems=4)}),
                   obj({**request, "trial_id": STR, "mode": {"enum": ["refit_targets"]},
                        "mutable_targets": MUTABLE_TARGETS, "variants": TRIAL_VARIANTS})]},
        free, ("profiling",), candidates))

    def trial_diagnostics(trial_id, count, seed, offset):
        from .diagnostics import ActivationDiagnostics
        from .evaluation import model_digest
        from .model import RTNExecutor
        # 只接受本会话程序封存的 search 候选，不允许传任意模型路径或读取 final。
        feedback = next((r["data"] for r in session.state.get("profile_feedback", [])
                         if r["source"] == "program_verified" and r["data"]["trial_id"] == trial_id), None)
        require(feedback is not None, "未知 search trial；请使用 execution_feedback 中的 trial")
        trial = session._feedback_trial(feedback)
        require(trial is not None and trial["status"] == "completed"
                and trial["evaluation"]["split"] == "search", "只能诊断已完成的本会话 search 候选")
        args = {"trial_id": trial_id, "count": count, "seed": seed, "offset": offset,
                "configuration_hash": trial["artifact"]["configuration_hash"]}

        def compute():
            data = CalibrationSubset(backend, session.config, count, seed, offset)
            executor = RTNExecutor(backend, session.config, data)
            model, _ = executor.load(session.run_dir.parent / "trials" / trial_id / "artifact")
            before, original = model_digest(model), model_digest(backend.model)
            probe = ActivationDiagnostics(model)
            errors = OutputErrors()
            try:
                for x, _ in data.batches("calibration"):
                    errors.add(forward(backend, backend.model, x), forward(backend, model, x))
                require(before == model_digest(model) and original == model_digest(backend.model),
                        "只读诊断改变了模型状态", code="protection_failed")
                catalog = site_catalog(backend, session.config)
                rows = [{**r, "semantic": catalog[r["site"]]["semantic"]} for r in probe.result()]
                # 全部行仍在 evidence；摘要优先显示误差最大的边，不能只截取前8层。
                rows.sort(key=lambda r: r["relative_mse"], reverse=True)
                return {"trial": trial_id, "rows": rows, "sample_ids": data.ids("calibration"),
                        "output_errors": errors.result(), "state_unchanged": True,
                        "scope": "frozen candidate on calibration subset; local Q/DQ input errors, not causal attribution or task accuracy"}
            finally:
                probe.close()
                del model, executor
                gc.collect()
                backend.reset()
        detail, ref, cached = registry._cached_measurement(
            "trial_diagnostics", args, count * len(trial["candidate"]["layers"]), compute, adjustable_request=True)
        return {"summary": detail["result"], "evidence": ref, "cached": cached, "source": detail["source"]}

    registry.register(ToolDefinition("profile_trial_diagnostics",
        "按需诊断已执行search候选的全部量化边：误差、负支/非零值置零率；只读取calibration小样本、不重拟合、不导出。反馈低精度时用于定位问题，不能单凭局部误差归因",
        obj({**request, "trial_id": STR}), free, ("profiling",), trial_diagnostics))

    def trial_ablation(trial_id, variants, count, seed, offset):
        from .diagnostics import frozen_ablation
        from .model import RTNExecutor
        feedback = next((r["data"] for r in session.state.get("profile_feedback", [])
                         if r["source"] == "program_verified" and r["data"]["trial_id"] == trial_id), None)
        require(feedback is not None, "只能旁路本会话已完成search候选")
        trial = session._feedback_trial(feedback)
        require(trial is not None and trial["status"] == "completed", "候选未完整完成")
        path = session.run_dir.parent / "trials" / trial_id / "artifact"
        require((path / "manifest.json").is_file(), "此候选大包已回收，不能伪造旁路诊断", code="asset_missing")
        require(not trial["candidate"].get("operators"), "整数算子绑定量化尺度，不能独立旁路；需专用消融接口",
                code="unsupported_ablation")
        args = {"trial_id": trial_id, "variants": variants, "count": count, "seed": seed, "offset": offset,
                "configuration_hash": trial["artifact"]["configuration_hash"]}
        def compute():
            data = CalibrationSubset(backend, session.config, count, seed, offset)
            executor = RTNExecutor(backend, session.config, data)
            model, _ = executor.load(path)
            try:
                return {"trial": trial_id, **frozen_ablation(backend, model, data, variants)}
            finally:
                del model, executor
                gc.collect()
                backend.reset()
        detail, ref, cached = registry._cached_measurement(
            "trial_ablation", args, count * (1 + len(variants)), compute, adjustable_request=True)
        return {"summary": detail["result"], "evidence": ref, "cached": cached, "source": detail["source"]}

    registry.register(ToolDefinition("profile_trial_ablation",
        "冻结已完成、可重载的search候选，低精度或未达标也可诊断；逐项只旁路一个W或A，其它参数/开关不变，不重新校准、不导出，只用calibration。绑定整数LN/LIS的候选不支持独立旁路；不能把旁路精度当作合法W4A4成绩",
        obj({**request, "trial_id": STR, "variants": arr(obj({"site": STR,
            "target": {"enum": ["weight", "activation"]}}), minItems=1, maxItems=8)}),
        free, ("profiling",), trial_ablation))
