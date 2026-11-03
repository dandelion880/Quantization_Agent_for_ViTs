"""冻结 search trial 背景中的局部重拟合；与完整 trial 共用 fit_target。

基准永不原地修改。每个变体独立 deepcopy，变体/异常路径均校验基准和
FP teacher；只读旁路仍由 diagnostics.frozen_ablation 负责，不混淆两者。
"""

from copy import deepcopy
import gc
import torch

from ..schemas import STR, arr, obj, require, fingerprint
from .planning import PLAN_PATCH, apply_plan_patch
from .plugin_contracts import site_id, calibration
from .pipeline import QuantizationExecutor, PipelineLinear, quantizers
from .evaluation import model_digest
from .model import tensor_digest, forward
from .metrics import OutputErrors
from .diagnostics import ActivationDiagnostics

MUTABLE_TARGETS = arr(obj({"site": STR, "targets": arr({"enum": ["weight", "activation"]},
                                                      minItems=1, uniqueItems=True)}), minItems=1, maxItems=32)
TRIAL_VARIANTS = arr(PLAN_PATCH, minItems=1, maxItems=4)


def state_signature(model):
    """named_buffers 包含 nonpersistent decoded W；补充配置、记录及开关。"""
    from .runtime_state import record_signature
    return fingerprint({"numeric": model_digest(model), "training": model.training,
                        "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
                        "hook_counts": {n: [len(m._forward_hooks), len(m._forward_pre_hooks)]
                                        for n, m in model.named_modules() if m._forward_hooks or m._forward_pre_hooks}, "sites": {
        sid: {"layer": q.layer, "execution": q.execution, "enabled": q.enabled,
              "weight_enabled": getattr(q, "weight_enabled", None), "gradient_probe": q.gradient_probe,
              "decoded_bias": getattr(q, "bias_uses_decoded_weight", False),
              "weight_record": record_signature(getattr(q, "weight_record", None))}
        for sid, q in quantizers(model).items()}})


def validate_scope(base, plan, mutable_targets):
    """局部范围按 W/A 分开；不把相邻层联动隐藏在 implementation 中。"""
    before, after = ({site_id(l): l for l in p["layers"]} for p in (base, plan))
    require(set(before) == set(after), "局部比较不能增删量化位点")
    require(all(base.get(k) == plan.get(k) for k in set(base) | set(plan) if k not in ("layers", "preprocess")),
            "局部比较不能修改执行阶段、算子、评价或全局字段")
    allowed = {r["site"]: set(r["targets"]) for r in mutable_targets}
    require(len(allowed) == len(mutable_targets) and set(allowed) <= set(before), "局部范围含重复/未知位点")
    for sid, targets in allowed.items():
        require(all(before[sid][t] is not None and after[sid][t] is not None for t in targets),
                "局部重拟合只允许已有量化目标；关闭目标请用只读旁路工具")
    from .runtime_state import target_options
    for sid, old in before.items():
        new = after[sid]
        if sid not in allowed:
            require(old == new, f"修改越界：{sid}", code="protection_failed")
        for target in ("weight", "activation"):
            if target in allowed.get(sid, set()):
                continue
            fields = (target, target + "_method", target + "_calibration")
            require(all(old.get(k) == new.get(k) for k in fields)
                    and target_options(old, target) == target_options(new, target),
                    f"修改越界：{sid}/{target}", code="protection_failed")
    preps = [{p["module"]: p for p in config["preprocess"]} for config in (base, plan)]
    changed = {name for name in set(preps[0]) | set(preps[1]) if preps[0].get(name) != preps[1].get(name)}
    for name in changed:
        require(name in allowed and {t for t in ("weight", "activation") if before[name][t]} <= allowed[name],
                f"输入变换联动 {name} 的 W/A 未全部授权", code="protection_failed")
        require(preps[1].get(name, {}).get("observation", "float") == "float",
                "局部重拟合不能运行全模型 quantized_prefix preview；请用独立完整 trial")
    require(not plan.get("operators"), "绑定整数算子需要联动尺度，当前局部接口拒绝隐式改写")
    require(all(calibration(after[sid], target)["strategy"] != "coordinate" for sid, targets in allowed.items() for target in targets),
            "coordinate 需算子联合范围，当前局部接口不隐式重拟合其它操作数")
    return allowed, changed


def refit_variant(backend, config, data, baseline, base_plan, plan, mutable_targets, shapes):
    from .capabilities import validate_plan, detect
    from . import plugin_math
    from .runtime_state import target_signature
    allowed, changed_preps = validate_scope(base_plan, plan, mutable_targets)
    validate_plan(backend, plan, config, detect(backend), shapes)
    model = deepcopy(baseline)
    executor = QuantizationExecutor(backend, config, data, diagnostic=True)
    executor.input_shapes = deepcopy(shapes)
    executor.observer_batch_size = plan["execution"].get("observer_batch_size")
    qmap = quantizers(model)
    before = {(sid, target): target_signature(q, target) for sid, q in qmap.items()
              for target in ("weight", "activation") if target not in allowed.get(sid, set())}
    report = []
    try:
        positions = []
        handles = [q.register_forward_pre_hook(lambda m, a: positions.append(m.ptq_site_id))
                   for q in quantizers(baseline).values()]
        try:
            forward(backend, baseline, next(iter(data.batches("calibration")))[0])
        finally:
            for handle in handles:
                handle.remove()
        prep_plan = dict(plan, preprocess=[p for p in plan["preprocess"] if p["module"] in changed_preps])
        values, prep_report = executor.preprocess(prep_plan)
        for prep in plan["preprocess"]:
            if prep["method"] == "repq_ln" and prep["module"] not in changed_preps:
                executor.repq_activation_params[prep["module"]] = None  # 保留仿射语义，不复用旧observer。
        for layer in plan["layers"]:
            q, sid = qmap[site_id(layer)], site_id(layer)
            if sid in changed_preps:
                source = backend.model.get_submodule(layer["module"]).weight.detach()
                affine = values.get(sid, {"scale": torch.ones(source.shape[1]), "shift": torch.zeros(source.shape[1])})
                q.input_scale, q.input_shift = (affine[k].to(backend.device) for k in ("scale", "shift"))
                q.reference_weight = source * q.input_scale
                q.bias_uses_decoded_weight = any(p["module"] == sid and p["method"] == "adalog_bias" for p in plan["preprocess"])
                # 输入仿射变化使该边原W/A网格失效；只关闭已显式声明联动的目标。
                q.enabled, q.weight_enabled = False, False
        require(set(allowed) <= set(positions), "局部目标没有经过真实 forward")
        desired = {site_id(l): l for l in plan["layers"]}
        order = sorted(allowed, key=positions.index)
        stages = ("activation", "weight") if plan["execution"]["order"] == "a_then_w" else ("weight", "activation")
        schedule = ([(sid, target) for sid in order for target in stages] if plan["execution"]["order"] == "joint"
                    else [(sid, target) for target in stages for sid in order])
        for sid, target in schedule:
            if target not in allowed[sid]:
                continue
            q = qmap[sid]
            # 未拟合的另一目标仍使用基准方法/参数，不能新方法配旧参数运行。
            from .runtime_state import target_options
            options = {**target_options(q.layer, "activation" if target == "weight" else "weight"),
                       **target_options(desired[sid], target)}
            for key in (target, target + "_method", target + "_calibration"):
                q.layer.pop(key, None)
                if key in desired[sid]:
                    q.layer[key] = deepcopy(desired[sid][key])
            if options:
                q.layer["options"] = options
            else:
                q.layer.pop("options", None)
            field = "enabled" if target == "activation" else "weight_enabled"
            previous = getattr(q, field)
            setattr(q, field, False)  # 只关闭当前正在拟合的目标；其余背景保持冻结。
            try:
                value, detail = executor.fit_target(model, q.layer, target)
                if target == "weight":
                    q.weight_record = value
                    q.decoded_weight = plugin_math.decode_weight(value, backend.device)
                else:
                    q.set_params(value)
            finally:
                setattr(q, field, previous)
            setattr(q, field, True)
            report.append({"site": sid, "target": target, **detail})
        for sid, layer in desired.items():
            qmap[sid].layer = deepcopy(layer)
        require(all(target_signature(qmap[sid], target) == signature for (sid, target), signature in before.items()),
                "局部重拟合改变了非目标量化状态", code="protection_failed")
        audit = executor.audit(model, plan)
        return model, {"preprocessing": prep_report, "calibration": report, "audit": audit,
                       "non_target_state_unchanged": True, "mutable_targets": mutable_targets,
                       "calibration_input_source": "frozen quantized trial outside declared scope; current target and declared invalidated targets disabled until refit",
                       "preprocessing_input_source": "original FP model; declared changed edges only",
                       "teacher_source": "original FP module output for AdaRound/GPTAQ; current-context unquantized local operator for scale search"}
    except BaseException:
        del model
        raise
    finally:
        backend.reset()
        if str(backend.device).startswith("cuda"):
            torch.cuda.empty_cache()


def measure_candidate(backend, model, data):
    """局部比较共用的测量器；完整证据保留具名输出误差。"""
    errors, ids = OutputErrors(), []
    correct, teacher_correct, kl = 0, 0, 0.0
    probe = ActivationDiagnostics(model)
    try:
        for x, sample_id in data.batches("calibration"):
            a, b = errors.add(forward(backend, backend.model, x), forward(backend, model, x))
            if data.labels:
                require(list(a) == ["output"] and a["output"].shape == (1, 1000), "需要真实分类 logits")
                ref, value = a["output"].double(), b["output"].double()
                correct += value.argmax(-1).item() == data.labels[sample_id]
                teacher_correct += ref.argmax(-1).item() == data.labels[sample_id]
                kl += torch.nn.functional.kl_div(value.log_softmax(-1), ref.softmax(-1), reduction="batchmean").item()
            ids.append(sample_id)
        require(ids == data.ids("calibration"), "局部测量未完整结束")
        return {**errors.result(), "classification_diagnostic": {
            "samples": len(ids), "top1": 100 * correct / len(ids), "baseline_top1": 100 * teacher_correct / len(ids),
            "teacher_kl": kl / len(ids), "scope": "calibration-only selection diagnostic; not independent search"} if data.labels else None,
            "activation_errors": probe.result()}
    finally:
        probe.close()


def compare_trial(session, trial, data, variants, mutable_targets):
    from .model import RTNExecutor
    backend = session.backend
    base = trial["candidate"]
    plans = [apply_plan_patch(base, **variant) for variant in variants]
    for plan in plans:
        validate_scope(base, plan, mutable_targets)
    loader = RTNExecutor(backend, session.config, data)
    model, manifest = loader.load(session.run_dir.parent / "trials" / trial["trial_id"] / "artifact")
    require(base["schema_version"] == 3, "背景局部重拟合仅支持v3插件；旧v2仍可只读旁路")
    frozen, teacher = state_signature(model), model_digest(backend.model)
    rows = []
    try:
        rows.append({"plan": base, "variant": "frozen_baseline", **measure_candidate(backend, model, data)})
        for plan in plans:
            variant = None
            try:
                variant, detail = refit_variant(backend, session.config, data, model, base, plan, mutable_targets, manifest["input_shapes"])
                rows.append({"plan": plan, **measure_candidate(backend, variant, data), "calibration": detail})
            finally:
                del variant
                require(state_signature(model) == frozen and model_digest(backend.model) == teacher,
                        "基准/FP teacher 状态被改变", code="protection_failed")
                gc.collect()
                backend.reset()
        return {"rows": rows, "trial": trial["trial_id"], "sample_ids": data.ids("calibration"),
                "shapes": manifest["input_shapes"], "baseline_state_sha256": frozen,
                "measurement_context": {"mode": "target_refit_in_frozen_trial", "test_input_source": "quantized candidate path",
                    "teacher_output_source": "original FP model", "same_inputs_all_variants": True,
                    "independent_from_same_baseline": True, "non_target_state_unchanged": True,
                    "fitting_and_test_split": "same calibration subset; never search/final"},
                "scope": "local screening only; complete candidate requires independent search evaluation"}
    finally:
        require(state_signature(model) == frozen and model_digest(backend.model) == teacher,
                "结束/异常时基准状态保护失败", code="protection_failed")
        del model, loader
        gc.collect()
        backend.reset()
