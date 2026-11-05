"""v2 真实实验服务。模型/位宽/粒度/训练选择均来自冻结实验，数值后端保持原样。"""
from copy import deepcopy
import gc
import importlib.util
from pathlib import Path

from runtime.paths import ROOT, QAVIT
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256, read_json
from tools.experiments import asset, verify_assets
from tools.profiling import profile_host, register_trials, restore_profile_evidence
from tools.safeguards import validate_approval


def tensor_state(value):
    import torch
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {str(k): tensor_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_state(v) for v in value]
    return value


def backend_configuration(execution):
    """Resolve the existing effective numerical configuration without loading assets."""
    from ptq_agent.configuration import load_config
    adapter = execution["backend"]
    path = adapter.get("config_asset", {}).get("path") or adapter.get("config_path")
    if path:
        path = Path(path) if Path(path).is_absolute() else ROOT / path
        cfg = load_config(path)
    else:
        cfg = deepcopy(adapter["config"])
    # 后端原有 search/selection 是MVP政策，不是v2科研约束。
    # 保留其数值配置，移除另一个工作流的固定选择门禁。
    loop = cfg.setdefault("loop", {})
    for key in ("required_bits", "site_bit_overrides", "required_granularity", "required_execution"):
        loop[key] = {}
    loop.update(required_sites=[], required_site_targets={}, require_decision_notes=False,
                stop_on_goal=False, min_top1=None, search_max_top1_drop_pp=None,
                max_relative_mse=None, max_top1_drop_pp=None, max_weight_bytes_ratio=None,
                max_latency_ms=None, max_peak_memory_mib=None)
    from ptq_agent.execution.contracts import FORMATS
    cfg["constraints"].update(precision_mode="mixed", allowed_formats=list(FORMATS),
                              fixed_weight_format=None, fixed_activation_format=None, fixed_layers={})
    cfg["backend"].update(deepcopy(adapter.get("backend_overrides", {})))
    return cfg


def backend_and_data(execution):
    from ptq_agent.backends.server import ServerBackend
    from ptq_agent.execution.evaluation import VerifiedManifestData
    adapter = execution["backend"]
    cfg = backend_configuration(execution)
    if adapter.get("kind") in ("fixture", "factory"):
        factory = adapter.get("factory_asset", {}).get("path")
        if not factory:
            filename, _, function = adapter["factory"].partition(":")
            factory = str(ROOT / filename)
        else:
            function = adapter["factory_function"]
        spec = importlib.util.spec_from_file_location("qavit_explicit_fixture", factory)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if adapter.get("kind") == "fixture":
            cfg["loop"].update(evaluation="feature_fidelity", quantize_classifier=True)
            cfg["loop"].pop("min_top1", None)
            cfg["task"]["metric"] = "feature_relative_mse"
        backend = getattr(module, function)(cfg["backend"])
    else:
        backend = ServerBackend(cfg["backend"])
    return backend, cfg, VerifiedManifestData(backend, cfg)


def plan_constraints(plan, choices, backend=None, cfg=None, contract=None, *, native=True):
    from core.execution_contract import check_plan
    return check_plan(plan, choices, backend, cfg, contract, native=native)


def module_from_approval(approval, experiment):
    validate_approval(approval, experiment)
    spec = importlib.util.spec_from_file_location("qavit_approved_implementation", approval["candidate"]["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_view(data, selection, split=None):
    from tools.evaluating import DataView
    name = split or selection["split"]
    count = selection.get("count")
    offset = selection.get("offset", 0)
    # 原数据清单保持不变；只在显式视图中取样，避免改变重载身份。
    ids = data.ids(name)
    if count is None:
        count = len(ids) - offset
    if not 0 <= offset < len(ids) or not 0 < count <= len(ids) - offset:
        raise ContractError("样本视图越界；不能静默缩小样本数")
    if offset:
        # 原DataView支持精确shard索引；该索引仅是显式子集，不修改原manifest。
        class Selection:
            def __init__(self):
                self.labels, self.identity = data.labels, data.identity
                self.selected = ids[offset:offset + count]
            def ids(self, split):
                return self.selected if split == name else data.ids(split)
            def batches(self, split):
                wanted = set(self.ids(split))
                for x, sid in data.batches(split):
                    if sid in wanted:
                        yield x, sid
        return Selection()
    return DataView(data, name, count=count)


def profile_service(experiment, backend, cfg, data, directory):
    from dataclasses import asdict
    from ptq_agent.schemas import ToolCall
    from tools.profiling import PROFILE_NAMES
    spec = experiment["payload"]["execution"]
    name, args = spec["tool"], spec["arguments"]
    if name not in PROFILE_NAMES:
        raise ContractError("Profiling 工具未注册")
    cfg = deepcopy(cfg)
    cfg["output_dir"] = str(directory / "profile")
    cfg["memory_dir"] = str(directory / "profile_memory")
    host = profile_host(backend, cfg, directory / "profile")
    register_trials(host, spec.get("trial_bindings", []))
    restore_profile_evidence(host, spec.get("profile_evidence_bindings", []), experiment["run_id"])
    call = ToolCall("v2_profile", name, args)
    result = asdict(host.registry.invoke("profiling", call))
    host.state["tool_results"].append({"call": asdict(call), "result": result})
    host.save()
    if result["status"] != "success":
        raise ContractError("Profiling 内层工具失败: " + str(result.get("error")))
    atomic_json(directory / "profile_result.json", result)
    from ptq_agent.profiling.evidence import read_returned_evidence
    artifacts = [asset(directory / "profile_result.json")]
    measured, sites, parameters = {}, set(), {}
    for reference in result["artifacts"]:
        if reference.startswith("evidence/"):
            verified = read_returned_evidence(host, reference)
            facts = verified["result"]
            parameters[reference] = deepcopy(verified.get("arguments", {}))
            ids = facts.get("sample_ids")
            if ids:
                splits = [split for split, pool in data.identity["split_ids"].items() if set(ids) <= set(pool)]
                measured[reference] = {"split": splits[0] if len(splits) == 1 else "unresolved_split",
                                       "sample_ids": deepcopy(ids),
                                       "sample_count": len(ids), "full_dataset": False}
            for row in facts.get("rows", []):
                if isinstance(row, dict) and isinstance(row.get("module"), str):
                    sites.add(row["module"])
            for row in facts.get("modules", []):
                if isinstance(row, dict) and isinstance(row.get("name"), str):
                    sites.add(row["name"])
            artifacts.append(asset(host.run_dir / reference))
    return {"observations": {"tool": name, "result": result["data"]},
            "artifacts": artifacts,
            "sample_selection": deepcopy(measured),
            "execution_facts": {"measurement_kind": "profile_measurement" if parameters else "unverified_scope",
                                "data": measured, "sites": sorted(sites), "parameters": parameters}}


def tensor_probe(experiment, backend, cfg, data, directory, approval):
    import torch
    import torch.nn.functional as F
    from ptq_agent.execution.calibration import capture, combined_input
    from ptq_agent.execution.sites import site_catalog
    from ptq_agent.execution.plugin_contracts import WEIGHT_KINDS
    from ptq_agent.execution.model import forward
    from ptq_agent.execution.evaluation import OutputErrors
    spec = experiment["payload"]["execution"]
    sid = spec["site"]
    catalog = site_catalog(backend, cfg)
    if sid not in catalog:
        raise ContractError("模型没有指定量化位点: " + sid)
    info = catalog[sid]
    layer = {"module": info["name"], "kind": info["kind"]}
    selected = data_view(data, spec["calibration"], "calibration")
    records, capture_report = capture(backend, backend.model, layer, selected,
                                      spec.get("cache_mib", 256))
    x = combined_input(records).float()
    params = deepcopy(spec["parameters"])
    if params.get("granularity") == "per_channel":
        axis = params.get("axis")
        if type(axis) is not int or not -x.ndim <= axis < x.ndim:
            raise ContractError("per_channel必须给出有效通道轴")
        # PyTorch负轴的规范化是等价索引转换。旧数值库按非负维度归约，
        # 直接传-1会错误地归约全部维度，伪装成per_channel但实际是per_tensor。
        params["axis"] = axis % x.ndim
        if params["axis"] == 0:
            raise ContractError("静态激活校准不能把batch索引当作通道轴")
    if params.get("bits") != spec["resolved_choices"].get("activation_bits"):
        raise ContractError("实际量化器参数位宽与冻结选择不同")
    from core.execution_contract import compare_fields
    from ptq_agent.execution.recipes import spec as quant_spec
    contract = spec.get("quantization_contract", {})
    frozen_activation = deepcopy(contract.get("specs", {}).get("activation", {}))
    if type(frozen_activation.get("axis")) is int and frozen_activation["axis"] < 0:
        frozen_activation["axis"] %= x.ndim
    qs = quant_spec(params["bits"], params["signed"], params["granularity"], params.get("axis"), params.get("group_size"))
    compare_fields(frozen_activation, qs, "tensor_probe.activation")
    if contract.get("methods", {}).get("activation", spec.get("method")) != spec.get("method"):
        raise ContractError("tensor_probe.method 与冻结 activation_method 不一致")
    if approval:
        plugin = module_from_approval(approval, experiment)
        state = plugin.fit(x.clone(), deepcopy(params))
        apply = lambda value: plugin.apply(value, state, deepcopy(params))
    else:
        from ptq_agent.execution.plugin_math import fit_activation, activation_qdq
        from ptq_agent.execution import grids
        method = spec["method"]
        if method == "rtn" or method in grids.METHODS:
            grids.validate_spec(qs, "activation", tuple(x.shape))
        elif method != "ptf" and params["granularity"] != "per_tensor":
            raise ContractError("该已有非均匀网格仅实现per_tensor；其它粒度须经Coding扩展，不能静默忽略")
        options = spec.get("method_options", {})
        state = fit_activation(x, qs, method, options)
        apply = lambda value: activation_qdq(value, qs, method, state, options)
    state_before = fingerprint(tensor_state(state))
    module = backend.model.get_submodule(info["name"])
    input_error, output_error = OutputErrors(), OutputErrors()
    zeros, total = 0, 0
    with torch.no_grad():
        for row in records:
            a = row["x"].float()
            qa = apply(a.clone())
            if not torch.is_tensor(qa) or qa.shape != a.shape or not torch.isfinite(qa).all():
                raise ContractError("实际量化返回形状/有限性非法")
            input_error.add(a, qa)
            zeros += int(((a != 0) & (qa == 0)).sum())
            total += a.numel()
            if layer["kind"] == "linear":
                local = F.linear(qa, module.weight.detach().cpu().float(), None if module.bias is None else module.bias.detach().cpu().float())
            elif layer["kind"] == "conv2d":
                local = F.conv2d(qa, module.weight.detach().cpu().float(), None if module.bias is None else module.bias.detach().cpu().float(),
                                 module.stride, module.padding, module.dilation, module.groups)
            elif layer["kind"] in ("post_softmax", "attention_query"):
                local = qa @ row["rhs"].float()
            elif layer["kind"] in ("attention_key", "attention_value"):
                local = row["rhs"].float() @ qa
            else:
                raise ContractError("此局部输出路径尚未验证；请请求自定义统计能力")
            output_error.add(row["y"].float(), local)
    observations = {"site": sid, "input_error": input_error.result(), "operator_output_error": output_error.result(),
                    "applied_quantization": {"activation_bits": params["bits"], "granularity": params["granularity"],
                                             "resolved_axis": params.get("axis"),
                                             "weight_state": "unchanged FP32; no weight quantization in this local probe",
                                             "activation_state": "fit once on declared calibration view; frozen during replay"},
                    "nonzero_to_zero_fraction_of_all_elements": zeros / max(1, total),
                    "elements": total, "calibration_samples": len(selected.ids("calibration")),
                    "scope": "controlled local intervention with FP reference; not full-model quantized accuracy"}
    if experiment["payload"]["level"] == "L2":
        if layer["kind"] not in ("linear", "conv2d"):
            raise ContractError("整网单边回放当前仅验证Linear/Conv2d输入；其它位点需专用适配器")
        pilot = data_view(data, spec["evaluation"])
        errors = OutputErrors()
        counts = {"baseline_top1": 0, "candidate_top1": 0, "samples": 0}
        for inputs, sample_id in pilot.batches(spec["evaluation"]["split"]):
            baseline = forward(backend, backend.model, inputs)
            def replace(_module, args):
                value = apply(args[0].detach().cpu().float()).to(args[0])
                return (value, *args[1:])
            handle = module.register_forward_pre_hook(replace)
            try:
                candidate = forward(backend, backend.model, inputs)
            finally:
                handle.remove()
            reference_outputs, candidate_outputs = errors.add(baseline, candidate)
            counts["samples"] += 1
            if sample_id in data.labels:
                label = data.labels[sample_id]
                if set(reference_outputs) != {"output"} or reference_outputs["output"].ndim != 2:
                    raise ContractError("分类Pilot需要明确分类logits；特征输出不能冒充分类准确率")
                counts["baseline_top1"] += int(reference_outputs["output"].argmax(-1).item() == label)
                counts["candidate_top1"] += int(candidate_outputs["output"].argmax(-1).item() == label)
        observations["pilot"] = {"output_error": errors.result(), "counts": counts,
                                  "sample_ids": pilot.ids(spec["evaluation"]["split"]),
                                  "task_accuracy_measured": bool(data.labels), "full_dataset": False,
                                  "scope": "single-site pilot; no full-quantized-model claim"}
    if fingerprint(tensor_state(state)) != state_before:
        raise ContractError("实验运行改变了冻结量化状态")
    atomic_json(directory / "quantizer_state.json", tensor_state(state))
    atomic_json(directory / "capture_report.json", capture_report)
    return {"observations": observations, "artifacts": [asset(directory / "quantizer_state.json"), asset(directory / "capture_report.json")],
            "sample_selection": {"calibration_ids": selected.ids("calibration"), "evaluation": spec.get("evaluation")},
            "quantizer_state_sha256": state_before,
            "execution_facts": {"measurement_kind": "local_activation_probe", "quantization": {
                sid: {"weight": None, "activation": deepcopy(qs),
                      "activation_method": "approved_plugin" if approval else spec["method"],
                      **({"method_options": deepcopy(options)} if not approval else {})}},
                "parameters": deepcopy(params),
                "data": {"calibration": {"split": "calibration", "sample_count": len(selected.ids("calibration")),
                                          "sample_ids": selected.ids("calibration"), "full_dataset": False},
                         **({"pilot": {"split": spec["evaluation"]["split"], "sample_count": counts["samples"],
                                       "sample_ids": pilot.ids(spec["evaluation"]["split"]), "full_dataset": False}}
                            if experiment["payload"]["level"] == "L2" else {})}}}


def evaluate_services(experiment, backend, cfg, data, directory, approval):
    from ptq_agent.execution.pipeline import QuantizationExecutor, plan_uses_calibration
    from ptq_agent.execution.model import RTNExecutor
    from core.execution_contract import validate_plan_schema
    from ptq_agent.execution.evaluation import Evaluator
    from ptq_agent.execution.capabilities import validate_plan, detect
    from tools.evaluating import evaluate
    from ptq_agent.execution.evaluation import model_digest
    spec = experiment["payload"]["execution"]
    choices = spec["resolved_choices"]
    from ptq_agent.execution.plugin_contracts import site_id
    native = True
    model_files = []
    preparation_reads = {}
    if spec["service"] == "quantize_evaluate":
        from tools.runtime_interface import ObservedData
        plugin = module_from_approval(approval, experiment) if spec.get('requires_code') else None
        plan = (plugin.build_plan(backend, deepcopy(cfg), deepcopy(experiment["payload"]), deepcopy(spec.get("parameters", {})))
                if plugin else deepcopy(spec['plan']))
        selection = data_view(data, spec["calibration"], "calibration") if spec.get("calibration") else data
        observed = ObservedData(selection)
        factory = getattr(plugin, "make_executor", RTNExecutor)
        executor = factory(backend, deepcopy(cfg), observed)
        native = type(executor) in (RTNExecutor, QuantizationExecutor)
        if native:
            plan = validate_plan_schema(plan)
            if type(executor) is QuantizationExecutor and plan["schema_version"] == 2 and not executor.diagnostic:
                # An explicitly returned built-in factory has the same declared
                # protocol range as direct/default built-in execution.
                factory = RTNExecutor
                executor = factory(backend, deepcopy(cfg), observed)
        coverage = plan_constraints(plan, choices, backend, cfg, spec.get("quantization_contract"), native=native)
        if native:
            validate_plan(backend, plan, cfg, detect(backend), {}, allow_unknown_shapes=True)
            if plan_uses_calibration(plan) and not spec.get('calibration'):
                raise ContractError("原生量化执行器需要明确 calibration 数据视图")
        model, weights, activations, audit = executor.prepare(plan, {})
        verifier = factory(backend, deepcopy(cfg), observed)
        if native:
            verifier.input_shapes = executor.input_shapes
            verifier.audit(model, plan)
        model_dir = directory / "model"
        executor.export(model, plan, weights, activations, model_dir, audit)
        model_files = sorted(p for p in model_dir.rglob('*') if p.is_file())
        if not model_files:
            raise ContractError("执行器未生成可重载的实际模型资产")
        calibration_ids = list(dict.fromkeys(observed.reads.get('calibration', [])))
        # 原生 manifest 的 calibration_ids 记录真实捕获，而非可用数据池。
        if native:
            exported_manifest = read_json(model_dir / 'manifest.json')
            exported_manifest['calibration_ids'] = calibration_ids
            exported_manifest.pop('content_sha256', None)
            exported_manifest['content_sha256'] = fingerprint(exported_manifest)
            atomic_json(model_dir / 'manifest.json', exported_manifest)
        frozen_files = {str(p.relative_to(model_dir)): file_sha256(p) for p in model_files}
        loaded, manifest = verifier.load(model_dir)
        preparation_reads = {split: {"split": split, "sample_ids": list(dict.fromkeys(ids)),
            "sample_count": len(set(ids)), "full_dataset": split == "final" and list(dict.fromkeys(ids)) == data.ids(split)
                and data.identity.get("full_validation") is True}
            for split, ids in observed.reads.items()}
        if not native:
            calibration_ids = list(dict.fromkeys(observed.reads.get('calibration', [])))
        if loaded is model:
            raise ContractError("执行器 load 必须返回实际重建实例，不能复用导出前模型")
        if frozen_files != {str(p.relative_to(model_dir)): file_sha256(p)
                            for p in model_dir.rglob('*') if p.is_file()}:
            raise ContractError("执行器重载改写了已导出的模型资产")
        reload_check = Evaluator(backend, cfg, data).reload_check(model, loaded, split="search")
        model = loaded
        if native:
            plan = manifest["plan"]
            coverage = plan_constraints(plan, choices, backend, cfg, spec.get("quantization_contract"))
            verifier.audit(model, plan)
        else:
            reload_check['loader'] = 'approved_custom_executor'
    else:
        executor = RTNExecutor(backend, cfg, data)
        model_dir = Path(spec["model_dir"])
        model, manifest = executor.load(model_dir)
        plan = manifest["plan"]
        coverage = plan_constraints(plan, choices, backend, cfg, spec.get("quantization_contract"))
        calibration_ids = manifest["calibration_ids"]
        reload_check = {"scope": "existing artifact validated by loader; no new export"}
    from core.execution_contract import check_fp_state
    check_fp_state(model, backend, spec.get("quantization_contract", {}))
    reload_check.update(host_verified=True, evaluated_instance="reloaded")
    if spec.get("calibration"):
        if calibration_ids != data_view(data, spec["calibration"], "calibration").ids("calibration"):
            raise ContractError("导出包实际校准样本与冻结 calibration 视图不一致")
    before = model_digest(model)
    evaluations = {}
    for requested in spec["evaluations"]:
        if requested["kind"] == "classification":
            value = evaluate(backend, cfg, data, model, plan, requested["split"], requested.get("count"), None,
                             **({"native": False, "sites": coverage["declared_sites"]} if not native else {}))
            atomic_json(directory / "classification.json", value)
            evaluations["classification"] = {"metrics": {k: v for k, v in value["metrics"].items()
                                                           if isinstance(v, (int, float, bool)) or v is None},
                                              "full_dataset": value["full_dataset_evaluation"],
                                              "sample_count": value["sample_count"], "sample_ids": value["sample_ids"],
                                              "split": value["split"]}
        elif requested["kind"] == "ade_frozen":
            from tools.segmentation import evaluate as ade
            value = ade(model, backend.model, directory, requested.get("count"))
            atomic_json(directory / "segmentation.json", value)
            # 原适配器目前只核验该冻结head协议，不能冒称适配所有ViT分割头。
            metric = value["metrics"].get("quantized", value["metrics"].get("quant"))
            if metric is None:
                raise ContractError("未知ADE结果字段；不能猜指标归属")
            evaluations["segmentation"] = {"metrics": {k: metric[k] for k in ("mIoU", "mAcc", "aAcc")},
                                             "full_dataset": value["full_dataset_evaluation"], "sample_count": value["images"],
                                             "sample_ids": value["sample_ids"], "split": "ade_validation"}
        else:
            raise ContractError("尚未注册评测适配器: " + requested["kind"])
    if model_digest(model) != before:
        raise ContractError("评测改变了候选模型状态")
    files = [asset(p) for p in sorted(directory.glob("*.json")) if p.name not in ("request.json", "started.json", "result.json")]
    if model_dir.parent == directory:
        files.extend(asset(p) for p in model_files or sorted(p for p in model_dir.rglob('*') if p.is_file()))
    return {"observations": {"evaluations": evaluations, "coverage": coverage, "reload": reload_check,
                              "representation": (plan["execution"]["mode"] if native else "approved_custom_executor") + "; native_low_precision_kernel_not_verified"},
            "artifacts": files, "model_ref": fingerprint({str(p.relative_to(model_dir)): file_sha256(p)
                                                          for p in model_dir.rglob('*') if p.is_file()}),
            "model_dir": str(model_dir), "sample_selection": {k: v["sample_ids"] for k, v in evaluations.items()},
            "execution_facts": {"measurement_kind": "full_model", "quantization": deepcopy(coverage["quantization"]),
                "data": {**preparation_reads, **({"calibration": {"split": "calibration", "sample_count": len(calibration_ids),
                                          "sample_ids": calibration_ids, "full_dataset": False}} if calibration_ids else {}),
                         **{name: {k: deepcopy(row[k]) for k in ("split", "sample_count", "sample_ids", "full_dataset")}
                            for name, row in evaluations.items()}}, "plan_sha256": fingerprint(plan)}}


def run_experiment(request, directory):
    import torch
    from runtime.guards import protect_files, install_gradient_guard, GRADIENT_AUDIT
    from ptq_agent.execution.evaluation import model_digest
    experiment, approval = request["experiment"], request.get("approval")
    payload, spec = experiment["payload"], experiment["payload"]["execution"]
    choices = spec["resolved_choices"]
    if "training" not in choices or type(choices["training"]) is not bool:
        raise ContractError("执行卡片须明确本次训练选择；通用框架不提供默认值")
    verify_assets(spec)
    if spec.get("requires_code") or spec["service"] == "custom_experiment":
        validate_approval(approval, experiment)
    seed = spec.get("seed")
    if type(seed) is not int:
        raise ContractError("实验必须指定随机种子")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(payload["resources"]["cpu_threads"])
    install_gradient_guard(not choices["training"])
    backend, cfg, data = backend_and_data(spec)
    # 模型构造器可能设置自身seed；算法随机流必须在加载后恢复为冻结实验的seed。
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    original = model_digest(backend.model)
    isolation = protect_files(directory)
    try:
        if spec["service"] == "profile":
            result = profile_service(experiment, backend, cfg, data, directory)
        elif spec["service"] == "tensor_probe":
            result = tensor_probe(experiment, backend, cfg, data, directory, approval)
        elif spec["service"] in ("evaluate_artifact", "quantize_evaluate"):
            result = evaluate_services(experiment, backend, cfg, data, directory, approval)
        else:
            plugin = module_from_approval(approval, experiment)
            from tools.runtime_interface import execute_custom
            result = execute_custom(plugin, payload, backend, cfg, data, directory)
            # 标准评测由宿主从导出包重新加载并执行；不接受插件自报重载或分类成绩。
            standard = any(r["kind"] in ("classification", "ade_frozen") for r in spec.get("evaluations", []))
            reload_required = (spec.get("result_protocol", {}).get("reload_before_evaluation")
                               or spec.get("reload_before_evaluation") or payload.get("validation", {}).get("export_reload"))
            if payload["level"] == "L3" and (standard or reload_required or spec.get("measurement_kind") == "full_model"):
                if not result.get("model_dir"):
                    raise ContractError("custom_experiment L3 缺少供宿主重载核验的 model_dir")
                exported = Path(result["model_dir"]).resolve()
                if directory.resolve() not in exported.parents:
                    raise ContractError("custom_experiment 导出包必须位于本作业目录")
                loaded_experiment = deepcopy(experiment)
                loaded_experiment["payload"]["execution"].update(service="evaluate_artifact", model_dir=str(exported))
                measured = evaluate_services(loaded_experiment, backend, cfg, data, directory, None)
                result["observations"].update(measured["observations"])
                result.update({k: v for k, v in measured.items() if k != "observations"})
        if model_digest(backend.model) != original:
            raise ContractError("实验改变了冻结基准模型")
        result["provenance"] = {"model_identity": backend.identity(), "baseline_state_sha256": original,
                                 "data_identity": data.identity, "sample_selection": result.get("sample_selection"),
                                 "backend_data": {k: cfg["backend"].get(k) for k in ("data_protocol", "data_sha256")},
                                 "seed": seed, "model_unchanged": True, "isolation": isolation,
                                 "rng_seeds": {"python": seed, "torch": seed, "numpy": seed % (2 ** 32)},
                                 "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                             "device": str(backend.device), "cpu_threads": torch.get_num_threads(),
                                             "gpu_name": torch.cuda.get_device_name() if str(backend.device).startswith("cuda") else None},
                                 "training_declared": choices["training"], "gradient_audit": dict(GRADIENT_AUDIT),
                                 "protocol_hash": spec.get("protocol_hash"), "model_ref": result.get("model_ref"),
                                 "quantizer_state_sha256": result.get("quantizer_state_sha256"),
                                 "execution_facts": result.pop("execution_facts", {})}
        result["measurement_source"] = "synthetic_computation" if spec["backend"].get("kind") == "fixture" else "real"
        from tools.result_contract import check_results
        check_results(payload, {"evaluation_protocol": spec.get("result_protocol", {})}, result)
        atomic_json(directory / "gradient_audit.json", dict(GRADIENT_AUDIT))
        result.setdefault("artifacts", []).append(asset(directory / "gradient_audit.json"))
        return result
    finally:
        backend.reset()
        del backend
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
