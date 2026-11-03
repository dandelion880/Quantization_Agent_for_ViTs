"""隔离运行一个候选或一个参考。父进程比较两个独立回执；不共享全局模块。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copy import deepcopy
import importlib.util
import os
import signal
import traceback

from runtime.paths import ROOT
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, read_json
from tools.safeguards import verify_artifact


def load_module(artifact):
    spec = importlib.util.spec_from_file_location("qavit_isolated_implementation", artifact["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_state(value):
    import torch
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {str(k): tensor_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_state(v) for v in value]
    return value


def run(request, directory):
    import torch
    import random
    import numpy as np
    from runtime.guards import protect_files, install_gradient_guard, GRADIENT_AUDIT
    experiment, artifact = request["experiment"], request["artifact"]
    verify_artifact(artifact, experiment)
    execution = experiment["payload"]["execution"]
    choices = execution["resolved_choices"]
    if type(choices.get("training")) is not bool:
        raise ContractError("执行前必须明确本实验是否训练，不能默认为禁止或允许")
    install_gradient_guard(not choices["training"])
    # TaskRuntime 已将唯一授权物理 GPU 映射为可见设备；在加载生成源码前初始化设备。
    # Landlock 禁止新建设备写句柄，不能把保护后的首次 CUDA 初始化误报成设备不可用。
    if artifact["interface"] == "executor_v1" and experiment["payload"]["resources"].get("gpu") is not None:
        torch.cuda.init()
        device_ready = torch.ones(1, device="cuda:0")
        torch.cuda.synchronize()
        del device_ready
    isolation = protect_files(directory)
    torch.set_num_threads(experiment["payload"]["resources"]["cpu_threads"])
    module = load_module(artifact)
    cases = experiment["payload"]["validation"].get("cases", [])
    if not cases:
        raise ContractError("新实现缺少冻结的独立数值测试案例")
    rows = []
    for index, case in enumerate(cases):
        seed = case["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        params = deepcopy(execution.get("parameters", {}))
        params.update(case.get("parameters", {}))
        if artifact["interface"] == "quantizer_v1":
            generator = torch.Generator().manual_seed(seed)
            if "values" in case:
                x = torch.tensor(case["values"], dtype=torch.float32)
                if case.get("shape"):
                    x = x.reshape(case["shape"])
            else:
                shape = tuple(case["shape"])
                if case.get("distribution", "normal") == "uniform":
                    x = torch.rand(shape, generator=generator) * 2 - 1
                else:
                    x = torch.randn(shape, generator=generator)
                x *= case.get("scale", 1)
            calibration = torch.tensor(case["calibration_values"], dtype=torch.float32) if "calibration_values" in case else x.clone()
            state = module.fit(calibration.clone(), deepcopy(params))
            before = fingerprint(tensor_state(state))
            original_x = x.clone()
            y = module.apply(x, state, deepcopy(params))
            if not torch.is_tensor(y) or y.shape != x.shape or not bool(torch.isfinite(y).all()):
                raise ContractError("quantizer 输出形状或有限性不符合接口")
            if not torch.equal(original_x, x) or fingerprint(tensor_state(state)) != before:
                raise ContractError("apply 修改了输入或冻结量化状态")
            repeated = module.apply(original_x.clone(), state, deepcopy(params))
            if not torch.equal(y, repeated) and not execution.get("stochastic", False):
                raise ContractError("未声明的非确定性 apply")
            row = {"index": index, "shape": list(y.shape), "output": y.detach().cpu().tolist(),
                   "finite": True, "input_unchanged": True, "state_unchanged": True,
                   "dtype": str(y.dtype), "unique_count_global": int(torch.unique(y).numel())}
            if params.get("granularity") == "per_tensor" and "code_count" in experiment["payload"]["validation"].get("invariants", []):
                if row["unique_count_global"] > 2 ** params["bits"]:
                    raise ContractError("per-tensor 码本超出声明位宽；零也必须计入编码")
            rows.append(row)
        elif artifact["interface"] == "executor_v1":
            from tools.experiment_worker import backend_and_data, plan_constraints
            from ptq_agent.execution.pipeline import QuantizationExecutor
            from ptq_agent.execution.model import forward
            from ptq_agent.execution.evaluation import model_digest
            from tools.evaluating import DataView
            backend, cfg, data = backend_and_data(execution)
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            baseline_before = model_digest(backend.model)
            plan = module.build_plan(backend, deepcopy(cfg), deepcopy(experiment["payload"]), deepcopy(params))
            plan_constraints(plan, choices, backend, cfg, execution.get("quantization_contract"))
            from ptq_agent.execution.capabilities import validate_plan, detect
            validate_plan(backend, plan, cfg, detect(backend), {})
            small = DataView(data, "calibration", count=case["calibration_count"])
            factory = getattr(module, "make_executor", QuantizationExecutor)
            executor = factory(backend, deepcopy(cfg), small)
            model, _, _, _ = executor.prepare(plan, {})
            verifier = QuantizationExecutor(backend, cfg, small)
            verifier.input_shapes = executor.input_shapes
            verifier.audit(model, plan)
            from core.execution_contract import check_fp_state
            check_fp_state(model, backend, execution.get("quantization_contract", {}))
            candidate_before = model_digest(model)
            outputs = []
            for x, sid in DataView(data, "search", count=case["evaluation_count"]).batches("search"):
                original_x = x.clone()
                y = forward(backend, model, x)
                value = tensor_state(y)
                fingerprint(value)  # 同时支持其它ViT的命名特征输出；严格拒绝非有限值。
                if not torch.equal(x, original_x):
                    raise ContractError("独立执行器验证改写了输入")
                outputs.append(value)
            baseline_unchanged = baseline_before == model_digest(backend.model)
            candidate_unchanged = candidate_before == model_digest(model)
            if not baseline_unchanged or not candidate_unchanged:
                raise ContractError("执行器修改了FP基准或评测期间改变了冻结候选")
            rows.append({"index": index, "output": outputs, "finite": True,
                         "plan": plan, "input_unchanged": True, "state_unchanged": candidate_unchanged,
                         "baseline_unchanged": baseline_unchanged})
            del model, executor, backend
        else:
            context = deepcopy(case["context"])
            before = fingerprint(tensor_state(context))
            result = tensor_state(module.execute(deepcopy(experiment["payload"]), context))
            fingerprint(result)  # 严格JSON数值检查；NaN/Inf不能被记录为finite。
            unchanged = before == fingerprint(tensor_state(context))
            if not unchanged and "input_unchanged" in experiment["payload"]["validation"].get("invariants", []):
                raise ContractError("自定义实验改变了冻结案例输入")
            rows.append({"index": index, "output": result, "finite": True,
                         "input_unchanged": unchanged, "state_unchanged": None,
                         "state_check_scope": "JSON test context only; full-model freeze checked by experiment worker"})
    interfaces = []
    if artifact["interface"] == "experiment_v1":
        from tools.runtime_interface import validate_interface_cases, interface_fixture, execute_custom
        validate_interface_cases(experiment["payload"])
        for index, case in enumerate(experiment["payload"]["validation"]["interface_cases"]):
            backend, cfg, data = interface_fixture(execution, case)
            value = execute_custom(module, experiment["payload"], backend, cfg, data, directory)
            interfaces.append({"index": index, "output": tensor_state(value["observations"]),
                               "execution_facts": value["execution_facts"], "source": "CPU_interface_fixture_only"})
    return {"cases": rows, "interface_cases": interfaces, "gradient_audit": dict(GRADIENT_AUDIT), "isolation": isolation,
            "source_sha256": artifact["source_sha256"], "experiment_sha256": experiment["content_sha256"]}


def main():
    request_path = Path(sys.argv[1]).resolve()
    request, directory = read_json(request_path), request_path.parent
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("独立验证超时")))
    signal.setitimer(signal.ITIMER_REAL, request["timeout_seconds"])
    try:
        result = {"status": "completed", "data": run(request, directory)}
    except BaseException as exc:
        traceback.print_exc()
        result = {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}
    atomic_json(directory / "result.json", result)


if __name__ == "__main__":
    main()
