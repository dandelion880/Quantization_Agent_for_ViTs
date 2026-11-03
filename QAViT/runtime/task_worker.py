"""单个持久任务尝试的工作进程。只写本次作业回执，不直接修改科研日志。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlib import ExitStack
import math
import os
import resource
import signal
import subprocess
import time
import traceback

from runtime.paths import ROOT, QAVIT
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256, file_lock, read_json
from runtime.task_queue import process_identity
from runtime.identity import code_identity
from runtime.subscription import StructuredOutputError


def numeric_comparison(left, right, atol, rtol, path="$"):
    maximum = 0.0
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ContractError("独立对照出现非有限数: " + path)
        error = abs(left - right)
        if error > atol + rtol * abs(right):
            raise ContractError(f"数值不符 {path}: candidate={left}, reference={right}, abs_error={error}")
        return error
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        for key in left:
            maximum = max(maximum, numeric_comparison(left[key], right[key], atol, rtol, path + "." + key))
        return maximum
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for index, (a, b) in enumerate(zip(left, right)):
            maximum = max(maximum, numeric_comparison(a, b, atol, rtol, path + f"[{index}]"))
        return maximum
    if left != right or type(left) is not type(right):
        raise ContractError("独立对照的结构/非数值结果不同: " + path)
    return maximum


def verification(request, directory):
    from runtime.codex_cli import parent_guard, terminate
    experiment = request["experiment"]
    config = experiment["payload"]["validation"]
    atol, rtol = config["atol"], config["rtol"]
    if not 0 <= atol < 1 or not 0 <= rtol < 1:
        raise ContractError("数值对照容差必须在冻结规格中合理规定")
    results = {}
    for name in ("candidate", "reference"):
        child = directory / name
        child.mkdir(exist_ok=True)
        remaining = request["deadline_unix"] - time.time()
        if remaining <= 0:
            raise TimeoutError("数值验证预算耗尽")
        atomic_json(child / "request.json", {"experiment": experiment, "artifact": request[name],
                                             "timeout_seconds": remaining})
        with (child / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(QAVIT / "tools/verification_worker.py"), str(child / "request.json")],
                                       cwd=child, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=parent_guard(os.getpid()))
            try:
                process.wait(timeout=remaining)
            finally:
                terminate(process)
        value = read_json(child / "result.json")
        if value["status"] != "completed":
            return {"status": "rejected", "experiment_sha256": experiment["content_sha256"],
                    "candidate_sha256": request["candidate"]["source_sha256"],
                    "reference_sha256": request["reference"]["source_sha256"],
                    "cases": [], "failure": {"side": name, "error": value.get("error")},
                    "receipts": {k: str(directory / k / "result.json") for k in results}}
        results[name] = value["data"]
    reports = []
    try:
        a, b = results["candidate"]["cases"], results["reference"]["cases"]
        if len(a) != len(b) or len(a) != len(config["cases"]):
            raise ContractError("对照测试案例数量不一致")
        for index, (candidate, reference) in enumerate(zip(a, b)):
            error = numeric_comparison(candidate["output"], reference["output"], atol, rtol)
            reports.append({"index": index, "status": "passed", "max_abs_error": error,
                            "atol": atol, "rtol": rtol, "finite": True,
                            "state_unchanged": candidate["state_unchanged"],
                            "input_unchanged": candidate["input_unchanged"]})
        interfaces = []
        a, b = results["candidate"].get("interface_cases", []), results["reference"].get("interface_cases", [])
        if len(a) != len(b) or len(a) != len(config.get("interface_cases", [])):
            raise ContractError("实际 Data/Backend 接口案例数量不一致")
        for index, (candidate, reference) in enumerate(zip(a, b)):
            error = numeric_comparison(candidate["output"], reference["output"], atol, rtol)
            if candidate["execution_facts"] != reference["execution_facts"]:
                raise ContractError("独立接口案例的实际数据/前向范围不一致")
            interfaces.append({"index": index, "status": "passed", "max_abs_error": error})
        status, failure = "passed", None
    except ContractError as exc:
        status, failure = "rejected", str(exc)
    return {"status": status, "cases": reports, "interface_cases": locals().get("interfaces", []), "failure": failure,
            "host_implementation_sha256": code_identity()["core_sha256"],
            "experiment_sha256": experiment["content_sha256"],
            "candidate_sha256": request["candidate"]["source_sha256"],
            "reference_sha256": request["reference"]["source_sha256"],
            "receipts": {k: str(directory / k / "result.json") for k in results},
            "method": "separate clean Python processes; reference never imports candidate",
            "scope": "finite numerical cases plus independent Safeguard semantic review; not a universal proof"}


def work(request, directory):
    kind = request["kind"]
    if kind == "llm":
        from runtime.model_router import call_once
        return call_once(request["spec"], directory, request["deadline_unix"])
    if kind == "experiment":
        from tools.experiment_worker import run_experiment
        return run_experiment(request, directory)
    if kind == "verification":
        return verification(request, directory)
    if kind == "literature":
        from tools.literature import search, fetch
        remaining = min(30, request["deadline_unix"] - time.time())
        if request["action"] == "search":
            return search(request["query"], directory / "literature", timeout=remaining)
        return fetch(request["url"], directory / "literature", timeout=remaining)
    if kind == "test_control":
        # 仅供基础设施故障注入，明确不生成真实科学 EvidenceCard。
        if request.get("sleep_seconds"):
            time.sleep(request["sleep_seconds"])
        if request.get("fail"):
            raise RuntimeError("explicit infrastructure fault injection")
        return {"source": "test_double", "value": request.get("value")}
    raise ContractError("未知工作进程任务类型")


def main():
    path = Path(sys.argv[1]).resolve()
    request, directory = read_json(path), path.parent
    atomic_json(directory / "started.json", {"process": process_identity(os.getpid()),
                                             "request_sha256": file_sha256(path), "started_unix": time.time()})
    start = time.monotonic()
    before = code_identity()
    def interrupted(signum, frame):
        if signum == signal.SIGALRM and request["kind"] in ("experiment", "verification"):
            raise TimeoutError("numeric task deadline exceeded")
        raise InterruptedError("task stopped by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_unix"] - time.time()))
    try:
        if before["core_sha256"] != request["core_sha256"]:
            raise ContractError("工作进程启动时实现版本已过期")
        with ExitStack() as stack:
            for key in sorted(request.get("resource_keys", [])):
                if key.startswith("gpu:"):
                    lock = ROOT / "EXP/cache/v2_resources" / (fingerprint(key) + ".lock")
                    stack.enter_context(file_lock(lock, timeout=max(0, request["deadline_unix"] - time.time())))
            data = work(request, directory)
        after = code_identity()
        if before != after:
            raise ContractError("任务执行期间核心或数值实现变化，结果不得进入科学证据")
        result = {"status": "completed", "data": data}
    except BaseException as exc:
        traceback.print_exc()
        failed_usage = exc.usage if isinstance(exc, StructuredOutputError) else None
        if isinstance(exc, StructuredOutputError) and before != code_identity():
            exc = ContractError("任务执行期间核心或数值实现变化，格式错误不得进入修复流程")
        error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": type(exc).__name__, "message": str(exc)}
        result = {"status": "interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed", "error": error}
        if failed_usage is not None:
            # A failed decode is still a paid/accounted model attempt, not an empty action.
            result["data"] = {"usage": failed_usage}
    signal.setitimer(signal.ITIMER_REAL, 0)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(task_id=request["task_id"], execution_id=request["execution_id"],
                  request_sha256=file_sha256(path), core_sha256=before["core_sha256"],
                  resources={"wall_seconds": time.monotonic() - start, "cpu_seconds": usage.ru_utime + usage.ru_stime,
                             "peak_rss_kib": usage.ru_maxrss, "physical_gpu": request.get("gpu"), "energy_joules": None},
                  finished_unix=time.time())
    if "experiment" in request:
        result.update(experiment_ref=request["experiment"]["id"], experiment_sha256=request["experiment"]["content_sha256"])
    try:
        if "torch" in sys.modules:
            import torch
            result["resources"].update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None)
    except ImportError:
        pass
    atomic_json(directory / "result.json", result)


if __name__ == "__main__":
    main()
