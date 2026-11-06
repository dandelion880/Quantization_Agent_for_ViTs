"""单个持久任务尝试的工作进程。只写本次作业回执，不直接修改科研日志。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlib import ExitStack
import os
import resource
import signal
import time
import traceback

from runtime.paths import ROOT
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256, file_lock, read_json
from runtime.task_queue import process_identity
from runtime.identity import code_identity
from runtime.subscription import StructuredOutputError


def failure_origin(exc, request):
    """Use actual failing frames, not an exception's name, to route code repair."""
    candidate = (request.get('approval') or {}).get('candidate') or {}
    source = candidate.get('path')
    if not source:
        return 'host'
    frames = traceback.extract_tb(exc.__traceback__)
    if any(Path(frame.filename).resolve() == Path(source).resolve() for frame in frames):
        return 'implementation'
    # These contracts check returned values/state after the plugin frame has exited.
    validators = {'tools/result_contract.py', 'core/execution_contract.py'}
    if isinstance(exc, ContractError) and frames:
        last = frames[-1]
        if (any(last.filename.endswith('/' + name) for name in validators)
                or last.name in ('execute_custom', 'tensor_probe', 'evaluate_services')
                or (last.name == 'run_experiment' and any(term in str(exc) for term in
                    ('缺少供宿主重载', '导出包必须位于', '改变了冻结基准模型')))):
            return 'implementation'
    return 'host'


def work(request, directory):
    kind = request["kind"]
    if kind == "llm":
        from runtime.model_router import call_once
        return call_once(request["spec"], directory, request["deadline_unix"])
    if kind == "experiment":
        from tools.experiment_worker import run_experiment
        return run_experiment(request, directory)
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
    os.environ["QAVIT_PROCESS_REGISTRY"] = str(directory / "processes")
    atomic_json(directory / "started.json", {"process": process_identity(os.getpid()),
                                             "request_sha256": file_sha256(path), "started_unix": time.time()})
    start = time.monotonic()
    before = code_identity()
    def interrupted(signum, frame):
        if signum == signal.SIGALRM and request["kind"] == "experiment":
            raise TimeoutError("numeric task deadline exceeded")
        raise InterruptedError("task stopped by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_unix"] - time.time()))
    try:
        if before["core_sha256"] != request["core_sha256"]:
            raise ContractError("工作进程启动时实现版本已过期")
        with ExitStack() as stack:
            if request["kind"] == "experiment":
                # CPU/GPU数值任务均共用单槽；Coding及Safeguard可在其它请求阶段推进。
                stack.enter_context(file_lock(ROOT / "EXP/cache/v2_resources/numeric.lock",
                    timeout=max(0, request["deadline_unix"] - time.time())))
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
        if failed_usage is None and request["kind"] == "llm" and (directory / "llm_audit.json").exists():
            failed_usage = read_json(directory / "llm_audit.json").get("usage")
        if isinstance(exc, StructuredOutputError) and before != code_identity():
            exc = ContractError("任务执行期间核心或数值实现变化，格式错误不得进入修复流程")
        error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": type(exc).__name__, "message": str(exc)}
        if request['kind'] == 'experiment':
            error['origin'] = failure_origin(exc, request)
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
