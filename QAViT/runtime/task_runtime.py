"""持久工作进程生命周期；调度决策与科学解释在 Workflow 中。"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from runtime.paths import QAVIT
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256, read_json
from runtime.task_queue import process_identity, process_alive
from runtime.identity import code_identity


class TaskRuntime:
    def __init__(self, root, queue, budget, settings):
        self.root, self.queue, self.budget, self.settings = Path(root), queue, budget, settings
        self.children = {}

    def launch(self, task, body, *, input_versions):
        from runtime.codex_cli import parent_guard
        attempt = task["attempt"] + 1
        execution = f"{task['task_id']}-a{attempt:04d}"
        directory = self.root / "jobs" / execution
        directory.mkdir(parents=True, exist_ok=True)
        estimated = body.pop("estimated_seconds", self.settings.get("task_timeout_seconds", 600))
        state = self.budget.reserve(execution, task["kind"], estimated_seconds=estimated,
                                    resources={"keys": task["resource_keys"]})
        deadline = min(self.budget.snapshot()["worker_deadline_unix"], time.time() + estimated)
        request = {**body, "task_id": task["task_id"], "execution_id": execution,
                   "deadline_unix": deadline, "resource_keys": task["resource_keys"],
                   "input_versions": input_versions, "memory_version": self.queue.journal.memory_version,
                   "core_sha256": code_identity()["core_sha256"]}
        path = directory / "request.json"
        # 未领取的 launch intent 可复用；已领取的尝试只能由 poll 接收，绝不重新启动。
        if path.exists():
            saved = read_json(path)
            for key in ("task_id", "execution_id", "input_versions"):
                if saved.get(key) != request[key]:
                    raise ContractError("未完成启动意图已过期，需显式恢复而非覆盖请求")
            if saved["deadline_unix"] <= time.time() or saved["memory_version"] != request["memory_version"] or saved["core_sha256"] != request["core_sha256"]:
                path = directory / ("request-" + fingerprint(request)[:20] + ".json")
                atomic_json(path, request)
            else:
                request = saved
        else:
            atomic_json(path, request)
        self.queue.claim(task["task_id"], input_versions=input_versions,
                         memory_version=request["memory_version"], request_path=str(path))
        env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET"))}
        threads = str(body.get("cpu_threads", self.settings.get("cpu_threads", 4)))
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        env["CUDA_VISIBLE_DEVICES"] = str(body["gpu"]) if body.get("gpu") is not None else ""
        try:
            with (directory / "worker.log").open("a") as log:
                process = subprocess.Popen([sys.executable, "-B", "-u", str(QAVIT / "runtime/task_worker.py"), str(path)],
                                           cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, preexec_fn=parent_guard(os.getpid()))
            self.children[execution] = process
            self.queue.change(task["task_id"], "running", patch={"process": process_identity(process.pid)})
        except BaseException as exc:
            self.queue.fail(task["task_id"], {"code": "launch_failed", "message": str(exc)}, retryable=True)
            self.budget.settle(execution, status="launch_failed")
            raise
        return execution

    def poll(self):
        """先接收完整回执，再判失联；死进程的中间文件永远不能构成 Evidence。"""
        completed = []
        for task in self.queue.snapshot().values():
            if task["state"] != "running":
                continue
            if not task.get("request_path"):
                # 宿主操作在提交前中断：按幂等键重放，没有外部工作进程。
                self.queue.change(task["task_id"], "retry", patch={"retry_after": 0})
                continue
            path = Path(task["request_path"])
            request = read_json(path)
            directory = path.parent
            identity = task.get("process")
            started = directory / "started.json"
            if started.exists():
                marker = read_json(started)
                if marker["request_sha256"] != file_sha256(path):
                    raise ContractError("工作进程启动记录与请求不一致")
                identity = marker["process"]
            process = self.children.get(request["execution_id"])
            if process:
                process.poll()  # reap 防止 zombie 被误认为活动任务。
            receipt = directory / "result.json"
            if receipt.exists():
                result = read_json(receipt)
                if any(result.get(k) != request[k] for k in ("task_id", "execution_id", "core_sha256")) or result.get("request_sha256") != file_sha256(path):
                    raise ContractError("任务回执身份/请求版本不一致")
                completed.append((task, result, receipt))
            elif process_alive(identity):
                if time.time() >= request["deadline_unix"] + 2:
                    self.stop_identity(identity)
            elif time.time() - task["started_unix"] > 2:
                # 父死亡保护保证主进程崩溃不会留下无主实验。未知消耗不记为0。
                result = {"status": "interrupted", "task_id": task["task_id"],
                          "execution_id": request["execution_id"], "request_sha256": file_sha256(path),
                          "core_sha256": request["core_sha256"], "resources": {"wall_seconds": None},
                          "error": {"code": "worker_lost", "message": "进程已结束且无完整回执；保留部分文件"}}
                atomic_json(directory / "interrupted.json", result)
                completed.append((task, result, directory / "interrupted.json"))
        return completed

    def settle(self, result):
        self.budget.settle(result["execution_id"], status=result["status"], resources=result.get("resources"),
                           usage=result.get("data", {}).get("usage"))
        child = self.children.pop(result["execution_id"], None)
        if child and child.poll() is None:
            child.wait(timeout=2)

    @staticmethod
    def stop_identity(identity):
        if not process_alive(identity):
            return
        try:
            os.killpg(identity["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 3
            while process_alive(identity) and time.monotonic() < deadline:
                time.sleep(.05)
            if process_alive(identity):
                os.killpg(identity["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass

    def stop(self):
        try:
            for task in self.queue.snapshot().values():
                if task["state"] != "running" or not task.get("request_path"):
                    continue
                identity = task.get("process")
                marker = Path(task["request_path"]).parent / "started.json"
                if marker.exists():
                    identity = read_json(marker)["process"]
                self.stop_identity(identity)
        finally:
            # 文件系统故障不能阻止释放本进程实际创建的子进程。
            for child in self.children.values():
                if child.poll() is None:
                    self.stop_identity(process_identity(child.pid))
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass

