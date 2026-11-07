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
        self._gpu_bindings = None

    def launch(self, task, body, *, input_versions):
        from runtime.gpu_resources import acquire, gpu_bindings
        fds = []
        try:
            if task['kind'] == 'experiment' and 'gpu_pool' in body:
                from runtime.gpu_resources import ResourceBusy
                # An exited worker can release its OS lease before its durable
                # receipt is accepted. The queue claim must agree with placement.
                held = {key for current in self.queue.snapshot().values() if current['state'] == 'running'
                        for key in current['resource_keys']}
                pool = [gpu for gpu in body['gpu_pool'] if 'gpu:' + str(gpu) not in held]
                if body['gpu_pool'] and (not pool or body['pinned_gpu'] is not None and body['pinned_gpu'] not in pool):
                    raise ResourceBusy('GPU receipt still pending; task remains queued')
                # CUDA enumeration can briefly register GPU activity. Resolve once
                # per runtime (including resume), never on every busy-queue poll.
                if pool and self._gpu_bindings is None:
                    self._gpu_bindings = gpu_bindings(self.settings.get('allowed_gpus', pool), self.settings)
                by_index = {r['index']: r for r in self._gpu_bindings or []}
                bindings = [by_index[g] for g in pool if g in by_index]
                available = [row['index'] for row in bindings]
                if pool and (not available or body['pinned_gpu'] is not None and body['pinned_gpu'] not in available):
                    raise ResourceBusy('Authorized GPU is not currently visible; task remains queued')
                gpus, fds = acquire(available, body['gpu_count'], body['pinned_gpu'], bindings=bindings)
                bindings = [row for row in bindings if row['index'] in gpus]
                body.update(gpus=gpus, gpu=gpus[0] if gpus else None, gpu_bindings=bindings)
                if 'parallelism' in body:
                    body['parallelism'].update(allocated_gpus=len(gpus))
                keys = [k for k in task['resource_keys'] if not k.startswith(('gpu:', 'model:'))]
                self.queue.change(task['task_id'], task['state'], patch={
                    'resource_keys': keys + ['gpu:' + str(g) for g in gpus], 'allocated_gpus': gpus})
                task = self.queue.get(task['task_id'])
            return self._launch(task, body, input_versions=input_versions, lease_fds=fds)
        finally:
            # close, not LOCK_UN: the inherited open description owns the lease.
            for fd in fds:
                os.close(fd)

    def _launch(self, task, body, *, input_versions, lease_fds=()):
        from runtime.codex_cli import parent_guard
        budget = self.budget.check()
        attempt = task["attempt"] + 1
        execution = f"{task['task_id']}-a{attempt:04d}"
        while execution in budget["settlements"]:
            attempt += 1
            execution = f"{task['task_id']}-a{attempt:04d}"
        directory = self.root / "jobs" / execution
        directory.mkdir(parents=True, exist_ok=True)
        estimated = body.pop("estimated_seconds", self.settings.get("task_timeout_seconds", 600))
        state = self.budget.reserve(execution, task["kind"], estimated_seconds=estimated,
                                    resources={"keys": task["resource_keys"]})
        deadline = time.time() + estimated
        if not (task['kind'] == 'experiment' and self.budget.excludes_numeric):
            deadline = min(self.budget.snapshot()["worker_deadline_unix"], deadline)
        request = {**body, "task_id": task["task_id"], "execution_id": execution,
                   "deadline_unix": deadline, "resource_keys": task["resource_keys"],
                   "budget_segment_id": (budget.get("active_segment") or {}).get("segment_id"),
                   "input_versions": input_versions, "memory_version": self.queue.journal.memory_version,
                   "core_sha256": code_identity()["core_sha256"]}
        path = directory / "request.json"
        # 未领取的 launch intent 可复用；已领取的尝试只能由 poll 接收，绝不重新启动。
        if path.exists():
            saved = read_json(path)
            for key in ("task_id", "execution_id", "input_versions"):
                if saved.get(key) != request[key]:
                    raise ContractError("未完成启动意图已过期，需显式恢复而非覆盖请求")
            if (saved.get('gpus') != request.get('gpus') or saved.get('gpu_bindings') != request.get('gpu_bindings')
                    or saved['resource_keys'] != request['resource_keys']
                    or saved["deadline_unix"] <= time.time() or saved["deadline_unix"] > deadline
                    or saved.get("budget_segment_id") != request["budget_segment_id"]
                    or saved["memory_version"] != request["memory_version"] or saved["core_sha256"] != request["core_sha256"]):
                path = directory / ("request-" + fingerprint(request)[:20] + ".json")
                atomic_json(path, request)
            else:
                request = saved
        else:
            atomic_json(path, request)
        self.queue.claim(task["task_id"], input_versions=input_versions,
                         memory_version=request["memory_version"], request_path=str(path), execution_attempt=attempt)
        env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET"))}
        threads = str(body.get("cpu_threads", self.settings.get("cpu_threads", 4)))
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        env["CUDA_VISIBLE_DEVICES"] = str(body["gpu"]) if body.get("gpu") is not None else ""
        if body.get('gpu_bindings'):
            env['CUDA_VISIBLE_DEVICES'] = body['gpu_bindings'][0]['uuid']
            env['QAVIT_ASSIGNED_GPU_UUIDS'] = ','.join(g['uuid'] for g in body['gpu_bindings'])
        env['QAVIT_GPU_LEASE_FDS'] = ','.join(map(str, lease_fds))
        env['QAVIT_RESOURCE_ASSIGNED'] = '1'
        try:
            with (directory / "worker.log").open("a") as log:
                process = subprocess.Popen([sys.executable, "-B", "-u", str(QAVIT / "runtime/task_worker.py"), str(path)],
                                           cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, preexec_fn=parent_guard(os.getpid()), pass_fds=tuple(lease_fds))
            self.children[execution] = process
            atomic_json(directory / "processes/worker.json", process_identity(process.pid))
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
                # A durable result is independent of slow Python/CUDA teardown.
                # Keep the task/leases running until all registered processes exit.
                live = self.job_processes(directory, identity)
                if live:
                    seen = task.get('receipt_seen_unix')
                    if seen is None:
                        self.queue.change(task['task_id'], 'running', patch={'receipt_seen_unix': time.time()})
                    elif time.time() - seen >= self.settings.get('worker_exit_grace_seconds', 30):
                        for child_identity in live:
                            self.stop_identity(child_identity)
                    continue
                completed.append((task, result, receipt))
            elif process_alive(identity):
                if time.time() >= request["deadline_unix"] + 2:
                    self.stop_identity(identity)
            elif time.time() - task["started_unix"] > 2:
                live = self.job_processes(directory, identity)
                if live:
                    for child_identity in live:
                        self.stop_identity(child_identity)
                    continue
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
        child = self.children.get(result["execution_id"])
        if child and child.poll() is not None:
            self.children.pop(result['execution_id'], None)

    @staticmethod
    def job_processes(directory, identity=None):
        identities = [identity, *(read_json(p) for p in (directory / 'processes').glob('*.json'))]
        unique = {row['pid']: row for row in identities if process_alive(row)}
        return list(unique.values())

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
            # CLI/account preflights use separate sessions. Their identities remain
            # available even after a worker dies or a receipt clears task.process.
            for identity in self.live_processes():
                self.stop_identity(identity)
            deadline = time.monotonic() + 3
            while self.live_processes() and time.monotonic() < deadline:
                time.sleep(.05)
            if self.live_processes():
                raise ContractError("工作进程尚未完全退出，不能结算暂停")

    def live_processes(self):
        identities = [read_json(path) for path in (self.root / "jobs").glob("*/processes/*.json")]
        for task in self.queue.snapshot().values():
            if task.get("process"):
                identities.append(task["process"])
            if task.get("request_path"):
                marker = Path(task["request_path"]).parent / "started.json"
                if marker.exists():
                    identities.append(read_json(marker)["process"])
        return [identity for identity in identities if process_alive(identity)]

    def settle_abandoned_intents(self):
        """Only after all processes stopped and durable receipts were accepted."""
        if self.live_processes():
            raise ContractError("存在在途进程，不能结算旧启动意图")
        state = self.budget.snapshot()
        for key in state["reservations"].keys() - state["settlements"].keys():
            self.budget.settle(key, status="interrupted_unsettled_attempt",
                               resources={"wall_seconds": None}, usage=None)

