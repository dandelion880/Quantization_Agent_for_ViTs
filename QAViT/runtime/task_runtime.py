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
from runtime.configuration import numeric_task_timeout


class TaskRuntime:
    def __init__(self, root, queue, budget, settings):
        self.root, self.queue, self.budget, self.settings = Path(root), queue, budget, settings
        self.children = {}
        self._gpu_bindings = None

    def unserved_gpu_candidates(self, task):
        """Include L3 plans still passing through serial Coding, without leases.

        Candidate identity follows the immutable hypothesis version through the
        request's parent link. Completed/failed candidates free their reservation;
        a process with an unaccepted receipt still owns its primary card.
        """
        from runtime.task_queue import TERMINAL
        from tools.experiment_worker import backend_device
        tasks = self.queue.snapshot(fields=('task_id', 'kind', 'state', 'inputs'))

        def identity(row):
            seen = set()
            root_id = row['task_id']
            while row and row['task_id'] not in seen:
                seen.add(row['task_id'])
                root_id = row['task_id']
                inputs = row['inputs']
                context = inputs.get('context', {})
                ref = inputs.get('hypothesis_ref') or context.get('hypothesis_ref')
                if ref:
                    return (inputs.get('round', context.get('round')), ref)
                row = (tasks.get(inputs.get('parent_task')) or
                       tasks.get(inputs.get('origin_task')))
            return ('task', root_id)

        requests = {identity(t) for t in tasks.values() if t['kind'] == 'experiment_request'}
        waiting, served = set(), set()
        for row in tasks.values():
            inputs = row['inputs']
            key = identity(row)
            if row['kind'] == 'llm' and inputs.get('purpose') == 'plan_l3':
                if row['state'] not in TERMINAL or (row['state'] == 'completed' and key not in requests):
                    waiting.add(key)
            elif row['kind'] == 'experiment_request' and row['state'] not in TERMINAL:
                if inputs.get('hypothesis_ref') or inputs.get('draft', {}).get('level') == 'L3':
                    waiting.add(key)
            elif row['kind'] == 'experiment' and row['state'] not in TERMINAL:
                ref = inputs.get('experiment_ref')
                if not ref:
                    continue
                spec = self.queue.journal.card(ref)['payload']['execution']
                if str(backend_device(spec)).startswith('cuda'):
                    (served if row['state'] == 'running' else waiting).add(key)
        return len(waiting - served - {identity(task)})

    def launch(self, task, body, *, input_versions):
        from runtime.gpu_resources import acquire, gpu_bindings
        fds = []
        try:
            if task['kind'] == 'experiment' and 'gpu_pool' in body:
                from runtime.gpu_resources import ResourceBusy, ResourceUnavailable
                if body.get('requires_gpu') and not body['gpu_pool']:
                    raise ResourceUnavailable('No authorized GPU is available; preserve task for resource recovery')
                # An exited worker can release its OS lease before its durable
                # receipt is accepted. The queue claim must agree with placement.
                held = self.queue.held_resources()
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
                    raise ResourceUnavailable('Authorized GPU is not currently visible; resume after resource recovery')
                reserve = self.unserved_gpu_candidates(task) if body['gpu_count'] > 1 else 0
                gpus, fds = acquire(available, body['gpu_count'], body['pinned_gpu'], bindings=bindings,
                                    reserve=reserve)
                bindings = [row for row in bindings if row['index'] in gpus]
                body.update(gpus=gpus, gpu=gpus[0] if gpus else None, gpu_bindings=bindings)
                if 'parallelism' in body:
                    body['parallelism'].update(allocated_gpus=len(gpus),
                        unserved_candidate_reservations=reserve,
                        allocation_policy='candidate primary GPUs before intra-candidate parallelism')
                keys = [k for k in task['resource_keys'] if not k.startswith(('gpu:', 'model:'))]
                self.queue.change(task['task_id'], task['state'], patch={
                    'resource_keys': keys + ['gpu:' + str(g) for g in gpus], 'allocated_gpus': gpus,
                    **({'resource_wait':None} if task.get('resource_wait') else {})})
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
        default_timeout = (numeric_task_timeout(self.settings, body.get('experiment', {}).get('payload', {}).get('level', 'L1'))
                           if task['kind'] == 'experiment' else self.settings.get("task_timeout_seconds", 600))
        estimated = body.pop("estimated_seconds", default_timeout)
        state = self.budget.reserve(execution, task["kind"], estimated_seconds=estimated,
                                    resources={"keys": task["resource_keys"]})
        deadline = time.time() + estimated
        if budget.get('absolute_worker_deadline_unix') is not None:
            deadline = min(deadline, budget['absolute_worker_deadline_unix'])
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
            if ({k: v for k, v in saved.items() if k != 'deadline_unix'} !=
                    {k: v for k, v in request.items() if k != 'deadline_unix'}
                    or saved["deadline_unix"] <= time.time() or saved["deadline_unix"] > deadline):
                path = directory / ("request-" + fingerprint(request)[:20] + ".json")
                atomic_json(path, request)
            else:
                request = saved
        else:
            atomic_json(path, request)
        self.queue.claim(task["task_id"], input_versions=input_versions,
                         memory_version=request["memory_version"], request_path=str(path), execution_attempt=attempt)
        env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET"))}
        # Placement, thread settings and descendants all use the exact persisted
        # request selected above, including a recovered launch intent.
        threads = str(request.get("cpu_threads", self.settings.get("cpu_threads", 4)))
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        env["CUDA_VISIBLE_DEVICES"] = str(request["gpu"]) if request.get("gpu") is not None else ""
        env.pop('QAVIT_ASSIGNED_GPU_UUIDS', None)
        if request.get('gpu_bindings'):
            env['CUDA_VISIBLE_DEVICES'] = request['gpu_bindings'][0]['uuid']
            env['QAVIT_ASSIGNED_GPU_UUIDS'] = ','.join(g['uuid'] for g in request['gpu_bindings'])
        env['QAVIT_TASK_REQUEST_PATH'] = str(path.resolve())
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
        for task in self.queue.snapshot(states={"running"}).values():
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
            transferred = directory / 'dual_secondary_transferred.json'
            if len(request.get('gpus', [])) == 2 and not task.get('secondary_lease_transferred') and transferred.is_file():
                phase = read_json(transferred)
                secondary = request['gpus'][1]
                child = directory / 'processes' / 'dual-segmentation.json'
                if phase['gpu'] == secondary and child.is_file() and phase['process'] == read_json(child):
                    self.queue.change(task['task_id'], 'running', patch={'secondary_lease_transferred': True,
                        'resource_keys': [key for key in task['resource_keys'] if key != 'gpu:' + str(secondary)]})
                    task = self.queue.get(task['task_id'])
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
            for task in self.queue.snapshot(states={"running"}, fields=("process", "request_path")).values():
                if not task.get("request_path"):
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
        for task in self.queue.snapshot(fields=("process", "request_path")).values():
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

