"""持久依赖队列。状态变更与 TaskCard 快照在同一提交中发布。"""
from copy import deepcopy
import os
from pathlib import Path
import time

from core.cards import ContractError, canonical, fingerprint, make_card
from core.journal import read_json


TERMINAL = {"completed", "failed", "cancelled", "ineligible"}
TRANSITIONS = {
    "pending": {"ready", "cancelled", "failed"},
    "ready": {"running", "cancelled", "needs_clarification"},
    "running": {"completed", "failed", "retry", "waiting", "needs_clarification", "cancelled"},
    "retry": {"ready", "cancelled", "failed"},
    "waiting": {"ready", "failed", "cancelled", "needs_clarification"},
    "needs_clarification": {"ready", "cancelled"},
    "failed": {"ready"},  # 只有显式恢复操作可使用，不自动重置次数。
    "completed": set(), "cancelled": set(), "ineligible": set(),
}


def process_identity(pid):
    """PID + boot_id + /proc starttime，避免恢复时误认复用的 PID。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rindex(")") + 2:].split()
        return {"pid": int(pid), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": tail[19], "state": tail[0]}
    except (OSError, ValueError, IndexError):
        return None


def process_alive(identity):
    if not identity:
        return False
    actual = process_identity(identity["pid"])
    return bool(actual and actual["state"] != "Z" and all(actual.get(k) == identity.get(k)
                                                         for k in ("pid", "boot_id", "start_ticks")))


class GlobalTaskQueue:
    def __init__(self, journal):
        self.journal = journal
        self._tasks, self._event_index = {}, 0

    def _refresh_tasks(self):
        self.journal.refresh()
        for event in self.journal.events[self._event_index:]:
            if event["type"] == "task_snapshot":
                self._tasks[event["task"]["task_id"]] = deepcopy(event["task"])
        self._event_index = len(self.journal.events)
        return self._tasks

    def snapshot(self):
        self._refresh_tasks()
        return deepcopy(self._tasks)

    def get(self, task_id):
        try:
            return deepcopy(self._refresh_tasks()[task_id])
        except KeyError:
            raise ContractError("未知任务: " + task_id) from None

    def enqueue(self, key, kind, *, role=None, inputs=None, depends_on=(), resource_keys=(), max_retries=2):
        task_id = "task-" + fingerprint({"run": self.journal.run_id, "key": key})[:24]
        current = self.snapshot()
        if task_id in current:
            return task_id
        if set(depends_on) - set(current):
            raise ContractError("任务依赖不存在")
        task = {"task_id": task_id, "key": key, "kind": kind, "role": role,
                "inputs": deepcopy(inputs or {}), "depends_on": list(dict.fromkeys(depends_on)),
                "state": "pending", "input_versions": {}, "memory_version": self.journal.memory_version,
                "resource_keys": list(dict.fromkeys(resource_keys)), "attempt": 0,
                "step": 0, "retry_count": 0, "max_retries": max_retries,
                "outputs": [], "history": [], "tool_results": [], "created_unix": time.time()}
        self._publish(task, "enqueue/" + key)
        return task_id

    def _publish(self, task, key, *, cards=(), events=(), expected_sequence=None):
        previous = self.journal.card(task["task_card"]) if task.get("task_card") else None
        data = {k: v for k, v in task.items() if k != "task_card"}
        refs = list(dict.fromkeys(task.get("outputs", []) + list(task.get("input_versions", {}))))
        card = make_card("TaskCard", self.journal.run_id, data, previous=previous, refs=refs)
        path = self.journal.card_path(card['id'])
        if (task['inputs'].get('purpose') == 'record' and task['state'] == 'completed'
                and path.exists() and card['id'] not in self.journal.cards):
            pending = read_json(path)
            # cards_written 后断电：同一已验证草稿的重试只允许时间戳不同。
            stable = lambda value: {k: v for k, v in value.items() if k not in ('changed_unix', 'finished_unix')}
            if (stable(pending['payload']) == stable(data) and pending['refs'] == card['refs']
                    and pending.get('previous') == card.get('previous')):
                card = pending
                task = dict(pending['payload'], task_card=task.get('task_card'))
        value = dict(task, task_card=card["id"])
        return self.journal.commit(key, cards=[*cards, card],
                                   events=[*events, {"type": "task_snapshot", "task": value}],
                                   expected_sequence=expected_sequence)

    def change(self, task_id, state, *, patch=None, key=None, cards=(), events=()):
        task = self.get(task_id)
        if state != task["state"] and state not in TRANSITIONS[task["state"]]:
            raise ContractError(f"不合法任务转换 {task['state']} -> {state}")
        seq = self.journal.sequence
        task.update(deepcopy(patch or {}), state=state, changed_unix=time.time())
        return self._publish(task, key or f"task/{task_id}/{state}/{seq}", cards=cards, events=events,
                             expected_sequence=seq)

    def promote(self):
        for task_id, task in list(self._refresh_tasks().items()):
            if task["state"] not in ("pending", "retry", "waiting"):
                continue
            tasks = self._refresh_tasks()
            if task["state"] == "waiting":
                waiting = task.get("waiting_for", [])
                if waiting and all(tasks[x]["state"] in TERMINAL for x in waiting):
                    results = task["tool_results"] + [{"task": x, "state": tasks[x]["state"],
                                                       "call_id": tasks[x]["inputs"].get("call_id"),
                                                       "name": tasks[x]["inputs"].get("name", "request_experiment" if tasks[x]["kind"] == "experiment_request" else tasks[x]["kind"]),
                                                       "outputs": tasks[x]["outputs"],
                                                       "result": tasks[x].get("result"),
                                                       "error": tasks[x].get("error")} for x in waiting]
                    for row in results:
                        if len(canonical(row.get("result"))) > 24000:
                            child = tasks[row["task"]]
                            row["result"] = {"complete_result_asset": str(self.journal.card_path(child["task_card"])),
                                             "summary": "大型原始工具结果按完整TaskCard引用提供；使用read_asset分页读取。"}
                    self.change(task_id, "ready", patch={"tool_results": results, "waiting_for": [],
                                                         "step": task["step"] + 1, "retry_count": 0,
                        **({"repair_count": 0, "repair_feedback": {}} if task["kind"] == "llm" else {})})
            elif all(tasks[x]["state"] == "completed" for x in task["depends_on"]):
                if task.get("retry_after", 0) <= time.time():
                    self.change(task_id, "ready")
            elif any(tasks[x]["state"] in ("failed", "cancelled", "ineligible") for x in task["depends_on"]):
                self.change(task_id, "failed", patch={"error": {"code": "dependency_failed",
                                                                  "dependencies": task["depends_on"]}})

    def ready(self):
        self.promote()
        tasks = self.snapshot()
        held = {r for t in tasks.values() if t["state"] == "running" for r in t["resource_keys"]}
        return [t for t in tasks.values() if t["state"] == "ready" and not held.intersection(t["resource_keys"])]

    def claim(self, task_id, *, input_versions=None, memory_version=None, request_path=None, execution_attempt=None):
        task = self.get(task_id)
        if task["state"] != "ready":
            raise ContractError("只能领取 ready 任务")
        held = {r for t in self.snapshot().values() if t["state"] == "running" for r in t["resource_keys"]}
        if held.intersection(task["resource_keys"]):
            raise ContractError("相同资源已有运行任务")
        attempt = task["attempt"] + 1 if execution_attempt is None else execution_attempt
        if attempt <= task["attempt"]:
            raise ContractError("尝试编号不能复用已领取历史")
        patch = {"attempt": attempt, "started_unix": time.time(),
                 "input_versions": input_versions or {},
                 "memory_version": self.journal.memory_version if memory_version is None else memory_version,
                 "request_path": request_path, "process": None, "receipt_seen_unix": None}
        self.change(task_id, "running", patch=patch)
        return self.get(task_id)

    def finish(self, task_id, *, outputs=(), result=None, cards=(), events=()):
        task = self.get(task_id)
        return self.change(task_id, "completed", patch={"outputs": list(outputs), "result": result,
                                                        "finished_unix": time.time(), "process": None},
                           key=f"complete/{task_id}/{task['attempt']}/{task['step']}", cards=cards, events=events)

    def fail(self, task_id, error, *, retryable=False):
        task = self.get(task_id)
        retry = retryable and task["retry_count"] < task["max_retries"]
        self.change(task_id, "retry" if retry else "failed", patch={
            "error": error, "process": None, "retry_count": task["retry_count"] + int(retry),
            "retry_after": time.time() + min(2 ** task["retry_count"], 8) if retry else None})

    def cancel_unfinished(self, reason):
        for task_id, task in self.snapshot().items():
            if task["state"] not in TERMINAL:
                self.change(task_id, "cancelled", patch={"error": {"code": reason}})
