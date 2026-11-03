"""持久墙钟预算；L1/L2 没有固定次数限制，恢复不重置已用时间。"""
from pathlib import Path
import shutil
import time

from core.cards import ContractError


class BudgetExhausted(ContractError):
    pass


class RequestBudgetInsufficient(ContractError):
    """拒绝单个请求；剩余总预算仍可用于判断、复盘与收尾。"""


class BudgetLedger:
    def __init__(self, journal, config=None, *, clock=time.time, started_unix=None):
        self.journal, self.clock = journal, clock
        if not any(e["type"] == "budget_started" for e in journal.events):
            if config is None:
                raise ContractError("缺少已保存预算")
            cfg = dict(config)
            cfg.setdefault("max_rounds", 10)
            cfg.setdefault("wall_seconds", 6 * 3600)
            if type(cfg["max_rounds"]) is not int or cfg["max_rounds"] < 1 or cfg["wall_seconds"] <= 0:
                raise ContractError("预算轮次/时间非法")
            reserves = [cfg.get(k, 0) for k in ("closeout_reserve_seconds", "shutdown_reserve_seconds")]
            if any(v < 0 for v in reserves) or sum(reserves) >= cfg["wall_seconds"]:
                raise ContractError("复盘/停机预留须非负且小于总墙钟；仅由运行配置指定")
            started = min(clock(), started_unix) if started_unix is not None else clock()
            deadline = min(started + cfg["wall_seconds"], cfg.get("parent_deadline_unix", float("inf")))
            journal.commit("budget/start", events=[{"type": "budget_started", "config": cfg,
                                                     "started_unix": started, "deadline_unix": deadline}])

    def snapshot(self):
        self.journal.refresh()
        value = None
        reservations, settlements, rounds, extensions = {}, {}, set(), []
        for event in self.journal.events:
            kind = event["type"]
            if kind == "budget_started":
                value = {k: v for k, v in event.items() if k not in ("type", "commit_sequence")}
            elif kind == "budget_extended":
                value["deadline_unix"] += event["seconds"]
                extensions.append(event)
            elif kind == "budget_reserved":
                reservations[event["reservation_id"]] = event
            elif kind == "budget_settled":
                settlements[event["reservation_id"]] = event
            elif kind == "round_started":
                rounds.add(event["round"])
        now = self.clock()
        shutdown = value["config"].get("shutdown_reserve_seconds", 0)
        closeout = value["config"].get("closeout_reserve_seconds", 0)
        return {**value, "as_of_unix": now, "elapsed_seconds": max(0, now - value["started_unix"]),
                "remaining_seconds": max(0, value["deadline_unix"] - now),
                "worker_deadline_unix": value["deadline_unix"] - shutdown,
                "research_remaining_seconds": max(0, value["deadline_unix"] - now - shutdown - closeout),
                "rounds_started": sorted(rounds), "reservations": reservations,
                "settlements": settlements, "extensions": extensions,
                "l1_l2_call_limit": None, "llm_call_limit": value["config"].get("llm_call_limit")}

    def input_receipt(self):
        """同一账本快照的可读输入；不另设额度，不改变 check/reserve 的计算。"""
        state = self.snapshot()
        active = [r for key, r in state["reservations"].items() if key not in state["settlements"]]
        estimates = {}
        for row in active:
            estimates[row["kind"]] = estimates.get(row["kind"], 0) + row["estimated_seconds"]
        calls = sum(r["kind"] == "llm" for r in state["reservations"].values())
        cap = state["llm_call_limit"]
        return {"source": "BudgetLedger.snapshot/check/reserve", **{k: state[k] for k in (
                    "as_of_unix", "deadline_unix", "worker_deadline_unix", "elapsed_seconds",
                    "remaining_seconds", "research_remaining_seconds")},
                "closeout_reserve_seconds": state["config"].get("closeout_reserve_seconds", 0),
                "shutdown_reserve_seconds": state["config"].get("shutdown_reserve_seconds", 0),
                "unsettled_reservations": len(active), "unsettled_estimated_seconds_by_kind": estimates,
                "settled_reservations": len(state["settlements"]),
                "llm_calls": {"limit": cap, "reserved": calls, "remaining": max(0, cap - calls) if cap is not None else None},
                "meaning": "remaining_seconds 是硬截止余额；research_remaining_seconds 已扣复盘/停机预留，"
                           "非 LLM 请求的 estimated_seconds 按此余额检查。未结算估计仅展示，现有账本不再扣减它们；"
                           "LLM estimated_seconds 是调用超时上限，不按实验耗时预扣。null 调用上限表示未配置；"
                           "磁盘、资源与实际启动时预算仍按原入口检查。"}

    def check(self, *, estimated_seconds=0, check_disk=True):
        state = self.snapshot()
        if state["remaining_seconds"] <= 0:
            raise BudgetExhausted("墙钟硬预算耗尽；恢复不重置计时")
        if estimated_seconds > state["research_remaining_seconds"]:
            raise RequestBudgetInsufficient("本次请求声明耗时超过可用科研预算；总预算及复盘/停机预留未耗尽")
        cfg = state["config"]
        if check_disk:
            if shutil.disk_usage(self.journal.root).free < cfg.get("min_free_bytes", 0):
                raise BudgetExhausted("共享磁盘剩余空间低于运行门槛")
            cap = cfg.get("max_artifact_bytes")
            if cap is not None:
                size = 0
                for path in self.journal.root.rglob("*"):
                    try:
                        if path.is_file() and not path.is_symlink():
                            size += path.stat().st_size
                    except FileNotFoundError:
                        # worker 原子发布可在枚举后移走临时文件；其它存储错误仍上报。
                        continue
                if size >= cap:
                    raise BudgetExhausted("本次运行磁盘预算耗尽")
        return state

    def start_round(self, number):
        state = self.check()
        if number in state["rounds_started"]:
            return
        if not 1 <= number <= state["config"]["max_rounds"]:
            raise BudgetExhausted("大循环预算耗尽")
        self.journal.commit(f"budget/round/{number}", events=[{"type": "round_started", "round": number}])

    def reserve(self, reservation_id, kind, *, estimated_seconds, resources=None):
        state = self.snapshot()
        if reservation_id in state["reservations"]:
            return state["reservations"][reservation_id]
        # LLM耗时未知，estimated_seconds在这里是单次超时上限而非必须预留
        # 完整完成时间。实际worker deadline仍取绝对硬截止。否则将剩余时间
        # 作为timeout后，几毫秒启动开销就会触发 remaining < estimated 的竞态。
        self.check(estimated_seconds=0 if kind == "llm" else estimated_seconds)
        cap = state["config"].get("llm_call_limit")
        if kind == "llm" and cap is not None:
            count = sum(e["kind"] == "llm" for e in state["reservations"].values())
            if count >= cap:
                raise BudgetExhausted("显式配置的 LLM 调用预算耗尽")
        event = {"type": "budget_reserved", "reservation_id": reservation_id, "kind": kind,
                 "estimated_seconds": estimated_seconds, "resources": resources or {},
                 "created_unix": self.clock()}
        self.journal.commit("budget/reserve/" + reservation_id, events=[event])
        return event

    def settle(self, reservation_id, *, status, resources=None, usage=None):
        if reservation_id not in self.snapshot()["reservations"]:
            raise ContractError("不能结算不存在的预算预留")
        self.journal.commit("budget/settle/" + reservation_id, events=[{
            "type": "budget_settled", "reservation_id": reservation_id, "status": status,
            "resources": resources or {}, "reported_usage": usage,
            "settled_unix": self.clock()}])

    def extend(self, seconds, reason):
        if seconds <= 0 or not reason.strip():
            raise ContractError("显式预算扩展需要正数秒数和原因")
        state = self.snapshot()
        if state["deadline_unix"] + seconds > state["config"].get("parent_deadline_unix", float("inf")):
            raise ContractError("子运行不能延长到父验证预算以外；须先明确解决父预算授权")
        self.journal.commit(f"budget/extension/{self.journal.sequence}", events=[{
            "type": "budget_extended", "seconds": seconds, "reason": reason,
            "authorized_action": "explicit CLI budget extension", "when": self.clock()}])
