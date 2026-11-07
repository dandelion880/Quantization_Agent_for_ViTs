"""持久墙钟预算；L1/L2 没有固定次数限制，恢复不重置已用时间。"""
from pathlib import Path
import shutil
import time
import os

from core.cards import ContractError
from core.journal import read_json, file_sha256
from runtime.task_queue import process_identity, process_alive


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
            if cfg.get("clock_mode", "continuous") not in ("continuous", "active_segments"):
                raise ContractError("未知预算计时协议")
            if cfg.get('exclude_numeric_wait') and cfg.get('clock_mode') != 'active_segments':
                raise ContractError('exclude_numeric_wait 使用 active_segments 持久运行段计时')
            if type(cfg["max_rounds"]) is not int or cfg["max_rounds"] < 1 or cfg["wall_seconds"] <= 0:
                raise ContractError("预算轮次/时间非法")
            reserves = [cfg.get(k, 0) for k in ("closeout_reserve_seconds", "shutdown_reserve_seconds")]
            if any(v < 0 for v in reserves) or sum(reserves) >= cfg["wall_seconds"]:
                raise ContractError("复盘/停机预留须非负且小于总墙钟；仅由运行配置指定")
            started = min(clock(), started_unix) if started_unix is not None else clock()
            deadline = min(started + cfg["wall_seconds"], cfg.get("parent_deadline_unix", float("inf")))
            journal.commit("budget/start", events=[{"type": "budget_started", "config": cfg,
                                                     "started_unix": started, "deadline_unix": deadline}])

    @property
    def excludes_numeric(self):
        return next(e['config'].get('exclude_numeric_wait', False) for e in self.journal.events
                    if e['type'] == 'budget_started')

    def snapshot(self):
        self.journal.refresh()
        value = None
        reservations, settlements, rounds, extensions, segments = {}, {}, set(), [], {}
        closeout_releases = []
        wall_authorizations = []
        waits = {}
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
            elif kind == "budget_segment_started":
                segments[event["segment_id"]] = dict(event)
            elif kind == "budget_segment_stopped":
                segments[event["segment_id"]].update(event)
            elif kind == "budget_closeout_reserve_released":
                closeout_releases.append(event)
            elif kind == "budget_total_wall_authorized":
                wall_authorizations.append(event)
            elif kind == 'budget_numeric_wait_started':
                waits[event['wait_id']] = dict(event)
            elif kind in ('budget_numeric_wait_checkpoint', 'budget_numeric_wait_stopped'):
                waits[event['wait_id']].update(event)
        now = self.clock()
        elapsed = max(0, now - value["started_unix"])
        active = None
        excluded = 0
        configured_total = value["config"]["wall_seconds"]
        total = configured_total + sum(e['seconds'] for e in extensions)
        if value["config"].get("clock_mode") == "active_segments":
            if wall_authorizations:
                total = wall_authorizations[-1]["total_wall_seconds"]
            elapsed = sum(max(0, s.get("stopped_unix", now) - s["started_unix"]) for s in segments.values())
            if self.excludes_numeric:
                for wait in waits.values():
                    end = wait.get('stopped_unix', now if process_alive(wait.get('owner'))
                                   else wait.get('confirmed_unix', wait['started_unix']))
                    for segment in segments.values():
                        excluded += max(0, min(end, segment.get('stopped_unix', now))
                                        - max(wait['started_unix'], segment['started_unix']))
                elapsed = max(0, elapsed - excluded)
            active = next((s for s in segments.values() if "stopped_unix" not in s), None)
            # A paused ledger has no usable deadline. Never compute now+remaining
            # in snapshot: only begin_segment may fix a new segment's deadline.
            value["deadline_unix"] = active["deadline_unix"] if active else None
            remaining = max(0, total - elapsed)
            if active and self.excludes_numeric:
                # A live balance hint, never a numerical worker's hard deadline.
                value['deadline_unix'] = min(now + remaining, value['config'].get('parent_deadline_unix', float('inf')))
                remaining = min(remaining, max(0, value['deadline_unix'] - now))
            elif active:
                remaining = min(remaining, max(0, active["deadline_unix"] - now))
        else:
            remaining = max(0, value["deadline_unix"] - now)
        shutdown = value["config"].get("shutdown_reserve_seconds", 0)
        configured_closeout = value["config"].get("closeout_reserve_seconds", 0)
        closeout = (0 if value["config"].get("clock_mode") == "active_segments" and closeout_releases
                    else configured_closeout)
        return {**value, "as_of_unix": now, "elapsed_seconds": elapsed,
                "configured_total_wall_seconds": configured_total, "total_wall_seconds": total,
                "wall_budget_authorizations": wall_authorizations,
                "remaining_seconds": remaining,
                'excluded_numeric_wait_seconds': excluded,
                'numeric_waits': list(waits.values()),
                "worker_deadline_unix": value["deadline_unix"] - shutdown if value["deadline_unix"] is not None else None,
                "research_remaining_seconds": max(0, remaining - shutdown - closeout),
                "configured_closeout_reserve_seconds": configured_closeout,
                "effective_closeout_reserve_seconds": closeout,
                "closeout_reserve_releases": closeout_releases,
                "segments": list(segments.values()), "active_segment": active,
                "rounds_started": sorted(rounds), "reservations": reservations,
                "settlements": settlements, "extensions": extensions,
                "l1_l2_call_limit": None, "llm_call_limit": value["config"].get("llm_call_limit")}

    def numeric_wait(self, waiting):
        """Only the Supervisor marks pure numerical wait; reasoning overlap closes it.

        Checkpoints bound unknown crash time. Overlapping experiments share one
        interval; pause/recovery never resets the already charged balance.
        """
        if not self.excludes_numeric:
            return
        state, now = self.snapshot(), self.clock()
        current = next((w for w in state['numeric_waits'] if 'stopped_unix' not in w), None)
        if current and (not waiting or not process_alive(current.get('owner'))):
            end = now if process_alive(current.get('owner')) else current.get('confirmed_unix', current['started_unix'])
            self.journal.commit('budget/wait/stop/' + current['wait_id'], events=[{
                'type': 'budget_numeric_wait_stopped', 'wait_id': current['wait_id'], 'stopped_unix': end}])
            current = None
        if waiting and state['active_segment']:
            if current is None:
                key = str(self.journal.sequence)
                self.journal.commit('budget/wait/start/' + key, events=[{
                    'type': 'budget_numeric_wait_started', 'wait_id': key, 'started_unix': now,
                    'confirmed_unix': now, 'owner': process_identity(os.getpid())}])
            elif now - current['confirmed_unix'] >= 1:
                self.journal.commit('budget/wait/checkpoint/' + str(self.journal.sequence), events=[{
                    'type': 'budget_numeric_wait_checkpoint', 'wait_id': current['wait_id'], 'confirmed_unix': now}])

    def begin_segment(self, *, started_unix=None, owner=None):
        state = self.snapshot()
        if state["config"].get("clock_mode") != "active_segments":
            return None
        if state["active_segment"]:
            raise ContractError("上段未确认完全停机；须保守结算后才能恢复")
        now = self.clock()
        previous_end = max((s["stopped_unix"] for s in state["segments"]), default=0)
        started = max(previous_end, min(now, now if started_unix is None else started_unix))
        deadline = min(started + state["remaining_seconds"],
                       state["config"].get("parent_deadline_unix", float("inf")))
        identifier = "segment-" + str(len(state["segments"]) + 1)
        self.journal.commit("budget/segment/start/" + identifier, events=[{
            "type": "budget_segment_started", "segment_id": identifier, "started_unix": started,
            "deadline_unix": deadline, "remaining_at_start": state["remaining_seconds"], "owner": owner}])
        return identifier

    def end_segment(self, *, processes_stopped, reason, recovered=False):
        state = self.snapshot()
        segment = state["active_segment"]
        if segment is None:
            return
        if not processes_stopped:
            raise ContractError("Supervisor/LLM/实验进程未确认停止，不能暂停计时")
        self.numeric_wait(False)
        state = self.snapshot()
        now = self.clock()
        wall = max(0, now - segment['started_unix'])
        excluded = sum(max(0, min(now, w['stopped_unix']) - max(segment['started_unix'], w['started_unix']))
                       for w in state['numeric_waits']) if self.excludes_numeric else 0
        # A missing stop receipt is charged through recovery confirmation, including
        # unknown downtime. Never cap this unknown cost at zero or the deadline.
        self.journal.commit("budget/segment/stop/" + segment["segment_id"], events=[{
            "type": "budget_segment_stopped", "segment_id": segment["segment_id"],
            "stopped_unix": now, "elapsed_seconds": wall - excluded,
            'wall_seconds': wall, 'excluded_numeric_wait_seconds': excluded,
            "reason": reason, "recovered_without_stop": recovered, "processes_stopped": True}])

    def input_receipt(self):
        """同一账本快照的可读输入；不另设额度，不改变 check/reserve 的计算。"""
        state = self.snapshot()
        active = [r for key, r in state["reservations"].items() if key not in state["settlements"]]
        estimates = {}
        for row in active:
            estimates[row["kind"]] = estimates.get(row["kind"], 0) + row["estimated_seconds"]
        calls = sum(r["kind"] == "llm" for r in state["reservations"].values())
        cap = state["llm_call_limit"]
        receipt = {"source": "BudgetLedger.snapshot/check/reserve", **{k: state[k] for k in (
                    "as_of_unix", "deadline_unix", "worker_deadline_unix", "elapsed_seconds",
                    "remaining_seconds", "research_remaining_seconds", "configured_total_wall_seconds",
                    "total_wall_seconds", "wall_budget_authorizations")},
                "closeout_reserve_seconds": state["effective_closeout_reserve_seconds"],
                "configured_closeout_reserve_seconds": state["configured_closeout_reserve_seconds"],
                "closeout_reserve_releases": state["closeout_reserve_releases"],
                "shutdown_reserve_seconds": state["config"].get("shutdown_reserve_seconds", 0),
                'clock_mode': state['config'].get('clock_mode', 'continuous'),
                'exclude_numeric_wait': bool(self.excludes_numeric),
                "unsettled_reservations": len(active), "unsettled_estimated_seconds_by_kind": estimates,
                "settled_reservations": len(state["settlements"]),
                "llm_calls": {"limit": cap, "reserved": calls, "remaining": max(0, cap - calls) if cap is not None else None},
                "meaning": "remaining_seconds 是硬截止余额；research_remaining_seconds 已扣复盘/停机预留，"
                           "非 LLM 请求的 estimated_seconds 按此余额检查。未结算估计仅展示，现有账本不再扣减它们；"
                           "LLM estimated_seconds 是调用超时上限，不按实验耗时预扣。null 调用上限表示未配置；"
                           "磁盘、资源与实际启动时预算仍按原入口检查。"}
        if self.excludes_numeric:
            receipt.update(excluded_numeric_wait_seconds=state['excluded_numeric_wait_seconds'],
                meaning='累计智能体时间：纯数值执行、等待资源或结果不计时；同时进行的推理/Coding/审核照常计时，'
                        '重叠区间只计一次。estimated_seconds 是数值单任务超时，不要求小于智能体余额。'
                        'deadline_unix 仅是当前余额提示，会随纯数值等待后移；已启动实验保留自身超时、人工停止和资源限制。'
                        '暂停不计时；恢复保留消费，崩溃未知间隔保守计入。')
        return receipt

    def check(self, *, estimated_seconds=0, check_disk=True, kind=None, allow_numeric_drain=False):
        state = self.snapshot()
        if state["config"].get("clock_mode") == "active_segments" and not state["active_segment"]:
            raise BudgetExhausted("累计运行预算已暂停；必须通过受监控的启动入口创建运行段")
        if state["remaining_seconds"] <= 0 and not (self.excludes_numeric and allow_numeric_drain):
            raise BudgetExhausted("墙钟硬预算耗尽；恢复不重置计时")
        if estimated_seconds > state["research_remaining_seconds"] and not (self.excludes_numeric and kind == 'experiment'):
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
        self.check(estimated_seconds=0 if kind == "llm" else estimated_seconds, kind=kind)
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
        if state["config"].get("clock_mode") == "active_segments":
            raise ContractError("本次累计运行段协议不允许 extend-budget；总授权不因恢复而增加")
        if state["deadline_unix"] + seconds > state["config"].get("parent_deadline_unix", float("inf")):
            raise ContractError("子运行不能延长到父验证预算以外；须先明确解决父预算授权")
        self.journal.commit(f"budget/extension/{self.journal.sequence}", events=[{
            "type": "budget_extended", "seconds": seconds, "reason": reason,
            "authorized_action": "explicit CLI budget extension", "when": self.clock()}])

    def authorize_total_wall_seconds(self, total, *, reason, authorization_path):
        """Apply a new explicit user ceiling while stopped; never reset spending.

        This operator-only entry is not exposed as a research-agent tool. The old
        configuration, all segments and the original no-extend policy stay saved.
        A subsequent begin_segment fixes its deadline from the amended balance.
        """
        state = self.snapshot()
        if state['config'].get('clock_mode') != 'active_segments' or state['active_segment']:
            raise ContractError('累计总额新授权必须在已确认完全停机的 active_segments 账本上应用')
        path = Path(authorization_path).resolve()
        authorization = read_json(path)
        if (type(total) is not int or total <= 0 or not reason.strip()
                or authorization.get('author') != 'user'
                or authorization.get('total_wall_seconds') != total
                or authorization.get('all_other_budgets_unchanged') is not True):
            raise ContractError('新的累计总额须有明确用户授权记录，且不得变更其他预算')
        digest = file_sha256(path)
        previous = next((e for e in state['wall_budget_authorizations']
                         if e['authorization_sha256'] == digest), None)
        if previous:
            return previous
        if total <= state['total_wall_seconds']:
            raise ContractError('本入口只接受新的较大总额；不得重置或重复增加余额')
        event = {'type': 'budget_total_wall_authorized', 'total_wall_seconds': total,
                 'previous_total_wall_seconds': state['total_wall_seconds'],
                 'additional_seconds': total - state['total_wall_seconds'],
                 'elapsed_seconds_at_authorization': state['elapsed_seconds'],
                 'authorization_path': str(path), 'authorization_sha256': digest,
                 'reason': reason, 'authorized_by': 'user', 'when': self.clock()}
        self.journal.commit('budget/total_authorization/' + digest, events=[event])
        return event
