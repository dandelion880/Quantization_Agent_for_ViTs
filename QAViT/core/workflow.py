"""科研主流程：生成→筛查→初始三场→选 H*→加强为 H4→父子比较→最终选择→L3→复盘。

Workflow 拥有唯一状态；实验子流程、角色交互和运行驱动分别放在三个职责分区。
Journal 是事实源，所有阶段依据已提交事件推进，重复 advance 不重复发布任务。
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from runtime.paths import ROOT
from core.cards import ContractError, fingerprint, make_card
from core.journal import Journal, atomic_json, file_lock, read_json
from runtime.task_queue import GlobalTaskQueue, TERMINAL
from runtime.budget import BudgetLedger
from core.memory import ResearchMemory, MemoryRetriever
from core.debate import INITIAL_PAIRS, MAX_EXCHANGES, match_key, ranking_table, selection_pairs
from runtime.task_runtime import TaskRuntime
from tools.experiments import ExperimentBroker
from runtime.identity import code_identity, code_snapshot
from runtime.configuration import config_check
from agents.tasks import TASKS
from agents.interaction import RoleInteraction
from core.experiment_requests import ExperimentRequests
from runtime.driver import WorkflowDriver



class Workflow(RoleInteraction, ExperimentRequests, WorkflowDriver):
    def __init__(self, run_dir, config=None, *, intervention=None, read_only=False):
        self.run_dir = Path(run_dir).resolve()
        self.read_only = read_only
        if ROOT / "EXP" not in self.run_dir.parents:
            raise ContractError("运行产物必须保存在本项目 EXP 的独立目录")
        manifest = self.run_dir / "research_run.json"
        if config is not None:
            if self.run_dir.exists():
                raise ContractError("新运行目录已存在，不覆盖历史")
            config = config_check(deepcopy(config))
            self.run_dir.mkdir(parents=True)
            identity = code_identity()
            atomic_json(manifest, {"schema_version": "qavit.run/2", "run_id": self.run_dir.name,
                                   "config": config, "config_sha256": fingerprint(config),
                                   "code_identity": identity, "source_snapshot": code_snapshot(identity),
                                   "created_at": datetime.now(timezone.utc).isoformat(),
                                   "legacy_memory_imported": False, "legacy_incumbent_imported": False})
        saved = read_json(manifest)
        self.config = config_check(saved["config"])
        if fingerprint(self.config) != saved["config_sha256"]:
            raise ContractError("运行配置被改写")
        self.journal = Journal(self.run_dir, saved["run_id"])
        expected = saved["code_identity"]["core_sha256"]
        for event in self.journal.events:
            if event["type"] == "implementation_intervention":
                expected = event["current_core_sha256"]
        current = code_identity()["core_sha256"]
        self.current_implementation_matches = expected == current
        if expected != current and not read_only:
            if not intervention:
                raise ContractError("核心实现变化；恢复需要 --intervention 记录修复原因，并重新检查冻结资产")
            with file_lock(self.run_dir / "supervisor.lock"):
                identity = code_identity()
                self.journal.commit(f"intervention/{self.journal.sequence}", events=[{
                    "type": "implementation_intervention", "previous_core_sha256": expected,
                    "current_core_sha256": current, "reason": intervention, "code_identity": identity,
                    "source_snapshot": code_snapshot(identity)}])
                self.current_implementation_matches = True
        if not self.journal.find_cards("ResearchBrief"):
            if read_only:
                raise ContractError("运行在发布 ResearchBrief 前中断；没有已提交科研状态")
            brief = make_card("ResearchBrief", self.journal.run_id, self.config["research_brief"], author="user")
            self.journal.commit("research/brief", cards=[brief])
        self.brief = self.journal.find_cards("ResearchBrief")[-1]
        self.queue = GlobalTaskQueue(self.journal)
        self.budget = BudgetLedger(self.journal, None if read_only else self.config.get("budget", {}),
                                   started_unix=datetime.fromisoformat(saved["created_at"]).timestamp())
        self.memory, self.retriever = ResearchMemory(self.journal), MemoryRetriever(self.journal)
        self.broker = ExperimentBroker(self.journal, self.brief["id"], self.budget, self.config['runtime'])
        self.runtime = TaskRuntime(self.run_dir, self.queue, self.budget, self.config["runtime"])
        self.transport_blocked = None


    def event(self, kind, key, **value):
        return self.journal.commit(key, events=[{"type": kind, **value}])


    def tasks(self):
        return {task["key"]: task for task in self.queue.snapshot().values()}


    def result_card(self, task):
        if task and task["state"] == "completed" and task["outputs"]:
            return self.journal.card(task["outputs"][0])
        return None


    def enqueue_role(self, key, purpose, *, refs=(), context=None, depends=(), parent_task=None):
        role = TASKS[purpose]["role"]
        context = deepcopy(context or {})
        if purpose == 'record' and 'recording_batch' not in context:
            batch = self.memory.recording_batch()
            context.update(recording_batch=batch)
            refs = batch['source_refs']
        producers = [t["task_id"] for t in self.queue.snapshot().values()
                     if t["state"] == "completed" and set(t["outputs"]).intersection(refs)]
        return self.queue.enqueue(key, "llm", role=role, inputs={"purpose": purpose, "refs": list(dict.fromkeys(refs)),
                                  "context": context, **({"parent_task": parent_task} if parent_task else {})}, depends_on=list(dict.fromkeys([*depends, *producers])), resource_keys=["llm:subscription"],
                                  max_retries=self.config["llm"].get("max_transport_retries", 2))


    def matches(self, number):
        result = {}
        for event in self.journal.events:
            if event["type"] == "match_closed" and event["round"] == number:
                result[event["pair"]] = {k: v for k, v in event.items() if k not in ("type", "commit_sequence")}
        return result


    def round_hypotheses(self, number):
        return {card["payload"]["slot"]: card for card in self.journal.find_cards("HypothesisCard")
                if card["payload"]["round"] == number}


    def round_refs(self, number):
        refs = set()
        for task in self.queue.snapshot().values():
            if task["key"].startswith(f"r{number:03d}/"):
                refs.update(task["outputs"])
        return sorted(refs)


    def selection_matches(self, context):
        matches = self.matches(context["round"])
        return {"-".join(pair): matches["-".join(pair)]
                for pair in selection_pairs(context["selection_stage"], context.get("parent_slot"))}


    def progress_selection(self, number, stage, hypotheses, screens, parent=None):
        """复用 rank 任务；明确裁决由程序引用，未决才等待 Judger 说明选择。"""
        key = f"r{number:03d}/" + ("select_parent" if stage == "parent" else "rank")
        task = self.tasks().get(key)
        if task is None:
            parent_slot = parent["payload"]["selected_slot"] if parent else None
            candidates = ["H1", "H2", "H3"] if stage == "parent" else (["H4", parent_slot] if parent_slot else [])
            context = {"round": number, "selection_stage": stage, "parent_slot": parent_slot,
                       "parent_selection_ref": parent["id"] if parent else None,
                       "candidate_refs": {s: hypotheses[s]["id"] for s in candidates},
                       "eligible": [s for s in candidates if screens[s]["payload"]["eligible"]],
                       "incumbent_separate": True}
            context["matches"] = self.selection_matches(context)
            context["direct_results"] = ranking_table(context["matches"], context["eligible"])
            identifier = self.enqueue_role(key, "rank", refs=self.round_refs(number), context=context)
            task = self.queue.get(identifier)
        context = task["inputs"]["context"]
        if context.get("selection_stage") != stage:
            raise ContractError("既有选择任务未绑定当前阶段，不能静默改变 H* 或合并排名")
        result = self.result_card(task)
        if result:
            return result
        matches = self.selection_matches(context)
        table = ranking_table(matches, context["eligible"])
        winners = [s for s, wins in table["wins"].items() if wins == len(context["eligible"]) - 1]
        if (winners or not context["eligible"]) and not task.get("request_path"):
            ready = {t["task_id"] for t in self.queue.ready()}
            if task["task_id"] in ready:
                selected = winners[0] if winners else None
                verdicts = [self.journal.card(m["verdict_ref"]) for m in matches.values()]
                evidence = sorted({ref for c in verdicts for ref in c["payload"]["evidence_refs"]})
                basis = ("复用本阶段已提交直接裁决；所选方案胜过其余合格候选。" if len(context["eligible"]) > 1
                         else "资格审查后仅余一个合格方案，保留该方案。" if selected
                         else "初始方案全部不合格；无合格候选，跳过演化及 L3。")
                reasons = [c["payload"].get("decisive_reason", c["payload"]["rationale"]) for c in verdicts
                           if selected in c["payload"].get("pair", []) and c["payload"]["outcome"] in ("A", "B")]
                if reasons:
                    basis += " 既有裁决依据：" + "；".join(reasons)
                self.submit(self.queue.claim(task["task_id"]), {
                    "phase": "ranking", "selected_slot": selected,
                    "provisional": table["requires_provisional_basis"], "selection_basis": basis,
                    "rationale": basis, "evidence_refs": evidence,
                    "uncertainties": ["其余局部未决关系保留在 direct_results。"] if table["requires_provisional_basis"] else [],
                    "selection_source": "existing_verdicts_and_screening",
                    "reused_verdict_refs": [c["id"] for c in verdicts]}, None, author="program")
        return self.result_card(self.queue.get(task["task_id"]))


    def enqueue_recorder(self, number, *, incomplete=False, key=None):
        batch = self.memory.recording_batch()
        if not batch['review_required']:
            return None  # 工程状态已在 Journal；没有新的科学材料时无需模型复盘。
        if key:
            # 收尾复用用户已重试/回答的同批任务，不另开一个模型调用。
            review_refs = set(batch['scientific_refs'] + batch.get('scientific_failure_refs', []))
            for task in self.queue.snapshot().values():
                if (task['inputs'].get('purpose') == 'record'
                        and task['state'] in ('pending', 'ready', 'running', 'retry', 'waiting')
                        and review_refs <= set(task['inputs']['context']['recording_batch']['source_refs'])):
                    return task['task_id']
        refs = batch['source_refs']
        ranking = self.result_card(self.tasks().get(f"r{number:03d}/rank"))
        parent = self.result_card(self.tasks().get(f"r{number:03d}/select_parent"))
        outcome = self.l3_outcome(number)
        return self.enqueue_role(key or f"r{number:03d}/record", "record", refs=refs, context={
            "recording_batch": batch,
            "round": number, "selection_ref": ranking["id"] if ranking else None,
            "selection": ranking["payload"]["selected_slot"] if ranking else None,
            "parent_selection_ref": parent["id"] if parent else None,
            "parent_hypothesis_ref": parent["payload"]["selected_hypothesis_ref"] if parent else None,
            "l3_outcome": outcome, "incomplete_round": incomplete,
            "requirements": "事实与来源已由程序完整归档。综合关键发现、矛盾证据、适用条件和下一轮问题；"
                            "按需深读原件，允许空 knowledge_updates；不要返回全量覆盖清单或逐条审计。"})

    def defer_recorder(self, task):
        """复盘失败留下可恢复待办；不造 Meta、不推进科学综合游标。"""
        if task and task['state'] in ('failed', 'cancelled', 'needs_clarification'):
            batch = task['inputs']['context']['recording_batch']
            self.event('recorder_deferred', 'recorder/deferred/' + task['task_card'],
                       batch_id=batch['batch_id'], source_refs=batch['source_refs'],
                       review_refs=batch['scientific_refs'] + batch.get('scientific_failure_refs', []),
                       task_ref=task['task_card'], reason=task.get('error') or task.get('clarification'),
                       next_action='下一批 Recorder 合并未复盘来源；Generation 先使用原始事实并保留复盘缺口。')
            return True
        return False

    def recording_deferred(self, batch):
        review_refs = set(batch['scientific_refs'] + batch.get('scientific_failure_refs', []))
        if any(t['inputs'].get('purpose') == 'record'
               and t['state'] in ('pending', 'ready', 'running', 'retry', 'waiting')
               and review_refs <= set(t['inputs']['context']['recording_batch']['source_refs'])
               for t in self.queue.snapshot().values()):
            return False  # 显式重试/澄清回答恢复原任务，旧 deferred 事件不能将其跳过。
        return any(e['type'] == 'recorder_deferred'
                   and (e['batch_id'] == batch['batch_id']
                        or (review_refs and review_refs <= set(e.get('review_refs', e['source_refs']))))
                   for e in self.journal.events)


    def progress_match(self, number, pair, hypotheses, screens):
        if "H4" in pair:
            parent = self.result_card(self.tasks().get(f"r{number:03d}/select_parent"))
            if not parent or tuple(pair) != ("H4", parent["payload"]["selected_slot"]):
                raise ContractError("H4 只能与本轮已持久化的 H* 交锋")
        key = match_key(number, pair)
        short = "-".join(pair)
        if short in self.matches(number):
            return
        refs = [hypotheses[s]["id"] for s in pair] + [screens[s]["id"] for s in pair]
        if not all(screens[s]["payload"]["eligible"] for s in pair):
            card = make_card("ReviewDecisionCard", self.journal.run_id, {
                "phase": "verdict", "round": number, "pair": list(pair), "outcome": "ineligible",
                "rationale": "至少一个假设未通过硬性审查；保留配对记录，没有开展辩论或计算胜负。",
                "evidence_refs": [], "uncertainties": [], "exchanges": 0}, refs=refs)
            self.journal.commit(key + "/ineligible", cards=[card], events=[{"type": "match_closed", "round": number,
                "pair": short, "outcome": "ineligible", "exchanges": 0, "verdict_ref": card["id"]}])
            return
        tasks = self.tasks()
        history = []
        for exchange in range(1, MAX_EXCHANGES + 1):
            judge = tasks.get(key + f"/e{exchange}/arbitrate")
            verdict = self.result_card(judge)
            if verdict:
                if verdict["payload"]["outcome"] != "continue":
                    self.event("match_closed", key + "/closed", round=number, pair=short,
                               outcome=verdict["payload"]["outcome"], exchanges=exchange, verdict_ref=verdict["id"])
                    return
                history.extend([*judge["inputs"]["refs"], verdict["id"]])
                continue
            positions = []
            for side in ("A", "B"):
                position_key = key + f"/e{exchange}/{side}"
                self.enqueue_role(position_key, "position", refs=[*refs, *history], context={
                    "round": number, "pair": list(pair), "exchange": exchange, "side": side,
                    "position_for": pair[0 if side == "A" else 1], "max_exchanges": MAX_EXCHANGES,
                    "independence": "本次双方立场分别生成；此输入不包含本次另一方发言。"})
                positions.append(self.result_card(self.tasks().get(position_key)))
            if all(positions):
                self.enqueue_role(key + f"/e{exchange}/arbitrate", "arbitrate",
                                  refs=[*refs, *history, *[p["id"] for p in positions]], context={
                    "round": number, "pair": list(pair), "exchange": exchange, "max_exchanges": MAX_EXCHANGES,
                    "protocol": "核查双方预测和证据。明显优劣提前裁决；真实争议可请求实验。最多5回合。"})
            return


    def advance(self):
        """依已提交状态推进两阶段选择；恢复不重选父卡片或重复创建比赛。"""
        self.return_code_clarifications()
        self.release_measured_requests()
        finished = {e["round"] for e in self.journal.events if e["type"] == "round_finished"}
        number = max(finished, default=0) + 1
        if number > self.budget.snapshot()["config"]["max_rounds"]:
            batch = self.memory.recording_batch()
            if not batch['review_required'] or self.recording_deferred(batch):
                return True
            # 正常结尾接收迟到产出后仍处理剩余增量；沿用 Recorder 和原预算。
            identifier = self.enqueue_recorder(max(finished, default=1), key='closeout/record/' + batch['batch_id'])
            if self.defer_recorder(self.queue.get(identifier)):
                return True
            return False
        self.budget.start_round(number)
        self.journal.commit(f"r{number:03d}/memory_start", events=[self.memory.live_update(round_number=number)])
        prefix = f"r{number:03d}"
        feedback_refs = self.memory.snapshot()["L3"][-1:]
        prior = next((e for e in reversed(self.journal.events)
                      if e['type'] == 'round_finished' and e['round'] == number - 1), {})
        review_pending = prior.get('scientific_review') == 'pending'
        if review_pending:
            feedback_refs += [r for r in self.round_refs(number - 1)
                              if self.journal.cards[r]['kind'] in ('HypothesisCard', 'EvidenceCard')
                              or (self.journal.cards[r]['kind'] == 'ReviewDecisionCard'
                                  and self.journal.card(r)['payload']['phase'] == 'ranking')]
        for slot, kind in (("H1", "existing_method"), ("H2", "engineering"), ("H3", "new_algorithm")):
            self.enqueue_role(prefix + "/generate/" + slot, "generate", refs=[self.brief["id"], *feedback_refs],
                              context={"round": number, "slot": slot, "hypothesis_type": kind,
                                       "previous_review_pending": review_pending})
        hypotheses = self.round_hypotheses(number)
        screens = {}
        for slot, card in hypotheses.items():
            screening = prefix + "/screen/" + slot
            # 所有已生成假设可比较机制，但当前轮各类型没有预设优先级。
            self.enqueue_role(screening, "screen", refs=[self.brief["id"], card["id"]],
                              context={"round": number, "slot": slot, "hypothesis_ref": card["id"]})
            screen = self.result_card(self.tasks().get(screening))
            if screen:
                screens[slot] = screen
        if not {"H1", "H2", "H3"} <= screens.keys():
            return False
        for pair in INITIAL_PAIRS:
            self.progress_match(number, pair, hypotheses, screens)
            # 结案提交是下一场的发布门槛；恢复时从首个未结案配对继续。
            if "-".join(pair) not in self.matches(number):
                return False
        matches = self.matches(number)
        parent = self.progress_selection(number, "parent", hypotheses, screens)
        if not parent:
            return False
        parent_slot, parent_ref = parent["payload"]["selected_slot"], parent["payload"]["selected_hypothesis_ref"]
        if parent_slot:
            # 使用选择卡绑定的完整 ID，不能由同槽位后来出现的版本替换父方案。
            hypotheses[parent_slot] = self.journal.card(parent_ref)
            related = [self.journal.card(ref) for ref in self.round_refs(number)]
            critiques = [c for c in related if c["kind"] == "ReviewDecisionCard"
                         and (parent_ref in c["refs"] or parent_slot in c["payload"].get("pair", []))]
            evidence_refs = sorted({c["id"] for c in related if c["kind"] == "EvidenceCard"}
                                   | {ref for c in critiques for ref in c["payload"].get("evidence_refs", [])})
            critique_refs = [c["id"] for c in critiques]
            self.enqueue_role(prefix + "/evolve/H4", "evolve", refs=[self.brief["id"], parent["id"], parent_ref,
                              *critique_refs, *evidence_refs, *feedback_refs], context={
                "round": number, "slot": "H4", "hypothesis_type": "evolved",
                "parent_ref": parent_ref, "parent_hypothesis": hypotheses[parent_slot],
                "parent_selection_ref": parent["id"], "selection_basis": parent["payload"]["selection_basis"],
                "parent_provisional": parent["payload"]["provisional"], "critique_refs": critique_refs,
                "evidence_refs": evidence_refs, "stage1_matches": {"-".join(p): matches["-".join(p)] for p in INITIAL_PAIRS},
                "required_input": "仅加强已选 H*；结合所附选择理由、相关批评和证据说明改进目标。parent_refs 必须仅含 parent_ref。"})
            if "H4" not in screens:
                return False
            hypotheses["H4"] = self.result_card(self.tasks()[prefix + "/evolve/H4"])
            pair = ("H4", parent_slot)
            self.progress_match(number, pair, hypotheses, screens)
            if "-".join(pair) not in self.matches(number):
                return False
        ranking = self.progress_selection(number, "final", hypotheses, screens, parent)
        if not ranking:
            return False
        selected = ranking["payload"]["selected_slot"]
        if self.config["enable_l3"] and selected:
            if not self.progress_planned_request(prefix + '/plan_l3', prefix + '/L3',
                    refs=[ranking['id'], ranking['payload']['selected_hypothesis_ref'], self.brief['id']],
                    context={'round': number, 'selection': selected, 'selection_ref': ranking['id'],
                             'hypothesis_ref': ranking['payload']['selected_hypothesis_ref'],
                             'selection_stage': 'final', 'gate': 'remaining hard budget only'}):
                return False
        if prefix + "/record" not in self.tasks():
            self.enqueue_recorder(number)
        recording_task = self.tasks().get(prefix + '/record')
        recorder = self.result_card(recording_task)
        if recorder or self.defer_recorder(recording_task) or recording_task is None:
            self.event("round_finished", prefix + "/finished", round=number, metareview_ref=recorder['id'] if recorder else None,
                       scientific_review='completed' if recorder or recording_task is None else 'pending',
                       selection_ref=ranking["id"], selected_slot=selected)
            # round_finished 是内部事件，必须继续推进后继轮或终态；否则队列空闲
            # 会被主循环误判为 blocked，完成的最后一轮也永远无法结束。
            return self.advance()
        return False

