"""G-only：H1/H2/H3 自主研究→Generation 候选综合与修订→三候选 L3→复盘。

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
from core.debate import selection_pairs  # Historical card inspection only.
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
            config = deepcopy(config)
            config.setdefault("architecture", "gonly")
            if config["architecture"] != "gonly":
                raise ContractError("此分支只启动 G-only；Full 使用 main 或基线 tag")
            config = config_check(config, resolve_resources=True)
            self.run_dir.mkdir(parents=True)
            identity = code_identity()
            atomic_json(manifest, {"schema_version": "qavit.run/2", "run_id": self.run_dir.name,
                                   "config": config, "config_sha256": fingerprint(config),
                                   "code_identity": identity, "source_snapshot": code_snapshot(identity),
                                   "created_at": datetime.now(timezone.utc).isoformat(),
                                   "legacy_memory_imported": False, "legacy_incumbent_imported": False})
        saved = read_json(manifest)
        if not read_only and saved["config"].get("architecture") != "gonly":
            raise ContractError("不能将 Full 历史运行静默恢复为 G-only；请使用 main 或基线 tag")
        self.config = config_check(saved["config"], historical=read_only)
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
        from core.solutions import SolutionLibrary
        self.solutions = SolutionLibrary(self.journal, self.brief["payload"], shared=self.config.get("purpose") == "research")
        if not read_only:
            self.solutions.initialize()
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
        producers = [t["task_id"] for t in self.queue.snapshot(states={"completed"}, fields=("task_id", "outputs")).values()
                     if set(t["outputs"]).intersection(refs)]
        # 同场 A/B、仲裁仍各自调用并串行领取；独立配对不共享订阅互斥锁。
        resources = ([f"llm:judger:{context['round']}:" + "-".join(context['pair'])]
                     if role == "judger" and context.get("pair") else ["llm:subscription"])
        return self.queue.enqueue(key, "llm", role=role, inputs={"purpose": purpose, "refs": list(dict.fromkeys(refs)),
                                  "context": context, **({"parent_task": parent_task} if parent_task else {})}, depends_on=list(dict.fromkeys([*depends, *producers])), resource_keys=resources,
                                  max_retries=self.config["llm"].get("max_transport_retries", 2))


    def matches(self, number):
        result = {}
        for event in self.journal.events:
            if event["type"] == "match_closed" and event["round"] == number:
                result[event["pair"]] = {k: v for k, v in event.items() if k not in ("type", "commit_sequence")}
        return result


    def round_hypotheses(self, number):
        # Only initial publications and the host revision transaction change candidates.
        result = {card['payload']['slot']: card for card in self.journal.find_cards('HypothesisCard')
                  if card['payload']['round'] == number and card['version'] == 1}
        for event in self.journal.events:
            if event['type'] == 'candidate_revised' and event['round'] == number:
                result[event['slot']] = self.journal.card(event['candidate_ref'])
        return result


    def round_refs(self, number):
        refs = set()
        for task in self.queue.snapshot(fields=("key", "outputs")).values():
            if task["key"].startswith(f"r{number:03d}/"):
                refs.update(task["outputs"])
        return sorted(refs)


    def selection_matches(self, context):
        matches = self.matches(context["round"])
        return {"-".join(pair): matches["-".join(pair)]
                for pair in selection_pairs(context["selection_stage"], context.get("parent_slot"))}


    def candidate_refs(self, number):
        return {slot: card['id'] for slot, card in self.round_hypotheses(number).items()}

    def selection_key(self, number):
        base = f'r{number:03d}/select'
        initial = self.queue.get_by_key(base)
        refs = self.candidate_refs(number)
        return (base if initial is None or initial['inputs']['context']['candidate_refs'] == refs
                else base + '/' + fingerprint(refs)[:16])

    def selection_card(self, number):
        selected = self.result_card(self.queue.get_by_key(self.selection_key(number)))
        if selected and selected['payload'].get('candidate_refs') == self.candidate_refs(number):
            return selected
        # Keep historical fallbacks readable, but never create another H1 preference.
        return next((self.journal.card(e['selection_ref']) for e in self.journal.events
                     if e['type'] == 'selection_fallback' and e['round'] == number
                     and self.journal.card(e['selection_ref'])['payload'].get('candidate_refs') == self.candidate_refs(number)), None)

    def progress_preselection(self, number, hypotheses, failures):
        """Pin a fresh decision to every candidate version; B is retained separately."""
        candidates = self.candidate_refs(number)
        for task in self.queue.snapshot(kinds={'llm'}).values():
            context = task['inputs'].get('context', {})
            if (task['inputs'].get('purpose') == 'select' and context.get('round') == number
                    and context.get('candidate_refs') != candidates and task['state'] not in TERMINAL):
                self.queue.change(task['task_id'], 'cancelled', patch={'error': {
                    'code': 'stale_candidate_selection', 'message': '候选已有新版本，重新预选；旧任务与证据保留'}})
        selected = self.selection_card(number)
        if selected or not hypotheses:
            return selected
        identifier = self.enqueue_role(self.selection_key(number), 'select',
            refs=[self.brief['id'], *self.round_refs(number), *candidates.values(),
                  *[t['task_card'] for t in failures]], context={
                'round': number, 'selection_stage': 'gonly_synthesis',
                'candidate_refs': candidates,
                'revision_context': [{k: e[k] for k in ('parent_ref', 'candidate_ref', 'reason', 'change_type',
                    'validation_status', 'basis_evidence_refs')} for e in self.journal.events
                    if e['type'] == 'candidate_revised' and e['candidate_ref'] in candidates.values()],
                'incomplete_slots': [t['inputs']['context']['slot'] for t in failures],
                'meaning': 'Compare and revise all candidates; every final version enters L3. B is retained independently; a new revision is unvalidated.'})
        return self.result_card(self.queue.get(identifier))

    def consider_l3(self, number):
        """Persist valid L3 before Recorder/closeout; partial results never replace Top-3."""
        selection = self.selection_card(number)
        if selection:
            for ref in self.l3_outcome(number)['valid_evidence_refs']:
                evidence = self.journal.card(ref)
                experiment = self.journal.card(evidence['payload']['experiment_ref'])
                self.solutions.consider_round(number, evidence, experiment, selection,
                                             validator=self.broker.validate_evidence_for)


    def enqueue_recorder(self, number, *, incomplete=False, key=None):
        batch = self.memory.recording_batch()
        if not batch['review_required']:
            return None  # 工程状态已在 Journal；没有新的科学材料时无需模型复盘。
        if key:
            # 收尾复用用户已重试/回答的同批任务，不另开一个模型调用。
            review_refs = set(batch['scientific_refs'] + batch.get('scientific_failure_refs', []))
            for task in self.queue.snapshot(states={"pending", "ready", "running", "retry", "waiting"},
                                            fields=("task_id", "inputs", "state")).values():
                if (task['inputs'].get('purpose') == 'record'
                        and task['state'] in ('pending', 'ready', 'running', 'retry', 'waiting')
                        and review_refs <= set(task['inputs']['context']['recording_batch']['source_refs'])):
                    return task['task_id']
        refs = batch['source_refs']
        ranking = self.selection_card(number)
        outcome = self.l3_outcome(number)
        return self.enqueue_role(key or f"r{number:03d}/record", "record", refs=refs, context={
            "recording_batch": batch,
            "round": number, "selection_ref": ranking["id"] if ranking else None,
            "selection": ranking["payload"].get("selected_slot") if ranking else None,
            "candidate_refs": ranking["payload"].get("candidate_refs", {}) if ranking else {},
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
               for t in self.queue.snapshot(states={"pending", "ready", "running", "retry", "waiting"},
                                            fields=("inputs", "state")).values()):
            return False  # 显式重试/澄清回答恢复原任务，旧 deferred 事件不能将其跳过。
        return any(e['type'] == 'recorder_deferred'
                   and (e['batch_id'] == batch['batch_id']
                        or (review_refs and review_refs <= set(e.get('review_refs', e['source_refs']))))
                   for e in self.journal.events)


    def advance(self):
        """依已提交状态推进 G-only；恢复不改写候选或重复预选。"""
        if self.scope_target_reached():
            return True
        self.release_scope_closeout_reserve()
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
            if prior.get("selection_ref"):
                feedback_refs.append(prior["selection_ref"])
            feedback_refs += [r for r in self.round_refs(number - 1)
                              if self.journal.cards[r]['kind'] in ('HypothesisCard', 'EvidenceCard')
                              or (self.journal.cards[r]['kind'] == 'ReviewDecisionCard'
                                  and self.journal.card(r)['payload']['phase'] == 'ranking')]
        for slot, kind in (("H1", "existing_method"), ("H2", "engineering"), ("H3", "new_algorithm")):
            self.enqueue_role(prefix + "/generate/" + slot, "generate", refs=[self.brief["id"], *feedback_refs],
                              context={"round": number, "slot": slot, "hypothesis_type": kind,
                                       "previous_review_pending": review_pending})
        target = self.config.get("scope_target")
        if target and number == target["round"]:
            # This boundary precedes *every* preselection publication, including partial
            # generation, the final receipt, recovery and repeated advance/resume.
            return self.mark_scope_target_if_ready()
        generators = [self.queue.get_by_key(prefix + '/generate/' + slot) for slot in ('H1', 'H2', 'H3')]
        if any(t['state'] not in TERMINAL for t in generators):
            return False
        # Keep exactly the submitted task outputs, including their version IDs.
        hypotheses = self.round_hypotheses(number)
        failures = [t for t in generators if not self.result_card(t)]
        ranking = self.progress_preselection(number, hypotheses, failures)
        selection_task = self.queue.get_by_key(self.selection_key(number))
        if selection_task and selection_task['state'] not in TERMINAL:
            return False
        selected = ranking['payload'].get('selected_slot') if ranking else None
        if self.config["enable_l3"] and ranking:
            # Visit every branch before waiting: a slow/failed candidate cannot
            # prevent its siblings from planning, calibrating or returning results.
            pending = False
            for slot, hypothesis_ref in ranking['payload']['candidate_refs'].items():
                plan_key, request_key = self.l3_request_keys(number, slot, hypothesis_ref)
                done = self.progress_planned_request(plan_key, request_key,
                    refs=[ranking['id'], hypothesis_ref, self.brief['id']],
                    context={'round': number, 'slot': slot, 'selection': slot, 'selection_ref': ranking['id'],
                             'hypothesis_ref': hypothesis_ref, 'candidate_refs': ranking['payload']['candidate_refs'],
                             'selection_stage': 'gonly_synthesis', 'gate': 'remaining hard budget only'})
                pending = pending or not done
            self.consider_l3(number)
            if pending:
                return False
        self.consider_l3(number)
        if self.queue.get_by_key(prefix + "/record") is None:
            self.enqueue_recorder(number, incomplete=bool(failures) or not ranking or
                                  (self.config["enable_l3"] and not self.l3_outcome(number)["formal_validation_success"]))
        recording_task = self.queue.get_by_key(prefix + '/record')
        recorder = self.result_card(recording_task)
        if recorder or self.defer_recorder(recording_task) or recording_task is None:
            self.event("round_finished", prefix + "/finished", round=number, metareview_ref=recorder['id'] if recorder else None,
                       scientific_review='completed' if recorder or recording_task is None else 'pending',
                       selection_ref=ranking["id"] if ranking else None, selected_slot=selected,
                       candidate_refs=ranking["payload"].get("candidate_refs", {}) if ranking else {},
                       l3_outcome=self.l3_outcome(number),
                       incomplete_slots=[t["inputs"]["context"]["slot"] for t in failures])
            # round_finished 是内部事件，必须继续推进后继轮或终态；否则队列空闲
            # 会被主循环误判为 blocked，完成的最后一轮也永远无法结束。
            return self.advance()
        return False

    def scope_target_reached(self):
        self.journal.refresh()
        return any(e["type"] == "scope_target_reached" for e in self.journal.events)


    def mark_scope_target_if_ready(self):
        """Accept a last durable receipt during shutdown without scheduling work."""
        if self.scope_target_reached():
            return True
        target = self.config.get("scope_target")
        if not target:
            return False
        finished = {e["round"] for e in self.journal.events if e["type"] == "round_finished"}
        if not set(range(1, target["round"])) <= finished:
            return False
        tasks = self.tasks()
        prefix = f"r{target['round']:03d}"
        generators = [tasks.get(prefix + "/generate/" + slot) for slot in ("H1", "H2", "H3")]
        if not all(self.result_card(task) is not None for task in generators):
            return False
        related = {task["task_id"] for task in generators}
        while True:
            children = related | {task["task_id"] for task in tasks.values()
                                  if task["inputs"].get("parent_task") in related}
            if children == related:
                break
            related = children
        if any(task["state"] not in TERMINAL for task in tasks.values() if task["task_id"] in related):
            return False
        self.event("scope_target_reached", "scope/target_reached", target=target,
                   hypothesis_refs=[self.result_card(task)["id"] for task in generators],
                   task_ids=sorted(related), completed_rounds=sorted(finished))
        return True


    def release_scope_closeout_reserve(self):
        """Round-two Generation has no permitted Recorder to consume this reserve."""
        target = self.config.get("scope_target")
        state = self.budget.snapshot()
        if (target != {"round": 2, "stage": "generation"}
                or state["config"].get("clock_mode") != "active_segments"
                or not state["effective_closeout_reserve_seconds"]):
            return False
        recorder = self.queue.get_by_key("r001/record")
        first_finished = next((e for e in self.journal.events
                               if e["type"] == "round_finished" and e["round"] == 1), None)
        # Success is known as soon as poll accepts the ordinary Recorder receipt.
        # A formally deferred first-round review also leaves no allowed round-two
        # Recorder; its unfinished scientific status must remain unchanged.
        if not (recorder and recorder["state"] == "completed") and first_finished is None:
            return False
        self.event("budget_closeout_reserve_released", "budget/scope_closeout_release", target=target,
                   configured_seconds=state["configured_closeout_reserve_seconds"], effective_seconds=0,
                   recorder_task_ref=recorder["task_card"] if recorder else None,
                   recorder_state=recorder["state"] if recorder else None,
                   recorder_outputs=recorder["outputs"] if recorder else [],
                   first_round_finished=first_finished is not None,
                   reason="Only second-round Generation remains; no second-round Recorder is authorized. "
                          "Total budget, fixed segment deadline and shutdown reserve are unchanged.")
        return True
