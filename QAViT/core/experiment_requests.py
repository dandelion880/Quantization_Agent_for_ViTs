"""实验申请：原始需求→单份代码→执行绑定→Safeguard→实测→原研究任务续调。

这是 Workflow 的职责分区，不独立实例化；状态和生命周期统一由 Workflow 拥有。
"""
from pathlib import Path
import re
from core.journal import read_json
from core.cards import ContractError, fingerprint
from runtime.identity import code_identity, measurement_implementation
from runtime.task_queue import TERMINAL
from tools.safeguards import approval_receipt, validate_approval



class ExperimentRequests:
    def pending_execution_rebind(self, task, *, current_core=None, measurement=None):
        """Only explicit host recovery may refresh work without a settled result."""
        if (task['kind'] != 'experiment_request' or task['state'] in TERMINAL
                or not task.get('request_ref') or not task.get('experiment_ref') or not task.get('bound_code_id')):
            return False
        core = current_core or code_identity()['core_sha256']
        if not any(e['type'] == 'implementation_intervention' and e['current_core_sha256'] == core
                   for e in self.journal.events):
            return False
        execution = self.journal.card(task['experiment_ref'])['payload']['execution']
        previous = execution.get('measurement_implementation')
        if previous is None or previous == (measurement if measurement is not None else measurement_implementation()):
            return False
        suffix = '/revision' + str(task['code_revision']) if task.get('code_revision') else ''
        tasks = self.tasks()
        code = tasks.get(task['key'] + '/code' + suffix)
        candidate = (code.get('result') or {}).get('artifact') if code and code['state'] == 'completed' else None
        if not candidate or candidate.get('id') != task['bound_code_id']:
            return False  # A newer Coding revision must complete through its normal path.
        trial = tasks.get(task['key'] + '/execute' + suffix)
        return not trial or (trial['state'] in ('pending', 'ready', 'retry') and not trial['outputs'] and (
            trial['attempt'] == 0 or (trial['state'] in ('ready', 'retry')
                                      and (trial.get('error') or {}).get('code') == 'InterruptedError')))

    def refresh_pending_execution(self, task):
        if not self.pending_execution_rebind(task):
            return task
        from tools.experiments import verify_assets
        old = self.journal.card(task['experiment_ref'])
        # A host repair changes measurement identity, never the frozen data/model assets.
        verify_assets({k: v for k, v in old['payload']['execution'].items() if k != 'measurement_implementation'})
        suffix = '/revision' + str(task['code_revision']) if task.get('code_revision') else ''
        code = self.queue.get_by_key(task['key'] + '/code' + suffix)
        candidate = (code.get('result') or {}).get('artifact') if code and code['state'] == 'completed' else None
        if not candidate or candidate['id'] != task['bound_code_id']:
            raise ContractError('恢复执行绑定缺少同一原请求的已完成代码')
        request = self.journal.card(task['request_ref'])
        experiment = self.broker.bind_execution(request, candidate)
        core = code_identity()['core_sha256']
        previous = {'experiment_ref': old['id'], 'bound_code_id': candidate['id'], 'current_core_sha256': core}
        self.queue.change(task['task_id'], task['state'], patch={
            'experiment_ref': experiment['id'], 'rebound_execution_core': core,
            'execution_binding_history': [*task.get('execution_binding_history', []), previous]},
            cards=[experiment], events=[{'type': 'pending_execution_rebound', 'task_id': task['task_id'],
                                        **previous, 'new_experiment_ref': experiment['id']}])
        return self.queue.get(task['task_id'])

    def wake_pending_execution_rebinds(self):
        core, measurement = code_identity()['core_sha256'], measurement_implementation()
        held = set()
        for task in self.queue.snapshot(kinds={"experiment_request"}).values():
            if task['kind'] != 'experiment_request' or task['state'] in TERMINAL:
                continue
            suffix = '/revision' + str(task['code_revision']) if task.get('code_revision') else ''
            trial = self.queue.get_by_key(task['key'] + '/execute' + suffix)
            handoff_pending = (task.get('rebound_execution_core') == core and trial
                and trial['state'] in ('pending', 'ready', 'retry')
                and trial['inputs']['experiment_ref'] != task.get('experiment_ref'))
            if not handoff_pending and not self.pending_execution_rebind(task, current_core=core, measurement=measurement):
                continue
            if trial:
                held.add(trial['task_id'])
                if task['state'] == 'waiting' and trial['task_id'] in task.get('waiting_for', []):
                    self.queue.change(task['task_id'], 'ready', patch={'waiting_for': []})
        return held

    @staticmethod
    def safeguard_core_binding(task):
        """Identity fixed at review publication, never the latest retry's core.

        Old persisted program keys already encode that identity. Noncanonical
        legacy handoffs may use their last request if no publication binding exists.
        """
        if task['inputs'].get('purpose') != 'safeguard':
            return None
        context = task['inputs'].get('context', {})
        explicit = context.get('host_implementation_sha256')
        key = re.search(r'/safeguard/([0-9a-f]{16})(?:/revision[0-9]+)?$', task['key'])
        if explicit:
            if not isinstance(explicit, str) or not re.fullmatch(r'[0-9a-f]{64}', explicit):
                raise ContractError('Safeguard 的宿主身份绑定损坏')
            if key and not explicit.startswith(key.group(1)):
                raise ContractError('Safeguard 的发布身份与任务键冲突')
            return explicit
        if key:
            return key.group(1)
        path = task.get('request_path')
        return read_json(path).get('core_sha256') if path else None

    def stale_safeguard_owner(self, task, tasks=None, *, current_core=None):
        """Use only host parent links; scientific prose cannot create ownership."""
        tasks = self.queue.snapshot() if tasks is None else tasks
        current_core = current_core or code_identity()['core_sha256']
        current, seen = task, set()
        while current and current['task_id'] not in seen:
            seen.add(current['task_id'])
            bound = self.safeguard_core_binding(current)
            if bound and not current_core.startswith(bound):
                return current
            current = tasks.get(current['inputs'].get('parent_task'))
        return None

    def cancel_stale_review_task(self, task, owner, *, receipt=None, current_core=None):
        # Terminal facts (including old rejection cards) remain immutable.
        current = self.queue.get(task['task_id'])
        if current['state'] in TERMINAL:
            return
        error = {'code': 'stale_safeguard_identity',
                 'message': '宿主实现已变化；旧身份审查不再执行或用于批准，由原请求发布当前身份审查',
                 'review_task_id': owner['task_id'],
                 'bound_core': self.safeguard_core_binding(owner),
                 'current_core': current_core or code_identity()['core_sha256']}
        patch = {'error': error}
        if receipt is not None:
            patch['superseded_receipt'] = str(receipt)
        self.queue.change(current['task_id'], 'cancelled', patch=patch)

    def retire_stale_safeguards(self):
        """Stop scheduling obsolete reviews and their read-only children.

        A running worker must first reach poll/accept so its durable receipt and
        usage settle. Recovery/stop already confirms its process exit. No failed
        or completed task is reopened, no original request is rewritten.
        """
        tasks, core = self.queue.snapshot(), code_identity()['core_sha256']
        for task in tasks.values():
            if task['state'] in TERMINAL or task['state'] == 'running':
                continue
            owner = self.stale_safeguard_owner(task, tasks, current_core=core)
            if owner:
                self.cancel_stale_review_task(task, owner, current_core=core)

    def retire_stale_clarification_return(self, task):
        """Keep a pending old-host handoff as history, not a new science task."""
        returned = task.get('clarification_return')
        if not returned or task['state'] in TERMINAL:
            return False
        child = self.queue.get(returned['child'])
        if not self.stale_safeguard_owner(child):
            return False
        self.queue.change(task['task_id'], task['state'], patch={
            'clarification_return': None,
            'superseded_clarification_returns': [
                *task.get('superseded_clarification_returns', []), returned]})
        return True

    def request_author(self, task, tasks=None):
        """Scientific ownership is distinct from a live queue dependency."""
        tasks = tasks if tasks is not None else self.queue.snapshot()
        current, seen, author = task, set(), None
        while current and current.get('task_id') not in seen:
            seen.add(current.get('task_id'))
            author = (tasks.get(current['inputs'].get('parent_task')) or
                      tasks.get(current['inputs'].get('origin_task')))
            if not author:
                # Upgrade old final-L3 requests on recovery without rewriting them.
                prefix, marker, tail = current.get('key', '').partition('/L3')
                if marker and tail == '':
                    author = next((t for t in tasks.values() if t['key'] == prefix + '/plan_l3'), None)
            if not author or author['kind'] == 'llm':
                break
            current = author
        return author if author and author['kind'] == 'llm' and author['role'] == task['inputs'].get('requester', author['role']) else None

    def cached_experiment(self, experiment, task, approval=None):
        """Reuse only facts visible to the requesting researcher at this handoff."""
        if approval is not None:
            validate_approval(approval, experiment)
        tasks = self.queue.snapshot()
        parent = self.request_author(task, tasks)
        scoped = dict(parent, _read_view={'sequence': self.journal.sequence}, _acl_allowed_refs=set()) if parent else None
        def visible(card):
            if scoped and not self.can_read_card(scoped, card):
                return False
            binding = experiment['payload'].get('candidate_binding')
            if experiment['payload']['level'] == 'L3' and binding:
                source = self.journal.card(card['payload']['experiment_ref'])
                if source['payload'].get('candidate_binding') != binding:
                    return False
            return True
        return self.broker.cached(experiment, code_artifact=approval['candidate'] if approval else None,
                                  visible=visible)

    def return_code_clarifications(self):
        """冻结规格的问题返回原作者；已提交计划由新任务承接，绝不改写旧计划。"""
        self.retire_stale_safeguards()
        tasks = self.queue.snapshot()
        for task in tasks.values():
            if task.get("clarification_return") and task["state"] in ("waiting", "ready", "retry"):
                if self.retire_stale_clarification_return(task):
                    continue
                self.finish_clarified_request(task)
                continue
            if task["kind"] != "experiment_request" or task["state"] != "waiting":
                continue
            parent = self.request_author(task, tasks)
            if not parent or not (parent['state'] == 'completed' or (
                    parent['state'] == 'waiting' and task['task_id'] in parent.get('waiting_for', []))):
                continue
            for identifier in task.get("waiting_for", []):
                child = tasks[identifier]
                if self.stale_safeguard_owner(child, tasks):
                    continue
                if (child["inputs"].get("purpose") not in ("code", "safeguard")
                        or not (child["state"] == "needs_clarification"
                                or (child["state"] == "cancelled" and
                                    (child.get("error") or {}).get("code") == "frozen_specification_clarification"))):
                    continue
                descendants = set(task.get("waiting_for", []))
                while True:
                    expanded = descendants | {t["task_id"] for t in tasks.values()
                                               if t["inputs"].get("parent_task") in descendants}
                    if expanded == descendants:
                        break
                    descendants = expanded
                # Let in-flight receipts settle before returning the frozen request.
                if any(tasks[i]["state"] == "running" for i in descendants):
                    break
                clarification = child["clarification"]
                result = {"status": "rejected",
                    "reason": "frozen_specification_clarification", "experiment_ref": task.get("experiment_ref") or task.get("request_ref"),
                    "clarification": clarification, "clarification_purpose": child["inputs"]["purpose"],
                    "clarification_task_ref": child["task_card"],
                    **({"coding_task_ref": child["task_card"]}
                       if child["inputs"]["purpose"] == "code" else {}),
                    "review_ref": task.get("code_revision_feedback", {}).get("review_ref"),
                    "instruction": "冻结规格不改写；由原申请任务根据澄清决定是否正常提交修订草案，或保留证据缺口。"
                                   "不得跳过 Safeguard，不得将工程拒绝判作假说失败。"}
                # Durable intent makes a crash between child cancellation and return replayable.
                self.queue.change(task["task_id"], task["state"], patch={"clarification_return": {
                    "result": result, "child": identifier, "descendants": sorted(descendants)}})
                self.finish_clarified_request(self.queue.get(task["task_id"]))
                break

    def finish_clarified_request(self, task):
        self.retire_stale_safeguards()
        task = self.queue.get(task['task_id'])
        if self.retire_stale_clarification_return(task):
            return
        returned = task["clarification_return"]
        for identifier in returned["descendants"]:
            child = self.queue.get(identifier)
            if child["state"] in ("pending", "ready", "retry", "waiting", "needs_clarification"):
                error = ({"code": "frozen_specification_clarification", **returned["result"]["clarification"]}
                         if identifier == returned["child"] else {"code": "request_returned_to_author"})
                self.queue.change(identifier, "cancelled", patch={"error": error})
        if task["state"] != "running":
            self.queue.change(task["task_id"], "ready")
            self.queue.claim(task["task_id"])
        self.queue.finish(task["task_id"], result=returned["result"])

    def request_outcome(self, trial, tasks=None, *, required_level=None, expected_hypothesis_ref=None):
        """Rebuild numerical facts independently of any interpretation task state."""
        lookup = tasks.get if tasks is not None else self.queue.get_by_key
        key, result = trial['key'], trial.get('result') or {}
        revision = trial.get('code_revision', 0)
        suffix = f'/revision{revision}' if revision else ''
        execution = lookup(key + '/execute' + suffix)
        interpretation = lookup(key + '/interpret')
        code = lookup(key + '/code' + suffix)
        artifact = (code.get('result') or {}).get('artifact') if code else None
        cache_reused = bool(result.get('cache_reused') or
            (execution and (execution.get('result') or {}).get('cache_reused')) or
            (interpretation and interpretation['inputs'].get('context', {}).get('cache_reused')))
        refs = set(trial['outputs'])
        if execution:
            refs.update(execution['outputs'])
        if interpretation:
            refs.update(interpretation['inputs'].get('refs', []))
        if result.get('evidence_ref'):
            refs.add(result['evidence_ref'])
        experiment_ref, valid, issues = trial.get('experiment_ref'), [], []
        for ref in sorted(refs):
            if self.journal.cards.get(ref, {}).get('kind') != 'EvidenceCard':
                continue
            try:
                evidence = self.journal.card(ref)
                if not experiment_ref:
                    raise ContractError('请求缺少冻结实验引用')
                experiment = self.journal.card(experiment_ref)
                if required_level and experiment['payload']['level'] != required_level:
                    raise ContractError('非 ' + required_level + ' 证据不能作为该级实验验证')
                self.broker.validate_evidence_for(experiment, evidence, code_artifact=artifact)
                binding = experiment['payload'].get('candidate_binding')
                if required_level == 'L3' and binding:
                    if expected_hypothesis_ref and binding.get('hypothesis_ref') != expected_hypothesis_ref:
                        raise ContractError('L3 执行候选与结果所属候选确切版本冲突')
                    source = self.journal.card(evidence['payload']['experiment_ref'])
                    if source['payload'].get('candidate_binding') != binding:
                        raise ContractError('L3 Evidence 属于另一候选确切版本')
                    if trial['inputs'].get('hypothesis_ref', binding['hypothesis_ref']) != binding['hypothesis_ref']:
                        raise ContractError('L3 请求与执行候选版本冲突')
                valid.append(ref)
            except (ContractError, OSError) as exc:
                issues.append({'evidence_ref': ref, 'reason': str(exc)})
        return {'request_state': trial['state'], 'experiment_ref': experiment_ref,
                'execution_state': execution['state'] if execution else ('cache_reused' if cache_reused else 'not_started'),
                'interpretation_state': interpretation['state'] if interpretation else 'not_separately_requested',
                'cache_reused': cache_reused, 'valid_evidence_refs': valid,
                'valid_evidence_missing': not bool(valid), 'evidence_issues': issues,
                'numerical_validation_status': 'validated' if valid else 'not_validated',
                'formal_validation_success': bool(valid)}

    def release_measured_requests(self):
        """Recover legacy waiting-for-interpret requests without another experiment/call."""
        tasks = self.queue.snapshot()
        for task in tasks.values():
            children = [tasks[i] for i in task.get('waiting_for', [])]
            if (task['kind'] == 'experiment_request' and task['state'] == 'waiting' and children
                    and all(t['inputs'].get('purpose') == 'interpret' for t in children)
                    and self.request_outcome(task)['valid_evidence_refs']):
                # Independent old interpreter receipts can still settle; their failure
                # must not determine the numerical request's completion.
                for child in children:
                    if child['state'] in ('pending', 'ready', 'retry', 'needs_clarification'):
                        self.queue.change(child['task_id'], 'cancelled', patch={'error': {
                            'code': 'interpretation_not_required',
                            'message': '有效测量直接返回原研究流程；解释交由现有科研步骤及 Recorder。'}})
                self.queue.change(task['task_id'], 'ready', patch={'waiting_for': []})

    def progress_planned_request(self, plan_key, request_key, *, refs, context):
        """Submit/continue an already authored plan through immutable task versions.

        Active tool callers resume via the queue. Program-created requests use this
        same result/error handoff in a new task belonging to their original author.
        """
        revision, previous = 0, None
        while True:
            suffix = f'/revision{revision}' if revision else ''
            current_plan, current_request = plan_key + suffix, request_key + suffix
            plan_refs, plan_context = list(refs), dict(context)
            if previous:
                prior_plan, prior_trial = previous
                result = prior_trial.get('result') or {}
                source_refs = [prior_plan['task_card'], prior_trial['task_card'],
                               *prior_trial['outputs']]
                if prior_trial.get('experiment_ref') or prior_trial.get('request_ref'):
                    source_refs.append(prior_trial.get('experiment_ref') or prior_trial['request_ref'])
                plan_refs.extend(source_refs)
                plan_context['request_feedback'] = {
                    'request_task_ref': prior_trial['task_card'], 'plan_task_ref': prior_plan['task_card'],
                    'state': prior_trial['state'], 'result': result, 'error': prior_trial.get('error'),
                    'original_request': {'ref': prior_trial['task_card'], 'field': '/payload/inputs/original_request'
                                         if 'original_request' in prior_trial['inputs'] else '/payload/inputs/draft'},
                    'instruction': '依据真实结果/失败调整计划，保留所选假说及正式协议。工程失败不是假说反证；'
                                   '不得无依据重复相同失败执行。可读取来源核实，不能执行时用 defer_reason 说明；'
                                   '只有缺少用户才能决定的信息才 clarify。'}
                plan_context['plan_revision'] = revision
                plan_context['previous_request_task'] = prior_trial['task_id']
            self.enqueue_role(current_plan, 'plan_l3', refs=plan_refs, context=plan_context)
            plan = self.tasks()[current_plan]
            if plan['state'] not in TERMINAL:
                return False
            if plan['state'] != 'completed' or (plan.get('result') or {}).get('defer_reason'):
                return True
            self.queue.enqueue(current_request, 'experiment_request', role='supervisor', inputs={
                **self.inspiration_request_binding(plan, plan['result']['draft']),
                **self.research_request_binding(plan, plan['result']['draft']),
                'draft': plan['result']['draft'],
                'original_request': plan['result'].get('original_request', plan['result']['draft']),
                'requester': plan['role'], 'refs': plan['inputs']['refs'], 'origin_task': plan['task_id'],
                'plan_revision': revision, 'request_root': request_key,
                **{k: context[k] for k in ('round', 'slot', 'hypothesis_ref', 'candidate_refs', 'selection_ref') if k in context},
                **({'previous_request_task': previous[1]['task_id']} if previous else {})})
            trial = self.tasks()[current_request]
            if trial['state'] not in TERMINAL:
                return False
            result = trial.get('result') or {}
            # A durable valid measurement also resolves old requests failed only by
            # interpretation. No duplicate execution or scientific decision rewrite.
            if (trial['state'] == 'cancelled' or result.get('status') == 'completed'
                    or self.request_outcome(trial)['valid_evidence_refs']):
                return True
            previous, revision = (plan, trial), revision + 1

    def l3_request_keys(self, number, slot, hypothesis_ref):
        """Versioned identities, with in-place continuation of a historical L3."""
        prefix = f'r{number:03d}'
        legacy = self.queue.get_by_key(prefix + '/plan_l3')
        if legacy and legacy['inputs'].get('context', {}).get('hypothesis_ref') == hypothesis_ref:
            return prefix + '/plan_l3', prefix + '/L3'
        trial = self.queue.get_by_key(prefix + '/L3')
        binding = (trial['inputs'].get('draft') or {}).get('candidate_binding', {}) if trial else {}
        if hypothesis_ref and binding.get('hypothesis_ref') == hypothesis_ref:
            return prefix + '/plan_l3', prefix + '/L3'
        suffix = '/' + slot + '/' + fingerprint(hypothesis_ref)[:16]
        return prefix + '/plan_l3' + suffix, prefix + '/L3' + suffix

    def l3_candidate_outcome(self, number, slot, hypothesis_ref):
        """Rebuild one exact candidate's attempts without completion-order coupling."""
        tasks = self.tasks()
        plan_key, key = self.l3_request_keys(number, slot, hypothesis_ref)
        trials = [t for k, t in tasks.items() if t['kind'] == 'experiment_request'
                  and (k == key or t['inputs'].get('request_root') == key)]
        trials.sort(key=lambda t: t['inputs'].get('plan_revision', 0))
        plans = [t for k, t in tasks.items() if k == plan_key or k.startswith(plan_key + '/revision')]
        plans.sort(key=lambda t: t['inputs'].get('context', {}).get('plan_revision', 0))
        related = plans + [t for k, t in tasks.items() if k == key or k.startswith(key + '/')]
        trial, plan = (trials[-1] if trials else None), (plans[-1] if plans else None)
        # A new planning revision is current even before it publishes a request.
        # Keep the earlier execution facts separately instead of reporting its
        # failed state as the state of the resumed planning step.
        newer_plan = plan and (not trial or plan['inputs'].get('context', {}).get('plan_revision', 0)
                              > trial['inputs'].get('plan_revision', 0))
        current = plan if newer_plan or (plan and (plan.get('result') or {}).get('defer_reason')) else trial or plan
        result = (current.get('result') or {}) if current else {}
        status = ('deferred' if result.get('defer_reason') else
                  result.get('status', current['state']) if current['state'] == 'completed' else current['state']) if current else (
                  'not_requested' if self.config['enable_l3'] else 'disabled')
        outcomes = [self.request_outcome(t, tasks, required_level='L3',
                                         expected_hypothesis_ref=hypothesis_ref) for t in trials]
        latest = outcomes[-1] if outcomes else {'request_state': 'not_started', 'execution_state': 'not_started',
                  'interpretation_state': 'not_separately_requested', 'experiment_ref': None, 'cache_reused': False}
        valid = list(dict.fromkeys(ref for outcome in outcomes for ref in outcome['valid_evidence_refs']))
        facts = [{'task_id': t['task_id'], 'key': t['key'], 'task_ref': t['task_card'],
                  'state': t['state'], 'result': t.get('result'), 'error': t.get('error')} for t in related]
        return {**latest, 'status': status, 'slot': slot, 'hypothesis_ref': hypothesis_ref,
                'request_root': key,
                'planning_state': plan['state'] if plan else 'not_started',
                'plan_revision': plan['inputs'].get('context', {}).get('plan_revision', 0) if plan else None,
                'tasks': facts, 'task_refs': [t['task_card'] for t in related],
                'valid_evidence_refs': valid, 'valid_evidence_missing': not bool(valid),
                'evidence_issues': [issue for outcome in outcomes for issue in outcome['evidence_issues']],
                'numerical_validation_status': 'validated' if valid else 'not_validated',
                'formal_validation_success': bool(valid)}

    def l3_outcome(self, number):
        """All three slots must validate; partial completion is explicitly partial."""
        selection = self.selection_card(number)
        candidates = selection['payload'].get('candidate_refs', {}) if selection else self.candidate_refs(number)
        expected = ['H1', 'H2', 'H3']
        historical = bool(selection and selection['payload'].get('selection_stage') == 'gonly_preselection'
                          and any(e['type'] == 'round_finished' and e['round'] == number
                                  and 'candidate_refs' not in e for e in self.journal.events))
        if historical:
            selected = selection['payload'].get('selected_slot')
            expected = [selected] if selected else []
            candidates = {slot: ref for slot, ref in candidates.items() if slot in expected}
        rows = {slot: self.l3_candidate_outcome(number, slot, ref) for slot, ref in sorted(candidates.items())}
        # Read old isolated report fixtures/runs that have only a legacy plan.
        if not rows:
            legacy = self.queue.get_by_key(f'r{number:03d}/plan_l3')
            if legacy:
                c = legacy['inputs'].get('context', {})
                slot = c.get('slot', c.get('selection', 'H1'))
                rows[slot] = self.l3_candidate_outcome(number, slot, c.get('hypothesis_ref'))
        legacy_key = f'r{number:03d}/L3'
        unattributed = []
        roots = {row['request_root'] for row in rows.values()}
        if legacy_key not in roots:
            # Old engineering receipts without a candidate binding remain readable,
            # but cannot be guessed into one of the new candidate result slots.
            unattributed = [{'task_id': t['task_id'], 'key': t['key'], 'task_ref': t['task_card'],
                             'state': t['state'], 'result': t.get('result'), 'error': t.get('error')}
                            for key, t in self.tasks().items()
                            if (key == legacy_key or key.startswith(legacy_key + '/'))
                            and not any(key == root or key.startswith(root + '/') for root in roots)]
        valid = list(dict.fromkeys(ref for row in rows.values() for ref in row['valid_evidence_refs']))
        success = bool(expected) and set(rows) == set(expected) and all(row['formal_validation_success'] for row in rows.values())
        states = [row['status'] for row in rows.values()]
        status = ('disabled' if not self.config['enable_l3'] else 'completed' if success else
                  'partial' if valid else states[0] if states and len(set(states)) == 1 and states[0] != 'completed' else
                  'in_progress' if any(s in ('pending', 'ready', 'running', 'waiting', 'retry') for s in states) else
                  'incomplete' if states or unattributed else 'not_requested')
        # Legacy top-level state fields remain useful to old readers; per-candidate
        # rows are authoritative whenever more than one branch exists.
        common = {k: (next(iter(rows.values()))[k] if len(rows) == 1 else 'per_candidate')
                  for k in ('request_state', 'execution_state', 'interpretation_state', 'planning_state')}
        return {**common, 'status': status,
                'candidates': {slot: {k: v for k, v in row.items() if k != 'tasks'} for slot, row in rows.items()},
                'expected_slots': expected, 'historical_single_candidate': historical,
                'missing_slots': [slot for slot in expected if slot not in rows],
                'validated_slots': [slot for slot, row in rows.items() if row['formal_validation_success']],
                'tasks': [t for row in rows.values() for t in row['tasks']] + unattributed,
                'task_refs': [ref for row in rows.values() for ref in row['task_refs']] + [t['task_ref'] for t in unattributed],
                'unattributed_task_refs': [t['task_ref'] for t in unattributed],
                'valid_evidence_refs': valid, 'valid_evidence_missing': not bool(valid),
                'evidence_issues': [issue for row in rows.values() for issue in row['evidence_issues']],
                'numerical_validation_status': 'validated' if success else 'partial' if valid else 'not_validated',
                'formal_validation_success': success}

    def research_request_binding(self, author, draft=None):
        binding = self.topics.binding(author, self.read_view(author)['sequence'])
        for field in ('topic_ref','method_id'):
            explicit = (draft or {}).get(field)
            if explicit and field in binding and explicit != binding[field]:
                raise ContractError('请求与确切候选/当前主题身份冲突: '+field)
            if explicit: binding[field] = explicit
        if binding.get('topic_ref') and binding['topic_ref'] not in self.topics.versions(self.read_view(author)['sequence']):
            raise ContractError('请求引用未知主题版本')
        if binding.get('method_id') and binding['method_id'] not in self.topics.methods(self.read_view(author)['sequence']):
            raise ContractError('请求引用未知方法谱系')
        return binding

    def inspiration_request_binding(self, author, draft=None):
        explicit = (draft or {}).get('inspiration_id')
        if explicit and explicit not in self.inspirations.snapshot(self.read_view(author)['sequence']):
            raise ContractError('实验引用了未知或不可见的灵感')
        if author['inputs'].get('purpose') not in ('generate', 'plan_l3'):
            return {'inspiration_id': explicit} if explicit else {}
        identifier = self.inspirations.bound_id(author, self.read_view(author)['sequence'])
        if identifier and explicit and identifier != explicit:
            raise ContractError('实验灵感与当前 H3/确切候选身份冲突；先更新灵感决定')
        identifier = identifier or explicit
        return {'inspiration_id': identifier} if identifier else {}

    def implementation_handoff(self, task):
        """Bind only same-researcher history or explicitly cited public Evidence producers."""
        tasks = self.queue.snapshot()
        author = self.request_author(task, tasks)
        if not author:
            return []
        def actor(t):
            c = t['inputs'].get('context', {})
            return (t['inputs'].get('purpose'), c.get('round'), c.get('slot'), tuple(c.get('pair', [])), c.get('side'))
        cited = set(task['inputs'].get('refs', [])) | set(task['inputs'].get('draft', {}).get('evidence_refs', []))
        history = []
        identifier = task['inputs'].get('inspiration_id')
        for previous in tasks.values():
            if previous['kind'] != 'experiment_request' or previous['task_id'] == task['task_id']:
                continue
            owner = self.request_author(previous, tasks)
            shared = bool(cited.intersection(previous.get('outputs', [])))
            if identifier and previous['inputs'].get('inspiration_id') not in (None, identifier) and not shared:
                continue
            if not shared and (not owner or actor(owner) != actor(author)):
                continue
            ref = previous.get('experiment_ref') or previous.get('request_ref')
            if not ref or not self.can_read_card(author, self.journal.card(ref)):
                continue
            children = [t for t in tasks.values() if t['inputs'].get('parent_task') == previous['task_id']]
            codes = [t for t in children if t['inputs'].get('purpose') == 'code' and (t.get('result') or {}).get('artifact')]
            if not codes:
                continue
            candidate = codes[-1]['result']['artifact']
            trials = [t for t in children if t['kind'] == 'experiment']
            history.append({'candidate': candidate, 'request_ref': previous.get('request_ref', ref),
                'experiment_ref': ref, 'source_task_ref': codes[-1]['task_card'],
                'execution_task_refs': [t['task_card'] for t in trials],
                'failures': [{'error': t.get('error'), 'worker_log': str(Path(t['request_path']).with_name('worker.log')) if t.get('request_path') else None}
                             for t in trials if t.get('error')],
                'meaning': 'Prior implementation context, not a requirement to preserve its bug or algorithm choices.'})
        identifier = task['inputs'].get('inspiration_id')
        if identifier:
            # The idea handoff survives a different round/author and failed or
            # low-scoring attempts without requiring a complete solution entry.
            prior = self.inspirations.recovery(identifier)['implementation_history']
            own_ids = {row['candidate']['id'] for row in history}
            history = [r for r in prior if r['candidate']['id'] not in own_ids] + history
        method_id = task['inputs'].get('method_id')
        if method_id:
            prior = self.topics.method_recovery(method_id)['implementation_history']
            own_ids = {r['candidate']['id'] for r in history}
            history = [r for r in prior if r['candidate']['id'] not in own_ids] + history
        return history[-3:]

    def progress_request(self, task):
        self.retire_stale_safeguards()
        task = self.queue.get(task['task_id'])
        if task['state'] in TERMINAL:
            return
        task = self.refresh_pending_execution(task)
        if self.retire_stale_clarification_return(task):
            task = self.queue.get(task['task_id'])
        if task.get("clarification_return"):
            self.finish_clarified_request(task)
            return
        inputs, key = task["inputs"], task["key"]
        tasks = self.tasks()
        if not task.get("request_ref") and not task.get("experiment_ref"):
            draft = self.broker.preflight(inputs["draft"])
            if draft["level"] == "L3" and not self.config["enable_l3"]:
                raise ContractError("本次验证明确禁用 L3")
            if draft["execution"]["requires_code"]:
                request = self.broker.request(inputs["draft"], requester=inputs["requester"], refs=inputs["refs"],
                                              original_request=inputs.get("original_request", inputs["draft"]))
                patch = {"request_ref": request["id"]}
            else:
                request = self.broker.freeze(draft, requester=inputs["requester"], refs=inputs["refs"])
                patch = {"experiment_ref": request["id"]}
            self.queue.change(task["task_id"], "running", patch=patch, cards=[request])
            task = self.queue.get(task["task_id"])
        request = self.journal.card(task.get("request_ref") or task["experiment_ref"])
        execution, approval, experiment = request["payload"]["execution"], None, request
        suffix = ""
        if execution.get("requires_code"):
            revision = task.get("code_revision", 0)
            suffix = f"/revision{revision}" if revision else ""
            code_key = key + "/code" + suffix
            failed_execution = tasks.get(key + "/execute" + suffix)
            if failed_execution and failed_execution['state'] == 'failed' and (failed_execution.get('error') or {}).get('origin') == 'request':
                self.return_request_failure(task, failed_execution)
                return
            if failed_execution and failed_execution["state"] == "failed" and self.revisable_failure(failed_execution):
                previous = tasks[code_key]["result"]["artifact"]
                if self.revise_implementation(task, revision, previous, execution_error=failed_execution):
                    return
            code_context = {"experiment_ref": request["id"], "request_ref": request["id"]}
            code_context["implementation_history"] = self.implementation_handoff(task)
            if revision:
                code_context["revision_feedback"] = task["code_revision_feedback"]
            identifier = self.enqueue_role(code_key, "code", refs=[self.brief["id"], request["id"]],
                                           context=code_context, parent_task=task["task_id"])
            code_task = self.tasks()[code_key]
            if code_task["state"] != "completed":
                self.wait_children_or_fail(task, [identifier])
                return
            candidate = code_task["result"]["artifact"]
            prior = task.get("code_revision_feedback", {}).get("previous_candidate")
            if prior and (candidate["source_sha256"], candidate.get("execution_choices", {})) == (
                    prior["source_sha256"], prior.get("execution_choices", {})):
                self.queue.finish(task["task_id"], result={"status": "rejected",
                    "reason": "implementation_unchanged_after_failure", "artifact": candidate,
                    "feedback": task["code_revision_feedback"]})
                return
            if task.get("bound_code_id") == candidate["id"] and task.get("experiment_ref"):
                experiment = self.journal.card(task["experiment_ref"])
            else:
                try:
                    experiment = self.broker.bind_execution(request, candidate)
                except ContractError as exc:
                    from runtime.failures import FrozenSourceError, failure_details
                    if isinstance(exc, FrozenSourceError):
                        self.queue.fail(task['task_id'], failure_details(exc, {}))
                        return
                    if self.revise_implementation(task, revision, candidate, binding_error=str(exc)):
                        return
                    self.queue.finish(task["task_id"], result={"status": "rejected", "reason": str(exc)})
                    return
                self.queue.change(task["task_id"], "running", patch={"experiment_ref": experiment["id"],
                                  "bound_code_id": candidate["id"]}, cards=[experiment])
            execution = experiment["payload"]["execution"]
            self.check_experiment_resources(experiment)
            review_core = code_identity()["core_sha256"]
            implementation = review_core[:16]
            review_key = (key + '/binding/' + fingerprint(experiment['id'])[:16]
                          if task.get('rebound_execution_core') else key)
            safeguard_key = review_key + "/safeguard/" + implementation + suffix
            identifier = self.enqueue_role(safeguard_key, "safeguard", refs=[self.brief["id"], request["id"], experiment["id"]],
                context={"request_ref": request["id"], "experiment_ref": experiment["id"], "candidate": candidate,
                         "host_implementation_sha256": review_core,
                         "implementation_history": code_task["inputs"]["context"].get("implementation_history", [])},
                depends=[code_task["task_id"]], parent_task=task["task_id"])
            review_task = self.queue.get_by_key(safeguard_key)
            if self.stale_safeguard_owner(review_task):
                raise ContractError("当前 Safeguard key 仍绑定过期宿主身份，拒绝消费其审批")
            review = self.result_card(review_task)
            if not review:
                self.wait_children_or_fail(task, [identifier])
                return
            if not review["payload"]["approved"]:
                if self.revise_implementation(task, revision, candidate, review=review):
                    return
                self.queue.finish(task["task_id"], outputs=[review["id"]], result={"status": "rejected",
                    "reason": "Safeguard rejected", "review_ref": review["id"]})
                return
            approval = approval_receipt(experiment, candidate, review)
        self.check_experiment_resources(experiment)
        trial_key = key + "/execute" + suffix
        prior_trial = self.queue.get_by_key(trial_key)
        if prior_trial and prior_trial['inputs']['experiment_ref'] != experiment['id']:
            if (not task.get('rebound_execution_core') or prior_trial['state'] not in ('pending', 'ready', 'retry')
                    or prior_trial['outputs'] or (prior_trial['attempt'] and
                       (prior_trial.get('error') or {}).get('code') != 'InterruptedError')):
                raise ContractError('已有执行任务不能静默更换冻结输入')
            self.queue.change(prior_trial['task_id'], prior_trial['state'], patch={
                'inputs': {'experiment_ref': experiment['id'], 'approval': approval, 'parent_task': task['task_id']},
                'input_versions': {},
                'superseded_execution_inputs': [*prior_trial.get('superseded_execution_inputs', []), {
                    'inputs': prior_trial['inputs'], 'input_versions': prior_trial['input_versions'],
                    'request_path': prior_trial.get('request_path'), 'attempt': prior_trial['attempt'],
                    'error': prior_trial.get('error')}]})
            prior_trial = self.queue.get(prior_trial['task_id'])
        own_result = prior_trial and prior_trial["state"] == "completed"
        cache = self.journal.card(prior_trial["outputs"][0]) if own_result else self.cached_experiment(experiment, task, approval)
        cache_reused = bool(cache and not own_result) or bool(own_result and (prior_trial.get("result") or {}).get("cache_reused"))
        if cache:
            evidence = cache
        else:
            identifier = self.queue.enqueue(trial_key, "experiment", role=None, inputs={
                "experiment_ref": experiment["id"], "approval": approval, "parent_task": task["task_id"]}, resource_keys=self.experiment_resources(experiment))
            trial = self.tasks()[trial_key]
            if trial["state"] != "completed":
                if trial["state"] == "failed" and (trial.get("error") or {}).get("origin") == "request":
                    self.return_request_failure(task, trial)
                    return
                if (execution.get("requires_code") and trial["state"] == "failed" and revision < 2
                        and self.revisable_failure(trial)):
                    self.queue.change(task["task_id"], "retry", patch={"retry_after": 0})
                    return
                self.wait_children_or_fail(task, [identifier])
                return
            evidence = self.journal.card(trial["outputs"][0])
        # Evidence has already passed the execution protocol and immutable storage
        # checks. Interpretation belongs to the requester or the normal Recorder;
        # it is never a dependency for returning this numerical result.
        interpretation = self.queue.get_by_key(key + '/interpret')
        interpreted = self.result_card(interpretation)
        self.queue.finish(task['task_id'], outputs=[evidence['id']], result={
            'status': 'completed', 'experiment_ref': experiment['id'], 'evidence_ref': evidence['id'],
            **({'interpretation_ref': interpreted['id']} if interpreted else {}),
            'interpretation_state': interpretation['state'] if interpretation else 'not_separately_requested',
            'cache_reused': cache_reused, 'raw_fact_and_interpretation_separated': True})


    def return_request_failure(self, task, trial):
        """Return immutable failed execution to its scientific owner without Coding retries."""
        error = trial['error']
        self.queue.finish(task['task_id'], result={
            'status': 'rejected', 'reason': 'request_revision_required',
            'experiment_ref': task.get('experiment_ref') or task.get('request_ref'),
            'execution_task_ref': trial['task_card'], 'error': error,
            'numerical_validation_status': 'not_validated', 'recovery': 'revise_request',
            'mechanism_measurement_completed': False, 'partial_execution_possible': True,
            'instruction': 'Generation 根据明确来源/选择反馈提交修订实验请求；原请求、失败与源码保留。'
                           'Coding 无法修改已冻结请求；加载失败不构成机制反证。'})

    def check_experiment_resources(self, experiment):
        from tools.experiments import check_resources
        check_resources(experiment['payload']['resources'], self.config['runtime'])

    @staticmethod
    def revisable_failure(task):
        error = task.get("error") or {}
        if error.get("origin") in ("host", "resource", "request"):
            return False
        # Identity-checked provenance is stronger than exception wording/names.
        if error.get('origin') == 'implementation':
            return True
        if error.get('code') == 'EngineeringBlocked':
            return False
        # Resource, storage, process and host-version failures remain engineering
        # failures. Do not burn a Coding revision or rerun the same failed source.
        if any(term in error.get("message", "").lower() for term in
               ("out of memory", "cuda error", "核心实现", "资产已变化", "运行时/主机", "批准")):
            return False
        return error.get("code") in {"TypeError", "AttributeError", "KeyError", "IndexError", "ValueError",
                                      "NameError", "UnboundLocalError", "ImportError", "ModuleNotFoundError",
                                      "RuntimeError", "AssertionError", "ArithmeticError", "ZeroDivisionError",
                                      "OverflowError", "FloatingPointError", "NotImplementedError", "ContractError"}

    def revise_implementation(self, task, revision, candidate, *, execution_error=None, binding_error=None, review=None):
        if revision >= 2:
            return False
        feedback = {"previous_candidate": candidate,
            "instruction": "依据当前源码和具体问题修订同一实现及必要执行选择；保留原始科研需求。"
                           "仅原始科学需求确有歧义时 clarify；新实现重新绑定执行条件并经 Safeguard。"}
        if execution_error:
            path = execution_error.get("request_path")
            feedback.update(execution_task_ref=execution_error["task_card"], error=execution_error["error"],
                            worker_log=str(Path(path).with_name("worker.log")) if path else None)
        if binding_error:
            feedback["binding_error"] = binding_error
        if review:
            feedback.update(review_ref=review["id"], review=review["payload"])
        self.queue.change(task["task_id"], "retry", patch={"retry_after": 0, "code_revision": revision + 1,
                          "code_revision_feedback": feedback})
        return True

    def experiment_resources(self, experiment):
        resources = experiment["payload"]["resources"]
        # 每个 worker 加载独立模型；只读同一 checkpoint 不构成互斥资源。
        keys = []
        if resources.get("gpu") is not None:
            keys.append("gpu:" + str(resources["gpu"]))
        return keys

