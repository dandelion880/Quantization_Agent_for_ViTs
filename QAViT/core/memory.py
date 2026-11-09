"""四层科研 Memory 与程序化检索；Recorder 提议科研判断，程序核验出处。

L0 即 Journal，不允许 Recorder 改写。L1/L2/L3 是有版本的投影；迟到证据
按其模型、配置及观察时点归档，不用写入顺序覆盖当前事实。
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from .cards import ContractError, canonical, fingerprint, MEMORY_QUERY_SCHEMA, PAYLOAD_SCHEMAS, validate_schema
from .evidence_scope import matches as scope_matches, validate_claim_scope, bind_claim_scope, card_scope
from .evidence import project_observations


KNOWLEDGE_STATUSES = {"hypothesis", "observed_once", "replicated", "refuted", "conditional", "abandoned"}
ROLE_KINDS = {
    "supervisor": {"ResearchBrief", "MetareviewCard"},
    "generation": {"HypothesisCard", "ExperimentCard", "ReviewDecisionCard", "EvidenceCard", "MetareviewCard"},
    "judger": {"HypothesisCard", "ExperimentCard", "ReviewDecisionCard", "EvidenceCard", "MetareviewCard"},
    "evolution": {"HypothesisCard", "ExperimentCard", "ReviewDecisionCard", "EvidenceCard", "MetareviewCard"},
    "recorder": {"HypothesisCard", "ExperimentCard", "ReviewDecisionCard", "EvidenceCard", "MetareviewCard"},
    "coding": {"ExperimentCard", "ReviewDecisionCard"},
    "safeguard": {"ExperimentCard", "ReviewDecisionCard", "EvidenceCard"},
}


def flattened(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flattened(item, prefix + "." + key if prefix else key))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def lexical_tokens(text):
    words = re.findall(r"[a-z0-9_.-]+", text.lower())
    for part in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(part[i:i + 2] for i in range(max(1, len(part) - 1)))
    return set(words)


def short_field(value, ref, field, limit=1400):
    """Whole short fields, otherwise complete sentences plus an explicit source pointer."""
    if len(canonical(value)) <= limit:
        return deepcopy(value)
    result = {'read_card': {'ref': ref, 'field': field}, 'characters': len(canonical(value)), 'complete': False}
    if isinstance(value, str):
        sentences = re.split(r'(?<=[。！？!?\n])\s*|(?<=\.)\s+', value)
        preview = ''
        for sentence in sentences:
            if len(preview) + len(sentence) > limit:
                break
            preview += sentence
        if preview:
            result['excerpt'] = preview
    return result


class ResearchMemory:
    def __init__(self, journal):
        self.journal = journal

    def snapshot(self, sequence=None):
        current = {"L1": {"current_round": None, "round_selection": None, "incumbent": None,
                          "open_questions": [], "recent_hypotheses": [], "recent_evidence": []},
                   "L2": {}, "L3": [], "projection_version": 0}
        for event in reversed(self.journal.events):
            if event["type"] == "memory_projection" and (sequence is None or event["commit_sequence"] <= sequence):
                current = deepcopy(event["value"])
                break
        visible = [ref for ref, entry in self.journal.cards.items() if sequence is None or entry["sequence"] <= sequence]
        # L1 facts follow Journal commits, never the Recorder's selective citations.
        starts = [event for event in self.journal.events if event['type'] == 'round_started'
                  and (sequence is None or event['commit_sequence'] <= sequence)]
        start = starts[-1]['commit_sequence'] if starts else 0
        if starts:
            current['L1']['current_round'] = starts[-1]['round']
        for kind, key in (('HypothesisCard', 'recent_hypotheses'), ('EvidenceCard', 'recent_evidence'),
                          ('ExperimentCard', 'recent_experiments')):
            current['L1'][key] = [ref for ref in visible if self.journal.cards[ref]['kind'] == kind
                                  and self.journal.cards[ref]['sequence'] >= start]
        return {**current, "memory_version": max((self.journal.card_versions.get(r, 0) for r in visible), default=0),
                "L0": {"source": "immutable cards and commits", "card_count": len(visible)}}

    def recording_batch(self, sequence=None):
        """仅按已提交对象关系追踪未整理版本；不对科学相关性作自动判断。"""
        sequence = self.journal.sequence if sequence is None else sequence
        covered, latest_tasks = set(), {}
        for event in self.journal.events:
            if event["commit_sequence"] > sequence:
                continue
            if event["type"] == "recorder_batch_completed":
                covered.update(event["source_refs"])
            if event["type"] == "task_snapshot":
                latest_tasks[event["task"]["task_id"]] = event["task"]
        science = [ref for ref, entry in self.journal.cards.items() if entry["sequence"] <= sequence
                   and entry["kind"] in {"HypothesisCard", "ReviewDecisionCard", "ExperimentCard", "EvidenceCard"}
                   and ref not in covered]
        engineering, failures = [], []
        for task in latest_tasks.values():
            relevant = (task["kind"] in {"experiment", "experiment_request", "literature"}
                        or task["inputs"].get("purpose") in {"code", "safeguard", "plan_l3"})
            if relevant and task["state"] in {"completed", "failed", "cancelled", "ineligible"} and task["task_card"] not in covered:
                engineering.append(task["task_card"])
                result = task.get('result') or {}
                if task['state'] in {'failed', 'cancelled', 'ineligible'} or result.get('status') in {
                        'failed', 'rejected', 'interrupted', 'error', 'ineligible'}:
                    failures.append(task['task_card'])
        refs = sorted(science + engineering)
        judgments = [r for r in science if self.journal.cards[r]["kind"] == "HypothesisCard" or
                     self.journal.cards[r]["kind"] == "ReviewDecisionCard"]
        return {"batch_id": fingerprint(refs), "through_sequence": sequence, "source_refs": refs,
                "scientific_refs": sorted(science), "engineering_refs": sorted(engineering),
                "scientific_failure_refs": sorted(failures), "review_required": bool(science or failures),
                "judgment_refs": sorted(judgments)}

    def pending_knowledge(self, sequence=None, visible=None):
        """Latest unresolved drafts only; immutable old cards retain their original issues."""
        pending = {}
        for ref, entry in self.journal.cards.items():
            if entry['kind'] != 'MetareviewCard' or (sequence is not None and entry['sequence'] > sequence):
                continue
            card = self.journal.card(ref)
            if visible is not None and not visible(card):
                continue
            payload = card['payload']
            for row in payload.get('pending_knowledge_updates', []):
                submission = row.get('submission') if isinstance(row.get('submission'), dict) else {}
                identity = row.get('claim_id') or submission.get('claim_id') or self.claim_identity(submission)
                pending[identity] = {**deepcopy(row), 'claim_id': identity, 'metareview_ref': ref}
            for update in payload.get('knowledge_updates', []):
                pending.pop(update['claim_id'], None)
        return pending

    @staticmethod
    def claim_identity(claim):
        return 'knowledge-' + fingerprint({'text': claim.get('text'), 'scope': claim.get('scope')})[:20]

    def recording_status(self, sequence=None, *, visible=None, pending_offset=0, pending_limit=8):
        """归档是程序事实，复盘是 Recorder 产出；不把前者冒充后者。"""
        batch = self.recording_batch(sequence)
        deferred = [e for e in self.journal.events if e['type'] == 'recorder_deferred'
                    and (sequence is None or e['commit_sequence'] <= sequence)
                    and set(e['source_refs']) & set(batch['source_refs'])]
        pending = list(self.pending_knowledge(sequence, visible).values())
        displayed = [{**{k: row[k] for k in ('metareview_ref', 'index', 'claim_id')},
                      'issue': short_field(row['issue'], row['metareview_ref'],
                          '/payload/pending_knowledge_updates', 700),
                      'read_card': {'ref': row['metareview_ref'], 'field': '/payload/pending_knowledge_updates'}}
                     for row in pending[pending_offset:pending_offset + pending_limit]]
        return {'facts': 'archived_in_journal', 'scientific_review': 'pending' if batch['review_required'] else 'completed',
                'pending_source_count': len(batch['source_refs']),
                'pending_scientific_count': len(batch['scientific_refs']) + len(batch['scientific_failure_refs']),
                'engineering_source_count': len(batch['engineering_refs']),
                'deferred_task_refs': [e['task_ref'] for e in deferred],
                'knowledge': 'pending_verification' if pending else 'no_pending_updates',
                'pending_knowledge_count': len(pending), 'pending_knowledge_updates': displayed,
                'pending_knowledge_more': {'query': {'layer': 'L2'}, 'offset': pending_offset + pending_limit,
                                          'limit': pending_limit} if pending_offset + pending_limit < len(pending) else None,
                'meaning': 'completed 表示没有待复盘科学材料；纯工程补记只归档，不冒充已发布科学综合。pending 知识独立保留。'}

    def prepare_metareview(self, content, batch, round_number, *, resolve_claim=None):
        """编译科研提交；每条知识独立绑定、核验，问题条目保留但不进入 L2。"""
        value = deepcopy(content)
        if 'round' in value and value['round'] != round_number:
            raise ContractError('Meta-review 轮次错误')
        value['round'] = round_number
        # Legacy declarations are accepted but never treated as observed reading facts.
        value.pop('reviewed_refs', None)
        for field in ('judgment_audit', 'predictive_metrics', 'knowledge_updates', 'recurring_failures',
                      'blind_spots', 'evidence_refs'):
            value.setdefault(field, [])
        value.setdefault('generation_guidance', '')
        if 'pending_knowledge_updates' in value:
            raise ContractError('pending_knowledge_updates 由程序逐条核验后生成')
        accepted, pending, seen = [], [], set()
        # 用任务批次快照绑定旧版本；提交重放不能受本次发布后的投影影响。
        prior = self.snapshot(batch.get('through_sequence'))['L2']
        drafts = self.pending_knowledge(batch.get('through_sequence'))
        submitted_updates = value['knowledge_updates']
        if not isinstance(submitted_updates, list):
            submitted_updates = [submitted_updates]
        for index, submitted in enumerate(submitted_updates):
            claim = {}
            try:
                if not isinstance(submitted, dict):
                    raise ContractError('知识更新必须是对象')
                claim = resolve_claim(deepcopy(submitted)) if resolve_claim else deepcopy(submitted)
                if not claim.get('claim_id'):
                    candidates = {key for key, old in {**drafts, **prior}.items()
                                  if (claim.get('supersedes') and (old.get('revision_ref') or old.get('metareview_ref')) == claim['supersedes'])
                                  or (key in drafts and claim.get('text') and (old.get('text') or
                                      (old.get('submission') if isinstance(old.get('submission'), dict) else {}).get('text')) == claim['text'])}
                    if len(candidates) == 1:
                        claim['claim_id'] = next(iter(candidates))
                old_draft = drafts.get(claim.get('claim_id'))
                old_claim = prior.get(claim.get('claim_id'))
                if claim.get('status') == 'abandoned':
                    previous = old_claim or (old_draft or {}).get('submission')
                    if old_claim is None and old_draft is None:
                        raise ContractError('放弃须关联已有知识或 pending claim_id')
                    old_text = previous.get('text') if isinstance(previous, dict) else previous
                    old_scope = previous.get('scope') if isinstance(previous, dict) else None
                    claim.setdefault('text', old_text if isinstance(old_text, str) and old_text else 'Abandoned pending draft')
                    claim.setdefault('scope', deepcopy(old_scope) if isinstance(old_scope, dict) and old_scope else
                                     {'pending_draft': old_draft['metareview_ref']})
                    claim.setdefault('rationale', 'Recorder explicitly abandoned this draft')
                claim.setdefault('evidence_refs', [])
                validate_schema(claim['evidence_refs'], {'type': 'array', 'items': {'type': 'string'}}, 'evidence_refs')
                claim['evidence_refs'] = sorted(set(claim['evidence_refs']))
                records = self._evidence(claim['evidence_refs'])
                if claim.get('status') != 'abandoned':
                    claim['scope'] = self.bind_scopes(claim.get('scope'), records)
                if claim.get('supersedes') and not claim.get('claim_id'):
                    matches = [key for key, old in prior.items() if old['revision_ref'] == claim['supersedes']]
                    if len(matches) != 1:
                        raise ContractError('旧复盘含多个知识条目，请指定修订哪一条 claim_id')
                    claim['claim_id'] = matches[0]
                claim.setdefault('claim_id', self.claim_identity(claim))
                validate_schema(claim, PAYLOAD_SCHEMAS['MetareviewCard']['properties']['knowledge_updates']['items'], 'knowledge_update')
                self._validate_claim(claim)
                if claim['claim_id'] in seen:
                    raise ContractError('同一复盘不能重复更新同一 claim_id')
                old = prior.get(claim['claim_id'])
                if old:
                    if claim.get('supersedes', old['revision_ref']) != old['revision_ref']:
                        raise ContractError('supersedes 与当前知识版本冲突: ' + old['revision_ref'])
                    claim['supersedes'] = old['revision_ref']
                elif old_draft:
                    if claim.get('supersedes', old_draft['metareview_ref']) != old_draft['metareview_ref']:
                        raise ContractError('supersedes 与当前 pending 版本冲突')
                    claim['supersedes'] = old_draft['metareview_ref']
                elif claim.get('supersedes'):
                    raise ContractError('supersedes 未指向当前可修订知识')
                accepted.append(claim)
                seen.add(claim['claim_id'])
            except (ContractError, KeyError, TypeError, AttributeError) as exc:
                pending.append({'index': index, 'submission': deepcopy(submitted), 'issue': str(exc),
                                'claim_id': claim.get('claim_id') or (submitted.get('claim_id') or self.claim_identity(submitted)
                                    if isinstance(submitted, dict) else 'knowledge-' + fingerprint(submitted)[:20]),
                                'status': 'pending_verification'})
        value['knowledge_updates'] = accepted
        if pending:
            value['pending_knowledge_updates'] = pending
        evidence = set(value['evidence_refs'])
        for field in ('judgment_audit', 'predictive_metrics', 'knowledge_updates'):
            for row in value[field]:
                evidence.update(row.get('evidence_refs', []))
        value['evidence_refs'] = sorted(evidence)
        bindings = {'source_refs': batch['source_refs'], 'recording_batch_id': batch['batch_id'],
                    'analysis_scope': 'batch_synthesis; selective_judgment_analysis; not_exhaustive_audit'}
        for key, item in bindings.items():
            if key in value and value[key] != item:
                raise ContractError('Recorder 不得覆盖程序来源绑定: ' + key)
        return {**value, **deepcopy(bindings)}

    def bind_scopes(self, stated, records):
        if stated is not None and not isinstance(stated, dict):
            raise ContractError('知识 scope 必须是科学条件对象')
        scopes = [record['payload'].get('scope', {}) for record in records]
        if len({canonical(scope) for scope in scopes}) <= 1:
            return bind_claim_scope(stated, scopes)
        # Comparison arms retain separate true conditions; no first-scope or union shortcut.
        arms = []
        comparison = (stated or {}).get('comparison', [])
        if not isinstance(comparison, list) or any(not isinstance(arm, dict) or
                not isinstance(arm.get('evidence_ref'), str) or not isinstance(arm.get('scope'), dict) for arm in comparison):
            raise ContractError('比较 scope 须由逐 Evidence 的条件对象组成')
        requested = {arm['evidence_ref']: arm['scope'] for arm in comparison}
        if requested and set(requested) != {record['id'] for record in records}:
            raise ContractError('比较 scope 的 Evidence 绑定与 evidence_refs 不一致')
        for record in records:
            scope = record['payload'].get('scope', {})
            request = requested.get(record['id'], stated if not requested else None)
            arms.append({'evidence_ref': record['id'], 'scope': bind_claim_scope(request, [scope])})
        return {'comparison': arms}

    def _validate_claim(self, claim):
        if not all(k in claim for k in ('claim_id', 'text', 'status', 'scope', 'evidence_refs', 'rationale')):
            raise ContractError('知识必须具备 ID、状态、适用范围、证据与科研依据')
        if claim['status'] not in KNOWLEDGE_STATUSES or not isinstance(claim['scope'], dict) or not claim['scope']:
            raise ContractError('知识状态/适用范围非法')
        if claim['status'] == 'abandoned':
            return
        evidence = self._evidence(claim['evidence_refs'])
        if claim['status'] != 'hypothesis' and not evidence:
            raise ContractError('有实证含义的知识必须有测量支持')
        comparisons = claim['scope'].get('comparison')
        arm_scopes = {arm['evidence_ref']: arm['scope'] for arm in comparisons} if comparisons else {}
        if comparisons and set(arm_scopes) != set(claim['evidence_refs']):
            raise ContractError('比较条件必须逐一绑定各 Evidence 原件')
        for record in evidence:
            validate_claim_scope(arm_scopes.get(record['id'], claim['scope']), record['payload'].get('scope', {}))
        executions = {e['payload']['execution_id'] for e in evidence}
        if claim['status'] == 'replicated' and len(executions) < 2:
            raise ContractError('缓存复用或同一实验多次引用不算重复验证')
        if claim['status'] == 'observed_once' and len(executions) != 1 and not comparisons:
            raise ContractError('一次观察状态必须对应一次独立执行')
        if claim['status'] == 'replicated' and comparisons:
            raise ContractError('不同条件的对照不等于独立重复验证；可保存为 conditional 比较认识')
        if claim['status'] == 'replicated' and not claim.get('replication_analysis'):
            raise ContractError('重复验证须解释执行独立性、相同条件与差异')

    def _evidence(self, refs, *, require_real=True, require_completed=True):
        records = []
        for ref in refs:
            card = self.journal.card(ref)
            if card["kind"] != "EvidenceCard" or (require_completed and card["payload"]["status"] != "completed"):
                raise ContractError("知识引用必须是完成的 EvidenceCard")
            if require_real and card["payload"]["measurement_source"] == "test_double":
                raise ContractError("模拟结果不能支撑真实科研知识")
            records.append(card)
        return records

    def live_update(self, *, round_number=None, hypothesis_ref=None, evidence_ref=None, selection=None):
        state = self.snapshot()
        for key in ("L0", "memory_version"):
            state.pop(key, None)
        if round_number is not None and round_number != state["L1"]["current_round"]:
            state["L1"].update(current_round=round_number, recent_hypotheses=[], recent_evidence=[], round_selection=None)
        for key, ref in (("recent_hypotheses", hypothesis_ref), ("recent_evidence", evidence_ref)):
            if ref and ref not in state["L1"][key]:
                state["L1"][key].append(ref)
        if selection is not None:
            state["L1"]["round_selection"] = selection
        state["projection_version"] += 1
        return {"type": "memory_projection", "value": state}

    def validate_metareview(self, card):
        if card["kind"] != "MetareviewCard" or card["author"] != "recorder":
            raise ContractError("Meta-review 必须由 Recorder 产生")
        payload = card["payload"]
        # 科学分析可引用失败或模拟回执；只有发布知识才要求真实完成测量。
        self._evidence(payload["evidence_refs"], require_real=False, require_completed=False)
        for claim in payload["knowledge_updates"]:
            self._validate_claim(claim)
            if not set(claim["evidence_refs"]) <= set(payload["evidence_refs"]):
                raise ContractError("知识证据必须包含在 Meta-review 证据列表")
        # 对科研判断要求可审查证据；不从引用数量推断结论真假。
        for row in payload["judgment_audit"]:
            if not all(k in row for k in ("judgment_ref", "assessment", "evidence_refs", "reasoning")):
                raise ContractError("Judger 预测审计缺少原判断或分析")
            original = self.journal.card(row["judgment_ref"])
            if original["kind"] not in ("ReviewDecisionCard", "HypothesisCard"):
                raise ContractError("预测审计指向非科研判断")
            self._evidence(row["evidence_refs"], require_real=False, require_completed=False)
        for row in payload['predictive_metrics']:
            self._evidence(row.get('evidence_refs', []), require_real=False, require_completed=False)

    def apply_metareview(self, card, *, hypothesis_refs=(), evidence_refs=(), selection=None):
        self.validate_metareview(card)
        state = self.snapshot()
        for key in ("L0", "memory_version"):
            state.pop(key, None)
        payload = card["payload"]
        state["projection_version"] += 1
        state['L1']['current_round'] = max(payload['round'], state['L1']['current_round'] or 0)
        if selection is not None:
            state['L1']['round_selection'] = selection
        closed = set(payload.get('resolved_questions', [])) | set(payload.get('abandoned_questions', []))
        state['L1']['open_questions'] = list(dict.fromkeys(q for q in
            [*state['L1']['open_questions'], *payload.get('next_questions', [])] if q not in closed))
        for update in payload["knowledge_updates"]:
            old = state["L2"].get(update["claim_id"])
            if old and update.get("supersedes") != old["revision_ref"]:
                raise ContractError("更新稳定知识须明确 supersedes 原版本，不能覆盖")
            state["L2"][update["claim_id"]] = {**deepcopy(update), "revision_ref": card["id"],
                                               "version": (old["version"] + 1 if old else 1)}
        state["L3"].append(card["id"])
        return {"type": "memory_projection", "value": state}

    def consider_incumbent(self, candidate, protocol_hash):
        """只接受同协议完整双任务；L1/L2 不能晋升完整评测 incumbent。"""
        if not all(k in candidate for k in ("model_ref", "evidence_refs", "protocol_hash")):
            raise ContractError("incumbent 候选缺少身份")
        if candidate["protocol_hash"] != protocol_hash:
            raise ContractError("incumbent 协议不一致")
        evidence = self._evidence(candidate["evidence_refs"])
        scores = {}
        for card in evidence:
            data = card["payload"]
            if data["provenance"].get("protocol_hash") != protocol_hash:
                raise ContractError("完整评测证据协议不一致")
            if data["provenance"].get("model_ref") != candidate["model_ref"]:
                raise ContractError("双任务评测不是同一模型")
            for name, result in data["observations"].get("evaluations", {}).items():
                if result.get("full_dataset") is True:
                    scores[name] = result["metrics"]
        if "classification" not in scores or "segmentation" not in scores:
            raise ContractError("未完成同模型双任务完整评测")
        primary, secondary = scores["classification"]["top1"], scores["segmentation"]["mIoU"]
        state = self.snapshot()
        old = state["L1"]["incumbent"]
        if old and (secondary, primary) <= (old["mIoU"], old["top1"]):
            return None
        for key in ("L0", "memory_version"):
            state.pop(key, None)
        state["projection_version"] += 1
        state["L1"]["incumbent"] = {**candidate, "top1": primary, "mIoU": secondary,
                                    "ade_regression": bool(old and secondary < old["mIoU"]),
                                    "previous": old}
        return {"type": "memory_projection", "value": state}

    def current_research_state(self, sequence=None, visible=None, *, question_offset=0, question_limit=8,
                               excluded_refs=(), compact=False, finding_refs=None):
        """Small role-visible L1 view; source facts and researcher judgments stay distinct."""
        visible = visible or (lambda card: True)
        state = self.snapshot(sequence)
        l1 = state['L1']
        groups = {}
        for key in ('recent_hypotheses', 'recent_experiments', 'recent_evidence'):
            groups[key] = [ref for ref in l1.get(key, []) if visible(self.journal.card(ref))]
        incumbent = deepcopy(l1.get('incumbent'))
        if incumbent:
            incumbent.pop('previous', None)
            if any(not visible(self.journal.card(ref)) for ref in incumbent.get('evidence_refs', [])):
                incumbent = None
        reviews = [self.journal.card(ref) for ref in state['L3'] if visible(self.journal.card(ref))]
        latest = reviews[-1] if reviews else None
        questions, question_sources = [], {}
        for review in reviews:
            payload = review['payload']
            closed = set(payload.get('resolved_questions', [])) | set(payload.get('abandoned_questions', []))
            questions = list(dict.fromkeys(q for q in [*questions, *payload.get('next_questions', [])] if q not in closed))
            for index, question in enumerate(payload.get('next_questions', [])):
                question_sources[question] = {'ref': review['id'], 'field': '/payload/next_questions/' + str(index)}
        if compact:
            question_limit = min(question_limit, 4)
        selected_questions = questions[question_offset:question_offset + question_limit]
        def finding(ref):
            card = self.journal.card(ref)
            if not compact:
                return MemoryRetriever(self.journal).entry(card, include_body=ref not in excluded_refs)
            payload = card['payload']
            result = {key: payload[key] for key in ('status', 'measurement_source')}
            result.update(ref=ref, read_card={'ref': ref})
            if ref in excluded_refs:
                return result
            scope = card_scope(card)
            result['actual_conditions'] = {key: short_field(scope[key], ref, '/payload/scope/' + key, 250)
                for key in ('model', 'measurement_kind', 'site', 'weight_bits', 'activation_bits', 'parameters', 'data', 'quantization')
                if key in scope}
            result['conditions_source'] = {'ref': ref, 'field': '/payload/scope'}
            result['observations'] = project_observations(payload['observations'], ref, max_characters=1200)
            for key in ('limitations', 'warnings', 'error'):
                if key in payload:
                    result[key] = short_field(payload[key], ref, '/payload/' + key, 350)
            result['basis'] = 'Recorded observations, not scientific conclusions; precise source remains available.'
            return result
        result = {'current_round': l1['current_round'],
                  'open_questions': [short_field(q, **question_sources[q], limit=700) for q in selected_questions],
                  'question_sources': [question_sources[q] for q in selected_questions],
                  'open_question_count': len(questions),
                  'more_questions': {'query': {'layer': 'L1'}, 'offset': question_offset + question_limit,
                                     'limit': question_limit} if question_offset + question_limit < len(questions) else None,
                  'incumbent': incumbent, 'progress': {key: len(refs) for key, refs in groups.items()},
                  'latest_findings': [finding(ref) for ref in groups['recent_evidence']
                                      if finding_refs is None or ref in finding_refs][-2:],
                  'history': {'retrieve_memory': {'query': {}, 'limit': 8}},
                  'fact_basis': 'Journal commits; review citations do not replace executed experiments'}
        if latest:
            result['latest_review'] = {'ref': latest['id'], **{key: short_field(latest['payload'][key], latest['id'],
                '/payload/' + key, 700 if compact else 1400) for key in ('scientific_analysis', 'generation_guidance', 'blind_spots')
                if key in latest['payload']}, 'read_card': {'ref': latest['id']}}
        return result


class MemoryRetriever:
    def __init__(self, journal):
        self.journal = journal
        self.memory = ResearchMemory(journal)

    def handle(self, ref):
        version = self.journal.card_versions.get(ref)
        return 'C' + str(version) if version is not None else ref

    def entry(self, card, *, include_body=True):
        """Programmatic short body with precise sources, conditions and recorded limitations."""
        body, ref = card["payload"], card["id"]
        scope = card_scope(card)
        scope_view = deepcopy(scope) if len(canonical(scope)) <= 1600 else {
            "read_card": {"ref": ref, "field": "/payload/scope"}, "sha256": fingerprint(scope),
            "characters": len(canonical(scope)), "complete": False}
        title = next((body[k] for k in ("title", "question", "research_question", "phase", "scientific_analysis")
                      if isinstance(body.get(k), str)), card["kind"])
        if card['kind'] == 'TaskCard' and body['kind'] == 'literature':
            title = body['inputs'].get('query', body['inputs'].get('url', 'literature'))
        return {"ref": ref, "handle": self.handle(ref), "kind": card["kind"], "title": title[:160], "title_truncated": len(title) > 160,
                "version": card["version"], "memory_version": self.journal.card_versions.get(ref, 0),
                "commit_sequence": self.journal.cards[ref]["sequence"], "content_sha256": card["content_sha256"],
                "scope": scope_view,
                **({'body': self.short_body(card)} if include_body else {}),
                'scope_basis': 'measurement' if card['kind'] == 'EvidenceCard' else 'declared_or_recorded',
                "scope_verified": card['kind'] == 'EvidenceCard'
                    and scope.get("schema_version") == "qavit.evidence_scope/1"
                    and scope.get('measurement_kind') != 'unverified_scope' and body.get('measurement_source') != 'test_double',
                "source_refs": card["refs"][:6], 'source_count': len(card['refs']),
                'all_sources': {'ref': ref, 'field': '/refs'}, "read_card": {"ref": ref},
                "full_card_asset": str(self.journal.card_path(ref)),
                **{k: body[k] for k in ("round", "slot", "phase", "experiment_ref", "measurement_source", "status") if k in body}}

    def short_body(self, card):
        body, ref = card['payload'], card['id']
        fields = {
            'HypothesisCard': ('research_question', 'mechanism', 'intervention', 'predictions'),
            'ExperimentCard': ('question', 'prediction', 'measurements', 'controls', 'decision_impact', 'validation'),
            'ReviewDecisionCard': ('rationale', 'outcome', 'decisive_reason', 'uncertainties', 'objections', 'next_questions'),
            'MetareviewCard': ('scientific_analysis', 'generation_guidance', 'blind_spots', 'next_questions'),
            'ResearchBrief': ('scientific_question', 'hard_constraints', 'evaluation_protocol'),
            'EvidenceCard': ('status', 'measurement_source', 'limitations', 'warnings', 'error'),
        }.get(card['kind'], ())
        result = {key: short_field(body[key], ref, '/payload/' + key, 900) for key in fields if key in body}
        if card['kind'] == 'EvidenceCard':
            result['observations'] = project_observations(body['observations'], ref, max_characters=2400)
            result['interpretation'] = 'recorded observations; descriptive statistics are not scientific conclusions'
            spec = self.journal.card(body['experiment_ref'])['payload']
            result['experiment'] = {key: short_field(spec[key], body['experiment_ref'], '/payload/' + key, 700)
                                    for key in ('question', 'controls', 'validation') if key in spec}
        if card['kind'] == 'TaskCard':
            result['literature'] = {key: short_field(body.get('result', {})[key], ref, '/payload/result/' + key, 1200)
                                    for key in ('text_preview', 'title', 'url', 'asset') if key in body.get('result', {})}
        return result

    @staticmethod
    def method_documents(query='', limit=4):
        root = Path(__file__).resolve().parents[1] / 'docs' / 'methods'
        try:
            methods = json.loads((root / 'index.json').read_text())['methods']
            if not isinstance(methods, list) or any(not isinstance(row, dict) or
                    not all(k in row for k in ('id', 'title', 'document')) for row in methods):
                return {'entries': [], 'policy': 'optional method index unavailable; continue research'}
        except (OSError, ValueError, KeyError, TypeError):
            return {'entries': [], 'policy': 'optional method index unavailable; continue research'}
        tokens = lexical_tokens(query)
        ranked = sorted(methods, key=lambda row: len(tokens & lexical_tokens(canonical(row))), reverse=True)
        return {'entries': [{'id': row['id'], 'title': row['title'], 'applicable_problems': row.get('tags', []),
                              'targets': row.get('targets', []), 'read_asset': {'asset': str(root / row['document'])}}
                             for row in ranked[:limit]],
                'index': {'read_asset': {'asset': str(root / 'index.json')}},
                'policy': 'direct documentation reading; no experiment request or measurement required'}

    def retrieve(self, role, *, explicit_refs=(), query=None, after_version=None, limit=12,
                 offset=0, sequence=None, visible=None, include_body=True):
        """Permissions and exact conditions first; short usable bodies need no follow-up read."""
        if query is not None:
            validate_schema(query, MEMORY_QUERY_SCHEMA, "retrieve_memory.query")
        query = {} if query is None else query
        if after_version is not None and (type(after_version) is not int or after_version < 0):
            raise ContractError("retrieve_memory.after_version 必须为非负整数")
        if type(limit) is not int or not 1 <= limit <= 50 or type(offset) is not int or offset < 0:
            raise ContractError("retrieve_memory 分页需 offset>=0、1<=limit<=50")
        sequence = self.journal.sequence if sequence is None else sequence
        visible = visible or (lambda card: True)
        state = self.memory.snapshot(sequence)
        explicit = set(explicit_refs) | set(query.get("evidence_ids", []))
        tokens = lexical_tokens(query.get("text", ""))
        filters = {k: v for k, v in query.items() if k not in ("kind", "browse", "layer")}
        rows = []
        for ref, meta in self.journal.cards.items():
            if query.get('layer') in ('L1', 'L2'):
                break  # Knowledge retrieval does not need a scan of every scientific/engineering card.
            if meta["sequence"] > sequence or meta["kind"] not in ROLE_KINDS.get(role, set()) | {"ResearchBrief", "TaskCard"}:
                continue
            if meta['kind'] == 'TaskCard' and (meta.get('task_kind') != 'literature' or meta.get('state') != 'completed'):
                continue
            if query.get('kind') and meta['kind'] != query['kind']:
                continue
            if explicit and ref not in explicit:
                continue
            card = self.journal.card(ref)
            if card['kind'] == 'TaskCard' and (card['payload']['kind'] != 'literature' or card['payload']['state'] != 'completed'):
                continue
            if not visible(card):
                continue
            version = self.journal.card_versions.get(ref, 0)
            if after_version is not None and version <= after_version:
                continue
            if query.get("kind") and card["kind"] != query["kind"]:
                continue
            if explicit and ref not in explicit:
                continue
            if not scope_matches(card_scope(card), filters):
                continue
            score = len(tokens & lexical_tokens(str(card["payload"]))) / len(tokens) if tokens else 0
            rows.append((score, version, card))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        total = len(rows)
        selected = [self.entry(row[2], include_body=include_body) for row in rows[offset:offset + limit]]
        if query.get('layer') == 'L2':
            ranked_claims = []
            for claim in state['L2'].values():
                card = self.journal.card(claim['revision_ref'])
                if (card['kind'] not in ROLE_KINDS.get(role, set()) or not visible(card)
                        or (query.get('kind') and query['kind'] != card['kind'])
                        or not any(scope_matches(scope, filters) for scope in
                            [arm['scope'] for arm in claim['scope'].get('comparison', [])] or [claim['scope']])):
                    continue
                if claim['status'] == 'abandoned' and not explicit:
                    continue
                if after_version is not None and self.journal.card_versions[card['id']] <= after_version:
                    continue
                if explicit and card['id'] not in explicit:
                    continue
                index = next(i for i, row in enumerate(card['payload']['knowledge_updates']) if row['claim_id'] == claim['claim_id'])
                entry = {'ref': card['id'], 'kind': 'Knowledge', 'claim_id': claim['claim_id'],
                    'title': claim['text'][:160], 'status': claim['status'], 'version': claim['version'],
                    **({'body': {key: short_field(claim[key], card['id'], '/payload/knowledge_updates/' + str(index) + '/' + key, 1200)
                             for key in ('text', 'rationale', 'replication_analysis') if key in claim}} if include_body else {}),
                    'review_sequence': self.journal.cards[card['id']]['sequence'],
                    'check_new_evidence': {'query': {'kind': 'EvidenceCard'},
                        'after_version': self.journal.card_versions[card['id']]},
                    'freshness': 'recorded judgment at review_sequence; later evidence may require reassessment, not automatic invalidation',
                    'scope_sha256': fingerprint(claim['scope']), 'source_refs': claim['evidence_refs'],
                    'actual_scope': deepcopy(claim['scope']) if len(canonical(claim['scope'])) <= 1600 else {
                        'read_card': {'ref': card['id'], 'field': '/payload/knowledge_updates/' + str(index) + '/scope'}},
                    'read_card': {'ref': card['id'], 'field': '/payload/knowledge_updates/' + str(index)}}
                if claim['scope'].get('comparison'):
                    entry['comparison'] = [{'evidence_ref': arm['evidence_ref'],
                        'matches_requested_conditions': scope_matches(arm['scope'], filters),
                        'actual_scope': short_field(arm['scope'], arm['evidence_ref'], '/payload/scope', 1200),
                        'read_card': {'ref': arm['evidence_ref']}} for arm in claim['scope']['comparison']]
                    entry['comparison_limit'] = 'Different conditions are separate comparison arms, not independent replications; do not generalize to their union.'
                    if claim['status'] == 'observed_once':
                        entry['observation_unit'] = 'one comparison; each arm retains its own execution, not repeated validation'
                    scopes = [flattened(arm['scope']) for arm in claim['scope']['comparison']]
                    entry['different_conditions'] = [key for key in sorted(set().union(*(set(scope) for scope in scopes)))
                        if len({canonical(scope.get(key)) for scope in scopes}) > 1]
                score = len(tokens & lexical_tokens(canonical(claim))) / max(1, len(tokens))
                ranked_claims.append((score, self.journal.card_versions[card['id']], claim['claim_id'], entry))
            ranked_claims.sort(key=lambda row: row[:3], reverse=True)
            total = len(ranked_claims)
            selected = [row[3] for row in ranked_claims[offset:offset + limit]]
        latest = next((ref for ref in reversed(state["L3"]) if visible(self.journal.card(ref))), None)
        result = {"memory_version": state["memory_version"], "read_sequence": sequence,
                "mode": "exact_metadata_then_lexical; " + ('short_body_and_sources' if include_body else 'internal_directory') + "; vector_semantics_disabled",
                "latest_metareview_ref": latest, "cards": [], "entries": selected,
                "offset": offset, "total": total,
                "next_offset": offset + limit if offset + limit < total else None,
                "read_manifest": {"refs": [], "policy": "已提供程序组织的短正文；完整原件 read_card/read_asset 按需读取，不要求阅读确认。"}}
        if query.get('layer') == 'L1':
            result['current_research_state'] = self.memory.current_research_state(sequence, visible,
                question_offset=offset, question_limit=limit)
        if query.get('layer') == 'L2':
            pending = self.memory.recording_status(sequence, visible=visible, pending_offset=offset, pending_limit=limit)
            result.update({key: pending[key] for key in ('pending_knowledge_updates', 'pending_knowledge_count', 'pending_knowledge_more')})
        if (include_body and role in {'generation', 'evolution', 'judger', 'recorder'}
                and query.get('layer') not in ('L1', 'L2') and not query.get('kind') and not explicit):
            result['method_documents'] = self.method_documents(query.get('text', ''))
        return result

    def relevant_delta(self, version, *, refs=(), scope=None):
        related = set(refs)
        result = []
        for ref, changed in self.journal.card_versions.items():
            if changed <= version:
                continue
            card = self.journal.card(ref)
            if card["kind"] not in ("EvidenceCard", "MetareviewCard", "ResearchBrief"):
                continue
            measured_scope = card_scope(card)
            if related.intersection(card["refs"]) or (scope and scope_matches(measured_scope, scope)):
                result.append(ref)
        return result
