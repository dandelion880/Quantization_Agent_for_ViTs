"""四层科研 Memory 与程序化检索；Recorder 提议科研判断，程序核验出处。

L0 即 Journal，不允许 Recorder 改写。L1/L2/L3 是有版本的投影；迟到证据
按其模型、配置及观察时点归档，不用写入顺序覆盖当前事实。
"""
from copy import deepcopy
import re

from .cards import ContractError, canonical, fingerprint, MEMORY_QUERY_SCHEMA, validate_schema
from .evidence_scope import matches as scope_matches, validate_claim_scope, card_scope


KNOWLEDGE_STATUSES = {"hypothesis", "observed_once", "replicated", "refuted", "conditional"}
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


class ResearchMemory:
    def __init__(self, journal):
        self.journal = journal

    def snapshot(self, sequence=None):
        current = {"L1": {"current_round": None, "round_selection": None, "incumbent": None,
                          "open_questions": [], "recent_hypotheses": [], "recent_evidence": []},
                   "L2": {}, "L3": [], "projection_version": 0}
        for event in self.journal.events:
            if event["type"] == "memory_projection" and (sequence is None or event["commit_sequence"] <= sequence):
                current = deepcopy(event["value"])
        visible = [ref for ref, entry in self.journal.cards.items() if sequence is None or entry["sequence"] <= sequence]
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
        engineering = []
        for task in latest_tasks.values():
            relevant = (task["kind"] in {"experiment", "verification", "experiment_request", "literature"}
                        or task["inputs"].get("purpose") in {"code", "reference", "safeguard", "plan_l3"})
            if relevant and task["state"] in {"completed", "failed", "cancelled", "ineligible"} and task["task_card"] not in covered:
                engineering.append(task["task_card"])
        refs = sorted(science + engineering)
        judgments = [r for r in science if self.journal.cards[r]["kind"] == "HypothesisCard" or
                     self.journal.cards[r]["kind"] == "ReviewDecisionCard"]
        return {"batch_id": fingerprint(refs), "through_sequence": sequence, "source_refs": refs,
                "scientific_refs": sorted(science), "engineering_refs": sorted(engineering),
                "judgment_refs": sorted(judgments)}

    def recording_status(self, sequence=None):
        """归档是程序事实，复盘是 Recorder 产出；不把前者冒充后者。"""
        batch = self.recording_batch(sequence)
        deferred = [e for e in self.journal.events if e['type'] == 'recorder_deferred'
                    and (sequence is None or e['commit_sequence'] <= sequence)
                    and set(e['source_refs']) & set(batch['source_refs'])]
        return {'facts': 'archived_in_journal', 'scientific_review': 'pending' if batch['source_refs'] else 'completed',
                'pending_source_count': len(batch['source_refs']),
                'deferred_task_refs': [e['task_ref'] for e in deferred],
                'meaning': 'completed 表示批次科学综合已发布；不表示逐条事实经过科学认可。'}

    def prepare_metareview(self, content, batch, round_number):
        """编译精简科研提交；只补身份、来源与精确范围，不代写科研结论。"""
        value = deepcopy(content)
        if 'round' in value and value['round'] != round_number:
            raise ContractError('Meta-review 轮次错误')
        value['round'] = round_number
        # 历史封套仍可读取；新产出不再宣称 reviewed_refs 全量逐项审查。
        if 'reviewed_refs' in value:
            raise ContractError('reviewed_refs 已由程序归档取代；请省略该字段，只提交科学分析')
        for field in ('judgment_audit', 'predictive_metrics', 'knowledge_updates', 'recurring_failures',
                      'blind_spots', 'evidence_refs'):
            value.setdefault(field, [])
        evidence = set(value['evidence_refs'])
        for field in ('judgment_audit', 'predictive_metrics', 'knowledge_updates'):
            for row in value[field]:
                evidence.update(row.get('evidence_refs', []))
        value['evidence_refs'] = sorted(evidence)
        for claim in value['knowledge_updates']:
            if claim.get('supersedes') and not claim.get('claim_id'):
                raise ContractError('修订知识须给已有 claim_id；新知识 ID 才由程序生成')
            # 重复相同陈述/范围会命中同一稳定 ID；修订仍须显式指向旧知识。
            if 'scope' not in claim:
                records = self._evidence(claim.get('evidence_refs', []))
                if not records:
                    raise ContractError('省略知识 scope 需要明确 Evidence；纯假说须显式给出适用范围')
                scopes = [r['payload'].get('scope', {}) for r in records]
                if not scopes[0] or any(s != scopes[0] for s in scopes):
                    raise ContractError('Evidence 范围不同或缺失，不能自动合并 scope；不同条件的比较保留在分析中')
                claim['scope'] = deepcopy(scopes[0])
            claim.setdefault('claim_id', 'knowledge-' + fingerprint({'text': claim.get('text'), 'scope': claim['scope']})[:20])
        ids = [c['claim_id'] for c in value['knowledge_updates']]
        if len(ids) != len(set(ids)):
            raise ContractError('同一复盘不能重复更新同一 claim_id')
        bindings = {'source_refs': batch['source_refs'], 'recording_batch_id': batch['batch_id'],
                    'analysis_scope': 'batch_synthesis; selective_judgment_analysis; not_exhaustive_audit'}
        for key, item in bindings.items():
            if key in value and value[key] != item:
                raise ContractError('Recorder 不得覆盖程序来源绑定: ' + key)
        return {**value, **deepcopy(bindings)}

    def _evidence(self, refs, *, require_real=True):
        records = []
        for ref in refs:
            card = self.journal.card(ref)
            if card["kind"] != "EvidenceCard" or card["payload"]["status"] != "completed":
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
        self._evidence(payload["evidence_refs"])
        for claim in payload["knowledge_updates"]:
            if not all(k in claim for k in ("claim_id", "text", "status", "scope", "evidence_refs", "rationale")):
                raise ContractError("知识必须具备 ID、状态、适用范围、证据与科研依据")
            if claim["status"] not in KNOWLEDGE_STATUSES or not isinstance(claim["scope"], dict) or not claim["scope"]:
                raise ContractError("知识状态/适用范围非法")
            evidence = self._evidence(claim["evidence_refs"])
            if claim["status"] != "hypothesis" and not evidence:
                raise ContractError("有实证含义的知识必须有测量支持")
            for record in evidence:
                measured_scope = record["payload"].get("scope", {})
                validate_claim_scope(claim["scope"], measured_scope)
            executions = {e["payload"]["execution_id"] for e in evidence}
            if claim["status"] == "replicated" and len(executions) < 2:
                raise ContractError("缓存复用或同一实验多次引用不算重复验证")
            if claim["status"] == "observed_once" and len(executions) != 1:
                raise ContractError("一次观察状态必须对应一次独立执行")
            if claim["status"] == "replicated" and not claim.get("replication_analysis"):
                raise ContractError("重复验证须解释执行独立性、相同条件与差异")
            if not set(claim["evidence_refs"]) <= set(payload["evidence_refs"]):
                raise ContractError("知识证据必须包含在 Meta-review 证据列表")
        # 对科研判断要求可审查证据；不从引用数量推断结论真假。
        for row in payload["judgment_audit"]:
            if not all(k in row for k in ("judgment_ref", "assessment", "evidence_refs", "reasoning")):
                raise ContractError("Judger 预测审计缺少原判断或分析")
            original = self.journal.card(row["judgment_ref"])
            if original["kind"] not in ("ReviewDecisionCard", "HypothesisCard"):
                raise ContractError("预测审计指向非科研判断")
            self._evidence(row["evidence_refs"])
        for row in payload['predictive_metrics']:
            self._evidence(row.get('evidence_refs', []))

    def apply_metareview(self, card, *, hypothesis_refs=(), evidence_refs=(), selection=None):
        self.validate_metareview(card)
        state = self.snapshot()
        for key in ("L0", "memory_version"):
            state.pop(key, None)
        payload = card["payload"]
        state["projection_version"] += 1
        state["L1"].update(current_round=payload["round"], round_selection=selection,
                           recent_hypotheses=list(hypothesis_refs), recent_evidence=list(evidence_refs),
                           open_questions=list(payload["next_questions"]))
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
        if old and (primary, secondary) <= (old["top1"], old["mIoU"]):
            return None
        for key in ("L0", "memory_version"):
            state.pop(key, None)
        state["projection_version"] += 1
        state["L1"]["incumbent"] = {**candidate, "top1": primary, "mIoU": secondary,
                                    "ade_regression": bool(old and secondary < old["mIoU"]),
                                    "previous": old}
        return {"type": "memory_projection", "value": state}


class MemoryRetriever:
    def __init__(self, journal):
        self.journal = journal
        self.memory = ResearchMemory(journal)

    def handle(self, ref):
        version = self.journal.card_versions.get(ref)
        return 'C' + str(version) if version is not None else ref

    def entry(self, card):
        """索引是可重建导航，不把未核验标签升级为证据或省略范围的科学摘要。"""
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
                'scope_basis': 'measurement' if card['kind'] == 'EvidenceCard' else 'declared_or_recorded',
                "scope_verified": card['kind'] == 'EvidenceCard'
                    and scope.get("schema_version") == "qavit.evidence_scope/1"
                    and scope.get('measurement_kind') != 'unverified_scope' and body.get('measurement_source') != 'test_double',
                "source_refs": card["refs"][:6], 'source_count': len(card['refs']),
                'all_sources': {'ref': ref, 'field': '/refs'}, "read_card": {"ref": ref},
                "full_card_asset": str(self.journal.card_path(ref)),
                **{k: body[k] for k in ("round", "slot", "phase", "experiment_ref", "measurement_source", "status") if k in body}}

    def retrieve(self, role, *, explicit_refs=(), query=None, after_version=None, limit=12,
                 offset=0, sequence=None, visible=None):
        """先权限/提交快照，再精确范围过滤和原有词法排序；只返回可分页目录。"""
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
            if meta["sequence"] > sequence or meta["kind"] not in ROLE_KINDS.get(role, set()) | {"ResearchBrief", "TaskCard"}:
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
            score = len(tokens & lexical_tokens(str(card["payload"]))) / max(1, len(tokens))
            rows.append((score, version, card))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        entries = [self.entry(row[2]) for row in rows]
        if query.get('layer') == 'L2':
            ranked_claims = []
            for claim in state['L2'].values():
                card = self.journal.card(claim['revision_ref'])
                if (card['kind'] not in ROLE_KINDS.get(role, set()) or not visible(card)
                        or (query.get('kind') and query['kind'] != card['kind'])
                        or not scope_matches(claim['scope'], filters)):
                    continue
                if after_version is not None and self.journal.card_versions[card['id']] <= after_version:
                    continue
                if explicit and card['id'] not in explicit:
                    continue
                index = next(i for i, row in enumerate(card['payload']['knowledge_updates']) if row['claim_id'] == claim['claim_id'])
                entry = {'ref': card['id'], 'kind': 'Knowledge', 'claim_id': claim['claim_id'],
                    'title': claim['text'][:160], 'status': claim['status'], 'version': claim['version'],
                    'scope_sha256': fingerprint(claim['scope']), 'source_refs': claim['evidence_refs'],
                    'actual_scope': deepcopy(claim['scope']) if len(canonical(claim['scope'])) <= 1600 else {
                        'read_card': {'ref': card['id'], 'field': '/payload/knowledge_updates/' + str(index) + '/scope'}},
                    'read_card': {'ref': card['id'], 'field': '/payload/knowledge_updates/' + str(index)}}
                score = len(tokens & lexical_tokens(canonical(claim))) / max(1, len(tokens))
                ranked_claims.append((score, self.journal.card_versions[card['id']], claim['claim_id'], entry))
            ranked_claims.sort(key=lambda row: row[:3], reverse=True)
            entries = [row[3] for row in ranked_claims]
        selected = entries[offset:offset + limit]
        latest = next((ref for ref in reversed(state["L3"]) if visible(self.journal.card(ref))), None)
        return {"memory_version": state["memory_version"], "read_sequence": sequence,
                "mode": "exact_metadata_then_lexical; index_only; vector_semantics_disabled",
                "latest_metareview_ref": latest, "cards": [], "entries": selected,
                "offset": offset, "total": len(entries),
                "next_offset": offset + limit if offset + limit < len(entries) else None,
                "read_manifest": {"refs": [], "policy": "目录不等于正文；read_card/read_asset 按需读取，可同批多调用。"}}

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
