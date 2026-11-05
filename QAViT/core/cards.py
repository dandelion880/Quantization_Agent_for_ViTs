"""QAViT v2 科研卡片：完整正文、不可变版本、显式来源与引用。

这里只验证通信合同；位宽、粒度、训练方式等科学选择来自 ResearchBrief，
绝不从对照实验配置推导通用默认值。数学正确性由实验和 Safeguard 判断。
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid

from jsonschema import Draft202012Validator


KINDS = ("ResearchBrief", "HypothesisCard", "ReviewDecisionCard", "ExperimentCard",
         "EvidenceCard", "MetareviewCard", "TaskCard")
ROLES = ("supervisor", "generation", "judger", "evolution", "recorder", "coding", "safeguard")
TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
STRINGS = {"type": "array", "items": TEXT}
OBJECT = {"type": "object", "additionalProperties": True}


class ContractError(ValueError):
    """不可自动修正科学定义的合同错误。"""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_loads(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ContractError("JSON 存在重复字段: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ContractError("JSON 不允许非有限数: " + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)  # 同时拒绝 1e999 解析产生的 inf。
    return value


def object_schema(properties, required):
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": True}


PAYLOAD_SCHEMAS = {
    "ResearchBrief": object_schema({
        "scientific_question": TEXT, "goals": OBJECT, "model": OBJECT,
        "hard_constraints": OBJECT, "search_space": OBJECT, "evaluation_protocol": OBJECT,
    }, ["scientific_question", "goals", "model", "hard_constraints", "search_space", "evaluation_protocol"]),
    "HypothesisCard": object_schema({
        "round": {"type": "integer", "minimum": 1},
        "slot": {"enum": ["H1", "H2", "H3", "H4"]},
        "hypothesis_type": {"enum": ["existing_method", "engineering", "new_algorithm", "evolved"]},
        "title": TEXT, "research_question": TEXT, "mechanism": TEXT,
        "intervention": {"type": ["object", "string"]},
        "predictions": {"type": "array", "items": object_schema({
            "id": TEXT, "statement": TEXT, "measurement": TEXT, "falsification": TEXT,
        }, ["id", "statement"])},
        "implementation": {"type": ["object", "string"]},
        "validation_design": {"type": ["object", "string"]},
        "sources": {"type": "array", "items": OBJECT},
        "mechanism_relations": {"type": "array", "items": object_schema({
            "relation": TEXT, "target": TEXT, "explanation": TEXT,
        }, ["relation", "target", "explanation"])},
        "feedback_used": {"type": "array", "items": OBJECT}, "scope": OBJECT,
        "evidence_refs": STRINGS,
    }, ["round", "slot", "hypothesis_type", "title", "research_question", "mechanism",
        "intervention", "predictions", "sources", "mechanism_relations", "feedback_used", "scope"]),
    "ReviewDecisionCard": object_schema({
        "phase": {"enum": ["screening", "position", "verdict", "ranking", "interpretation",
                            "experiment_approval", "safeguard", "reference_design", "revision"]},
        "rationale": TEXT, "evidence_refs": STRINGS, "uncertainties": STRINGS,
    }, ["phase", "rationale", "evidence_refs", "uncertainties"]),
    "ExperimentCard": object_schema({
        "stage": {"enum": ["request", "execution"]},
        "request_ref": TEXT, "request_sha256": TEXT,
        "level": {"enum": ["L1", "L2", "L3"]}, "question": TEXT,
        "prediction": TEXT, "measurements": {"type": "array", "minItems": 1},
        "controls": OBJECT, "decision_impact": TEXT, "execution": OBJECT,
        "resources": object_schema({"estimated_seconds": {"type": "number", "exclusiveMinimum": 0},
                                    "cpu_threads": {"type": "integer", "minimum": 1}},
                                   ["estimated_seconds", "cpu_threads"]),
        "replication": object_schema({"requested": {"type": "boolean"}, "purpose": {"type": "string"},
                                      "configuration_difference": OBJECT},
                                     ["requested", "purpose", "configuration_difference"]),
        "scope": OBJECT, "validation": OBJECT,
    }, ["level", "question", "measurements", "controls",
        "execution", "resources", "replication", "scope"]),
    "EvidenceCard": object_schema({
        "experiment_ref": TEXT, "execution_fingerprint": TEXT,
        "status": {"enum": ["completed", "failed", "interrupted"]},
        "observations": OBJECT, "provenance": OBJECT, "resources": OBJECT,
        "artifacts": {"type": "array", "items": OBJECT},
        "measurement_source": {"enum": ["real", "synthetic_computation", "test_double"]},
        "execution_id": TEXT,
    }, ["experiment_ref", "execution_fingerprint", "status", "observations", "provenance",
        "resources", "artifacts", "measurement_source", "execution_id"]),
    "MetareviewCard": object_schema({
        "round": {"type": "integer", "minimum": 1}, "scientific_analysis": TEXT,
        "judgment_audit": {"type": "array", "items": OBJECT},
        "predictive_metrics": {"type": "array", "items": OBJECT},
        "knowledge_updates": {"type": "array", "items": OBJECT},
        "pending_knowledge_updates": {"type": "array", "items": object_schema({
            "index": {"type": "integer", "minimum": 0}, "submission": {}, "issue": TEXT,
            "status": {"const": "pending_verification"}, "claim_id": TEXT,
        }, ["index", "submission", "issue", "status"])},
        "recurring_failures": STRINGS, "blind_spots": STRINGS,
        "next_questions": STRINGS, "resolved_questions": STRINGS, "abandoned_questions": STRINGS,
        "generation_guidance": {"type": "string"},
        "evidence_refs": STRINGS, "reviewed_refs": {**STRINGS, "uniqueItems": True},
    }, ["round", "scientific_analysis", "judgment_audit", "predictive_metrics", "knowledge_updates",
        "recurring_failures", "blind_spots", "generation_guidance", "evidence_refs"]),
    "TaskCard": object_schema({
        "task_id": TEXT, "kind": TEXT, "role": {"type": ["string", "null"]},
        "inputs": OBJECT, "depends_on": STRINGS, "state": TEXT,
        "input_versions": OBJECT, "memory_version": {"type": "integer", "minimum": 0},
        "resource_keys": STRINGS, "attempt": {"type": "integer", "minimum": 0},
        "outputs": STRINGS,
    }, ["task_id", "kind", "role", "inputs", "depends_on", "state", "input_versions",
        "memory_version", "resource_keys", "attempt", "outputs"]),
}

PAYLOAD_SCHEMAS["ReviewDecisionCard"]["allOf"] = [{
    "if": {"properties": {"phase": {"const": "screening"}}, "required": ["phase"]},
    "then": object_schema({"eligible": {"type": "boolean"}, "hard_rejection_reasons": STRINGS,
                           "similarity_analysis": TEXT, "evidence_gaps": STRINGS},
                          ["eligible", "hard_rejection_reasons"])
}]

PAYLOAD_SCHEMAS["ResearchBrief"]["properties"].update(quantization_contract=OBJECT, fixed_quantization=OBJECT)
PAYLOAD_SCHEMAS["ExperimentCard"]["properties"]["execution"] = object_schema({
    "service": TEXT, "seed": {"type": "integer", "minimum": 0}, "requires_code": {"type": "boolean"},
    "resolved_choices": OBJECT, "parameters": OBJECT, "quantization_contract": OBJECT,
    "evaluations": {"type": "array", "items": OBJECT}, "reload_before_evaluation": {"type": "boolean"},
}, ["service", "resolved_choices"])
PAYLOAD_SCHEMAS["MetareviewCard"]["properties"]["knowledge_updates"]["items"] = object_schema({
    "claim_id": TEXT, "text": TEXT, "status": {"enum": ["hypothesis", "observed_once", "replicated", "refuted", "conditional", "abandoned"]},
    "scope": {**OBJECT, "minProperties": 1}, "evidence_refs": STRINGS, "rationale": TEXT,
    "replication_analysis": TEXT, "supersedes": TEXT,
}, ["claim_id", "text", "status", "scope", "evidence_refs", "rationale"])
PAYLOAD_SCHEMAS["MetareviewCard"]["properties"]["judgment_audit"]["items"] = object_schema({
    "judgment_ref": TEXT, "assessment": TEXT, "evidence_refs": STRINGS, "reasoning": TEXT,
}, ["judgment_ref", "assessment", "evidence_refs", "reasoning"])

OPTIONAL_ANALYSIS_FIELDS = {
    'HypothesisCard': ('predictions', 'sources', 'mechanism_relations', 'feedback_used', 'scope',
                       'implementation', 'validation_design'),
    'ReviewDecisionCard': ('uncertainties', 'similarity_analysis', 'evidence_gaps',
                           'arguments', 'objections', 'responses_to_prior', 'key_predictions', 'changed_judgments', 'next_questions', 'continuation_assessment',
                           'decisive_reason', 'selection_basis'),
    'MetareviewCard': ('judgment_audit', 'predictive_metrics', 'recurring_failures', 'blind_spots',
                       'next_questions', 'resolved_questions', 'abandoned_questions', 'generation_guidance'),
}

MEMORY_QUERY_SCHEMA = object_schema({"text": {"type": "string"}, "evidence_ids": {**STRINGS, "minItems": 1},
    "browse": {"const": True}, "kind": {"enum": list(KINDS)}, "layer": {"enum": ["L0", "L1", "L2"]}}, [])
MEMORY_QUERY_SCHEMA["additionalProperties"] = {"anyOf": [TEXT, {"type": ["number", "boolean", "null"]},
    {"type": "array", "minItems": 1, "items": {"anyOf": [TEXT, {"type": ["number", "boolean", "null"]},
                                                                  {"type": "object", "minProperties": 1}]}},
    {"type": "object", "minProperties": 1}]}


def validate_schema(value, schema, path):
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error:
        raise ContractError(path + "." + ".".join(map(str, error.path)) + ": " + error.message)
    return value

ENVELOPE = object_schema({
    "schema_version": {"const": "qavit.cards/2"}, "kind": {"enum": list(KINDS)},
    "id": TEXT, "logical_id": TEXT, "version": {"type": "integer", "minimum": 1},
    "run_id": TEXT, "created_at": TEXT, "author": TEXT,
    "parents": STRINGS, "refs": STRINGS, "payload": OBJECT, "content_sha256": TEXT,
}, ["schema_version", "kind", "id", "logical_id", "version", "run_id", "created_at",
    "author", "parents", "refs", "payload", "content_sha256"])


def validate_payload(kind, payload):
    if kind not in PAYLOAD_SCHEMAS:
        raise ContractError("未知卡片类型 " + str(kind))
    schema = PAYLOAD_SCHEMAS[kind]
    if kind == "ExperimentCard" and payload.get("stage") == "request":
        # 原始需求独立持久化；执行字段只有 Coding 交付后才绑定。
        schema = deepcopy(schema)
        schema["required"] = ["question", "measurements"]
        schema["properties"]["execution"]["required"] = []
        schema["properties"]["resources"]["required"] = []
    errors = sorted(Draft202012Validator(schema).iter_errors(payload),
                    key=lambda x: str(list(x.path)))
    if errors:
        err = errors[0]
        raise ContractError(f"{kind}.{'.'.join(map(str, err.path))}: {err.message}")
    canonical(payload)
    if kind == "HypothesisCard":
        expected = dict(H1="existing_method", H2="engineering", H3="new_algorithm", H4="evolved")
        if payload["hypothesis_type"] != expected[payload["slot"]]:
            raise ContractError("假设 slot 与探索类型不一致")
        ids = [p["id"] for p in payload["predictions"]]
        if len(ids) != len(set(ids)):
            raise ContractError("预测 ID 重复")
    if kind == "ExperimentCard" and payload.get("stage") != "request":
        if payload["replication"]["requested"] and not payload["replication"]["purpose"].strip():
            raise ContractError("重复验证必须说明科学目的")
        if not payload["execution"].get("service"):
            raise ContractError("实验必须指定已注册或待开发的 service")
    if kind == "ReviewDecisionCard" and payload["phase"] == "screening":
        if payload["eligible"] == bool(payload["hard_rejection_reasons"]):
            raise ContractError("screen.eligible 与 hard_rejection_reasons 不一致；拒绝必须有非空理由")
    if kind == "TaskCard":
        if payload["role"] is not None and payload["role"] not in ROLES:
            raise ContractError("TaskCard 引用了未知角色")
        if payload["state"] not in ("pending", "ready", "running", "retry", "waiting", "needs_clarification", "completed", "failed", "cancelled", "ineligible"):
            raise ContractError("TaskCard 状态非法")
    return payload


def make_card(kind, run_id, payload, *, author="program", parents=(), refs=(), previous=None):
    validate_payload(kind, payload)
    logical = previous["logical_id"] if previous else kind + "-" + uuid.uuid4().hex
    version = previous["version"] + 1 if previous else 1
    card = {"schema_version": "qavit.cards/2", "kind": kind,
            "id": logical + f"@{version}", "logical_id": logical, "version": version,
            "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "author": author, "parents": list(dict.fromkeys(parents)),
            "refs": list(dict.fromkeys([*refs, *([previous["id"]] if previous else [])])),
            "payload": deepcopy(payload)}
    if previous:
        card["previous"] = previous["id"]
    card["content_sha256"] = fingerprint(card)
    validate_card(card)
    return card


def validate_card(card):
    error = next(Draft202012Validator(ENVELOPE).iter_errors(card), None)
    if error:
        raise ContractError("卡片封套无效: " + error.message)
    if card["content_sha256"] != fingerprint({k: v for k, v in card.items() if k != "content_sha256"}):
        raise ContractError("卡片内容哈希不一致")
    if card["id"] != card["logical_id"] + f"@{card['version']}":
        raise ContractError("卡片 ID/版本不一致")
    if set(card["parents"]) - set(card["refs"]):
        raise ContractError("父假设必须同时出现在 refs 中")
    if card["kind"] == "EvidenceCard" and card["author"] != "program":
        raise ContractError("原始 EvidenceCard 只能由宿主程序生成")
    if card["kind"] == "HypothesisCard" and card["payload"]["slot"] == "H4" and not card["parents"]:
        raise ContractError("H4 必须保留父假设引用")
    validate_payload(card["kind"], card["payload"])
    return card


def check_constraints(brief, resolved):
    """仅执行用户明确的约束；未指定不等于禁止训练或固定4bit。

    nested 字典为逐字段约束；列表表示允许集合；{min,max}表示数值范围。
    其它复杂约束保留原文，交给 Judger/Safeguard，不把自然语言猜成数值限制。
    """
    constraints = brief.get("hard_constraints", {})
    failures = []
    def compare(rule, value, path):
        if isinstance(rule, dict) and set(rule) <= {"min", "max"} and rule:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                failures.append(path + " 缺少数值选择")
            elif ("min" in rule and value < rule["min"]) or ("max" in rule and value > rule["max"]):
                failures.append(path + " 超出用户范围")
        elif isinstance(rule, dict):
            for key, child in rule.items():
                compare(child, value.get(key) if isinstance(value, dict) else None, path + "." + key)
        elif isinstance(rule, list):
            if value not in rule:
                failures.append(path + " 不在用户允许集合")
        elif value != rule:
            failures.append(path + " 不符合用户硬约束")
    for key, rule in constraints.items():
        if key != "notes":
            compare(rule, resolved.get(key), key)
    if failures:
        raise ContractError("; ".join(failures))
