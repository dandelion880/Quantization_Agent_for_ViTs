"""实测范围的唯一表示，供证据封装、知识边界和精确检索共同使用。"""
from copy import deepcopy
from core.cards import ContractError, canonical, fingerprint


VERSION = "qavit.evidence_scope/1"
BOUNDARIES = ("model", "model_identity_sha256", "measurement_kind", "measurement_source",
              "site", "weight_bits", "activation_bits", "quantization", "parameters", "data", "implementation")


def canonical_scope(experiment, data):
    spec = experiment["payload"]["execution"]
    provenance = data.get("provenance", {})
    facts = provenance.get("execution_facts", {})
    # 自定义回执不能用申请范围或自填 observations 代替宿主核验事实。
    result = {"schema_version": VERSION, "service": spec["service"],
              "measurement_source": data["measurement_source"],
              "model": spec.get("model_scope", {}).get("model") or "adapter:" + fingerprint(spec["backend"]),
              "model_identity_sha256": fingerprint(provenance.get("model_identity", {})),
              "measurement_kind": facts.get("measurement_kind", "unverified_scope"),
              "data": deepcopy(facts.get("data", {}))}
    implementation = {}
    if spec.get("measurement_implementation"):
        implementation["measurement_sha256"] = fingerprint(spec["measurement_implementation"])
    artifact = provenance.get("code_artifact")
    if artifact:
        implementation["code"] = {key: artifact[key] for key in ("source_sha256", "interface")}
    elif spec.get("requires_code"):
        raise ContractError("自定义实测范围缺少程序绑定的代码身份")
    if implementation:
        result["implementation"] = implementation
    quantization = facts.get("quantization", {})
    if facts.get("sites"):
        result["site"] = deepcopy(facts["sites"])
    if facts.get("parameters"):
        result["parameters"] = deepcopy(facts["parameters"])
    if quantization:
        result["quantization"] = deepcopy(quantization)
        result["site"] = sorted(quantization)
        for target in ("weight", "activation"):
            bits = sorted({q[target]["bits"] for q in quantization.values() if q.get(target) is not None})
            if bits:
                result[target + "_bits"] = bits
    return result


def fields(scope, prefix=""):
    result = {}
    for key, value in scope.items():
        name = prefix + "." + key if prefix else key
        if isinstance(value, dict):
            result.update(fields(value, name))
        else:
            result[name] = value
    return result


def contains(measured, stated):
    """列表表示已证实集合；声明不得用更大集合包住一条测量。"""
    if isinstance(measured, dict):
        return isinstance(stated, dict) and all(k in measured and contains(measured[k], v) for k, v in stated.items())
    if isinstance(stated, list):
        available = measured if isinstance(measured, list) else [measured]
        return bool(stated) and all(any(canonical(v) == canonical(a) for a in available) for v in stated)
    if isinstance(measured, list):
        return any(canonical(stated) == canonical(a) for a in measured)
    return canonical(measured) == canonical(stated)


def validate_claim_scope(stated, measured):
    if measured.get("schema_version") != VERSION or measured.get("measurement_kind") == "unverified_scope":
        raise ContractError("引用证据缺少宿主核验的规范范围；历史原件可读，不能据此扩大实证知识")
    for key in BOUNDARIES:
        if key in measured and key not in stated:
            raise ContractError("知识范围遗漏实测边界: " + key)
    if not contains(measured, stated):
        raise ContractError("知识适用范围超出引用测量或包含未核验字段")
    # 样本上的整体统计不能凭空推出子集统计；数据范围须原样绑定。
    if not equal_data(stated.get("data"), measured.get("data")):
        raise ContractError("知识数据范围必须与实际测量一致")
    for key in ("quantization", "parameters", "implementation"):
        if key in measured and not equal_data(stated.get(key), measured[key]):
            raise ContractError("知识不得省略或改写实际量化规格/冻结参数: " + key)


def bind_claim_scope(stated, measured_scopes):
    """补全真实证据已经确定的边界；未知科学条件和显式冲突不能被覆盖。"""
    if not measured_scopes:
        if not isinstance(stated, dict) or not stated:
            raise ContractError("没有明确 Evidence 的假说须说明科学适用条件")
        return deepcopy(stated)
    measured = measured_scopes[0]
    if any(not equal_data(measured, scope) for scope in measured_scopes[1:]):
        raise ContractError("Evidence 范围不同，不能绑定为单一共同范围；比较知识应逐份绑定各 Evidence 条件")
    if stated is None:
        stated = {}
    if not isinstance(stated, dict):
        raise ContractError("知识 scope 必须是科学条件对象")
    if not contains(measured, stated):
        # 指明最小冲突路径，避免要求模型重写整个 scope。
        def conflict(actual, requested, path='scope'):
            if isinstance(actual, dict) and isinstance(requested, dict):
                for key, value in requested.items():
                    if key not in actual:
                        return path + '.' + key + ' 未在引用证据中核验'
                    if not contains(actual[key], value):
                        return conflict(actual[key], value, path + '.' + key)
            return path + ': 声明=' + canonical(requested) + '，实测=' + canonical(actual)
        raise ContractError('知识范围冲突: ' + conflict(measured, stated))

    def complete(actual, requested):
        result = deepcopy(actual)
        for key, value in requested.items():
            result[key] = (complete(actual[key], value) if isinstance(value, dict)
                           else deepcopy(value))
        return result
    bound = complete(measured, stated)
    validate_claim_scope(bound, measured)
    return bound


def equal_data(left, right):
    return canonical(left) == canonical(right)


def matches(scope, query):
    flattened = fields(scope)
    for key, value in query.items():
        if key in ("text", "evidence_ids"):
            continue
        # 精确字段路径；不通过 suffix 回落到 research_target 等不同语义的字段。
        actual = scope[key] if key in scope else flattened.get(key)
        if (key not in scope and key not in flattened) or not contains(actual, value):
            return False
    return True


def card_scope(card):
    payload = card["payload"]
    scope = payload.get("scope", payload.get("provenance", {}))
    if card["kind"] == "EvidenceCard" and scope.get("schema_version") != VERSION:
        return {}  # 历史原件仍可按 ID/文本读取，但其未经核验标签不作为精确证据范围。
    return scope
