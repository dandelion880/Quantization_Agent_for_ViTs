"""所有 L3 服务共用的冻结结果协议；completed 不是验证结论。"""
import math
import json
from jsonschema import Draft202012Validator, SchemaError

from core.cards import ContractError


def evaluation_name(row):
    return "segmentation" if row["kind"] == "ade_frozen" else row["kind"]


def classification_shard_fallback(payload, brief):
    """Placement capability, never an admission check or a weaker result contract.

    Composite required_results schemas are intentionally handled by the existing
    serial evaluator: guessing their metric requirements would change semantics.
    """
    spec = payload['execution']
    if (spec['service'] not in ('quantize_evaluate', 'evaluate_artifact')
            or spec.get('requires_code') or spec.get('resolved_choices', {}).get('training')
            or spec.get('backend', {}).get('kind', 'server') != 'server'):
        return 'existing single-device backend/executor'
    rows = spec.get('evaluations', [])
    if not rows or any(row['kind'] != 'classification' for row in rows):
        return 'existing single-device evaluation (non-classification or mixed)'
    # These are exactly the scalar measures produced by evaluating.aggregate.
    metrics = {'top1', 'top5', 'baseline_top1', 'baseline_top5', 'delta_top1_pp',
               'top1_drop_pp', 'relative_mse', 'aggregate_relative_mse', 'mse', 'cosine_error'}
    unsupported = {m for row in rows for m in row.get('metrics', [])} - metrics
    if payload.get('level') == 'L3':
        for row in spec.get('result_protocol', brief.get('evaluation_protocol', {})).get('required_results', []):
            parts = row['path'].split('.')
            if parts[:3] == ['evaluations', 'classification', 'metrics'] and len(parts) == 4:
                if parts[3] not in metrics:
                    unsupported.add(parts[3])
            elif (parts in (['evaluations'], ['evaluations', 'classification'])
                  or parts[:3] == ['evaluations', 'classification', 'metrics']):
                return 'existing single-device evaluation for composite required_results schema'
    if unsupported:
        return 'existing single-device evaluation for required metrics: ' + ', '.join(sorted(unsupported))
    return None


def result_requirements(payload, brief):
    spec = payload["execution"]
    protocol = brief["evaluation_protocol"] if payload["level"] == "L3" else {}
    requirements = protocol.get("required_results", [])
    if not isinstance(requirements, list):
        raise ContractError("evaluation_protocol.required_results 必须是数组")
    for row in requirements:
        if (not isinstance(row, dict) or set(row) != {"path", "schema"}
                or not isinstance(row["path"], str) or not row["path"].strip() or not isinstance(row["schema"], dict)):
            raise ContractError("required_results 必须包含非空 path 和 schema 对象")
        try:
            Draft202012Validator.check_schema(row["schema"])
        except SchemaError as exc:
            raise ContractError("required_results.schema 非法: " + exc.message) from exc
        # 不允许验证过程去网络解析外部引用。
        if '"$ref"' in json.dumps(row["schema"]):
            raise ContractError("required_results.schema 不支持外部/递归 $ref")
    rows = spec.get("evaluations", [])
    if not isinstance(rows, list):
        raise ContractError("execution.evaluations 必须是数组")
    names = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str) or not row["kind"].strip():
            raise ContractError("execution.evaluations.kind 必须是非空字符串")
        if set(row) - {"kind", "split", "count", "offset", "metrics", "full_dataset"}:
            raise ContractError("execution.evaluations 存在未识别的必需字段")
        if "count" not in row or (row["count"] is not None and (type(row["count"]) is not int or row["count"] <= 0)):
            raise ContractError("execution.evaluations.count 必须显式为正整数或 null")
        if row["kind"] != "ade_frozen" and (not isinstance(row.get("split"), str) or not row["split"].strip()):
            raise ContractError("execution.evaluations.split 必须是非空字符串")
        if "offset" in row and row["offset"] != 0:
            raise ContractError("当前评测接口不支持非零 offset；不能静默忽略")
        if "metrics" in row and (not isinstance(row["metrics"], list)
                                  or any(not isinstance(m, str) or not m.strip() for m in row["metrics"])):
            raise ContractError("execution.evaluations.metrics 必须是指标名数组")
        if "full_dataset" in row and type(row["full_dataset"]) is not bool:
            raise ContractError("execution.evaluations.full_dataset 必须为 boolean")
        names.append(evaluation_name(row))
    if len(names) != len(set(names)):
        raise ContractError("评测结果键重复；当前接口每个 kind 仅支持一个冻结数据视图")
    if payload["level"] == "L3" and not rows and not requirements:
        raise ContractError("L3 缺少可核验的 evaluations 或 evaluation_protocol.required_results")
    for key in ("full_dataset", "reload_before_evaluation"):
        if key in protocol and type(protocol[key]) is not bool:
            raise ContractError("evaluation_protocol." + key + " 必须为 boolean")
    for key in ("batch_size", "calibration_samples", "search_samples"):
        if key in protocol and (type(protocol[key]) is not int or protocol[key] <= 0):
            raise ContractError("evaluation_protocol." + key + " 必须为正整数")
    return requirements


def check_results(payload, brief, data):
    requirements = result_requirements(payload, brief)
    spec = payload["execution"]
    observations = data.get("observations")
    if not isinstance(observations, dict) or not observations:
        raise ContractError("测量证据缺少 observations")
    protocol = brief["evaluation_protocol"] if payload["level"] == "L3" else {}
    if observations.get("status") in ("engineering_blocked", "blocked", "failed"):
        raise ContractError("engineering failure is not completed numerical evidence")
    if payload['level'] == 'L3' and spec.get('quantization_contract', {}).get('protocol'):
        from core.static_w4a4 import check_audit
        expected = observations.get('coverage', {}).get('static_w4a4', {})
        if not expected.get('required_boundaries'):
            raise ContractError('final static W4A4 missing host boundary coverage')
        check_audit(observations.get('static_w4a4'), expected)
    evaluations = observations.get("evaluations", {})
    if not isinstance(evaluations, dict):
        raise ContractError("observations.evaluations 必须是对象")
    identity = data.get("provenance", {}).get("data_identity", {})
    for requested in spec.get("evaluations", []):
        name = evaluation_name(requested)
        measured = evaluations.get(name, {})
        split = "ade_validation" if requested["kind"] == "ade_frozen" else requested["split"]
        count, ids, metrics = measured.get("sample_count"), measured.get("sample_ids"), measured.get("metrics")
        if (type(count) is not int or count <= 0 or not isinstance(ids, list) or len(ids) != count
                or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != count
                or measured.get("split") != split or not isinstance(metrics, dict) or not metrics):
            raise ContractError("测量证据缺少符合冻结规格的评测结果: " + requested["kind"])
        expected_ids = identity.get("split_ids", {}).get(split)
        if expected_ids is not None and ids != expected_ids[:requested["count"]]:
            raise ContractError("评测实际 sample_ids 与冻结数据顺序/范围不一致")
        if requested["count"] is not None and count != requested["count"]:
            raise ContractError("评测 sample_count 与冻结规格不一致")
        observed_ranges = data.get("provenance", {}).get("execution_facts", {}).get("data", {})
        actual = next((row for row in observed_ranges.values() if row.get("split") == split
                       and row.get("sample_ids") == ids and row.get("sample_count") == count), None)
        if payload["level"] == "L3" and actual is None:
            raise ContractError("L3 评测范围缺少宿主实际数据读取/执行事实支持")
        if measured.get("full_dataset") is True and (actual is None or actual.get("full_dataset") is not True):
            raise ContractError("自报 full_dataset 与宿主实际完整数据范围不一致")
        if requested["count"] is None and expected_ids is None and measured.get("full_dataset") is not True:
            raise ContractError("完整视图缺少可核验的数据身份或完整评测回执")
        if ((protocol.get("full_dataset") is True or requested.get("full_dataset") is True
             or (requested["count"] is None and split in ("final", "ade_validation")))
                and measured.get("full_dataset") is not True):
            raise ContractError("冻结评测要求完整数据，实际证据仅为子集")
        required_metrics = list(requested.get("metrics", []))
        if requested["kind"] == "classification" and data.get("measurement_source") == "real":
            required_metrics.append("top1")
        if requested["kind"] == "ade_frozen":
            required_metrics.extend(("mIoU", "mAcc", "aAcc"))
        if not set(required_metrics) <= set(metrics):
            raise ContractError("评测缺少冻结协议必需指标: " + name)
        if not any(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()):
            raise ContractError("评测没有有限数值指标: " + name)
        for key in required_metrics:
            if type(metrics[key]) not in (int, float) or not math.isfinite(metrics[key]):
                raise ContractError("必需指标不是有限数值: " + key)
    for row in requirements:
        value = observations
        for part in row["path"].split("."):
            if not isinstance(value, dict) or part not in value:
                raise ContractError("缺少 L3 必需结果: " + row["path"])
            value = value[part]
        error = next(Draft202012Validator(row["schema"]).iter_errors(value), None)
        if error:
            raise ContractError("L3 必需结果不符合冻结合同 " + row["path"] + ": " + error.message)
    if protocol.get("reload_before_evaluation") or spec.get("reload_before_evaluation") or payload.get("validation", {}).get("export_reload"):
        reload = observations.get("reload", {})
        if reload.get("evaluated_instance") != "reloaded" or reload.get("host_verified") is not True:
            raise ContractError("冻结协议要求使用宿主核验的重载实例评测")
    provenance = data.get("provenance", {})
    measured_data = provenance.get("execution_facts", {}).get("data", {})
    bindings = {"batch_size": identity.get("batch_size"),
                "calibration_samples": measured_data.get("calibration", {}).get("sample_count"),
                "data_protocol": provenance.get("backend_data", {}).get("data_protocol"),
                "manifest_sha256": provenance.get("backend_data", {}).get("data_sha256")}
    for key, actual in bindings.items():
        applicable = (key != "calibration_samples" or "calibration" in spec or "calibration" in measured_data)
        applicable = applicable and (key != "batch_size" or bool(measured_data))
        if applicable and key in protocol and actual != protocol[key]:
            raise ContractError("实际执行不满足 evaluation_protocol." + key)
    if "search_samples" in protocol:
        counts = {row.get("sample_count") for row in measured_data.values() if row.get("split") == "search"}
        requested_search = any(row.get("split") == "search" for row in spec.get("evaluations", []))
        if (counts or requested_search) and counts != {protocol["search_samples"]}:
            raise ContractError("实际执行不满足 evaluation_protocol.search_samples")
    if protocol.get("full_dataset") and not spec.get("evaluations"):
        if not any(row.get("full_dataset") is True for row in measured_data.values()):
            raise ContractError("完整数据协议缺少实际完整测量范围")
    if payload["level"] == "L3" and not provenance.get("execution_facts", {}).get("measurement_kind"):
        raise ContractError("L3 缺少宿主核验的 execution_facts，不能凭 completed 放行")
