"""实测结果的确定性比较视图；保留原证据，不生成算法建议或改写指标。"""

from .search_context import evidence_facts, remember_facts, plan_delta
from ..schemas import fingerprint


def comparison_view(session, reference, payload):
    result = payload["result"]
    facts = evidence_facts(reference, payload)
    by_pointer = {f["pointer"]: f for f in facts}
    rows = result.get("rows", [])
    bypass = result.get("no_recalibration") is True and all(isinstance(r, dict) and "variant" in r for r in rows)
    if not rows or not (bypass or any(isinstance(r, dict) and "plan" in r for r in rows)):
        selected = remember_facts(session, facts[:32])
        return {"facts": selected, "facts_total": len(facts), "complete": len(facts) <= 32,
                "remaining_tool": "read_evidence_batch", "evidence": reference,
                "measurement_context": result.get("measurement_context", result.get("context", result.get("scope", "as_recorded"))),
                "sample_ids": result.get("sample_ids", []), "raw_evidence_retained": True}
    base = rows[0]
    context = (dict(mode="readonly_bypass", input_source="frozen quantized trial with declared single target bypass",
                    teacher_output_source="original FP model", split="calibration", recalibration=False)
               if bypass else result.get("measurement_context", result.get("context", result.get("scope"))))
    view = []
    for i, row in enumerate(rows):
        plan = row.get("plan")
        metrics = {}
        for suffix in ("relative_mse", "mse", "cosine_error", "classification_diagnostic/top1",
                       "classification_diagnostic/teacher_kl", "classification_diagnostic/baseline_top1"):
            f = by_pointer.get(f"/rows/{i}/{suffix}")
            if f is None:
                continue
            remember_facts(session, [f])
            item = {"fact": f}
            baseline = by_pointer.get(f"/rows/0/{suffix}")
            if baseline is not None:
                remember_facts(session, [baseline])
                item["difference_from_first"] = {
                    "value": f["value"] - baseline["value"], "definition": "candidate minus first candidate",
                    "derived": True, "operand_fact_ids": [f["fact_id"], baseline["fact_id"]]}
            metrics[suffix] = item
        from .planning import plan_view
        identity = plan if plan is not None else {"trial": result.get("trial"), "state_sha256": result["state_sha256"],
                                                 "temporary_bypass": row["variant"]}
        configuration = (plan_view(plan) if i == 0 else {"reconstruct_from_first_and_explicit_difference": True}) if not bypass else identity
        difference = plan_delta(base["plan"], plan) if not bypass else {"temporary_bypass": row["variant"], "numeric_parameters_changed": False}
        calibration_context = row.get("calibration_context", "see sealed calibration report")
        detail = row.get("calibration", {})
        if isinstance(detail, dict) and "mutable_targets" in detail:
            # 配置差异不等于实际拟合范围；未改配置的已声明目标也可能被重拟合。
            calibration_context = {k: detail[k] for k in (
                "mutable_targets", "calibration_input_source", "preprocessing_input_source",
                "teacher_source", "non_target_state_unchanged") if k in detail}
            calibration_context.update(recalibration=True, evidence_pointer=f"/rows/{i}/calibration",
                performed_refits=[{k: record[k] for k in (
                    "site", "target", "samples", "input_path", "teacher", "observer_batch_size") if k in record}
                    for record in detail["calibration"]])
        elif bypass or row.get("variant") == "frozen_baseline":
            calibration_context = {"recalibration": False, "performed_refits": [],
                                   "source": "sealed trial; no parameters refitted in this row"}
        view.append({"candidate_id": ("bypass_" if bypass else "candidate_") + fingerprint(identity)[:24], "row_index": i,
                     "configuration_hash": fingerprint(plan) if plan is not None else None,
                     "configuration": configuration,
                     "difference_from_first": difference,
                     "metrics": metrics, "metric_definition": row.get("relative_mse_reduction"),
                     "measurement_context": row.get("context", context),
                     "calibration_context": calibration_context,
                     "sample_ids": result.get("sample_ids", []), "evidence": reference})
    return {"candidates": view, "baseline": "first returned row, not an algorithm recommendation",
            "raw_evidence_retained": True, "local_only": True,
            "mode": "readonly_bypass" if bypass else "candidate_comparison",
            "note": "delta为程序派生值；引用原始标量fact_id。局部筛选不能代替独立search；只读旁路不是可提交的W4A4方案。"}
