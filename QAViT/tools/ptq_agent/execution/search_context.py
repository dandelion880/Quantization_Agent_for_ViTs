"""搜索的证据引用、调用理由和无损计划索引；不替LLM选择方法或配方。"""

from copy import deepcopy
from dataclasses import replace
import math

from ..schemas import STR, arr, obj, require, validate, fingerprint, canonical
from ..profiling.evidence import read_returned_evidence
from ..registry import ToolDefinition


FACT_SCHEMA = obj({"evidence": STR, "pointer": STR, "site": STR, "target": STR,
                   "statistic": STR, "reduction": STR, "unit": STR, "value": {"type": "number"}})
FACT_SCHEMA["properties"]["fact_id"] = STR  # 可选，旧的完整断言保持兼容。
# 公布的schema须与校验一致：只接受短引用或完整断言，不允许部分元数据。
FACT_INPUT_SCHEMA = {"anyOf": [FACT_SCHEMA, obj({"evidence": STR, "pointer": STR}), obj({"fact_id": STR})]}
NOTE_SCHEMA = obj({"purpose": STR, "expected_information": STR,
                   "facts": arr(FACT_INPUT_SCHEMA, maxItems=8), "hypothesis": STR,
                   "alternatives": arr(obj({"tool": STR, "reason_not_used_now": STR}), maxItems=4)})
SEMANTICS = {
    "mse": ("sum_squared_error / evaluated_elements", "squared_tensor_value"),
    "aggregate_relative_mse": ("sum_output_SSE / sum_output_reference_energy", "ratio"),
    "cosine_error": ("1 - cosine_over_all_evaluated_output_elements", "ratio"),
    "teacher_kl": ("mean_over_images", "nats"),
    "teacher_kl_sum": ("sum_over_images", "nats"),
    "entropy_nats_mean": ("mean_over_attention_rows", "nats"),
    "top1_mass_mean": ("mean_of_row_max_probability", "probability_mass"),
    "below_1e_minus3_fraction": ("count_below_threshold / elements", "element_fraction"),
    "negative_fraction": ("negative_elements / elements", "element_fraction"),
    "negative_zero_fraction": ("negative_zeroed / negative_elements", "element_fraction"),
    "nonzero_zero_fraction": ("nonzero_zeroed / nonzero_elements", "element_fraction"),
    "top1": ("correct1 / images * 100", "percent"),
    "baseline_top1": ("correct1 / images * 100", "percent"),
    "top5": ("correct5 / images * 100", "percent"),
    "baseline_top5": ("correct5 / images * 100", "percent"),
    "top1_drop_pp": ("baseline_top1 - top1", "percentage_points"),
    "rms": ("sqrt(mean(square(x)))", "tensor_value"),
    "rms_min": ("minimum_of_group_rms", "tensor_value"),
    "rms_max": ("maximum_of_group_rms", "tensor_value"),
    "range_min": ("minimum_over_selected_elements", "tensor_value"),
    "range_max": ("maximum_over_selected_elements", "tensor_value"),
}


def fact_payload(session, reference):
    """Profile读原测量；search反馈读已封存evaluation，不采信外部自述数字。"""
    if not reference.startswith("feedback/"):
        return read_returned_evidence(session, reference), "evidence.result"
    row = next((r for r in session.state.get("profile_feedback", [])
                if r["reference"] == reference and r["source"] == "program_verified"), None)
    require(row is not None and reference == "feedback/" + fingerprint(row["data"]) + ".json",
            "只能读取本会话程序核验的search反馈；external/final不可作为搜索事实", code="invalid_evidence")
    trial = session._feedback_trial(row["data"])
    require(trial is not None and trial["status"] == "completed", "事实引用需要完整search候选",
            code="invalid_evidence")
    artifact, evaluation = trial.get("artifact", {}), trial.get("evaluation", {})
    require(evaluation.get("status") == "completed" and evaluation.get("split") == "search"
            and artifact.get("identity") == session.state["identity"]
            and artifact.get("plan") == trial["candidate"]
            and artifact.get("content_sha256") == fingerprint(
                {k: v for k, v in artifact.items() if k != "content_sha256"}),
            "search事实的评价或模型身份校验失败", code="invalid_evidence")
    return {"result": evaluation}, "sealed search trial.evaluation"


def evidence_facts(reference, payload):
    """指针指向封存result；不猜未知统计的单位，不把配置数字当测量。"""
    facts = []
    omitted = {"plan", "configuration", "protocol", "sample_ids", "shapes", "input_shapes",
               "original_shape", "selected_shape", "shape", "execution", "token_layout", "histogram"}

    def append(number, path, site, target, statistic, container=None):
        reduction, unit = SEMANTICS.get(statistic, ("as_recorded_by_tool", "unspecified_by_tool"))
        if statistic == "relative_mse":
            reduction = (container or {}).get("relative_mse_reduction", "site_or_named_output_SSE / reference_energy")
            unit = "ratio"
        # Evaluator同时记录正确图像数和百分比；同名top1不能仅按字段名解释。
        if (len(path) >= 2 and path[-2] == "classification_counts"
                and statistic in ("top1", "top5", "baseline_top1", "baseline_top5")):
            reduction, unit = "number_of_correct_images", "images"
        output_metric = statistic in ("teacher_kl", "teacher_kl_sum", "top1", "top5", "baseline_top1",
                                      "baseline_top5", "top1_drop_pp")
        facts.append({"evidence": reference,
            "pointer": "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in path),
            "site": "model_output" if output_metric else str(site),
            "target": "model_output" if output_metric else str(target),
            "statistic": statistic, "reduction": reduction, "unit": unit, "value": number})

    def visit(value, path=(), site="model_output", target="unspecified"):
        if isinstance(value, dict):
            site = value.get("site", value.get("module", site))
            target = value.get("target", target)
            for key, child in value.items():
                if key in omitted:
                    continue
                child_target = key if key in ("weight", "activation") else target
                if (key == "range" and value.get("definition") == "numpy_float64_stats_v1"
                        and isinstance(child, list) and len(child) == 2
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                and math.isfinite(x) for x in child) and child[0] <= child[1]):
                    # 仅展开已定义统计的两个原始端点；差值是派生量，不冒充原始测量。
                    for index, number in enumerate(child):
                        append(number, (*path, key, index), site, child_target,
                               "range_min" if index == 0 else "range_max")
                elif isinstance(child, (int, float)) and not isinstance(child, bool) and math.isfinite(child):
                    append(child, (*path, key), site, child_target, key, value)
                else:
                    visit(child, (*path, key), site, child_target)
        elif isinstance(value, list):
            # 明细行可索引；不将任意tensor数组展开成成千上万个伪“统计”。
            for index, child in enumerate(value):
                if isinstance(child, dict):
                    visit(child, (*path, index), site, target)
    visit(payload["result"])
    for fact in facts:
        fact["fact_id"] = "fact_" + fingerprint(fact)[:24]
    return facts


def remember_facts(session, facts):
    """短 ID 只索引本会话实际返回的标量；使用时仍重新核对封存证据。"""
    catalog = session.state.setdefault("fact_catalog", {})
    for fact in facts:
        reference = {k: fact[k] for k in ("evidence", "pointer")}
        previous = catalog.get(fact["fact_id"])
        require(previous is None or previous == reference, "fact ID 冲突", code="invalid_evidence")
        catalog[fact["fact_id"]] = reference
    return facts


def resolve_fact_id(session, fact_id):
    catalog = session.state.setdefault("fact_catalog", {})
    if fact_id not in catalog:
        # journal 已提交、state 未落盘的恢复窗口：只从成功工具结果恢复索引。
        def visit(value):
            if isinstance(value, dict):
                if value.get("fact_id") == fact_id and {"evidence", "pointer", "value", "statistic"} <= value.keys():
                    require("fact_" + fingerprint({k: v for k, v in value.items() if k != "fact_id"})[:24] == fact_id,
                            "journal事实ID损坏", code="invalid_evidence")
                    remember_facts(session, [value])
                else:
                    for child in value.values():
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        for row in session.state.get("tool_results", []):
            result = row["result"]
            if result["status"] == "success" and (result["name"].startswith("profile") or
                                                   result["name"] in ("read_evidence_facts", "read_evidence_batch")):
                visit(result["data"])
    require(fact_id in catalog, "未知fact_id；只能引用本会话工具实际返回的事实", code="fact_mismatch")
    return catalog[fact_id]


def validate_note(session, note):
    validate(note, NOTE_SCHEMA)
    require(note["purpose"].strip() and note["expected_information"].strip(), "调用理由和预期信息不能为空")
    for alternative in note["alternatives"]:
        require(alternative["tool"] in session.registry.definitions, "未调用的替代工具名称必须真实存在")
        require(alternative["reason_not_used_now"].strip(), "暂不调用的理由不能为空")
    cache, verified = {}, []
    for index, fact in enumerate(note["facts"]):
        requested_id = fact.get("fact_id") if set(fact) == {"fact_id"} else None
        if requested_id:
            fact = resolve_fact_id(session, requested_id)
        reference_only = set(fact) == {"evidence", "pointer"}
        if not reference_only:
            validate(fact, FACT_SCHEMA, f"$._decision.facts[{index}]")
        ref = fact["evidence"]
        if ref not in cache:
            cache[ref] = {f["pointer"]: f for f in evidence_facts(ref, fact_payload(session, ref)[0])}
        actual = cache[ref].get(fact["pointer"])
        path = f"$._decision.facts[{index}]"
        require(actual is not None,
                f"第{index + 1}条事实pointer不是已登记标量：{ref}:{fact['pointer']}；"
                "请read_evidence_facts复制真实端点，不把推算的差值当原始测量",
                path=path + ".pointer", code="fact_mismatch")
        require(requested_id is None or requested_id == actual["fact_id"],
                "fact_id 与封存事实不符", path=path, code="fact_mismatch")
        if reference_only:
            # 引用由封存证据展开，避免要求LLM反复抄写浮点尾数；不猜测缺失值。
            verified.append(deepcopy(actual))
            continue
        different = [k for k, v in actual.items() if k != "value"
                     and not (k == "fact_id" and k not in fact) and fact[k] != v]
        if not math.isclose(fact["value"], actual["value"], rel_tol=1e-12, abs_tol=1e-12):
            different.append("value")
        # pointer已核验时直接返回原事实及短引用，避免为修正抄写错误再查询一次。
        # 错误断言仍被拒绝；不存在的pointer不会获得猜测的替代引用。
        reference = {k: actual[k] for k in ("evidence", "pointer")}
        require(not different, f"第{index + 1}条事实与证据不符的字段：{', '.join(different)}；"
                f"原事实：{canonical(actual)}；可改用短引用：{canonical(reference)}",
                path=path, code="fact_mismatch")
        verified.append(deepcopy(actual))
    return {"verified_fact_count": len(verified), "verified_facts": verified,
            "text_status": "LLM rationale/hypothesis, not program-certified causality"}


def plan_records(state):
    result = {}
    def visit(value):
        if isinstance(value, dict):
            if value.get("schema_version") in (2, 3) and "layers" in value and "execution" in value:
                result[fingerprint(value)] = value
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for key in ("analysis", "memories", "execution_feedback", "tool_results"):
        visit(state.get(key, []))
    return result


def failure_records(state):
    """旧日志按真实call_id补epoch；未知年代不冒充当前阻塞。"""
    calls = {"tool:" + row["result"]["name"] + ":" + row["result"]["call_id"]: row
             for row in state.get("tool_results", [])}
    records = []
    for failure in state.get("failures", []):
        row = calls.get(failure["location"])
        epoch = failure.get("epoch", row.get("epoch") if row else
                            state["analysis_epoch"] if not state.get("analysis") else None)
        records.append({**deepcopy(failure), "epoch": epoch,
                        "tool_call": deepcopy(row) if row else None})
    return records




def plan_delta(previous, current):
    from .plugin_contracts import site_id
    def numeric(plan):
        return {site_id(l): {k: v for k, v in l.items() if k not in ("reason", "evidence")}
                for l in plan.get("layers", [])}
    old, new = numeric(previous), numeric(current)
    groups = {}
    for sid in sorted(old.keys() | new.keys()):
        a, b = old.get(sid, {}), new.get(sid, {})
        fields = {k: {"before": a.get(k), "after": b.get(k)} for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if fields:
            groups.setdefault(canonical(fields), {"sites": [], "changes": fields})["sites"].append(sid)
    changes = {k: {"before": previous.get(k), "after": current.get(k)}
               for k in ("execution", "preprocess", "operators") if previous.get(k) != current.get(k)}
    return {"layer_changes": list(groups.values()), "global_changes": changes,
            "execution_changed": previous.get("execution") != current.get("execution"),
            "preprocess_changed": [{k: v for k, v in x.items() if k != "evidence"} for x in previous.get("preprocess", [])]
                != [{k: v for k, v in x.items() if k != "evidence"} for x in current.get("preprocess", [])],
            "operators_changed": previous.get("operators", []) != current.get("operators", []),
            "scope": "configuration difference, not tensor equality or causal attribution"}






def note_schema(schema, required=False):
    result = deepcopy(schema)
    for branch in result.get("anyOf", [result]):
        branch["properties"]["_decision"] = deepcopy(NOTE_SCHEMA)
        if required:
            branch["required"] = list(dict.fromkeys([*branch.get("required", []), "_decision"]))
    return result


def submission_budget(state):
    """只分配调用预算，不选择算法、替代候选或增加总预算。旧配置默认不阻断。"""
    config, used = state["config"], state["budget"]
    loop, limits = config.get("loop", {}), config["budget"]
    trials = max(0, min(loop.get("max_trials", limits["trials"]), limits["trials"]) - used["trials"])
    tools, agents = (max(0, limits[k] - used[k]) for k in ("tools", "agents"))
    reserve_tools = trials * loop.get("submission_reserve_tools_per_trial", 0)
    reserve_agents = trials * loop.get("submission_reserve_agents_per_trial", 0)
    phase = sum(row.get("epoch") == state.get("analysis_epoch") for row in state.get("tool_results", []))
    phase_allowance = math.ceil((tools + phase) / max(1, trials))
    blocked = bool(trials and ((reserve_tools and tools <= reserve_tools) or
                              (reserve_agents and agents <= reserve_agents)))
    return {"remaining_trials": trials, "remaining_tools": tools, "remaining_agents": agents,
            "reserved_tools_for_submission_and_feedback": reserve_tools,
            "reserved_agents_for_submission_and_feedback": reserve_agents,
            "current_phase_tools_used": phase, "suggested_phase_tool_allowance": phase_allowance,
            "new_measurements_blocked": blocked,
            "note": "已达到预留边界时停止新profile；仍可批量读既有证据/方法卡、修正草案并提交。"
                    "阶段额度是规划建议，不替LLM选择方案；完整候选和search反馈优先保留。"}


def register_search_context(session):
    registry = session.registry
    free = {"type": "object", "additionalProperties": True}
    def facts(evidence, site, statistic, offset, limit, pointers=None):
        payload, pointer_base = fact_payload(session, evidence)
        rows = evidence_facts(evidence, payload)
        if pointers is not None:
            require(set(pointers) <= {r["pointer"] for r in rows},
                    "请求包含不存在的精确标量 pointer", code="fact_mismatch")
        rows = [r for r in rows if (not site or r["site"] == site) and (not statistic or statistic in r["statistic"])]
        if pointers is not None:
            available = {r["pointer"]: r for r in rows}
            require(set(pointers) <= set(available),
                    "pointer存在但与site/statistic筛选冲突；跨位点/指标精确读取时请将筛选设为空字符串",
                    code="fact_filter_conflict")
            rows = [available[p] for p in pointers]
        require(offset == 0 or offset < len(rows), "事实分页offset越界")
        selected = remember_facts(session, rows[offset:offset+limit])
        return {"facts": selected, "references": [{k: r[k] for k in ("evidence", "pointer")} for r in selected],
                "fact_references": [{"fact_id": r["fact_id"]} for r in selected],
                "total": len(rows),
                "pointer_base": pointer_base,
                "next_offset": offset+limit if offset+limit < len(rows) else None,
                "note": "_decision.facts可仅复制references中的evidence/pointer，由程序绑定完整原始统计；"
                        "也可完整复制facts，不能舍入数值或只填写部分元数据。pointer相对于pointer_base声明的原始对象；"
                        "range_min/range_max是已定义分布的原始端点，"
                        "不提供推算的range差值或任意数组元素。结构化字段可核验；自由文字因果结论仍是LLM假设。"}
    page = {"offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 64}}
    # 空字符串表示不筛选；证据路径和事实本身仍要求非空。
    filter_text = {"type": "string"}
    fact_request = obj({"evidence": STR, "site": filter_text, "statistic": filter_text, **page,
                        "pointers": arr(STR, minItems=1, maxItems=64, uniqueItems=True)},
                       ["evidence", "site", "statistic", "offset", "limit"])
    registry.register(ToolDefinition("read_evidence_facts", "读取真实标量，返回稳定fact_id，可直接用于_decision.facts；可选pointers精确选择字段，空site/statistic不过滤；禁止external/final",
        fact_request, free, ("profiling",), facts))

    def batch(requests):
        # 每项继续走相同的封存身份/标量检查；失败不会被当作空的成功证据。
        require(sum(r["limit"] for r in requests) <= 128, "批量读取最多128条事实；请明确分页")
        return {"results": [facts(**r) for r in requests], "raw_evidence_retained": True}
    registry.register(ToolDefinition("read_evidence_batch", "一次读取1至8份本会话证据的精确标量；共享原事实校验，最多128条，不重新forward。各项返回fact_id与独立分页位置",
        obj({"requests": arr(fact_request, minItems=1, maxItems=8)}), free, ("profiling",), batch))

    def record(kind, key, offset, limit):
        if kind == "plan":
            plans = plan_records(session.state)
            require(key in plans, "未知本会话plan_ref，不能读取外部方案")
            plan = plans[key]
            return {"configuration_hash": key, "plan": {**deepcopy(plan), "layers": deepcopy(plan["layers"][offset:offset+limit])},
                    "layers_total": len(plan["layers"]), "next_offset": offset+limit if offset+limit < len(plan["layers"]) else None}
        rows = failure_records(session.state) if kind == "failure" else session.state.get(
            {"feedback": "profile_feedback", "analysis": "analysis", "memory": "memories"}[kind], [])
        rows = [r for r in rows if not key or key in (r.get("id"), r.get("reference"),
                                                     r.get("location"), str(r.get("epoch")))]
        return {"records_detail": deepcopy(rows[offset:offset+limit]), "total": len(rows)}
    registry.register(ToolDefinition("read_search_record", "读取本会话计划原文/历史分析/反馈/记忆/失败及原始工具参数；不含final、不读论文整套配方",
        obj({"kind": {"enum": ["plan", "feedback", "analysis", "memory", "failure"]}, "key": filter_text, **page}), free, ("profiling",), record))
    for name, definition in list(registry.definitions.items()):
        if "profiling" not in definition.roles:
            continue
        def wrap(handler, tool_name):
            def checked(_decision=None, **arguments):
                # 脚本内层RPC继承外层调用目的；仍逐次校验参数、权限和工具预算。
                require(_decision is not None or getattr(session, "active_program", None)
                        or not session.config.get("loop", {}).get("require_decision_notes", False),
                        "缺少本次调用的_decision依据", code="missing_decision_note")
                if _decision is not None:
                    validate_note(session, _decision)
                if tool_name.startswith("profile"):
                    require(not submission_budget(session.state)["new_measurements_blocked"],
                            "剩余调用已预留给完整候选提交和反馈；请批量读取已有证据或组合/提交草案，不能继续新profile",
                            code="submission_budget_reserved")
                result = handler(**arguments)
                if tool_name.startswith("profile") and isinstance(result, dict) and result.get("evidence"):
                    from ..profiling.evidence import read_measurement
                    from .comparison import comparison_view
                    path = (session.run_dir / result["evidence"]).resolve()
                    require(path.is_relative_to((session.run_dir / "evidence").resolve()), "证据路径越界")
                    payload = read_measurement(path, session.state["identity"])
                    result["comparison"] = comparison_view(session, result["evidence"], payload)
                return result
            return checked
        returns = deepcopy(definition.returns)
        if name.startswith("profile") and "properties" in returns:
            returns["properties"]["comparison"] = free
        registry.definitions[name] = replace(definition, parameters=note_schema(definition.parameters),
                                             returns=returns, handler=wrap(definition.handler, name))
