"""将 LLM 明确给出的位点组展开为普通 v3 JSON；不选择方法或修改插件接口。"""

from copy import deepcopy

from ..schemas import STR, arr, obj, require, validate, fingerprint
from ..storage import atomic_json, read_json
from ..profiling.evidence import seal
from ..registry import ToolDefinition
from .contracts import PLUGIN_PLAN_SCHEMA, PLUGIN_PLAN_INPUT_SCHEMA, normalize_plan
from .plugin_contracts import KINDS
from .sites import site_catalog


FIELDS = deepcopy(PLUGIN_PLAN_INPUT_SCHEMA)
del FIELDS["properties"]["layers"]
FIELDS["required"].remove("layers")
SETTINGS = deepcopy(PLUGIN_PLAN_INPUT_SCHEMA["properties"]["layers"]["items"])
for key in ("module", "kind"):
    del SETTINGS["properties"][key]
    SETTINGS["required"].remove(key)
SELECTOR = obj({
        "pool": {"enum": ["required_sites", "available_sites"]},
        "kinds": arr({"enum": list(KINDS)}, minItems=1, uniqueItems=True),
        "site_ids": arr(STR, uniqueItems=True),
        "module_prefixes": arr(STR, uniqueItems=True),
        "exclude_sites": arr(STR, uniqueItems=True),
        "semantics": arr(STR, minItems=1, uniqueItems=True),
    }, ["pool", "kinds"])
GROUPS = arr(obj({
    "selector": SELECTOR,
    "settings": SETTINGS,
}), minItems=1, maxItems=128)
PREPROCESS_SETTINGS = deepcopy(FIELDS["properties"]["preprocess"]["items"])
for key in ("module", "norm"):
    del PREPROCESS_SETTINGS["properties"][key]
    PREPROCESS_SETTINGS["required"].remove(key)
PREPROCESS_GROUPS = arr(obj({"selector": SELECTOR, "settings": PREPROCESS_SETTINGS}), maxItems=128)
GROUPED_PLAN = obj({"plan_fields": FIELDS, "groups": GROUPS, "preprocess_groups": PREPROCESS_GROUPS},
                   ["plan_fields", "groups"])
CHANGES = deepcopy(SETTINGS)
CHANGES["required"] = []
PATCH_FIELDS = deepcopy(FIELDS)
PATCH_FIELDS["required"] = []
PLAN_PATCH = obj({
    "updates": arr(obj({"site_ids": arr(STR, minItems=1, uniqueItems=True), "changes": CHANGES,
                        "remove_fields": arr({"enum": ["weight_calibration", "activation_calibration", "options"]},
                                             uniqueItems=True)}, ["site_ids", "changes"]), minItems=1, maxItems=128),
    "plan_fields": PATCH_FIELDS,
}, ["updates"])
DELTA_PLAN = obj({"base_trial": STR, "base_draft": STR, **deepcopy(PLAN_PATCH["properties"])}, ["updates"])


def session_trial(session, trial_id):
    """只有本会话完成的 search 模型能成为增量或局部比较基准。"""
    from .search_context import fact_payload
    feedback = next((r for r in session.state.get("profile_feedback", [])
                     if r["source"] == "program_verified" and r["data"]["trial_id"] == trial_id), None)
    require(feedback is not None, "未知本会话已完成search trial", code="invalid_evidence")
    fact_payload(session, feedback["reference"])
    return session._feedback_trial(feedback["data"])


def apply_plan_patch(base, updates, plan_fields=None):
    """字段级替换，不猜默认算法，不把部分spec当作完整spec自动补全。"""
    from .plugin_contracts import site_id
    validate({"updates": updates, **({"plan_fields": plan_fields} if plan_fields is not None else {})}, PLAN_PATCH)
    plan = deepcopy(base)
    plan.update(deepcopy(plan_fields or {}))
    layers = {site_id(layer): layer for layer in plan["layers"]}
    touched = set()
    for update in updates:
        require(set(update["site_ids"]) <= set(layers), "增量更新引用基准中不存在的位点")
        require(not touched.intersection(update["site_ids"]), "增量更新重复位点")
        require(not set(update.get("remove_fields", [])).intersection(update["changes"]), "不能同时删除和设置字段")
        for sid in update["site_ids"]:
            touched.add(sid)
            for key in update.get("remove_fields", []):
                layers[sid].pop(key, None)
            for target in ("weight", "activation"):
                if (target in update["changes"] and update["changes"][target] is None
                        and target + "_method" not in update["changes"]):
                    layers[sid].pop(target + "_method", None)
            layers[sid].update(deepcopy(update["changes"]))
    plan = normalize_plan(plan)
    validate(plan, PLUGIN_PLAN_SCHEMA)
    return plan


def expand_groups(plan_fields, groups, catalog, config, preprocess_groups=()):
    """选择器必须非空、互斥；逐字段深拷贝，不推测遗漏的 W/A 或校准策略。"""
    plan_fields = normalize_plan(plan_fields)
    validate(plan_fields, FIELDS)
    validate(groups, GROUPS)
    layers, seen = [], set()
    scope = config["loop"]["scope_modules"]
    protected = config["constraints"]["protected_modules"] + config["constraints"]["excluded_modules"]
    required = set(config["loop"]["required_sites"])
    for group in groups:
        selector = group["selector"]
        explicit = set(selector.get("site_ids", []))
        excluded = set(selector.get("exclude_sites", []))
        require(explicit <= set(catalog) and excluded <= set(catalog), "分组引用未知 site_id")
        matches = []
        for sid, info in catalog.items():
            name = info["name"]
            if selector["pool"] == "required_sites" and sid not in required:
                continue
            if info["kind"] not in selector["kinds"] or sid in excluded or (explicit and sid not in explicit):
                continue
            if selector.get("semantics") and info["semantic"] not in selector["semantics"]:
                continue
            prefixes = selector.get("module_prefixes", [])
            if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
                continue
            if info["protected"] or (scope and sid not in scope and name not in scope):
                continue
            if any(sid == p or name == p or name.startswith(p + ".") for p in protected):
                continue
            matches.append((sid, info))
        if explicit:
            rejected = sorted(explicit - {sid for sid, _ in matches})
            # 保持严格拒绝，只补足定位信息；不替 LLM 修改选择器或补漏位点。
            detail = [{"site_id": sid, "actual_kind": catalog[sid]["kind"],
                       "actual_semantic": catalog[sid]["semantic"],
                       "protected": catalog[sid]["protected"]} for sid in rejected]
            require(not rejected, f"显式 site_ids 被选择条件筛除：{detail}；selector={selector}；"
                    "kinds、semantics、范围和排除条件取交集。请按真实目录调整分组，不能静默忽略")
        require(bool(matches), "分组选择器没有合法位点；请核对 query_ptq_sites/required_sites")
        for sid, info in matches:
            require(sid not in seen, f"分组重复覆盖位点：{sid}；请用 exclude_sites 明确分区")
            seen.add(sid)
            layers.append({"module": info["name"], "kind": info["kind"], **deepcopy(group["settings"])})
    plan = {**deepcopy(plan_fields), "layers": layers}
    if preprocess_groups:
        validate(list(preprocess_groups), PREPROCESS_GROUPS)
        # 只展开 LLM 指定的方法，LN 名称取已验证的直接边；不按论文或层名选算法。
        for group in preprocess_groups:
            placeholder = {"weight": None, "activation": None, "weight_method": None,
                           "activation_method": None, "reason": "selector only",
                           "evidence": group["settings"].get("evidence", [])}
            selected = expand_groups(plan_fields, [{"selector": group["selector"], "settings": placeholder}],
                                     catalog, config)["layers"]
            for layer in selected:
                require(layer["kind"] == "linear", "预处理分组仅支持 Linear")
                info = catalog[layer["module"]]
                norm_edge = (info["semantic"] == "post_layernorm"
                             and (info.get("edge") or "").endswith(":output"))
                # SQ支持一般Linear输入的显式边缩放；RepQ-LN才要求直接LN前驱。
                # 下游仍检查特殊网格互斥与FP等价，不替模型消除非法组合。
                require(group["settings"]["method"] == "smoothquant" or norm_edge or
                        (group["settings"]["method"] == "adalog_bias" and info["semantic"] == "post_gelu"),
                        f"RepQ-LN预处理需要已验证的 LN→Linear 边：{layer['module']}")
                plan["preprocess"].append({**deepcopy(group["settings"]), "module": layer["module"],
                                           "norm": info["edge"].removesuffix(":output") if norm_edge else None})
    plan = normalize_plan(plan)
    validate(plan, PLUGIN_PLAN_SCHEMA)
    return plan


def register_planning_tool(session):
    def compose(plan_fields=None, groups=None, preprocess_groups=(), base_trial=None, base_draft=None, updates=None):
        base = None
        if updates is not None:
            require((base_trial is None) != (base_draft is None), "增量计划需要且仅需要base_trial或base_draft")
            require(groups is None and not preprocess_groups, "增量与全量分组不能混用")
            base = session_trial(session, base_trial)["candidate"] if base_trial else resolve_draft(session, base_draft)
            plan = apply_plan_patch(base, updates, plan_fields)
        else:
            require(base_trial is None and base_draft is None, "基准计划需要明确updates")
            plan = expand_groups(plan_fields, groups, site_catalog(session.backend, session.config), session.config,
                                 preprocess_groups)
        # 草案与最终提交共用证据、组合、完整覆盖、硬件和资源门禁。
        decision = dict(action="analysis_done", observations=[], hypotheses=[], directions=[], quantization_config=plan)
        session.validate_saved_recommendation(decision)
        session.validate_experiment(plan)
        key = fingerprint(plan)
        reference = f"drafts/{key}.json"
        construction = ({"base_trial": base_trial, "base_draft": base_draft, "updates": updates,
                         "plan_fields": plan_fields} if base is not None else
                        {"plan_fields": plan_fields, "groups": groups, "preprocess_groups": list(preprocess_groups)})
        payload = seal({"quantization_config": plan, "configuration_hash": key, "construction": construction})
        origin = getattr(session, "active_program", None)
        if origin:
            payload = seal({**{k: v for k, v in payload.items() if k != "content_sha256"},
                            "program_origin": deepcopy(origin)})
        path = session.run_dir / reference
        if path.exists():
            # 同一完整计划可由直接调用或不同适配代码构造；保留首次来源，
            # 每次调用的实际来源另在 calls/ 中保存，数值计划仍须严格相同。
            previous = read_json(path)
            require(previous.get("content_sha256") == fingerprint(
                {k: v for k, v in previous.items() if k != "content_sha256"}), "草案内容校验失败")
            require(previous["quantization_config"] == plan and previous["configuration_hash"] == key,
                    "同 hash 草案配置不同")
        else:
            atomic_json(path, payload)
        from .search_context import plan_delta
        return {"draft": reference, "configuration_hash": key, "layer_count": len(plan["layers"]),
                "plan_view": plan_view(plan), "valid": True,
                "base_trial": base_trial, "base_draft": base_draft,
                "configuration_difference": plan_delta(base, plan) if base is not None else None,
                "next_action": "核对分组结果后用 analysis_from_draft 提交原样 draft 路径；执行仍使用完整逐层 JSON"}
    session.registry.register(ToolDefinition(
        "compose_experiment", "全模型优先用此工具：按 LLM 明确的互斥位点组展开普通 v3 计划并严格校验。"
        "selector.pool=required_sites 表示用户必选位点；kinds 过滤类型，可选 semantics/site_ids/module_prefixes/exclude_sites。"
        "preprocess_groups 按显式方法展开：SmoothQuant支持一般Linear输入边，非LN前驱填norm=null；"
        "RepQ-LN只接受已验证LN→Linear边，norm取真实前驱。不选择算法，特殊网格冲突仍拒绝。"
        "settings 是去掉 module/kind 的 layer；格式唯一确定的 bits、per_tensor 的空 axis/group_size、禁用目标的空 method 由程序补齐。方法与粒度仍须明确。"
        "也可用base_trial或base_draft及updates增量更新已有方案；changes逐字段替换，remove_fields只删除可选校准/options。"
        "返回校验过的 draft及差异，可用 analysis_from_draft 提交；不执行量化。",
        {"anyOf": [GROUPED_PLAN, DELTA_PLAN]}, {"type": "object", "additionalProperties": True},
        ("profiling",), compose))


def resolve_draft(session, reference):
    known = {row["result"]["data"].get("draft") for row in session.state["tool_results"]
             if row["result"]["name"] == "compose_experiment" and row["result"]["status"] == "success"}
    require(reference in known, "只能提交本会话 compose_experiment 成功返回的 draft")
    payload = read_json(session.run_dir / reference)
    require(payload.get("content_sha256") == fingerprint({k: v for k, v in payload.items() if k != "content_sha256"}),
            "草案完整性校验失败", code="invalid_evidence")
    plan = payload["quantization_config"]
    require(fingerprint(plan) == payload["configuration_hash"], "草案配置 hash 不匹配")
    return deepcopy(plan)


def plan_view(plan):
    """LLM 上下文的分组视图，不是可提交配置；数值字段与完整位点集合无损。"""
    if plan.get("schema_version") != 3 or len(plan.get("layers", [])) <= 8:
        return deepcopy(plan)
    result = {k: deepcopy(v) for k, v in plan.items() if k != "layers"}
    groups = {}
    for layer in plan["layers"]:
        numerical = {k: v for k, v in layer.items() if k not in ("module", "kind", "reason", "evidence")}
        key = canonical(numerical)
        group = groups.setdefault(key, {"settings": deepcopy(numerical), "sites": []})
        # 展示实际默认值，不把文字reason中的“scale搜索”当作已提交配置。
        group["effective_calibration_strategy"] = {
            target: layer.get(target + "_calibration", {}).get("strategy", "minmax")
            for target in ("weight", "activation") if layer.get(target) is not None}
        group["sites"].append({"module": layer["module"], "kind": layer["kind"]})
    result.update(_context_view_only=True, configuration_hash=fingerprint(plan), layer_count=len(plan["layers"]),
                  effective_execution={"within_stage": "parallel", **deepcopy(plan["execution"])},
                  layer_groups=list(groups.values()),
                  note="这是反馈视图，完整逐层reason/evidence保存在原trial。可用compose_experiment构建新草案，"
                       "再用analysis_from_draft提交；只有直接analysis_done才需填写完整layers，不能直接提交此摘要。")
    return result
