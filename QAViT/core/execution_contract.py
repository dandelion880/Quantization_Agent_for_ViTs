"""冻结执行合同：只归并显式字段，不从模型名或科研文字猜科学配置。"""
from copy import deepcopy

from jsonschema import Draft202012Validator

from core.cards import ContractError, canonical


TARGETS = ("weight", "activation")
EXECUTION_FIELDS = {"mode", "compute_dtype", "activation_calibration", "order", "within_stage",
                    "operator_calibration", "observer_batch_size"}
SPEC_FIELDS = {"bits", "format", "granularity", "axis", "group_size", "symmetric", "scale"}
CONTRACT_FIELDS = {"required_sites", "required_site_targets", "required_kinds", "specs", "methods",
                   "calibrations", "site_overrides", "execution", "preprocess", "operators",
                   "unlisted_modules", "parameters", "parameter_bindings", "fp_exceptions"}


def equal(left, right):
    return canonical(left) == canonical(right)


def merge(left, right, path="quantization_contract"):
    """同一选择的所有镜像必须一致；禁止后写覆盖冻结值。"""
    for key, value in right.items():
        if key not in left:
            left[key] = deepcopy(value)
        elif isinstance(left[key], dict) and isinstance(value, dict):
            merge(left[key], value, path + "." + key)
        elif not equal(left[key], value):
            raise ContractError(f"{path}.{key} 显式冲突: {left[key]!r} != {value!r}")
    return left


def strings(value, path, *, empty=True):
    if (not isinstance(value, list) or (not empty and not value)
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(value) != len(set(value))):
        raise ContractError(path + " 必须是无重复的非空字符串数组")


def validate_contract(value):
    if not isinstance(value, dict):
        raise ContractError("quantization_contract 必须是对象")
    unknown = set(value) - CONTRACT_FIELDS
    if unknown:
        raise ContractError("未识别的必需执行约束: " + str(sorted(unknown)))
    for key in ("required_sites", "required_kinds"):
        if key in value:
            strings(value[key], "quantization_contract." + key)
    for key in ("specs", "methods", "calibrations", "site_overrides", "execution", "parameters", "required_site_targets"):
        if key in value and not isinstance(value[key], dict):
            raise ContractError("quantization_contract." + key + " 必须是对象")
    for key in ("specs", "methods", "calibrations"):
        if set(value.get(key, {})) - set(TARGETS):
            raise ContractError("未识别的量化目标: " + key)
    for target, spec in value.get("specs", {}).items():
        if not isinstance(spec, dict) or set(spec) - SPEC_FIELDS:
            raise ContractError("未识别的量化规格: " + target)
    if set(value.get("execution", {})) - EXECUTION_FIELDS:
        raise ContractError("未识别的必需 execution 字段")
    for sid, targets in value.get("required_site_targets", {}).items():
        strings(targets, "required_site_targets." + sid, empty=False)
        if set(targets) - set(TARGETS):
            raise ContractError("required_site_targets 包含未知目标")
    allowed_row = {*TARGETS, "weight_method", "activation_method", "weight_calibration", "activation_calibration", "options"}
    for sid, row in value.get("site_overrides", {}).items():
        if not isinstance(sid, str) or not sid.strip() or not isinstance(row, dict) or set(row) - allowed_row:
            raise ContractError("site_overrides 必须映射精确 site_id 到已注册 layer 字段")
    if "fp_exceptions" in value:
        fp = value["fp_exceptions"]
        if not isinstance(fp, dict) or set(fp) - {"sites", "modules", "parameters"}:
            raise ContractError("fp_exceptions 必须结构化为 sites/模块/参数；自然语言不能代替执行约束")
        for key in ("modules", "parameters"):
            if key in fp:
                strings(fp[key], "fp_exceptions." + key)
        if "sites" in fp:
            if not isinstance(fp["sites"], dict):
                raise ContractError("fp_exceptions.sites 必须映射 site_id 到目标数组")
            for sid, targets in fp["sites"].items():
                strings(targets, "fp_exceptions.sites." + sid, empty=False)
                if set(targets) - set(TARGETS):
                    raise ContractError("FP 例外包含未知目标")
    for key in ("preprocess", "operators", "parameter_bindings"):
        if key in value and not isinstance(value[key], list):
            raise ContractError(key + " 必须是数组")
    for binding in value.get("parameter_bindings", []):
        if (not isinstance(binding, dict) or set(binding) != {"parameter", "site", "target", "field", "as_list"}
                or binding["target"] not in TARGETS or type(binding["as_list"]) is not bool
                or any(not isinstance(binding[k], str) or not binding[k].strip() for k in ("parameter", "site", "field"))):
            raise ContractError("parameter_bindings 需要 parameter/site/target/field/as_list")
    # An execution contract describes intent. The native plan schema is applied
    # only when a native plan is actually consumed, never as an algorithm registry.
    for target, spec in value.get("specs", {}).items():
        if "bits" in spec and (type(spec["bits"]) is not int or spec["bits"] < 1):
            raise ContractError("specs." + target + ".bits 必须是正整数")
        for key in ("format", "granularity", "scale"):
            if key in spec and (not isinstance(spec[key], str) or not spec[key].strip()):
                raise ContractError("specs." + target + "." + key + " 必须是非空字符串")
    for target, method in value.get("methods", {}).items():
        if not isinstance(method, str) or not method.strip():
            raise ContractError("methods." + target + " 必须是非空字符串")
    for target, calibration in value.get("calibrations", {}).items():
        if not isinstance(calibration, dict):
            raise ContractError("calibrations." + target + " 必须是对象")
    for sid, row in value.get("site_overrides", {}).items():
        for key, item in row.items():
            if key.endswith("_method") and (not isinstance(item, str) or not item.strip()):
                raise ContractError("site_overrides." + sid + "." + key + " 必须是非空字符串")
            if key.endswith("_calibration") and not isinstance(item, dict):
                raise ContractError("site_overrides." + sid + "." + key + " 必须是对象")
    return value


def aliases(source, *, strict=False, validate=True, require_targets=True):
    """兼容已有字段名称，但只在这里转换一次。未知必需字段直接返回错误。"""
    if not isinstance(source, dict):
        raise ContractError("执行选择必须是对象")
    result, handled = {}, {"notes", "training", "training_allowed", "calibration_samples"}
    for key in CONTRACT_FIELDS:
        if key in source:
            result[key] = deepcopy(source[key]); handled.add(key)
    if "quantization_contract" in source:
        merge(result, validate_contract(source["quantization_contract"])); handled.add("quantization_contract")
    for target in TARGETS:
        for field in SPEC_FIELDS:
            key = target + "_" + field
            if key in source:
                merge(result, {"specs": {target: {field: source[key]}}}); handled.add(key)
        key = target + "_spec"
        if key in source:
            merge(result, {"specs": {target: source[key]}}); handled.add(key)
        key = target + "_method"
        if key in source:
            merge(result, {"methods": {target: source[key]}}); handled.add(key)
    if "granularity" in source:
        granularity = source["granularity"]
        if not isinstance(granularity, dict):
            raise ContractError("granularity 必须明确 weight/activation 目标")
        merge(result, {"specs": {t: {"granularity": g} for t, g in granularity.items()}})
        handled.add("granularity")
    for key in EXECUTION_FIELDS:
        if key in source:
            merge(result, {"execution": {key: source[key]}}); handled.add(key)
    if "quantization_scope" in source:
        scope = source["quantization_scope"]
        if not isinstance(scope, dict) or set(scope) - {"sites", "kinds"}:
            raise ContractError("quantization_scope 仅支持结构化 sites/kinds")
        for old, new in (("sites", "required_sites"), ("kinds", "required_kinds")):
            if old in scope:
                merge(result, {new: scope[old]})
        handled.add("quantization_scope")
    if "target_activation_calibration" in source:
        if not source.get("target_site"):
            if require_targets:
                raise ContractError("target_activation_calibration 缺少 target_site")
            validate_contract({"calibrations": {"activation": source["target_activation_calibration"]}})
        else:
            merge(result, {"site_overrides": {source["target_site"]: {"activation_calibration": source["target_activation_calibration"]}}})
        handled.update(("target_activation_calibration", "target_site"))
    if "other_calibrations" in source:
        merge(result, {"calibrations": {t: {"strategy": source["other_calibrations"]} for t in TARGETS}})
        handled.add("other_calibrations")
    if "exceptions" in source:
        merge(result, {"fp_exceptions": source["exceptions"]}); handled.add("exceptions")
    if strict and set(source) - handled:
        raise ContractError("未识别的必需执行约束，请结构化到 quantization_contract: " + str(sorted(set(source) - handled)))
    if not strict and any(k.startswith("required_") and k not in handled for k in source):
        raise ContractError("resolved_choices 存在未识别的必需执行约束")
    if "coverage" in source:
        raise ContractError("resolved_choices.coverage 不能代替结构化 required_sites/required_site_targets/fp_exceptions")
    return validate_contract(result) if validate else result


def declared_contract(execution, *, require_targets=True):
    """Check only choices already supplied, before the original request is immutable."""
    choices = execution.get("resolved_choices", {})
    own = aliases(choices, require_targets=require_targets)
    merge(own, validate_contract(execution.get("quantization_contract", {})))
    merge(own, aliases({k: v for k, v in execution.items() if k.startswith("required_")}, strict=True))
    params = execution.get("parameters", {})
    if not isinstance(params, dict):
        raise ContractError("execution.parameters 必须是对象")
    for key in (params.keys() & choices.keys()) - {"granularity"}:
        if not equal(params[key], choices[key]):
            raise ContractError("execution.parameters 与 resolved_choices 冲突: " + key)
    merge(own, {"parameters": params})
    return own


def research_contracts(brief):
    hard = aliases(brief.get("hard_constraints", {}), strict=True, validate=False)
    fixed = aliases(brief.get("fixed_quantization", {}), strict=True)
    merge(fixed, validate_contract(deepcopy(brief.get("quantization_contract", {}))))
    merge(fixed, aliases({k: v for k, v in brief.items() if k.startswith("required_")}, strict=True))
    return hard, fixed


def declared_rules(rules, supplied):
    """Project existing rules onto supplied fields; missing choices remain undecided."""
    return {key: (declared_rules(rule, supplied[key])
        if isinstance(rule, dict) and isinstance(supplied[key], dict) and not set(rule) <= {'min', 'max'}
        else deepcopy(rule)) for key, rule in rules.items() if key in supplied}


def uses_quantization_contract(execution):
    if execution['service'] == 'profile':
        from tools.profiling import uses_calibration
        return uses_calibration(execution)
    return True


def freeze_contract(execution, brief, *, level=None):
    """把研究目标与本次执行分别保存；局部实验不会自动获得全网覆盖声明。"""
    if not uses_quantization_contract(execution):
        # 能力/目录读取没有量化或校准过程，不把研究目标写成已执行事实。
        execution['quantization_contract'] = {}
        execution['research_quantization_contract'] = deepcopy(brief.get('quantization_contract', {}))
        return execution
    own = declared_contract(execution)
    # Brief quantization fields describe the final candidate. Exploration may
    # observe the whole model while intervening locally, ablating or keeping FP.
    # L3 is the existing final-candidate stage; no extra mode or approval is needed.
    hard, fixed = research_contracts(brief)
    # Preserve the declared target without demanding exploratory choices for it.
    goals = deepcopy(hard)
    for key, value in fixed.items():
        if isinstance(value, dict) and isinstance(goals.get(key), dict):
            goals[key].update(deepcopy(value))
        else:
            goals[key] = deepcopy(value)
    selected = {}
    if level == "L3":
        actual = deepcopy(fixed)
        merge(actual, own)
        def resolve(rule, actual, path):
            if isinstance(rule, list):
                if not any(equal(actual, option) for option in rule):
                    raise ContractError(path + " 缺少允许集合中的明确选择")
                return deepcopy(actual)
            if isinstance(rule, dict) and rule and set(rule) <= {"min", "max"}:
                if type(actual) not in (int, float) or any(
                        actual < bound if key == "min" else actual > bound for key, bound in rule.items()):
                    raise ContractError(path + " 缺少范围内的明确选择")
                return actual
            if isinstance(rule, dict):
                return {key: resolve(child, (actual or {}).get(key), path + "." + key)
                        for key, child in rule.items()}
            return deepcopy(rule)
        selected = deepcopy(hard)
        # Only hard-constraint choices use allowed-set/range semantics.
        for key in ("specs", "methods", "execution", "parameters"):
            if key in hard:
                selected[key] = resolve(hard[key], actual.get(key, {}), "hard_constraints." + key)
        merge(selected, fixed)
    merge(selected, own)
    params = execution.setdefault("parameters", {})
    if not isinstance(params, dict):
        raise ContractError("execution.parameters 必须是对象")
    for key, value in selected.get("parameters", {}).items():
        if key not in params:
            params[key] = deepcopy(value)
        elif not equal(value, params[key]):
            raise ContractError("冻结参数与 execution.parameters 不一致: " + key)
    # Every parameter is passed to the implementation and recorded. Native
    # parameter_bindings, when supplied, are checked at native-plan consumption.
    merge(selected, {"parameters": params})
    execution["quantization_contract"] = validate_contract(selected)
    execution["research_quantization_contract"] = goals
    if execution["service"] == "tensor_probe":
        sid = execution.get("site")
        if not isinstance(sid, str) or not sid.strip():
            raise ContractError("tensor_probe.site 必须是非空 site_id")
        required = set(selected.get("required_sites", [])) | set(selected.get("required_site_targets", {}))
        if required - {sid} or selected.get("required_kinds"):
            raise ContractError("tensor_probe 的实际局部位点不能满足所声明多位点范围")
        if any("weight" in targets for targets in selected.get("required_site_targets", {}).values()):
            raise ContractError("tensor_probe 只干预激活，不能满足必需 weight 量化")
        if choices_mode := selected.get("execution", {}).get("activation_calibration"):
            if choices_mode != "static":
                raise ContractError("tensor_probe fit-once 接口只实现 static 校准")
        if set(own.get("execution", {})) - {"activation_calibration", "compute_dtype"}:
            raise ContractError("tensor_probe 无整网执行阶段，不能核验所声明的 order/mode 等字段")
        if selected.get("execution", {}).get("compute_dtype", "FP32") != "FP32":
            raise ContractError("tensor_probe 当前局部统计明确使用 FP32，不能满足其它 compute_dtype")
    return execution


def validate_plan_schema(plan):
    from ptq_agent.execution.contracts import PLAN_SCHEMA, normalize_plan
    from ptq_agent.schemas import PTQError
    try:
        plan = normalize_plan(plan)
    except PTQError as exc:
        raise ContractError(str(exc)) from exc
    # 使用完整 JSON Schema，早于 detect、capture、prepare 和任何校准前向。
    error = next(Draft202012Validator(PLAN_SCHEMA).iter_errors(plan), None)
    if error:
        details = sorted(error.context, key=lambda e: len(e.path), reverse=True)
        detail = next((e for e in details if e.path and e.path[0] != "schema_version"), error)
        raise ContractError("plan Schema " + ".".join(map(str, detail.path)) + ": " + detail.message)
    if plan["unresolved"]:
        raise ContractError("计划仍有未解决的必需执行约束: " + str(plan["unresolved"]))
    return plan


def compare_fields(expected, actual, path):
    for key, value in expected.items():
        if key not in actual:
            raise ContractError(path + "." + key + " 缺少冻结字段")
        if isinstance(value, dict) and isinstance(actual[key], dict):
            compare_fields(value, actual[key], path + "." + key)
        elif not equal(value, actual[key]):
            raise ContractError(path + "." + key + " 与冻结选择不一致")


def check_plan(plan, choices, backend=None, cfg=None, contract=None, *, native=True):
    from ptq_agent.execution.plugin_contracts import site_id, KINDS
    contract = validate_contract(deepcopy(contract)) if contract is not None else aliases(choices)
    if native:
        plan = validate_plan_schema(plan)
    else:
        if not isinstance(plan, dict):
            raise ContractError("自定义执行器 plan 必须为对象")
        rows = plan.get("layers", [])
        described = (isinstance(rows, list) and isinstance(plan.get("execution", {}), dict)
            and all(isinstance(row, dict) and isinstance(row.get("module"), str) and row["module"].strip()
                and isinstance(row.get("kind", "linear"), str)
                and all(row.get(t) is None or isinstance(row[t], dict)
                    and type(row[t].get("bits")) is int and row[t]["bits"] > 0 for t in TARGETS)
                for row in rows))
        structural = (any(contract.get(k) for k in ("specs", "methods", "calibrations", "required_sites",
            "required_site_targets", "required_kinds", "site_overrides", "parameter_bindings", "execution"))
            or any(k in contract for k in ("preprocess", "operators", "unlisted_modules"))
            or bool(contract.get("fp_exceptions", {}).get("sites")))
        if not structural and (not described or not rows):
            # The custom implementation owns its plan representation. Without
            # structural intent, do not reinterpret similarly named algorithm fields.
            if backend is not None:
                modules = dict(backend.model.named_modules())
                for module in contract.get('fp_exceptions', {}).get('modules', []):
                    if module not in modules:
                        raise ContractError("FP 例外引用未知模块: " + module)
            return {"required_sites": [], "covered_sites": [], "weight_sites": 0, "activation_sites": 0,
                    "quantization": {}, "declared_sites": [], "plan": deepcopy(plan), "contract": deepcopy(contract)}
        if (not isinstance(plan, dict) or not isinstance(plan.get("layers", []), list)
                or not isinstance(plan.get("execution", {}), dict)):
            raise ContractError("自定义执行器 plan 必须为对象；提供 layers/execution 时须为数组/对象")
        for row in plan.get("layers", []):
            if (not isinstance(row, dict) or not isinstance(row.get("module"), str) or not row["module"].strip()
                    or not isinstance(row.get("kind", "linear"), str)):
                raise ContractError("自定义计划位点需要实际 module 和 kind")
            for target in TARGETS:
                spec = row.get(target)
                if spec is not None:
                    if not isinstance(spec, dict) or type(spec.get("bits")) is not int or spec["bits"] < 1:
                        raise ContractError("自定义计划量化位点需要正整数 bits")
    rows = plan.get("layers", [])
    if not native and not rows and any(contract.get(key) for key in ("specs", "methods", "calibrations")):
        raise ContractError("明确量化规格缺少可核验的实际位点描述")
    actual = {site_id(row): row for row in rows}
    if len(actual) != len(rows):
        raise ContractError("实际计划包含重复量化位点")
    for target in TARGETS:
        stated = any(target in contract.get(key, {}) for key in ("specs", "methods", "calibrations"))
        fp = contract.get("specs", {}).get(target, {}).get("bits") == 32
        if stated and not fp and not any(row.get(target) is not None for row in rows):
            raise ContractError("明确量化选择没有实际执行目标: " + target)
    catalog = None
    if backend is not None:
        from ptq_agent.execution.sites import site_catalog
        catalog = site_catalog(backend, cfg)
        if native and set(actual) - set(catalog):
            raise ContractError("实际计划包含未知位点")
        if not native:
            modules = dict(backend.model.named_modules())
            if any(row["module"] not in modules for row in actual.values()):
                raise ContractError("自定义计划包含模型中不存在的模块")
    required = set(contract.get("required_sites", [])) | set(contract.get("required_site_targets", {}))
    kinds = contract.get("required_kinds", [])
    if native and set(kinds) - set(KINDS):
        raise ContractError("required_kinds 未识别的内置位点类型")
    # Preflight has no model catalog. The worker repeats this check with the
    # actual backend before capture/execution and expands required kinds there.
    for kind in kinds if catalog is not None else ():
        sites = {sid for sid, info in catalog.items() if info["kind"] == kind and not info["protected"]}
        if not sites and not native and kind not in KINDS:
            sites = {sid for sid, row in actual.items() if row.get("kind") == kind}
        if not sites:
            raise ContractError("必需量化 kind 在模型中没有可执行位点: " + kind)
        required |= sites
    if required - set(actual):
        raise ContractError("实际计划遗漏必需量化位点: " + str(sorted(required - set(actual))))
    overrides = deepcopy(contract.get("site_overrides", {}))
    for binding in contract.get("parameter_bindings", []):
        key = binding["parameter"]
        if key not in contract.get("parameters", {}):
            raise ContractError("parameter_bindings 缺少冻结参数: " + key)
        value = contract["parameters"][key]
        merge(overrides, {binding["site"]: {binding["target"] + "_calibration": {
            binding["field"]: [value] if binding["as_list"] else value}}})
    if set(overrides) - set(actual):
        raise ContractError("实际计划遗漏有冻结参数/位点覆盖的目标")
    for sid, row in actual.items():
        targets = contract.get("required_site_targets", {}).get(sid)
        if targets is None:
            declared = {**contract.get("specs", {}),
                        **{t: spec for t, spec in overrides.get(sid, {}).items() if t in TARGETS}}
            targets = tuple(t for t, spec in declared.items()
                            if spec is not None and spec.get("bits", 1) < 32)
            if not declared:
                targets = tuple(t for t in TARGETS if row.get(t) is not None and row[t]["bits"] < 32)
            if sid in required and not targets:
                raise ContractError("指定量化位点没有实际量化目标: " + sid)
        for target in TARGETS:
            spec = row.get(target)
            if sid in required and target in targets and (spec is None or spec["bits"] >= 32):
                raise ContractError("指定量化位点被静默保留 FP: " + sid + "/" + target)
            if spec is None:
                continue
            expected = {**contract.get("specs", {}).get(target, {}), **(overrides.get(sid, {}).get(target) or {})}
            compare_fields(expected, spec, sid + "." + target)
            method = overrides.get(sid, {}).get(target + "_method", contract.get("methods", {}).get(target))
            if method is not None and row.get(target + "_method", row.get("method")) != method:
                raise ContractError(sid + "." + target + "_method 与冻结选择不一致")
            # 位点 observer 是显式覆盖；不能被通用 observer 再覆盖回来。
            calibration = overrides.get(sid, {}).get(target + "_calibration", contract.get("calibrations", {}).get(target))
            if calibration is not None:
                compare_fields(calibration, row.get(target + "_calibration", {}), sid + "." + target + "_calibration")
        compare_fields(overrides.get(sid, {}), row, sid)
    compare_fields(contract.get("execution", {}), plan.get("execution", {}), "plan.execution")
    for key in ("preprocess", "operators", "unlisted_modules"):
        if key in contract and not equal(contract[key], plan.get(key, [] if key == "operators" else None)):
            raise ContractError("plan." + key + " 与冻结选择不一致")
    fp = contract.get("fp_exceptions", {})
    for sid, targets in fp.get("sites", {}).items():
        if catalog is not None and sid not in catalog and (native or sid not in actual):
            raise ContractError("FP 例外引用未知位点: " + sid)
        if any(actual.get(sid, {}).get(t) is not None for t in targets):
            raise ContractError("FP 例外被量化: " + sid)
    for module in fp.get("modules", []):
        if backend is not None and module not in dict(backend.model.named_modules()):
            raise ContractError("FP 例外引用未知模块: " + module)
        if any(r["module"] == module or r["module"].startswith(module + ".") for r in rows + plan.get("operators", []) + plan.get("preprocess", [])):
            raise ContractError("FP 例外模块被干预: " + module)
    return {"required_sites": sorted(required), "covered_sites": sorted(actual),
            "weight_sites": sum(r.get("weight") is not None for r in rows),
            "activation_sites": sum(r.get("activation") is not None for r in rows),
            "quantization": {sid: {k: deepcopy(row[k]) for k in ("weight", "activation", "method", "weight_method",
                "activation_method", "weight_calibration", "activation_calibration", "options") if k in row}
                for sid, row in actual.items()},
            "declared_sites": [{"module": row["module"], "site_id": sid} for sid, row in actual.items()],
            "plan": deepcopy(plan), "contract": deepcopy(contract)}


def check_fp_state(model, backend, contract):
    from ptq_agent.execution.model import tensor_digest
    fp = contract.get("fp_exceptions", {})
    original, current = backend.model.state_dict(), model.state_dict()
    names = set(fp.get("parameters", []))
    for module in fp.get("modules", []):
        names.update(k for k in original if k.startswith(module + "."))
    for name in names:
        if name not in original or name not in current or tensor_digest(original[name]) != tensor_digest(current[name]):
            raise ContractError("FP 例外参数缺失或被改变: " + name)
