"""数值执行计划 v2/v3 与工具参数；这里的版本号是模型计划协议，不是旧工作流。"""

from copy import deepcopy
from ..schemas import NULL, STR, arr, obj, require, validate
from .plugin_contracts import WEIGHT_METHODS, ACTIVATION_METHODS

FORMATS = {
    "FP32": 32,
    "FP16": 16,
    "BF16": 16,
    "FP8_E4M3FN": 8,
    "FP8_E5M2": 8,
    "INT4": 4,
    "UINT4": 4,
    "INT6": 6,
    "UINT6": 6,
    "INT8": 8,
    "UINT8": 8,
    "NF4": 4,
    "FP4_E2M1": 4,
    "NVFP4": 4,
    "MXFP4": 4,
}
RAW_FLOATS = ("FP32", "FP16", "BF16")
BLOCKS = {"NVFP4": 16, "MXFP4": 32}
SPEC = obj(
    {
        "format": {"enum": list(FORMATS)},
        "bits": {"enum": [4, 6, 8, 16, 32]},
        "granularity": {"enum": ["per_tensor", "per_channel", "per_group", "per_token"]},
        "axis": {"anyOf": [{"type": "integer", "minimum": 0}, NULL]},
        "group_size": {"anyOf": [{"type": "integer", "minimum": 1}, NULL]},
        "symmetric": {"type": "boolean"},
        "scale": {"enum": ["minmax", "none"]},
    }
)
OPTIONAL_SPEC = {"anyOf": [NULL, SPEC]}
PLAN_SCHEMA = obj(
    {
        "schema_version": {"enum": [2]},
        "unlisted_modules": {"enum": ["keep_fp"]},
        "execution": obj(
            {
                "mode": {"enum": ["reference_qdq", "native"]},
                "compute_dtype": {"enum": list(RAW_FLOATS)},
                "activation_calibration": {"enum": ["static", "dynamic"]},
            }
        ),
        "layers": arr(
            obj(
                {
                    "module": STR,
                    "method": {"enum": ["rtn"]},
                    "weight": OPTIONAL_SPEC,
                    "activation": OPTIONAL_SPEC,
                    "reason": {"type": "string"},
                    "evidence": arr(STR, uniqueItems=True),
                }, ["module", "method", "weight", "activation"]
            ),
            minItems=0,
            maxItems=512,
        ),
        "unresolved": arr(STR, maxItems=32),
    }
)
LOOP_SCHEMA = obj(
    {
        "evaluation": {"enum": ["feature_fidelity", "classification"]},
        "max_relative_mse": {"anyOf": [{"type": "number", "minimum": 0}, NULL]},
        "max_top1_drop_pp": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 100}, NULL]},
        "max_weight_bytes_ratio": {"anyOf": [{"type": "number", "exclusiveMinimum": 0, "maximum": 1}, NULL]},
        "max_latency_ms": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, NULL]},
        "max_peak_memory_mib": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, NULL]},
        "execution_modes": arr({"enum": ["reference_qdq", "native"]}, minItems=1, uniqueItems=True),
        "max_trials": {"type": "integer", "minimum": 1, "maximum": 32},
        "warmup": {"type": "integer", "minimum": 1, "maximum": 20},
        "repeats": {"type": "integer", "minimum": 1, "maximum": 50},
        "disk_budget_mib": {"type": "integer", "minimum": 64},
        "min_free_disk_mib": {"type": "integer", "minimum": 64},
        "scope_modules": arr(STR, uniqueItems=True),
        "quantize_targets": arr({"enum": ["weight", "activation"]}, minItems=1, uniqueItems=True),
    }
)
# 可选展示/搜索代理，不改变最终精度门槛或量化数学。
LOOP_SCHEMA["properties"].update({
    "require_decision_notes": {"type": "boolean"},
    "submission_reserve_tools_per_trial": {"type": "integer", "minimum": 0, "maximum": 8},
    "submission_reserve_agents_per_trial": {"type": "integer", "minimum": 0, "maximum": 8},
    "search_max_top1_drop_pp": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 100}, NULL]},
})


def validate_loop(options):
    validate(options, LOOP_SCHEMA)
    require(options.get("min_top1") is None or options["evaluation"] == "classification",
            "min_top1 需要真实分类评价")
    require(options.get("search_max_top1_drop_pp") is None or options["evaluation"] == "classification",
            "search精度下降代理只适用于分类")
    # 内置 schema 校验器不解释 exclusiveMinimum，显式保证资源门槛严格为正。
    for key in ("max_weight_bytes_ratio", "max_latency_ms", "max_peak_memory_mib"):
        require(options[key] is None or options[key] > 0, f"{key} 必须大于 0")
    field = "max_relative_mse" if options["evaluation"] == "feature_fidelity" else "max_top1_drop_pp"
    if options.get("selection", "first_feasible") == "best_accuracy":
        require(options["evaluation"] == "classification", "best_accuracy 需要真实分类评价")
        if options.get("stop_on_goal", False):
            require(options.get("min_top1") is not None or options["max_top1_drop_pp"] is not None,
                    "best_accuracy提前停止必须明确min_top1或max_top1_drop_pp，不能把健康检查通过视为精度达标")
    else:
        require(options[field] is not None or options.get("min_top1") is not None,
                f"请明确达标门槛 {field} 或 min_top1")


def normalize_plan(plan):
    """Canonicalize only uniquely determined built-in representation fields.

    The caller chooses whether it is using the built-in executor. Custom plans
    must never pass through this function merely because they contain layers.
    Scientific choices (method, format, axis, calibration, order) remain explicit.
    """
    value = deepcopy(plan)
    if not isinstance(value, dict) or value.get("schema_version") not in (2, 3):
        return value  # The full schema produces the ordinary protocol error.

    def determined(target, field, expected, path):
        if field in target:
            require(target[field] == expected and type(target[field]) is type(expected),
                    f"{path}.{field}={target[field]!r} 与已确定值 {expected!r} 冲突",
                    path=path + "." + field)
        else:
            target[field] = expected

    determined(value, "unlisted_modules", "keep_fp", "$")
    if value["schema_version"] == 3 and isinstance(value.get("execution"), dict):
        determined(value["execution"], "mode", "reference_qdq", "$.execution")
    if not isinstance(value.get("layers"), list):
        return value
    from .plugin_contracts import KINDS, WEIGHT_KINDS
    for index, row in enumerate(value["layers"]):
        if not isinstance(row, dict):
            continue
        path = f"$.layers[{index}]"
        if value["schema_version"] == 2:
            determined(row, "method", "rtn", path)
        elif row.get("kind") in KINDS and row["kind"] not in WEIGHT_KINDS:
            determined(row, "weight", None, path)
        for target in ("weight", "activation"):
            spec = row.get(target)
            if target in row and spec is None and value["schema_version"] == 3:
                determined(row, target + "_method", None, path)
            if not isinstance(spec, dict):
                continue
            spec_path = path + "." + target
            if isinstance(spec.get("format"), str) and spec["format"] in FORMATS:
                fmt = spec["format"]
                determined(spec, "bits", FORMATS[fmt], spec_path)
                determined(spec, "scale", "none" if fmt in RAW_FLOATS else "minmax", spec_path)
                if fmt in RAW_FLOATS:
                    determined(spec, "granularity", "per_tensor", spec_path)
                if fmt.startswith("UINT") or not fmt.startswith("INT"):
                    determined(spec, "symmetric", not fmt.startswith("UINT"), spec_path)
                if fmt in BLOCKS:
                    determined(spec, "granularity", "per_group", spec_path)
                    determined(spec, "group_size", BLOCKS[fmt], spec_path)
            if spec.get("granularity") == "per_tensor":
                determined(spec, "axis", None, spec_path)
            if spec.get("granularity") in ("per_tensor", "per_channel", "per_token"):
                determined(spec, "group_size", None, spec_path)
    return value


def validate_spec(spec, target, shape, activation_calibration="static", *, activation_axis=None,
                  allow_unknown_shape=False):
    validate(spec, SPEC)
    fmt, mode, axis = spec["format"], spec["granularity"], spec["axis"]
    require(spec["bits"] == FORMATS[fmt], "format/bits 不一致")
    require(shape is None or len(shape) >= 2 and all(int(n) > 0 for n in shape), "缺少有效张量形状")
    require(shape is not None or mode == "per_tensor" or allow_unknown_shape,
            "准备后仍缺少真实张量形状")
    if mode == "per_tensor":
        require(axis is None and spec["group_size"] is None, "per_tensor 的 axis/group_size 必须为 null")
    else:
        require(type(axis) is int and axis >= 0 and (shape is None or axis < len(shape)),
                f"量化轴越界：axis={axis}, shape={shape}", path="$.axis")
        if mode == "per_channel":
            # Linear 默认末轴；v3 位点目录显式给出转置 K / NCHW 的 feature 轴。
            channel_axis = 0 if target == "weight" else (None if shape is None else
                len(shape) - 1 if activation_axis is None else activation_axis % len(shape))
            require(
                channel_axis is None or axis == channel_axis,
                f"channel 轴不符：axis={axis}, 该位点应为 {channel_axis}, shape={shape}",
                path="$.axis",
            )
            require(spec["group_size"] is None, "per_channel 没有 group_size")
        elif mode == "per_group":
            require(
                type(spec["group_size"]) is int and spec["group_size"] > 0
                and (shape is None or axis == len(shape) - 1 and shape[axis] % spec["group_size"] == 0),
                "group 必须沿最后一轴且整除该轴",
            )
        else:
            require(
                target == "activation" and (shape is None or axis == len(shape) - 1) and spec["group_size"] is None,
                "per_token 仅适用于输入激活，归约最后 feature 轴",
            )
        if target == "activation" and mode in ("per_group", "per_token"):
            require(
                activation_calibration == "dynamic",
                "逐 token/block 激活必须显式选择 dynamic，不能保存 batch/token 位置专属静态尺度",
            )
    if fmt in RAW_FLOATS:
        require(
            mode == "per_tensor" and spec["scale"] == "none" and spec["symmetric"],
            "FP32/FP16/BF16 是直接格式转换：per_tensor / scale=none / symmetric=true",
        )
    else:
        require(spec["scale"] == "minmax",
                f"{target}/{fmt} 的 spec.scale 必须为 minmax，当前为 {spec['scale']!r}；"
                "scale=none 仅用于 FP32/FP16/BF16 直接转换，搜索策略另填 calibration 字段",
                path=f"$.{target}.scale")
        if fmt.startswith("UINT"):
            require(not spec["symmetric"], "UINT 使用显式无符号 affine 网格，symmetric=false")
        elif not fmt.startswith("INT"):
            require(spec["symmetric"], "浮点/codebook RTN 使用 absmax 缩放，不支持 affine zero point")
    if fmt in BLOCKS:
        require(
            mode == "per_group" and spec["group_size"] == BLOCKS[fmt],
            f"{fmt} 必须为最后一轴的 {BLOCKS[fmt]} 元素块",
        )


# v2 原契约保留；v3 将 W/A 方法与前置变换分开，不能把新方法塞进旧 rtn 字段。
from .plugin_contracts import plan_schema, WEIGHT_METHODS, ACTIVATION_METHODS
LEGACY_PLAN_SCHEMA = PLAN_SCHEMA
PLUGIN_PLAN_SCHEMA = plan_schema(SPEC)
PLAN_SCHEMA = {"anyOf": [LEGACY_PLAN_SCHEMA, PLUGIN_PLAN_SCHEMA]}


def plan_input_schema(schema):
    """Input may omit mirrors; persisted artifacts still use the full schema."""
    value = deepcopy(schema)
    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            props = node.get("properties", {})
            optional = set()
            if "format" in props and "granularity" in props:
                optional.update(("bits", "axis", "group_size", "scale", "symmetric", "granularity"))
            if "weight" in props and "activation" in props:
                optional.update(("weight_method", "activation_method", "weight"))
                if props.get("method", {}).get("enum") == ["rtn"]:
                    optional.add("method")
            if props.get("unlisted_modules", {}).get("enum") == ["keep_fp"]:
                optional.add("unlisted_modules")
            if props.get("mode", {}).get("enum") == ["reference_qdq"]:
                optional.add("mode")
            if "required" in node:
                node["required"] = [key for key in node["required"] if key not in optional]
            for child in node.values():
                visit(child)
    visit(value)
    return value


PLUGIN_PLAN_INPUT_SCHEMA = plan_input_schema(PLUGIN_PLAN_SCHEMA)
OPTIONAL_SPEC_INPUT = plan_input_schema(OPTIONAL_SPEC)
LOOP_SCHEMA["properties"].update({
    "selection": {"enum": ["first_feasible", "best_accuracy"]},
    "stop_on_goal": {"type": "boolean"},
    "min_top1": {"type": "number", "minimum": 0, "maximum": 100},
    "require_method_documents": {"type": "boolean"},
    "required_bits": obj({t: {"enum": [4, 6, 8, 16, 32]} for t in ("weight", "activation")}, []),
    "required_granularity": obj({t: {"enum": [
        "per_tensor", "per_channel", "per_group", "per_token", "per_block"]}
        for t in ("weight", "activation")}, []),
    "site_bit_overrides": obj({}, [], additional=obj(
        {t: {"enum": [4, 6, 8, 16, 32]} for t in ("weight", "activation")}, [])),
    "quantize_classifier": {"type": "boolean"},
    "max_output_relative_mse": obj({}, [], additional={"type": "number", "minimum": 0}),
    "required_execution": obj({
        "compute_dtype": {"enum": list(RAW_FLOATS)},
        "activation_calibration": {"enum": ["static", "dynamic"]},
    }, []),
    "required_sites": arr(STR, uniqueItems=True),
    "required_site_targets": obj({}, [], additional=arr({"enum": ["weight", "activation"]}, minItems=1, uniqueItems=True)),
    "calibration_cache_mib": {"type": "integer", "minimum": 16, "maximum": 4096},
    "allowed_weight_methods": arr({"enum": list(WEIGHT_METHODS)}, minItems=1, uniqueItems=True),
    "allowed_activation_methods": arr({"enum": list(ACTIVATION_METHODS)}, minItems=1, uniqueItems=True),
    "allowed_preprocess": arr({"enum": ["smoothquant", "repq_ln", "adalog_bias"]}, uniqueItems=True),
})


def validate_execution_requirements(plan, options, *, partial=False):
    """用户固定的执行协议不是 objective 里的软提示，所有执行入口共用。"""
    from .plugin_contracts import site_id
    for key, value in options.get("required_execution", {}).items():
        require(plan["execution"][key] == value, f"违反固定执行要求：{key}={value}")
    # 论文的首层例外只作用于明确的位点/目标，不能因允许 UINT8 而放宽全网。
    # 局部诊断也遵守位宽；可用不同粒度测量差异，但不是正式提交候选。
    for layer in plan["layers"]:
        sid = site_id(layer)
        bits = {**options.get("required_bits", {}),
                **options.get("site_bit_overrides", {}).get(sid, {})}
        for target, expected in bits.items():
            if layer.get(target) is not None:
                require(layer[target]["bits"] == expected,
                        f"违反位点位宽约束：{sid}/{target} 必须为 {expected} 位")
    if partial:
        return  # 免除全网覆盖/部署粒度；static、compute、位宽仍然固定。
    for index, layer in enumerate(plan["layers"]):
        for target, expected in options.get("required_granularity", {}).items():
            if layer.get(target) is not None:
                require(layer[target]["granularity"] == expected,
                        f"违反部署粒度约束：{site_id(layer)}/{target} 必须为 {expected}",
                        path=f"$.layers[{index}].{target}.granularity")
    layers = {site_id(l): l for l in plan["layers"]}
    missing = sorted(set(options.get("required_sites", [])) - set(layers))
    require(not missing, f"遗漏用户要求的量化位点：{missing}", path="$.layers")
    for sid, targets in options.get("required_site_targets", {}).items():
        require(sid in layers and all(layers[sid][t] is not None for t in targets),
                f"遗漏固定量化目标：{sid} {targets}")


def planning_schema(schema, config, *, references=(), drafts=()):
    """向 LLM 展示当前任务合法的格式/执行值，不选择量化算法。

    仅收窄传输 schema；完整程序校验仍执行。不要修改全局 schema，否则不同
    会话的白名单会相互污染。位点级位宽例外仍由实际展开后的计划逐项校验。
    """
    result = deepcopy(schema)
    allowed = set(config["constraints"]["allowed_formats"])

    def visit(node):
        if isinstance(node, list):
            for value in node:
                visit(value)
        elif isinstance(node, dict):
            props = node.get("properties", {})
            # 不让 LLM 重写长哈希；提供本会话已返回的合法句柄，但不替它选证据。
            if references and props.get('evidence', {}).get('type') == 'array':
                props['evidence']['items'] = {'type': 'string', 'enum': sorted(set(references))}
            if drafts and props.get('draft', {}).get('type') == 'string':
                # deepcopy 仍保留 schema 内部共享的 STR 引用；替换字段，避免
                # 把 reason/module 等其它字符串一并限制成 draft 路径。
                props['draft'] = {**props['draft'], 'enum': sorted(set(drafts))}
            if {"format", "bits", "scale"} <= props.keys():
                formats = [f for f in props["format"]["enum"] if f in allowed]
                require(formats, "任务与方案 schema 没有共同可用格式")
                props["format"]["enum"] = formats
                props["bits"]["enum"] = sorted({FORMATS[f] for f in formats})
                if not set(formats).intersection(RAW_FLOATS):
                    props["scale"]["enum"] = ["minmax"]
            if {"mode", "compute_dtype", "activation_calibration"} <= props.keys():
                for key, value in config["loop"].get("required_execution", {}).items():
                    props[key]["enum"] = [value]
            for value in node.values():
                visit(value)

    visit(result)
    return result
