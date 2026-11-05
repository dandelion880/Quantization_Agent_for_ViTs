"""PTQ 插件 v3 契约。预处理、W、A、scale 搜索各有独立语义和组合规则。"""

from copy import deepcopy

from ..schemas import NULL, STR, PTQError, arr, obj, require, validate

WEIGHT_METHODS = ("rtn", "gptq", "uniform_full", "uniform_affine", "uniform_minmax", "adaround", "gptaq")
ACTIVATION_METHODS = ("rtn", "uniform_full", "uniform_affine", "uniform_minmax", "twin_softmax", "log2", "log_sqrt2", "twin_gelu", "asymmetric_gelu", "ptf", "adalog")
PREPROCESS_METHODS = ("smoothquant", "repq_ln", "adalog_bias")
WEIGHT_KINDS = ("linear", "conv2d")
ATTENTION_KINDS = ("attention_query", "attention_key", "attention_value", "post_softmax")
KINDS = (*WEIGHT_KINDS, *ATTENTION_KINDS, "post_gelu", "pre_layernorm", "post_layernorm", "module_output",
         "attention_logits", "attention_residual", "block_output", "embedding_tokens", "position_embedding", "embedding_sum")
METRICS = ("mse", "cosine", "hessian")
CALIBRATION = obj({
    "strategy": {"enum": ["minmax", "search", "manual", "percentile", "coordinate"]},
    "metric": {"enum": list(METRICS)},
    "factors": arr({"type": "number", "minimum": 0}, minItems=1, maxItems=128),
    "factor_grid": obj({"start": {"type": "number", "minimum": 0}, "stop": {"type": "number", "minimum": 0},
                        "count": {"type": "integer", "minimum": 1, "maximum": 128}}),
    "scale_values": arr({"type": "number", "minimum": 0}, minItems=1),
    "zero_values": arr({"type": "number"}, minItems=1),
    "loss": {"enum": ["teacher_kl", "classification_ce"]},
    "percentiles": arr({"type": "number", "minimum": 0.5, "maximum": 1}, minItems=1, maxItems=16),
    "rounds": {"type": "integer", "minimum": 1, "maximum": 16},
    "peer_precision": {"enum": ["quantized", "float"]},
}, ["strategy"])
OPTIONS = obj({
    "blocksize": {"type": "integer", "minimum": 1, "maximum": 1024},
    "damping": {"type": "number", "minimum": 0, "maximum": 1},
    "actorder": {"type": "boolean"},
    "negative_factor": {"type": "number", "minimum": 0},
    "ptf_k": {"type": "integer", "minimum": 0, "maximum": 8},
    "split": {"type": "number", "minimum": 0, "maximum": 1},
    "log_base_q": {"type": "integer", "minimum": 1, "maximum": 256},
    "log_base_candidates": arr({"type": "integer", "minimum": 1, "maximum": 256}, minItems=1, maxItems=128, uniqueItems=True),
    "gptaq_clip": {"type": "boolean"},
    "rounding_grid": {"enum": ["uniform_full", "uniform_affine", "uniform_minmax"]},
    "rounding_iterations": {"type": "integer", "minimum": 1, "maximum": 20000},
    "rounding_batch_size": {"type": "integer", "minimum": 1, "maximum": 32},
    "rounding_lr": {"type": "number", "minimum": 0, "maximum": 1},
    "rounding_reg": {"type": "number", "minimum": 0, "maximum": 1},
    "rounding_warmup": {"type": "number", "minimum": 0, "maximum": 0.9},
    "rounding_seed": {"type": "integer", "minimum": 0},
}, [])
OBJECTIVE_METRICS = ("relative_mse", "mse", "cosine_error", "top1_drop_pp", "weight_bytes_ratio", "native_latency_ms")
OBJECTIVE = obj({
    "weights": obj({k: {"type": "number", "minimum": 0} for k in OBJECTIVE_METRICS}, []),
    "normalizers": obj({k: {"type": "number", "minimum": 0} for k in OBJECTIVE_METRICS}, []),
    "reason": {"type": "string"},
}, ["weights", "normalizers"])


def plan_schema(spec):
    spec = deepcopy(spec)
    spec["properties"]["granularity"]["enum"].append("per_block")
    optional = {"anyOf": [NULL, spec]}
    return obj({
        "schema_version": {"enum": [3]},
        "unlisted_modules": {"enum": ["keep_fp"]},
        "execution": obj({
            "mode": {"enum": ["reference_qdq"]},
            "compute_dtype": {"enum": ["FP32", "FP16", "BF16"]},
            "activation_calibration": {"enum": ["static", "dynamic"]},
            "order": {"enum": ["w_then_a", "a_then_w", "joint"]},
            "within_stage": {"enum": ["parallel", "sequential"]},
            "operator_calibration": {"enum": ["float", "progressive"]},
            "observer_batch_size": {"type": "integer", "minimum": 1},
        }, ["mode", "compute_dtype", "activation_calibration", "order"]),
        "preprocess": arr(obj({
            "method": {"enum": list(PREPROCESS_METHODS)},
            "module": STR,
            "norm": {"anyOf": [STR, NULL]},
            "alpha": {"type": "number", "minimum": 0, "maximum": 1},
            "calibration": CALIBRATION,
            "observation": {"enum": ["float", "quantized_prefix"]},
            "evidence": arr(STR),
        }, ["method", "module", "norm"]), maxItems=512),
        "layers": arr(obj({
            "module": STR, "kind": {"enum": list(KINDS)},
            "weight": optional, "activation": optional,
            "weight_method": {"anyOf": [{"enum": list(WEIGHT_METHODS)}, NULL]},
            "activation_method": {"anyOf": [{"enum": list(ACTIVATION_METHODS)}, NULL]},
            "weight_calibration": CALIBRATION, "activation_calibration": CALIBRATION,
            "options": OPTIONS,
            "reason": {"type": "string"}, "evidence": arr(STR),
        }, ["module", "kind", "weight", "activation", "weight_method", "activation_method"]), minItems=0, maxItems=512),
        "evaluation": OBJECTIVE,
        "operators": arr(obj({"method": {"enum": ["integer_layernorm", "integer_softmax"]}, "module": STR,
                              "input_site": STR, "output_site": STR}), maxItems=128),
        "unresolved": arr(STR, maxItems=32),
    }, ["schema_version", "unlisted_modules", "execution", "preprocess", "layers", "unresolved"])


def site_id(layer):
    return layer["module"] if layer.get("kind", "linear") in WEIGHT_KINDS else layer["module"] + "::" + layer["kind"]


def calibration(layer, target):
    result = {"strategy": "minmax", "metric": "mse", "loss": "teacher_kl"}
    result.update(deepcopy(layer.get(target + "_calibration", {})))
    if "factor_grid" in result:
        require("factors" not in result, "factors 与 factor_grid 不能同时指定")
        grid = result.pop("factor_grid")
        require(0 < grid["start"] < grid["stop"], "factor_grid 需要 0<start<stop")
        result["factors"] = [grid["start"] + i * (grid["stop"] - grid["start"]) / grid["count"] for i in range(grid["count"])]
    if result["strategy"] in ("search", "coordinate"):
        result.setdefault("factors", [0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.2])
    if result["strategy"] == "coordinate":
        result.setdefault("rounds", 3)
        result.setdefault("peer_precision", "quantized")
    if result["strategy"] == "percentile":
        result.setdefault("percentiles", [0.999, 0.9999, 0.99999])
    return result


def _validate_site_spec(validator, spec, target, shape, sid, path, *args, **kwargs):
    """保留原错误类别和门禁，补上实际位点、张量形状及完整字段路径。"""
    try:
        validator(spec, target, shape, *args, **kwargs)
    except PTQError as exc:
        suffix = exc.path.removeprefix("$")
        suffix = suffix.removeprefix("." + target)
        raise PTQError(exc.code, f"{sid}/{target}, shape={shape}: {exc}",
                       path=path + "." + target + suffix,
                       status=exc.status, retryable=exc.retryable) from exc


def validate_composition(plan, known, config, *, classification_available=False, diagnostic=False,
                         allow_unknown_shapes=False):
    """先做无模型执行的组合检查，拒绝重叠位点、非法阶段和未实现的数学组合。"""
    from .contracts import SPEC, RAW_FLOATS, validate_spec
    require(plan["execution"]["mode"] in config["loop"]["execution_modes"], "执行方式不在用户允许范围")
    catalog = known
    protected = set(config["constraints"]["protected_modules"] + config["constraints"]["excluded_modules"])
    scope = config["loop"]["scope_modules"]
    constraints = config["constraints"]
    names = {row["name"] for row in catalog.values()}
    require(set(config["loop"].get("site_bit_overrides", {})) <= set(catalog),
            "位宽例外必须引用当前模型的真实 site_id")
    activation_edges = {catalog[site_id(l)].get("edge"): l["activation"] for l in plan["layers"]
                        if l["activation"] and site_id(l) in catalog}
    require(all(p in catalog or p in names or any(n.startswith(p + ".") for n in names) for p in protected), "保护/排除范围不存在")
    require(all(p in catalog or p in names for p in scope), "量化范围引用未知位点")
    def is_protected(name, sid):
        return any(sid == p or name == p or name.startswith(p + ".") for p in protected)
    seen, linear = set(), {}
    for index, layer in enumerate(plan["layers"]):
        sid, name = site_id(layer), layer["module"]
        path = f"$.layers[{index}]"
        require(sid in catalog, f"未知或未支持位点：{sid}；请查询实际位点目录",
                code="unsupported_site", path=path)
        require(sid not in seen, f"重复位点：{sid}。同一 plan 只能为同一位点指定一次量化；"
                "比较不同方法时，把每种方法放在 plans 数组中的独立 plan，不能作为同一 plan 的多个 layer",
                code="duplicate_site", path=path)
        info = catalog[sid]
        require(info["kind"] == layer["kind"], "位点 kind 与模型实际算子不一致")
        require(not info["protected"] and not is_protected(name, sid), f"受保护位点：{sid}")
        require(not scope or name in scope or sid in scope, f"位点不在量化范围：{sid}")
        seen.add(sid)
        require(layer["weight"] or layer["activation"], "不能提交空量化位点")
        if layer["kind"] in WEIGHT_KINDS:
            linear[name] = layer
        else:
            require(layer["weight"] is None, "纯激活位点不能包含 W 量化")
        for target in ("weight", "activation"):
            spec, method = layer[target], layer[target + "_method"]
            # 概率/GELU/LN 等纯激活位点没有 W；固定 W 格式只约束 Linear。
            # 激活固定格式、Linear 完整覆盖与 fixed_layers 检查仍然保留。
            if constraints["precision_mode"] == "fixed" and (layer["kind"] in WEIGHT_KINDS or target == "activation"):
                actual = spec
                if target == "activation" and actual is None and constraints["fixed_activation_format"] is not None:
                    actual = activation_edges.get(info.get("edge"))
                require((actual["format"] if actual else None) == constraints["fixed_" + target + "_format"], "违反固定 W/A 格式或该输入边缺少量化")
            require((spec is None) == (method is None), f"{target} method/spec 必须同时为空或同时指定")
            if spec is None:
                require(target + "_calibration" not in layer, "空目标不能含 scale 策略")
                continue
            if target == "activation" and spec["granularity"] != "per_tensor":
                require(allow_unknown_shapes or info.get("activation_shape") is not None,
                        f"{sid}: 准备后仍缺少真实激活形状", path=path + ".activation")
            allowed = config["loop"].get("allowed_" + target + "_methods", WEIGHT_METHODS if target == "weight" else ACTIVATION_METHODS)
            require(method in allowed, f"{target} 方法不在用户允许范围")
            require(target in config["loop"]["quantize_targets"], "目标违反用户 W/A 范围")
            require(spec["format"] in config["constraints"]["allowed_formats"], "格式违反用户约束")
            from .contracts import FORMATS
            require(spec["bits"] == FORMATS[spec["format"]], "格式/位宽不一致")
            fixed_format = constraints.get("fixed_" + target + "_format")
            require(fixed_format is None or spec["format"] == fixed_format, "违反固定格式")
            if method in ("uniform_full", "uniform_affine", "uniform_minmax"):
                from . import grids
                shape = info.get("weight_shape") if target == "weight" else info.get("activation_shape")
                _validate_site_spec(grids.validate_spec, spec, target, shape, sid, path)
                require(method == "uniform_minmax" or spec["format"].startswith("INT" if method == "uniform_full" else "UINT"), "uniform 网格有符号性不符")
                require(spec["symmetric"] == spec["format"].startswith("INT"), "uniform 网格 symmetric 不符")
                require(plan["execution"]["activation_calibration"] == "static", "论文参考 uniform 网格必须冻结尺度")
            elif method == "rtn":
                shape = info.get("weight_shape") if target == "weight" else info.get("activation_shape")
                _validate_site_spec(validate_spec, spec, target, shape, sid, path,
                                    plan["execution"]["activation_calibration"],
                                    activation_axis=info.get("activation_channel_axis"),
                                    allow_unknown_shape=allow_unknown_shapes)
            elif method in ("gptq", "gptaq"):
                require(layer["kind"] == "linear",
                        f"{sid}: GPTQ 当前只实现 Linear 权重，实际 kind={layer['kind']}、"
                        f"weight_shape={info.get('weight_shape')}；请把该位点单独分组选择已支持的方法",
                        path=path + ".weight_method", code="unsupported_site")
                require(target == "weight" and spec["format"].startswith(("INT", "UINT")), "GPTQ 当前实现仅支持整数权重")
                require(spec["granularity"] in ("per_channel", "per_group"), "GPTQ 使用输出通道或静态输入组尺度")
                _validate_site_spec(validate_spec, spec, target, info["weight_shape"], sid, path)
                require(calibration(layer, target)["strategy"] == "minmax", "GPTQ 使用原文量化器尺度；不可静默替换为 RTN scale 搜索")
                if method == "gptaq":
                    require(spec["granularity"] == "per_channel", "GPTAQ 当前核对的是官方 per-channel 网格")
                    require(plan["execution"]["order"] == "a_then_w" and plan["execution"].get("within_stage") == "sequential",
                            "GPTAQ 需要已冻结 A 和逐层更新的真实量化输入：使用 a_then_w/sequential")
            elif method == "adaround":
                from . import grids
                require(target == "weight" and layer["kind"] == "linear", "AdaRound 当前已核对 Linear 权重")
                _validate_site_spec(grids.validate_spec, spec, target, info["weight_shape"], sid, path)
                require(spec["granularity"] in ("per_tensor", "per_channel"), "AdaRound 当前支持固定 tensor/channel 网格")
                options = layer.get("options", {})
                require({"rounding_grid", "rounding_iterations"} <= set(options), "AdaRound 必须显式给出冻结网格和迭代预算")
                grid = options["rounding_grid"]
                require(grid == "uniform_minmax" or spec["format"].startswith("INT" if grid == "uniform_full" else "UINT"), "AdaRound 冻结网格有符号性不符")
                require(spec["symmetric"] == spec["format"].startswith("INT"), "AdaRound symmetric 不符")
                require(calibration(layer, target)["strategy"] in ("minmax", "manual", "search"), "AdaRound 先固定 minmax/manual/search 网格再优化舍入")
            else:
                require(target == "activation" and spec["format"] == ("INT" if method in ("twin_gelu", "asymmetric_gelu") else "UINT") + str(spec["bits"]), "特殊激活格式的有符号性/位宽不符")
                require(spec["scale"] == "minmax" and spec["bits"] in (4, 6, 8), "特殊激活仅实现 4/6/8 位的明示网格")
                require(spec["granularity"] == ("per_channel" if method == "ptf" else "per_tensor"), "特殊激活粒度不符")
                require(spec["group_size"] is None, "特殊激活不支持分组")
                require(spec["symmetric"] == (method in ("twin_gelu", "asymmetric_gelu")), "特殊激活 symmetric 字段不符")
                require(plan["execution"]["activation_calibration"] == "static", "特殊激活参数须静态冻结")
                if method in ("twin_softmax", "log2", "log_sqrt2"):
                    require(layer["kind"] == "post_softmax", "softmax 方法只可用于实际 post-softmax")
                elif method in ("twin_gelu", "asymmetric_gelu"):
                    require(info["semantic"] == "post_gelu", "GELU 方法需要已核对的 post-GELU 位点")
                elif method == "ptf":
                    require(info["semantic"] in ("pre_layernorm", "post_layernorm"), "PTF 需要明确 LayerNorm 输入/输出语义")
                    shape = info.get("activation_shape")
                    require(shape is None or spec["axis"] == len(shape) - 1, "PTF 必须使用实际输入的最后 feature 轴")
                elif method == "adalog":
                    require(layer["kind"] == "post_softmax" or info["semantic"] == "post_gelu", "AdaLog 需要已验证的 post-softmax/GELU 语义")
                require(spec["axis"] is None if method != "ptf" else type(spec["axis"]) is int, "特殊激活 axis 不符")
            c = calibration(layer, target)
            provided = layer.get(target + "_calibration", {})
            fields = {"minmax": {"strategy"}, "manual": {"strategy", "scale_values", "zero_values"},
                      "search": {"strategy", "metric", "factors", "factor_grid", "loss"},
                      "percentile": {"strategy", "percentiles"},
                      "coordinate": {"strategy", "metric", "factors", "factor_grid", "loss", "rounds", "peer_precision"}}[c["strategy"]]
            require(set(provided) <= fields,
                    f"{sid}/{target}_calibration: strategy={c['strategy']} 不执行字段"
                    f" {sorted(set(provided) - fields)}；允许字段={sorted(fields)}。"
                    "search倍率用factors/factor_grid，manual绝对尺度才用scale_values；"
                    "rounds/peer_precision仅用于coordinate，不会自动转换策略。",
                    path=path + "." + target + "_calibration")
            require("loss" not in provided or c["metric"] == "hessian", "只有 Hessian metric 使用 loss")
            require(all(v > 0 for v in c.get("factors", [1])), "scale 因子须为正")
            require(all(v > 0 for v in c.get("scale_values", [1])), "scale 须为正")
            if c["strategy"] == "manual":
                require("scale_values" in c, "manual 必须给出真实 scale_values")
            if c["strategy"] == "percentile":
                require(method in ("uniform_affine", "log_sqrt2"), "percentile observer 只支持 affine/log-sqrt2 网格")
            if c["strategy"] == "coordinate":
                require(plan["execution"]["order"] == "joint" and plan["execution"]["activation_calibration"] == "static",
                        "coordinate 需要 joint/static，不能与分离 W/A 阶段冲突")
                require(method in ("uniform_full", "twin_gelu", "twin_softmax"), "coordinate 仅开放已核对的均匀/twin 网格")
                require(layer["kind"] in (*WEIGHT_KINDS, *ATTENTION_KINDS), "coordinate 需要显式 Linear/Conv/MatMul 输出")
                require(c["peer_precision"] == "quantized" or target == "activation", "只有激活搜索允许浮点另一操作数")
            if c["strategy"] != "minmax":
                require(spec["format"] not in RAW_FLOATS, "浮点直转没有可搜索 scale")
                require(spec["format"] not in ("NVFP4", "MXFP4"), "块浮点尺度遵循既有格式约束，当前不开放任意 scale 修改")
            if c["metric"] == "hessian":
                require(c["strategy"] in ("search", "coordinate") and classification_available, "Hessian-guided 需要已接入的分类输出；不得以特征 MSE 或伪梯度替代")
            if target == "activation" and plan["execution"]["activation_calibration"] == "dynamic":
                require(c["strategy"] == "minmax" and method == "rtn", "动态激活仅支持运行时 RTN minmax")
        options = layer.get("options", {})
        legal_options = set()
        if layer["weight_method"] in ("gptq", "gptaq"):
            legal_options.update(("blocksize", "damping", "actorder"))
        if layer["weight_method"] == "gptaq":
            legal_options.add("gptaq_clip")
        if layer["weight_method"] == "adaround":
            legal_options.update(k for k in OPTIONS["properties"] if k.startswith("rounding_"))
            require(options.get("rounding_lr", .001) > 0, "AdaRound 学习率必须为正")
        legal_options.update({"ptf": ("ptf_k",), "twin_softmax": ("split",),
                              "adalog": ("log_base_q", "log_base_candidates"),
                              "twin_gelu": ("negative_factor",), "asymmetric_gelu": ("negative_factor",)}.get(layer["activation_method"], ()))
        require(set(options) <= legal_options, "方法 options 包含当前组合不会执行的参数")
        require(options.get("damping", 0.01) > 0 and options.get("negative_factor", 1) > 0, "damping/negative_factor 须为正")
        require(0 < options.get("split", 0.01) <= 1, "twin softmax split 须在 (0,1]")
        if "log_base_candidates" in options:
            require(calibration(layer, "activation")["strategy"] == "search", "AdaLog base_candidates 需要显式 search 评分")
            require("log_base_q" not in options, "固定 base_q 与 base_candidates 不能同时指定")
    transformed = set()
    require(not (any(p["method"] == "repq_ln" for p in plan["preprocess"]) and
                 any(l["activation_method"] == "ptf" for l in plan["layers"])),
            "当前组合策略将 PTF 与 RepQ-LN 设为方案级互斥，不在同一方案叠加")
    for prep in plan["preprocess"]:
        require(prep["method"] in config["loop"].get("allowed_preprocess", PREPROCESS_METHODS), "预处理不在用户允许范围")
        name = prep["module"]
        require(name in linear and name not in transformed, "预处理目标须在 Linear 计划中，且同一边仅一种重参数化")
        require(linear[name]["kind"] == "linear", "预处理当前只支持 Linear 输入边")
        transformed.add(name)
        layer = linear[name]
        require(layer["activation_method"] not in ("ptf", "twin_gelu", "asymmetric_gelu"), "重参数化改变输入语义，与该特殊网格互斥")
        if prep["method"] == "repq_ln":
            require("alpha" not in prep, "RepQ-LN 不使用 SmoothQuant 的 alpha")
            require(prep["norm"] is not None and layer["activation"] is not None, "RepQ-LN 需指定 LayerNorm 和激活量化")
            require(layer["activation_method"] in ("rtn", "uniform_affine") and layer["activation"]["format"].startswith("UINT"), "RepQ-LN 转为无符号 affine 网格")
            require(layer["activation"]["granularity"] == "per_tensor", "RepQ-LN 最终使用 layer-wise scale")
            require(plan["execution"]["activation_calibration"] == "static", "RepQ-LN 需要冻结静态共享尺度")
            require(prep.get("calibration", {"strategy": "minmax"})["strategy"] in ("minmax", "percentile"), "RepQ observer 只能 minmax/percentile")
            require(set(prep.get("calibration", {})) <= {"strategy", "percentiles"}, "预处理 observer 含未执行参数")
        elif prep["method"] == "adalog_bias":
            require(catalog[name]["semantic"] == "post_gelu" and prep["norm"] is None,
                    "AdaLog bias 需要真实 GELU→Linear 输入边；不能修改相邻 LN")
            require("alpha" not in prep and "calibration" not in prep and prep.get("observation", "float") == "float",
                    "AdaLog bias 使用固定 GELU 负界，不使用 SQ alpha 或 RepQ observer")
        else:
            require("calibration" not in prep, "SmoothQuant 使用原始 absmax，不使用额外 observer")
            require(prep.get("observation", "float") == "float", "SmoothQuant 在 FP 输入上测量")
        if layer["activation_method"] == "adalog":
            require(prep["method"] == "adalog_bias", "AdaLog GELU 语义不能隐式叠加其它输入重参数化")
        edge = catalog[name].get("edge")
        require(not any(l["activation_method"] == "ptf" and
                        (catalog[site_id(l)].get("edge") == edge or edge == l["module"] + ":output")
                        for l in plan["layers"]), "同一 LayerNorm 链不允许 PTF 与重参数化叠加")
        if prep["norm"] is not None:
            require(prep["norm"] + "::post_layernorm" in catalog, "预处理源不是已支持 LayerNorm")
            require(not any(l["module"] == prep["norm"] and l["activation_method"] == "ptf" for l in plan["layers"]), "同一 LayerNorm 链不允许 PTF 与重参数化叠加")
    # GELU 输出与 fc2 输入是同一张量边，禁止重复 QDQ；LN 输出亦然。
    active_edges = [catalog[site_id(l)].get("edge") for l in plan["layers"] if l["activation"]]
    active_edges = [e for e in active_edges if e]
    require(len(active_edges) == len(set(active_edges)), "同一激活边配置了两个量化器")
    for name, fixed in constraints["fixed_layers"].items():
        if diagnostic and name not in linear:
            continue  # 局部诊断不补其它层，但被诊断层的固定规格仍须一致。
        require(name in linear, f"遗漏固定层：{name}")
        for target, required in fixed.items():
            actual = linear[name][target]
            require((actual is None) == (required is None), "固定目标被增删")
            if required is not None:
                require({k: v for k, v in actual.items() if k != "bits"} == required, "固定规格被修改")
    if not diagnostic and constraints["precision_mode"] == "fixed":
        eligible = {n for n, info in catalog.items() if info["kind"] in WEIGHT_KINDS and not info["protected"]
                    and not is_protected(n, n) and (not scope or n in scope)}
        require(set(linear) == eligible, "固定模式必须覆盖全部允许 Linear")
    from .coordinate import operator_id, validate_group
    groups = {}
    for layer in plan["layers"]:
        groups.setdefault(operator_id(layer), []).append(layer)
    for rows in groups.values():
        if any(calibration(l, t)["strategy"] == "coordinate" for l in rows for t in ("weight", "activation") if l[t]):
            validate_group(rows)
    if any(p.get("observation") == "quantized_prefix" for p in plan["preprocess"]):
        require(all(p["method"] == "repq_ln" and p.get("observation") == "quantized_prefix" for p in plan["preprocess"]),
                "同一 preview 只允许声明相同观察协议的 RepQ-LN 边")
        require(plan["execution"]["order"] == "joint" and plan["execution"].get("within_stage") == "sequential",
                "quantized_prefix observer 需要 joint/sequential 校准")
        require(all(l[t + "_method"] in ("uniform_affine", "log_sqrt2") and calibration(l, t)["strategy"] == "percentile"
                    for l in plan["layers"] for t in ("weight", "activation") if l[t]),
                "quantized_prefix preview 当前只核对 affine/log-sqrt2 percentile 初始化")
        require(all(p.get("calibration", {}).get("strategy") == "percentile" for p in plan["preprocess"]),
                "quantized_prefix 的逐通道 observer 必须显式声明 percentile")
    operator_modules = set()
    if "observer_batch_size" in plan["execution"]:
        require(plan["execution"]["activation_calibration"] == "static"
                and plan["execution"]["order"] == "joint"
                and plan["execution"].get("within_stage", "parallel") == "parallel",
                "分批 running observer 需要 joint/parallel 静态校准")
        require(all(calibration(l, t)["strategy"] == "minmax" and l[t + "_method"] in ("uniform_minmax", "ptf", "log2")
                    for l in plan["layers"] for t in ("weight", "activation") if l[t]),
                "分批 running observer 当前只核对 minmax/PTF/log2；不能静默套到其它校准器")
    require(plan["execution"].get("operator_calibration", "float") == "float" or plan.get("operators"),
            "progressive operator 校准需要明确算子声明")
    layers_by_id = {site_id(l): l for l in plan["layers"]}
    for operator in plan.get("operators", []):
        name = operator["module"]
        require(name not in operator_modules, "同一算子被重复替换")
        operator_modules.add(name)
        require(plan["execution"]["activation_calibration"] == "static" and plan["execution"]["order"] == "joint"
                and plan["execution"].get("within_stage", "parallel") == "parallel", "整数参考算子先在 FP 路径校准全部静态尺度")
        require(operator["input_site"] in layers_by_id and operator["output_site"] in layers_by_id, "算子引用未声明的量化位点")
        src, dst = (layers_by_id[operator[key]] for key in ("input_site", "output_site"))
        require(src["activation"] is not None and dst["activation"] is not None, "算子尺度源必须实际量化")
        if operator["method"] == "integer_layernorm":
            require((dst["module"] == name and dst["kind"] == "post_layernorm") or
                    (name == "norm" and dst["module"] == "head" and dst["kind"] == "linear"),
                    "整数 LN 输出尺度须来自其直接输出或已核对的最终 CLS→head 边")
            require(src["activation_method"] == "ptf" or src["activation"]["granularity"] == "per_tensor", "整数 LN 输入尺度须共享或满足 PTF 关系")
            require(dst["activation_method"] in ("uniform_minmax", "uniform_affine", "rtn")
                    and dst["activation"]["format"].startswith("UINT"), "整数 LN 需要明确的 affine 输出网格")
        else:
            require(src["module"] == dst["module"] == name and src["kind"] == "attention_logits" and dst["kind"] == "post_softmax",
                    "整数 softmax 需要同一 attention 的 logits/概率位点")
            require(src["activation"]["granularity"] == "per_tensor" and dst["activation_method"] == "log2"
                    and calibration(dst, "activation")["strategy"] == "minmax", "LIS 需要共享输入尺度和未缩放 log2 输出网格")
    return [{"module": site_id(l), "execution_ready": True} for l in plan["layers"]]
