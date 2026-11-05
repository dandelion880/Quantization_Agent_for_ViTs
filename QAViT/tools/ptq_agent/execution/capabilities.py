"""实测能力、位点合法性和资源预估；声明与执行共用此门禁。"""

import shutil

from ..schemas import require, validate, fingerprint
from ..profiling.contracts import linear_forward_supported
from .contracts import FORMATS, RAW_FLOATS, BLOCKS, PLAN_SCHEMA, validate_spec, validate_execution_requirements
from .methods import get_method, versions, plugin_versions


def detect(backend):
    import torch

    device = torch.device(backend.device)
    dtype_results = {}
    for name, dtype in (("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)):
        native = name == "FP32" or (
            device.type == "cuda" and (name == "FP16" or torch.cuda.get_device_capability(device)[0] >= 8)
        )
        try:
            x = torch.ones(8, 16, device=device, dtype=dtype)
            y = torch.nn.functional.linear(x, torch.ones(8, 16, device=device, dtype=dtype))
            finite = bool(torch.isfinite(y).all())
            dtype_results[name] = {"forward_available": finite, "native_matmul": native and finite}
        except (RuntimeError, TypeError) as exc:
            dtype_results[name] = {"forward_available": False, "native_matmul": False, "reason": str(exc)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        info = {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "free_bytes": free,
            "total_bytes": total,
        }
    else:
        import os

        info = {
            "device": str(device),
            "name": "CPU",
            "compute_capability": None,
            "free_bytes": os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            "total_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        }
    return {
        "hardware": info,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "formats": {
            fmt: {
                "bits": bits,
                "reference_qdq": True,
                "native": fmt in RAW_FLOATS and dtype_results[fmt]["native_matmul"],
                "required_group_size": BLOCKS.get(fmt),
            }
            for fmt, bits in FORMATS.items()
        },
        "compute_dtypes": dtype_results,
        "methods": {name: {"version": version, "implemented": True} for name, version in versions().items()},
        "sites": "v2 verified Linear; v3 adds explicit post-softmax/GELU/pre-LN/post-LN sites via query_ptq_plugins/query_ptq_sites",
        "plan_versions": [2, 3],
        "plugin_versions": plugin_versions(),
        "weight_plugins": list(plugin_versions()["weight"]),
        "activation_plugins": list(plugin_versions()["activation"]),
        "preprocess_plugins": list(plugin_versions()["preprocess"]),
        "storage": "packed 4/6/8-bit weights plus explicit scales; reference caches decoded weights at runtime",
        "native_note": "仅已实现浮点 Linear 路径；未登记的 INT/FP8/FP4 kernel 均不推断可用",
    }


def validate_plan(backend, plan, config, capabilities, shapes=None, *, diagnostic=False,
                  allow_unknown_shapes=False):
    from .contracts import normalize_plan
    plan = normalize_plan(plan)
    validate(plan, PLAN_SCHEMA)
    if not diagnostic:
        validate_execution_requirements(plan, config["loop"])
    if plan.get("schema_version") == 3:
        if diagnostic:
            validate_execution_requirements(plan, config["loop"], partial=True)
        from .plugin_contracts import validate_composition
        from .sites import site_catalog
        from .objectives import validate_objective
        validate(plan, PLAN_SCHEMA)
        catalog = site_catalog(backend, config)
        for name, shape in (shapes or {}).items():
            if name in catalog:
                catalog[name]["activation_shape"] = shape
        classification_available = (backend.spec.get("classification_head") == "official pretrained ImageNet-1k head retained"
                                    or bool(getattr(backend, "task_head", None)))
        rows = validate_composition(plan, catalog, config, classification_available=classification_available,
                                    diagnostic=diagnostic, allow_unknown_shapes=allow_unknown_shapes)
        dtype = plan["execution"]["compute_dtype"]
        require(capabilities["compute_dtypes"][dtype]["forward_available"], "设备不能执行该计算类型")
        validate_objective(plan.get("evaluation"), config, native_available=False, classification_available=classification_available)
        return rows
    validate(plan, PLAN_SCHEMA)
    modes = config["loop"]["execution_modes"]
    execution, constraints = plan["execution"], config["constraints"]
    require(diagnostic or execution["mode"] in modes, "执行模式不在用户允许范围", code="unsupported_execution")
    require(
        capabilities["compute_dtypes"][execution["compute_dtype"]]["forward_available"],
        "计算 dtype 未通过设备 forward 检测",
        code="unsupported_execution",
    )
    known = {m["name"]: m for m in backend.modules()}
    from .sites import site_catalog
    sites = site_catalog(backend)
    scope_names = set(sites) | {m["name"] for m in sites.values()}
    protected = set(constraints["protected_modules"] + constraints["excluded_modules"])
    require(all(p in sites or p in dict(backend.model.named_modules()) for p in protected), "保护/排除模块名称不存在")
    scope = config["loop"]["scope_modules"]
    # v2 RTN 局部诊断仍可在 v3 的混合范围中使用；范围中的 attention 不等于本次量化它。
    require(set(scope) <= scope_names, "量化范围引用未知位点")
    def is_protected(name):
        return any(name == p or name.startswith(p + ".") for p in protected)
    rows, seen = [], set()
    for layer in plan["layers"]:
        get_method(layer["method"])
        for target in ("weight", "activation"):
            if layer[target]:
                require(layer["method"] in config["loop"].get("allowed_" + target + "_methods", ["rtn"]), "v2 方法违反用户白名单")
        name = layer["module"]
        require(name in known and name not in seen, "位点未知或重复", code="invalid_site")
        info = known[name]
        require(
            not info["protected"] and not is_protected(name) and (not scope or name in scope),
            f"位点 {name} 违反保护/量化范围；必须匹配 scope_modules 中的精确位点",
            code="protected_site",
        )
        require(
            linear_forward_supported(backend.model.get_submodule(name)),
            "该位点 forward 语义尚未接入",
            code="unsupported_site",
        )
        require(layer["weight"] or layer["activation"], "空量化位点")
        seen.add(name)
        for target in ("weight", "activation"):
            spec = layer[target]
            if spec:
                require(target in config["loop"]["quantize_targets"], "目标张量违反用户 W/A 范围")
                require(spec["format"] in constraints["allowed_formats"], "格式违反用户约束")
                shape = info["weight_shape"] if target == "weight" else (shapes or {}).get(name)
                validate_spec(spec, target, shape, execution["activation_calibration"],
                              allow_unknown_shape=allow_unknown_shapes)
            if constraints["precision_mode"] == "fixed":
                require(
                    (spec["format"] if spec else None) == constraints[f"fixed_{target}_format"],
                    "违反固定 W/A 格式",
                )
            fixed = constraints["fixed_layers"].get(name, {})
            if target in fixed:
                expected = fixed[target]
                plain = {k: v for k, v in spec.items() if k != "bits"} if spec else None
                require(plain == expected, "违反固定层规格")
        if execution["mode"] == "native":
            dtype = execution["compute_dtype"]
            require(
                capabilities["compute_dtypes"][dtype]["native_matmul"]
                and all(
                    layer[t] is not None and layer[t]["format"] == dtype for t in ("weight", "activation")
                ),
                "native 目前要求 W/A/compute 为相同的已验证浮点格式；没有低比特 kernel 回退",
                code="unsupported_execution",
            )
        rows.append(
            {
                "module": name,
                "weight_site": name + ".weight" if layer["weight"] else None,
                "activation_site": name + ".input" if layer["activation"] else None,
                "execution_ready": True,
            }
        )
    if not diagnostic and constraints["precision_mode"] == "fixed":
        eligible = {
            n
            for n, m in known.items()
            if not m["protected"] and not is_protected(n) and (not scope or n in scope)
        }
        require(seen == eligible, "固定模式必须覆盖全部允许位点")
    for name, fixed in constraints["fixed_layers"].items():
        require(name in known and (diagnostic or name in seen or all(x is None for x in fixed.values())), "遗漏固定层")
    return rows


def plan_key(plan):
    """自然语言理由/引用的变化不构成新的数值实验，避免重复候选绕过预算。"""
    if plan.get("schema_version") == 3:
        from .plugin_contracts import site_id, calibration
        layers = []
        for layer in plan["layers"]:
            item = {k: v for k, v in layer.items() if k not in ("reason", "evidence")}
            for target in ("weight", "activation"):
                if layer[target]:
                    item[target + "_calibration"] = calibration(layer, target)
            layers.append(item)
        # 省略默认值不是新实验；与pipeline实际默认值一致，不改提交的plan。
        execution = {"within_stage": "parallel", "operator_calibration": "float", **plan["execution"]}
        preprocess = []
        for prep in plan["preprocess"]:
            item = {k: v for k, v in prep.items() if k != "evidence"}
            item.setdefault("observation", "float")
            if prep["method"] == "smoothquant":
                item.setdefault("alpha", 0.5)
            elif prep["method"] == "repq_ln":
                item["calibration"] = calibration({"activation_calibration": prep.get("calibration", {})}, "activation")
            preprocess.append(item)
        return fingerprint({"execution": execution, "operators": plan.get("operators", []),
                            "layers": sorted(layers, key=site_id), "preprocess": preprocess})
    return fingerprint(
        {
            "execution": plan["execution"],
            "unlisted_modules": plan["unlisted_modules"],
            "layers": sorted(
                [{k: l[k] for k in ("module", "method", "weight", "activation")} for l in plan["layers"]],
                key=lambda x: x["module"],
            ),
        }
    )


def estimate(backend, plan):
    original = sum(t.numel() * t.element_size() for t in backend.model.state_dict().values())
    packed = original
    for layer in plan["layers"]:
        if layer["weight"]:
            w = backend.model.get_submodule(layer["module"]).weight
            packed += (w.numel() * layer["weight"]["bits"] + 7) // 8 - w.numel() * w.element_size()
    return {
        "original_state_bytes": original,
        "packed_weights_lower_bound_bytes": packed,
        "candidate_resident_lower_bound_bytes": original,
        "working_memory_estimate_bytes": original * 4,
        "note": "尺度、临时激活与 allocator 开销另计；reference 会缓存解码权重，不承诺低比特显存或加速",
    }


def check_resources(backend, plan, config, directory):
    report = estimate(backend, plan)
    if plan.get("schema_version") == 3:
        cache = config["loop"].get("calibration_cache_mib", 256) * 1024 ** 2
        hessian = max((backend.model.get_submodule(l["module"]).in_features ** 2 * 4 *
                       (8 if l["weight_method"] == "gptaq" else 4)
                       for l in plan["layers"] if l.get("weight_method") in ("gptq", "gptaq")), default=0)
        rounding = max((backend.model.get_submodule(l["module"]).weight.numel() * 4 * 10
                        for l in plan["layers"] if l.get("weight_method") == "adaround"), default=0)
        report["calibration_cache_limit_bytes"] = cache
        report["gptq_matrix_workspace_estimate_bytes"] = hessian
        report["rounding_workspace_estimate_bytes"] = rounding
        report["working_memory_estimate_bytes"] += max(hessian, rounding) + cache * 2
    minimum = config["loop"]["min_free_disk_mib"] * 1024**2
    free = shutil.disk_usage(directory).free
    required = minimum + report["original_state_bytes"] * 2
    require(
        free > required,
        f"磁盘余量不足以原子导出与重载：可用 {free / 1024**2:.1f} MiB，"
        f"要求大于 {required / 1024**2:.1f} MiB（含保留量 {minimum / 1024**2:.1f} MiB）；请清理无用产物",
        code="disk_budget",
    )
    used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    require(
        used + report["original_state_bytes"] * 2 <= config["loop"]["disk_budget_mib"] * 1024**2,
        "本运行磁盘预算不足；不会删除历史运行或自动扩大预算",
        code="disk_budget",
    )
    cap = detect(backend)
    require(
        cap["hardware"]["free_bytes"] > report["working_memory_estimate_bytes"],
        "当前设备可用内存不足以构建和验证候选",
        code="memory_budget",
    )
    return report
