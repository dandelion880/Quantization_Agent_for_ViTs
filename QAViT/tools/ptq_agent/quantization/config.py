"""能力声明、候选合法性和逐层展开；不允许静默降级或改变硬约束。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..schemas import JSON, INT, NULL, POS, STR, PTQError, arr, obj, require, validate

SPEC_SCHEMA = obj({
    "format": STR,
    "granularity": {"enum": ["per_tensor", "per_channel", "per_group"]},
    "axis": {"anyOf": [{"type": "integer"}, NULL]},
    "group_size": {"anyOf": [POS, NULL]},
    "symmetric": {"type": "boolean"}, "scale": {"enum": ["minmax"]},
})
OPTIONAL_SPEC = {"anyOf": [NULL, SPEC_SCHEMA]}

def format_capabilities() -> JSON:
    """特殊格式保留明确变体名，绝不映射到 INT4。"""
    formats = {}
    for name in ("INT4", "INT8", "FP8_E4M3FN", "FP8_E5M2", "FP4_E2M1", "NF4", "NVFP4"):
        available = name.startswith("INT")
        formats[name] = {
            "numerical_available": available,
            "test_status": "offline_tests_provided" if available else "not_implemented",
            "targets": ["weight", "activation"] if available else [],
            "granularities": {"weight": ["per_tensor", "per_channel", "per_group"],
                              "activation": ["per_tensor", "per_channel"]} if available else {},
            "parameters": {"scale": ["minmax"], "symmetric": [True, False],
                           "weight_axes": [0, 1], "activation_channel_axis": 1,
                           "group_size": "positive divisor of weight axis"} if available else {},
            "native_backend": False, "current_backend_available": available,
            "reason": "CPU Q/DQ; NumPy float matmul" if available else "未实现编码、校准与后端适配",
        }
    return formats


def validate_spec(spec: JSON, target: str, shape: list[int], path: str) -> None:
    validate(spec, SPEC_SCHEMA, path)
    cap = format_capabilities().get(spec["format"])
    if cap is None or not cap["numerical_available"]:
        raise PTQError("unsupported_format", f"{spec['format']} 未实现数值量化", path=path + ".format", status="unsupported")
    require(spec["granularity"] in cap["granularities"][target],
            f"{target} 不支持 {spec['granularity']}", path + ".granularity", "unsupported_combination")
    axis, size, gran = spec["axis"], spec["group_size"], spec["granularity"]
    if gran == "per_tensor":
        require(axis is None and size is None, "per_tensor 的 axis/group_size 必须为 null", path)
        return
    require(type(axis) is int and 0 <= axis < len(shape), f"axis 必须在 [0,{len(shape) - 1}]", path + ".axis")
    if target == "activation":
        require(axis == 1, "静态激活仅允许 feature axis=1，不能依赖 batch 位置", path + ".axis")
    if gran == "per_channel":
        require(size is None, "per_channel 的 group_size 必须为 null", path + ".group_size")
    else:
        require(type(size) is int and size > 0 and shape[axis] % size == 0,
                "group_size 必须整除该张量轴", path + ".group_size")

