"""CPU min/max 整数量化。Q/DQ 后仍使用浮点 NumPy 运算，不是低比特 kernel。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from ..schemas import JSON, PTQError, require


@dataclass
class QuantizedTensor:
    codes: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray
    spec: JSON

    def dequantize(self) -> np.ndarray:
        return (self.codes.astype(np.float64) - self.zero_point) * self.scale


def _finite(x: Any) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64)
    require(data.size > 0 and bool(np.isfinite(data).all()),
            "张量必须非空且全部有限", code="numerical_error")
    return data


def _range(x: np.ndarray, spec: JSON) -> tuple[np.ndarray, np.ndarray]:
    """channel axis 保留该轴，其余轴归约；group 沿指定轴分组，其他坐标独立。"""
    granularity = spec["granularity"]
    if granularity == "per_tensor":
        return np.asarray(x.min()), np.asarray(x.max())
    axis = spec["axis"]
    if not isinstance(axis, int) or isinstance(axis, bool) or not -x.ndim <= axis < x.ndim:
        raise PTQError("invalid_axis", f"axis={axis} 不适用于 {x.shape}")
    axis %= x.ndim
    if granularity == "per_channel":
        reduce = tuple(a for a in range(x.ndim) if a != axis)
        return x.min(axis=reduce, keepdims=True), x.max(axis=reduce, keepdims=True)
    if granularity == "per_group":
        size = spec["group_size"]
        require(type(size) is int and size > 0 and x.shape[axis] % size == 0,
                "group_size 必须整除指定轴长度")
        moved = np.moveaxis(x, axis, -1)
        grouped = moved.reshape(*moved.shape[:-1], moved.shape[-1] // size, size)
        low = np.repeat(grouped.min(axis=-1), size, axis=-1)
        high = np.repeat(grouped.max(axis=-1), size, axis=-1)
        return np.moveaxis(low, -1, axis), np.moveaxis(high, -1, axis)
    raise PTQError("unsupported_granularity", granularity, status="unsupported")


def quantize(x: Any, spec: JSON, *, reference: Any | None = None) -> QuantizedTensor:
    """reference 用于静态激活校准；推理数据不能重新拟合 scale。"""
    data = _finite(x)
    fmt = spec.get("format")
    if fmt not in ("INT4", "INT8"):
        raise PTQError("unsupported_format", f"{fmt} 没有数值实现", status="unsupported")
    require(spec.get("scale") == "minmax", "只实现 minmax 缩放")
    bits = int(fmt[3:])
    low, high = _range(data if reference is None else _finite(reference), spec)
    if spec["symmetric"]:
        # 对称窄范围 [-7,7] / [-127,127]；0 精确表示，round 使用 ties-to-even。
        qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1) - 1)
        scale = np.maximum(np.abs(low), np.abs(high)) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        zero = np.zeros_like(scale)
    else:
        qmin, qmax = 0, 2 ** bits - 1
        low, high = np.minimum(low, 0), np.maximum(high, 0)
        scale = (high - low) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        zero = np.clip(np.rint(-low / scale), qmin, qmax)
    require(bool(np.isfinite(scale).all()) and bool((scale > 0).all()),
            "缩放因子溢出或下溢", code="numerical_error")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            codes = np.clip(np.rint(data / scale) + zero, qmin, qmax).astype(np.int16)
    except (FloatingPointError, ValueError) as exc:
        raise PTQError("numerical_error", f"量化失败: {exc}") from exc
    return QuantizedTensor(codes, np.asarray(scale), np.asarray(zero), dict(spec))
