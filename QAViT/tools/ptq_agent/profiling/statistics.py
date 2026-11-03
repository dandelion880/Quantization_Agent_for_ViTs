"""定向 CPU 统计；详细通道/分组统计保存在证据文件，不直接塞进 LLM 上下文。"""

from __future__ import annotations

import numpy as np
from ..schemas import JSON, require

STATISTICS = ["range", "mean", "std", "rms", "quantiles", "zero_fraction"]
DEFINITION_VERSION = "numpy_float64_stats_v1"


def tensor_statistics(tensor: np.ndarray, statistics: list[str], grouping: str = "tensor",
                      axis: int | None = None, group_size: int | None = None) -> JSON:
    x = np.asarray(tensor, dtype=np.float64)
    require(x.size > 0 and bool(np.isfinite(x).all()), "统计张量必须非空且有限", code="numerical_error")
    require(set(statistics) <= set(STATISTICS), "未知统计项")
    result: JSON = {"shape": list(x.shape), "definition": DEFINITION_VERSION}
    if "range" in statistics:
        result["range"] = [float(x.min()), float(x.max())]
    if "mean" in statistics:
        result["mean"] = float(x.mean())
    if "std" in statistics:
        result["std"] = float(x.std(ddof=0))
    if "rms" in statistics:
        result["rms"] = float(np.sqrt(np.mean(x ** 2)))
    if "quantiles" in statistics:
        levels = [0.0, 0.01, 0.5, 0.99, 1.0]
        result["quantiles"] = {"levels": levels, "values": np.quantile(x, levels, method="linear").tolist()}
    if "zero_fraction" in statistics:
        result["zero_fraction"] = float(np.mean(x == 0))
    require(grouping in ("tensor", "channel", "group"), "未知统计粒度")
    if grouping == "tensor":
        require(axis is None and group_size is None, "tensor 统计不接受 axis/group_size")
    else:
        require(type(axis) is int and 0 <= axis < x.ndim, "统计 axis 越界")
        size = 1 if grouping == "channel" else group_size
        require(grouping != "channel" or group_size is None, "channel 不接受 group_size")
        require(type(size) is int and size > 0 and x.shape[axis] % size == 0, "统计组大小必须整除指定轴")
        grouped = []
        for start in range(0, x.shape[axis], size):
            part = np.take(x, range(start, start + size), axis=axis)
            grouped.append({"start": start, "stop": start + size, "statistics": tensor_statistics(part, statistics)})
        result["grouping"] = {"kind": grouping, "axis": axis, "group_size": size, "groups": grouped}
    return result
