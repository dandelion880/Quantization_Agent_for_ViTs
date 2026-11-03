"""可选的多目标评分；用户硬门槛独立保留，评分策略在第一次试验前冻结。"""

from dataclasses import dataclass
from typing import Callable
import math

from ..schemas import require, validate
from .plugin_contracts import OBJECTIVE


@dataclass(frozen=True)
class NativeKernelAdapter:
    """未来真实 kernel 的显式扩展点，不能用 reference 计时填写 native 指标。

    supports(plan, hardware)->bool；build(artifact, device)->可执行模型；
    benchmark(model, protocol)->包含原生算子证据、协议及 latency_ms 的结构化结果。
    注册后仍需接入 native 执行契约和等价测试；当前 v3 只开放 reference_qdq。
    """
    version: str
    supports: Callable
    build: Callable
    benchmark: Callable


NATIVE_KERNELS = {}


def register_native_kernel(name, adapter):
    require(name not in NATIVE_KERNELS and isinstance(adapter, NativeKernelAdapter), "kernel 重复或接口不符")
    NATIVE_KERNELS[name] = adapter


def validate_objective(policy, config, *, native_available, classification_available=None):
    if policy is None:
        return
    validate(policy, OBJECTIVE)
    weights, normalizers = policy["weights"], policy["normalizers"]
    selected = {k for k, v in weights.items() if v > 0}
    if config["loop"].get("selection") == "best_accuracy":
        require(selected == {"top1_drop_pp"},
                "best_accuracy 只允许 top1_drop_pp 为正权重，且须给出同名的正 normalizer；"
                "不能使用空 weights 或加入 MSE/存储/延迟。evaluation=null 可沿用默认准确率策略")
    require(selected and set(normalizers) == selected, "正权重指标必须逐项给出且仅给出归一化常数")
    require(all(v > 0 and math.isfinite(v) for v in normalizers.values()), "评价归一化常数必须有限且为正")
    classification_available = config["loop"]["evaluation"] == "classification" if classification_available is None else classification_available
    require("top1_drop_pp" not in selected or classification_available, "没有下游 head/标签协议，不能选择 Top-1 权重")
    require("native_latency_ms" not in selected or native_available,
            "未测得原生 kernel 延迟；不能用 fake QDQ 耗时代替加速评价")


def weighted_score(metrics, policy):
    if policy is None:
        return None
    components = {}
    total = sum(policy["weights"].values())
    for key, weight in policy["weights"].items():
        if weight == 0:
            continue
        value = metrics.get(key)
        require(value is not None and math.isfinite(value), f"所选评价指标缺失：{key}")
        components[key] = {"raw": value, "weight": weight / total,
                           "normalizer": policy["normalizers"][key],
                           "contribution": weight / total * value / policy["normalizers"][key]}
    return {"value": sum(x["contribution"] for x in components.values()), "direction": "minimize",
            "components": components, "policy": policy,
            "scope": "comparison/feedback only; cannot override hard gates"}
