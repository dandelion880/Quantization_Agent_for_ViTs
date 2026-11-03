"""AdaLog 推理 LUT 与独立 bias 重参数化；来源和适配见方法卡。

GoatWu/AdaLog 75a57d2 的 q/r 参数化、LUT 舍入和 underflow 规则保持不变。
搜索由现有输出重建评分器负责，不在这里预置整篇论文的组合配方。
"""

import torch

from ..schemas import require
from .plugin_math import GELU_NEGATIVE_BOUND


def parameters(scale, bits, q=37, shift=0.0):
    index = torch.arange(2 ** bits, device=scale.device, dtype=scale.dtype)
    resolution = 2 ** (bits + 1) - 2
    return {"scale": scale.detach().clone(), "base_q": scale.new_tensor(q),
            "shift": scale.new_tensor(shift), "table1": (index * q / 37).floor(),
            "table2": (2.0 ** (-((index * q) % 37) / 37) * resolution).round() / resolution}


def fit(x, spec, options=None):
    options = options or {}
    shift = options.get("_shift", 0.0)
    require(bool(torch.isfinite(x).all()) and bool((x + shift >= -1e-7).all()),
            "AdaLog 输入不在已声明的非负/shifted-GELU 域")
    return parameters((x + shift).max().clamp_min(1e-12), spec["bits"],
                      options.get("log_base_q", 37), shift)


def apply(x, spec, params=None, options=None):
    p = fit(x, spec, options) if params is None else {k: v.to(x.device) for k, v in params.items()}
    positive = x + p["shift"]
    require(bool((positive >= -1e-7).all()), "AdaLog 出现超出已声明 shift 的负值")
    code = (-(positive / p["scale"]).clamp(1e-15, 1).log2() * 37 / p["base_q"]).round()
    valid = code < 2 ** spec["bits"]
    index = code.clamp(0, 2 ** spec["bits"] - 1).long()
    y = 2.0 ** (-p["table1"][index]) * p["table2"][index] * p["scale"]
    return (y * valid - p["shift"]).to(x.dtype)


def bias_transform(x, weight):
    """只生成输入仿射。最终补偿由 PipelineLinear 使用当前 decoded W 计算。"""
    require(bool((x >= -GELU_NEGATIVE_BOUND - 1e-7).all()), "AdaLog bias 仅支持精确 GELU 的有界负尾")
    scale = weight.new_ones(weight.shape[1])
    return {"scale": scale, "shift": torch.full_like(scale, -GELU_NEGATIVE_BOUND)}, None


def validate_params(params, spec):
    require(set(params) == {"scale", "base_q", "shift", "table1", "table2"}, "AdaLog 参数字段不符")
    require(all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in params.values()), "AdaLog 参数非有限")
    require(all(params[k].numel() == 1 for k in ("scale", "base_q", "shift")), "AdaLog 使用静态标量参数")
    q = params["base_q"].item()
    require(1 <= q <= 256 and q == int(q) and params["scale"].item() > 0, "AdaLog base/scale 非法")
    require(params["shift"].item() in (0.0, float(torch.tensor(GELU_NEGATIVE_BOUND))), "AdaLog shift 非法")
    expected = parameters(params["scale"], spec["bits"], q, params["shift"].item())
    require(all(torch.equal(params[k], expected[k]) for k in ("table1", "table2")), "AdaLog LUT 与 base/bit 不一致")
