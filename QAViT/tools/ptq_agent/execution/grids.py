"""独立的完整有符号/affine 网格与 percentile observer。

完整有符号网格沿用 PTQ4ViT 的 [-2^(b-1), 2^(b-1)-1]、amax/(qmax-.5)。
affine/percentile 沿用 RepQ 的 round(x/delta)+zero，不把 zero 强行截到编码范围。
它们不选择模型位点、不调用其它量化插件；组合由执行器负责。
"""

from copy import deepcopy
import math
import numpy as np
import torch

from ..schemas import require
from . import codecs

METHODS = ("uniform_full", "uniform_affine", "uniform_minmax")


def quantile(x, q):
    """沿用 RepQ 的大张量 NumPy 回退；不抽样，不合并分块分位数。"""
    flat = x.reshape(-1)
    if flat.numel() > 2 ** 24:
        # torch.quantile 的元素数上限会挡住 DeiT-B 的完整32图 GELU observer。
        # 仅回退已知尺寸限制，不吞掉 OOM 等其它错误；小输入保持原 torch 数值。
        value = np.percentile(flat.detach().cpu().numpy(), q * 100, method="linear")
        return x.new_tensor(value)
    return torch.quantile(flat, q)


def layout(x, spec):
    if spec["granularity"] == "per_block":
        # 连续输出行分块，可表达 Q/K/V 的三个独立权重尺度。
        size = spec["group_size"]
        return x.reshape(x.shape[0] // size, size, *x.shape[1:]), 0
    return x, spec["axis"] if spec["granularity"] == "per_channel" else None


def parameter_shape(shape, spec):
    if spec["granularity"] == "per_tensor":
        return ()
    if spec["granularity"] == "per_block":
        return (shape[0] // spec["group_size"],) + (1,) * len(shape)
    return tuple(n if i == spec["axis"] else 1 for i, n in enumerate(shape))


def validate_spec(spec, target, shape=None):
    require(spec["format"].startswith(("INT", "UINT")),
            f"{target} uniform 网格只接受 INT/UINT，当前为 {spec['format']}", path=f"$.{target}.format")
    require(spec["scale"] == "minmax",
            f"{target}/{spec['format']} 的 spec.scale 必须为 minmax，当前为 {spec['scale']!r}",
            path=f"$.{target}.scale")
    mode, axis, size = spec["granularity"], spec["axis"], spec["group_size"]
    require(mode in ("per_tensor", "per_channel", "per_block"), "该静态网格不支持 token/输入组尺度")
    if mode == "per_tensor":
        require(axis is None and size is None, "per_tensor 不使用 axis/group_size")
    elif mode == "per_channel":
        require(type(axis) is int and axis >= 0 and size is None, "per_channel 需要真实通道轴")
        require(target != "weight" or axis == 0, "权重通道轴必须为输出轴 0")
        require(shape is None or axis < len(shape), "通道轴越界")
        require(target != "activation" or axis != 0, "静态激活尺度不能绑定 batch 位置")
    else:
        require(target == "weight" and axis == 0 and type(size) is int and size > 0,
                "per_block 仅表示沿权重输出行的连续块")
        require(shape is None or shape[0] % size == 0, "输出块必须整除权重输出行")


def fit(x, spec, method, *, percentiles=None):
    require(method in METHODS and bool(torch.isfinite(x).all()), "网格/输入非法")
    x, axis = layout(x.detach().float(), spec)
    dims = tuple(i for i in range(x.ndim) if i != axis)
    if method == "uniform_minmax":
        require(percentiles is None, "minmax observer 不接受 percentile")
        lo, hi = (x.min(), x.max()) if axis is None else (x.amin(dims, keepdim=True), x.amax(dims, keepdim=True))
        if spec["symmetric"]:
            scale = (torch.maximum(-lo, hi) / ((2 ** spec["bits"] - 1) / 2)).clamp_min(torch.finfo(x.dtype).eps)
            return {"scale": scale, "zero": torch.zeros_like(scale, dtype=torch.int64)}
        scale = ((hi - lo) / (2 ** spec["bits"] - 1)).clamp_min(torch.finfo(x.dtype).eps)
        return {"scale": scale, "zero": (-torch.round(lo / scale)).clamp(0, 2 ** spec["bits"] - 1)}
    if method == "uniform_full":
        require(percentiles is None, "完整有符号网格不隐式改用 affine percentile")
        scale = x.abs().amax() if axis is None else x.abs().amax(dims, keepdim=True)
        scale = scale / (2 ** (spec["bits"] - 1) - .5)
        # 仅全零退化区间使用 eps；保留正的 sub-epsilon 候选。
        return {"scale": torch.where(scale == 0, torch.full_like(scale, torch.finfo(x.dtype).eps), scale)}
    if percentiles is not None:
        rows = x.reshape(1, -1) if axis is None else x.movedim(axis, 0).reshape(x.shape[axis], -1)
        selected = []
        for row in rows:
            best, params = None, None
            for percentile in percentiles:
                lo, hi = quantile(row, 1 - percentile), quantile(row, percentile)
                p = affine_params(lo, hi, spec["bits"])
                error = (affine_qdq(row, spec["bits"], p) - row).abs().pow(2).mean()
                if best is None or error < best:
                    best, params = error, p
            selected.append(params)
        shape = () if axis is None else tuple(n if i == axis else 1 for i, n in enumerate(x.shape))
        return {key: torch.stack([p[key] for p in selected]).reshape(shape) for key in ("scale", "zero")}
    lo, hi = (x.min(), x.max()) if axis is None else (x.amin(dims, keepdim=True), x.amax(dims, keepdim=True))
    return affine_params(lo, hi, spec["bits"])


def affine_params(lo, hi, bits):
    require(bool(torch.isfinite(lo).all()) and bool(torch.isfinite(hi).all()) and bool((hi >= lo).all()),
            "affine observer 遇到非有限/反向区间")
    # Defined constant-channel extension: include zero only for degenerate ranges.
    # Nondegenerate upstream affine arithmetic is unchanged.
    dead = hi == lo
    lo = torch.where(dead, torch.minimum(lo, torch.zeros_like(lo)), lo)
    hi = torch.where(dead, torch.maximum(hi, torch.zeros_like(hi)), hi)
    scale = (hi - lo) / (2 ** bits - 1)
    scale = torch.where(scale == 0, torch.full_like(scale, torch.finfo(scale.dtype).eps), scale)
    return {"scale": scale, "zero": (-lo / scale).round()}


def affine_qdq(x, bits, params):
    code = (torch.round(x / params["scale"]) + params["zero"]).clamp(0, 2 ** bits - 1)
    return (code - params["zero"]) * params["scale"]


def apply(x, spec, method, params=None, options=None):
    params = fit(x, spec, method) if params is None else {k: v.to(x.device) for k, v in params.items()}
    value, _ = layout(x, spec)
    if method == "uniform_minmax":
        qmin = -2 ** (spec["bits"] - 1) if spec["symmetric"] else 0
        result = ((value / params["scale"] + params["zero"]).round().clamp(qmin, qmin + 2 ** spec["bits"] - 1) - params["zero"]) * params["scale"]
    elif method == "uniform_full":
        qmax = 2 ** (spec["bits"] - 1)
        result = (value / params["scale"]).round().clamp(-qmax, qmax - 1) * params["scale"]
    else:
        require(method == "uniform_affine", "未知网格")
        result = affine_qdq(value, spec["bits"], params)
    require(bool(torch.isfinite(result).all()), "uniform QDQ 输出非有限")
    return result.reshape_as(x).to(x.dtype)


def encode(weight, spec, params=None, *, method):
    params = fit(weight, spec, method) if params is None else params
    x, _ = layout(weight.detach().float(), spec)
    p = {k: v.to(x.device) for k, v in params.items()}
    if method == "uniform_minmax":
        qmin = -2 ** (spec["bits"] - 1) if spec["symmetric"] else 0
        codes = (x / p["scale"] + p["zero"]).round().clamp(qmin, qmin + 2 ** spec["bits"] - 1) - qmin
    elif method == "uniform_full":
        qmax = 2 ** (spec["bits"] - 1)
        codes = (x / p["scale"]).round().clamp(-qmax, qmax - 1) + qmax
    else:
        codes = ((x / p["scale"]).round() + p["zero"]).clamp(0, 2 ** spec["bits"] - 1)
    return {"method": method, "spec": deepcopy(spec), "shape": list(weight.shape),
            "data": codecs.pack(codes.reshape_as(weight).to(torch.uint8), spec["bits"]),
            "params": {k: v.detach().cpu() for k, v in p.items()}}


def decode(record, device):
    spec = record["spec"]
    codes = codecs.unpack(record["data"], spec["bits"], record["shape"], device).float()
    value, _ = layout(codes, spec)
    p = {k: v.to(device) for k, v in record["params"].items()}
    if record["method"] == "uniform_minmax":
        qmin = -2 ** (spec["bits"] - 1) if spec["symmetric"] else 0
        return ((value + qmin - p["zero"]) * p["scale"]).reshape(record["shape"])
    zero = 2 ** (spec["bits"] - 1) if record["method"] == "uniform_full" else p["zero"]
    return ((value - zero) * p["scale"]).reshape(record["shape"])


def validate_params(params, spec, shape, method):
    require(set(params) == ({"scale"} if method == "uniform_full" else {"scale", "zero"}), "uniform 参数字段不符")
    expected = parameter_shape(shape, spec)
    for value in params.values():
        require(tuple(value.shape) == expected and bool(torch.isfinite(value).all()), "uniform 参数形状/有限性不符")
    require(bool((params["scale"] > 0).all()), "uniform scale 非正")
    if "zero" in params:
        require(bool((params["zero"] == params["zero"].round()).all()), "affine 原始 observer 的 zero 必须为整数")
        if method == "uniform_minmax":
            z = params["zero"]
            require(bool((z == 0).all()) if spec["symmetric"] else bool(((z >= 0) & (z <= 2 ** spec["bits"] - 1)).all()),
                    "minmax observer 的 zero 与 signed/unsigned 网格不符")


def validate_record(record, spec, shape):
    require(record["spec"] == spec and record["shape"] == shape, "uniform 权重规格不符")
    require(record["data"].dtype == torch.uint8 and record["data"].numel() == (math.prod(shape) * spec["bits"] + 7) // 8,
            "uniform packed 权重长度不符")
    validate_params(record["params"], spec, shape, record["method"])


def percentile_log(x, spec, apply_fn, percentiles):
    """observer 与 log 编解码独立；严格按候选顺序和平分保留首项。"""
    best, params = None, None
    for percentile in percentiles:
        scale = quantile(x, percentile)
        require(bool(torch.isfinite(scale)) and scale > 0, "log percentile scale 非正")
        candidate = {"scale": scale}
        error = (apply_fn(x, spec, params=candidate) - x).abs().pow(2).mean()
        if best is None or error < best:
            best, params = error, candidate
    return params
