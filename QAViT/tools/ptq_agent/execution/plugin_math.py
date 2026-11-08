"""PTQ 数值插件；原文/代码与适配差异见 docs/PTQ_METHODS.md。

PTQ4ViT twin 网格、FQ-ViT PTF/log2、RepQ log√2 和 GPTQ 更新保持原实现
的舍入及搜索先后约定。这里不训练权重，也不把 fake QDQ 当作整数 kernel。
"""

from copy import deepcopy
from contextlib import contextmanager
import math
import torch

from ..schemas import require
from . import codecs
from . import grids

GELU_NEGATIVE_BOUND = 0.16997124254703522


@contextmanager
def fp32_statistics(device):
    """GPTQ 的 Gram/误差更新使用完整 FP32；在支持 TF32 的 GPU 上也不改变原式。

    仅作用于这段统计，退出（含异常）恢复调用方设置；不改变模型评价协议。
    """
    if torch.device(device).type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def smoothquant(x, weight, *, alpha=0.5):
    """SmoothQuant 原式；返回边仿射参数，LN/fan-out 保持原状态。"""
    channels = x.reshape(-1, x.shape[-1])
    scale = (channels.abs().amax(0).pow(alpha) / weight.abs().amax(0).clamp_min(1e-5).pow(1-alpha)).clamp_min(1e-5)
    return {"scale": scale, "shift": torch.zeros_like(scale)}, None


def repq_ln(x, weight, *, bits, observer=None, channel_params=None):
    """RepQ Eq. 11–14；初始化用 minmax，保留官方可为非整数的 mean(z)。"""
    channels = x.reshape(-1, x.shape[-1])
    if channel_params is not None:
        delta, zero = (channel_params[k].reshape(-1) for k in ("scale", "zero"))
    elif observer and observer["strategy"] == "percentile":
        spec = {"bits": bits, "granularity": "per_channel", "axis": x.ndim - 1, "group_size": None}
        p = grids.fit(x, spec, "uniform_affine", percentiles=observer.get("percentiles", [0.999, 0.9999, 0.99999]))
        delta, zero = (p[k].reshape(-1) for k in ("scale", "zero"))
    else:
        lo, hi = channels.amin(0), channels.amax(0)
        delta = ((hi-lo) / (2 ** bits - 1)).clamp_min(1e-12)
        zero = (-lo / delta).round()
    target_delta, target_zero = delta.mean(), zero.mean()
    scale = delta / target_delta
    shift = -zero * delta / scale + target_zero * target_delta
    return {"scale": scale, "shift": shift}, {"scale": target_delta, "zero": target_zero,
                                             "repq_affine": torch.tensor(True)}


def fit_activation(x, spec, method, options=None):
    options = options or {}
    require(bool(torch.isfinite(x).all()), "校准值非有限", code="numerical_error")
    if method == "rtn":
        return codecs.fit(x, spec)
    if method in grids.METHODS:
        return grids.fit(x, spec, method)
    bits = spec["bits"]
    if method == "twin_softmax":
        require(bool(((x >= 0) & (x <= 1 + 1e-6)).all()), "post-softmax 输入不在概率域")
        return {"split": x.new_tensor(options.get("split", 2.0 ** -8))}
    if method in ("log2", "log_sqrt2"):
        require(bool((x >= 0).all()), "log 量化不接受负数")
        # FQ-ViT log2 无缩放；RepQ log√2 初始尺度为 max，搜索可再调整。
        return {"scale": x.new_tensor(1.0) if method == "log2" else x.max().clamp_min(1e-12)}
    if method in ("twin_gelu", "asymmetric_gelu"):
        qmax = 2 ** (bits - 1)
        positive = x.clamp_min(0).max() / (qmax - 0.5)
        negative = (x.new_tensor(GELU_NEGATIVE_BOUND) if method == "twin_gelu" else -x.clamp_max(0).min()) / qmax
        return {"scale": positive.clamp_min(1e-12),
                "negative_scale": (negative * options.get("negative_factor", 1.0)).clamp_min(1e-12)}
    require(method == "ptf", "未知激活数值插件")
    require(x.ndim >= 2 and spec["axis"] == x.ndim - 1, "PTF 需要最后 feature 轴")
    k = options.get("ptf_k", 3)
    lo, hi = x.min(), x.max()
    full = ((hi - lo) / (2 ** bits - 1)).clamp_min(torch.finfo(torch.float32).eps)
    zero = (-lo / full).round().clamp(0, 2 ** bits - 1)
    base = full / 2 ** k
    # 与官方逐通道 argmin 等价；候选从小到大，平分时保留最小指数。
    # min/max 累积全部样本；可选最后 observer batch 仅用于指数误差评分。
    # 这是校准数据选择，不改变 PTF 的候选网格或舍入。
    score_input = options.get("score_input", x)
    rows = score_input.reshape(-1, x.shape[-1])
    scores = []
    for exponent in range(k + 1):
        scale = base * 2 ** exponent
        q = ((rows / scale + zero).round().clamp(0, 2 ** bits - 1) - zero) * scale
        scores.append((rows - q).square().mean(0))
    exponent = torch.stack(scores).argmin(0)
    return {"base_scale": base, "zero": zero, "exponent": exponent,
            "scale": (base * 2.0 ** exponent).reshape([1] * (x.ndim - 1) + [-1])}


def activation_qdq(x, spec, method, params=None, options=None):
    p = fit_activation(x, spec, method, options) if params is None else {k: v.to(x.device) for k, v in params.items()}
    if method in ("rtn", "uniform_affine") and "repq_affine" in p:
        y = ((x / p["scale"]).round() + p["zero"]).clamp(0, 2 ** spec["bits"] - 1)
        return ((y - p["zero"]) * p["scale"]).to(x.dtype)
    if method in grids.METHODS:
        return grids.apply(x, spec, method, p)
    if method == "rtn":
        return codecs.qdq(x, spec, p)
    bits = spec["bits"]
    if method == "twin_softmax":
        split, qmax = p["split"], 2 ** (bits - 1) - 1
        # 官方 SoSPTQSLBatchingQuantMatMul 的双网格相加，保持其边界行为；不额外归一化。
        high = (x.clamp(min=split.item(), max=1) * qmax).round().clamp(0, qmax) / qmax
        interval = split / qmax
        low = (x.clamp(min=0, max=split.item()) / interval).round().clamp(0, qmax) * interval
        y = high + low
    elif method in ("twin_gelu", "asymmetric_gelu"):
        qmax = 2 ** (bits - 1)
        pos = (x / p["scale"]).round().clamp(0, qmax - 1) * p["scale"]
        neg = (x / p["negative_scale"]).round().clamp(-qmax, 0) * p["negative_scale"]
        y = pos + neg
    elif method in ("log2", "log_sqrt2"):
        require(bool((x >= 0).all()), "log 量化遇到负数")
        multiplier = 1 if method == "log2" else 2
        exponent = torch.round(-torch.log2(x / p["scale"]) * multiplier)
        underflow = exponent >= 2 ** bits - int((options or {}).get("reserve_zero", False))  # x=0 的 +inf 也进入零掩码。
        exponent = exponent.clamp(0, 2 ** bits - 1)
        if method == "log2":
            y = 2.0 ** (-exponent) * p["scale"]
        else:
            odd = (exponent % 2) * (math.sqrt(2) - 1) + 1
            y = 2.0 ** (-torch.ceil(exponent / 2)) * odd * p["scale"]
        y = y.masked_fill(underflow, 0)
    else:
        require(method == "ptf", "未知激活方法")
        y = ((x / p["scale"] + p["zero"]).round().clamp(0, 2 ** bits - 1) - p["zero"]) * p["scale"]
    require(bool(torch.isfinite(y).all()), "激活 QDQ 非有限", code="numerical_error")
    return y.to(x.dtype)


def adjusted_params(params, factor, method, *, manual=None, zero=None):
    p = {k: v.clone() for k, v in params.items()}
    key = "split" if method == "twin_softmax" else "scale"
    require(key in p, "该方法不提供可调 scale")
    if manual is not None:
        values = torch.as_tensor(manual, device=p[key].device, dtype=p[key].dtype)
        require(values.numel() in (1, p[key].numel()), "手动 scale 数量与实际粒度不符")
        p[key] = values.reshape(p[key].shape) if values.numel() == p[key].numel() else torch.ones_like(p[key]) * values[0]
    else:
        p[key] *= factor
    if method == "ptf":
        require(manual is None or len(manual) == 1, "PTF 手动设置只能调整共享尺度，不能破坏 power-of-two 关系")
        if manual is not None:
            p["base_scale"] = p["base_scale"].new_tensor(manual[0])
            p["scale"] = p["base_scale"] * (2.0 ** p["exponent"]).reshape(p["scale"].shape)
        else:
            p["base_scale"] *= factor
    if zero is not None:
        require("zero" in p, "该网格没有 zero point")
        values = torch.as_tensor(zero, device=p["zero"].device, dtype=p["zero"].dtype)
        require(values.numel() in (1, p["zero"].numel()), "手动 zero point 数量不符")
        p["zero"] = values.reshape(p["zero"].shape) if values.numel() == p["zero"].numel() else torch.ones_like(p["zero"]) * values[0]
    require(bool((p[key] > 0).all()) and bool(torch.isfinite(p[key]).all()), "scale 非有限或非正")
    if method == "twin_softmax":
        require(p[key].numel() == 1 and p[key] <= 1, "twin split 越界")
    return p


def score(reference, candidate, metric, gradient=None):
    a, b = reference.double(), candidate.double()
    if metric == "cosine":
        a, b = a.reshape(-1), b.reshape(-1)
        norm = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        return float(1 - (torch.dot(a, b) / norm).clamp(-1, 1)) if norm else float((a - b).square().mean())
    if metric == "hessian":
        require(gradient is not None and gradient.shape == reference.shape, "Hessian-guided 缺少实际输出梯度")
        return float(((b - a) * gradient.double()).square().mean())
    require(metric == "mse", "未知 scale 搜索 metric")
    return float((a - b).square().mean())


def encode_rtn(weight, spec, params=None):
    codes, params = codecs.encode(weight, spec, params)
    return {"spec": deepcopy(spec), "shape": list(weight.shape),
            "data": codes.cpu() if spec["format"] in codecs.RAW_FLOATS else codecs.pack(codes, spec["bits"]),
            "params": {k: v.cpu() for k, v in params.items()}}


def _gptq_grid(w, bits, symmetric):
    """直接适配 IST-DASLab Quantizer.find_params 的 perchannel/minmax 分支。"""
    xmin = torch.minimum(w.min(1).values, torch.zeros(w.shape[0], device=w.device))
    xmax = torch.maximum(w.max(1).values, torch.zeros(w.shape[0], device=w.device))
    if symmetric:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    dead = (xmin == 0) & (xmax == 0)
    xmin, xmax = xmin.masked_fill(dead, -1), xmax.masked_fill(dead, 1)
    scale = (xmax - xmin) / (2 ** bits - 1)
    zero = torch.full_like(scale, 2 ** (bits - 1)) if symmetric else (-xmin / scale).round()
    return scale, zero


def gptq(weight, hessian, spec, options):
    """GPTQ 原始 Cholesky/block error update；输入组静态冻结，支持 act-order。

    来自 IST-DASLab/GPTQ (Apache-2.0) 和本地已验证 gptq_reference.py。
    移除原模型原地写入/CUDA 全局同步；编码记录代替原地赋值，数学顺序不变。
    """
    w, h = weight.detach().float().clone(), hessian.detach().float().clone()
    rows, columns = w.shape
    bits = spec["bits"]
    group = spec["group_size"] if spec["granularity"] == "per_group" else columns
    initial = _gptq_grid(w, bits, spec["symmetric"])
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    groups = [_gptq_grid(w[:, i:i+group], bits, spec["symmetric"]) for i in range(0, columns, group)] if spec["granularity"] == "per_group" else [initial]
    order = torch.argsort(torch.diag(h), descending=True) if options.get("actorder", False) else torch.arange(columns, device=w.device)
    inverse = torch.argsort(order)
    w, h = w[:, order], h[order][:, order]
    damp = options.get("damping", 0.01) * torch.mean(torch.diag(h))
    idx = torch.arange(columns, device=w.device)
    h[idx, idx] += damp
    require(bool(torch.isfinite(h).all()), "GPTQ Hessian 非有限")
    h = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    result = torch.zeros_like(w)
    codes = torch.zeros_like(w, dtype=torch.uint8)
    blocksize = options.get("blocksize", 128)
    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        local = w[:, start:end].clone()
        error = torch.zeros_like(local)
        for j in range(end - start):
            scale, zero = groups[int(order[start+j]) // group]
            q = (local[:, j] / scale).round().add(zero).clamp(0, 2 ** bits - 1)
            decoded = scale * (q - zero)
            result[:, start+j], codes[:, start+j] = decoded, q.to(torch.uint8)
            err = (local[:, j] - decoded) / h[start+j, start+j]
            local[:, j:] -= err[:, None] @ h[start+j, start+j:end][None, :]
            error[:, j] = err
        w[:, end:] -= error @ h[start:end, end:]
    result, codes = result[:, inverse], codes[:, inverse]
    require(bool(torch.isfinite(result).all()), "GPTQ 输出非有限")
    return {"method": "gptq", "spec": deepcopy(spec), "shape": list(weight.shape),
            "data": codecs.pack(codes, bits),
            "params": {"scale": torch.stack([g[0] for g in groups], 1).cpu(),
                       "zero": torch.stack([g[1] for g in groups], 1).cpu()}}, result


def decode_weight(record, device):
    if record.get("method") == "adaround":
        from .adaround import decode
        return decode(record, device)
    if record.get("method") in grids.METHODS:
        return grids.decode(record, device)
    if record.get("method") not in ("gptq", "gptaq"):
        return codecs.decode_weight(record, device)
    spec, shape = record["spec"], record["shape"]
    q = codecs.unpack(record["data"], spec["bits"], shape, device).float()
    group = spec["group_size"] if spec["granularity"] == "per_group" else shape[1]
    s, z = (record["params"][k].to(device).repeat_interleave(group, 1) for k in ("scale", "zero"))
    return (q - z) * s


def validate_weight(record, spec, shape):
    if record.get("method") == "adaround":
        from .adaround import validate_record
        return validate_record(record, spec, shape)
    if record.get("method") in grids.METHODS:
        return grids.validate_record(record, spec, shape)
    if record.get("method") not in ("gptq", "gptaq"):
        return codecs.validate_record(record, spec, shape)
    require(record["spec"] == spec and record["shape"] == shape, "GPTQ 编码规格不匹配")
    group = spec["group_size"] if spec["granularity"] == "per_group" else shape[1]
    require(record["data"].dtype == torch.uint8 and record["data"].numel() == (math.prod(shape) * spec["bits"] + 7) // 8, "GPTQ bit packing 不符")
    require(set(record["params"]) == {"scale", "zero"}, "GPTQ 参数字段不符")
    for k, v in record["params"].items():
        require(tuple(v.shape) == (shape[0], shape[1] // group) and bool(torch.isfinite(v).all()), "GPTQ 参数形状/有限性不符")
    require(bool((record["params"]["scale"] > 0).all()), "GPTQ scale 非正")
    z = record["params"]["zero"]
    require(bool(((z >= 0) & (z <= 2 ** spec["bits"] - 1) & (z == z.round())).all()), "GPTQ zero point 非法")


def validate_activation(params, layer, shape):
    spec, method = layer["activation"], layer["activation_method"]
    if method == "adalog":
        from .adalog import validate_params
        return validate_params(params, spec)
    if method in ("rtn", "uniform_affine") and "repq_affine" in params:
        require(set(params) == {"scale", "zero", "repq_affine"} and params["repq_affine"].dtype == torch.bool and params["repq_affine"].numel() == 1 and params["repq_affine"].item(), "RepQ affine 标记非法")
        require(spec["format"].startswith("UINT") and spec["granularity"] == "per_tensor", "RepQ grid 规格非法")
        require(all(v.numel() == 1 and bool(torch.isfinite(v).all()) for v in params.values()) and params["scale"].item() > 0, "RepQ 参数非法")
        return
    if method in grids.METHODS:
        return grids.validate_params(params, spec, shape, method)
    if method == "rtn":
        return codecs.validate_params(params, spec, shape)
    fields = {"twin_softmax": {"split"}, "log2": {"scale"}, "log_sqrt2": {"scale"},
              "twin_gelu": {"scale", "negative_scale"}, "asymmetric_gelu": {"scale", "negative_scale"},
              "ptf": {"scale", "base_scale", "zero", "exponent"}}[method]
    require(set(params) == fields and all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in params.values()), "激活参数字段/有限性不符")
    for key, value in params.items():
        if "scale" in key or key == "split":
            require(bool((value > 0).all()), "激活 scale 非正")
        if method != "ptf" or key in ("zero", "base_scale"):
            require(value.numel() == 1, "激活参数必须为标量")
    if method == "twin_softmax":
        require(params["split"].item() <= 1, "twin split 非法")
    if method == "ptf":
        exp = params["exponent"]
        require(exp.shape == (shape[-1],) and bool(((exp >= 0) & (exp <= layer.get("options", {}).get("ptf_k", 3)) & (exp == exp.round())).all()), "PTF exponent 非法")
        require(tuple(params["scale"].shape) == tuple([1] * (len(shape)-1) + [shape[-1]]), "PTF scale 形状不符")
        require(torch.equal(params["scale"].reshape(-1), params["base_scale"] * 2.0 ** exp), "PTF power-of-two 关系不符")
        z = params["zero"]
        require(bool(((z >= 0) & (z <= 2 ** spec["bits"] - 1) & (z == z.round())).all()), "PTF zero point 非法")
