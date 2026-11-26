"""Software integer-accumulator reference for channel-affine folding and signed requantization."""

import math
from types import MethodType
import torch
import torch.nn.functional as F
from ..schemas import require

FIELDS = {
    "multiplier",
    "acc_bias",
    "constant_code",
    "is_constant",
    "output_scale",
    "output_zero",
    "output_bits",
    "shift",
    "sx",
    "sw",
    "acc_bits",
}


def compile_state(q, alpha, beta, cfg):
    spec = q.layer
    ap = q.params()
    record = q.weight_record
    require(
        spec["activation_method"]
        in ("rtn", "uniform_affine", "uniform_minmax", "uniform_full")
        and spec["activation"]["granularity"] == "per_tensor"
        and record["method"]
        in ("uniform_affine", "uniform_minmax", "uniform_full", "rtn")
        and spec["weight"]["granularity"] in ("per_channel", "per_tensor"),
        "integer affine folding requires scalar X units and scalar/per-output W units",
    )
    require(
        all(
            "zero" not in p or torch.equal(p["zero"], p["zero"].round())
            for p in (ap, record["params"])
        ),
        "fractional zero point cannot be an integer accumulator offset",
    )
    device = q.reference_weight.device
    alpha, beta = (
        alpha.flatten().double().to(device),
        beta.flatten().double().to(device),
    )
    n = len(alpha)
    sx = ap["scale"].double().flatten().to(device)
    sw = record["params"]["scale"].double().flatten().to(device).expand(n)
    require(sx.numel() == 1 and torch.all(sw > 0) and sx > 0, "invalid input units")
    sy = float(cfg["output_scale"])
    zy = int(cfg["output_zero"])
    bits = cfg["output_bits"]
    shift = cfg["fractional_bits"]
    require(
        sy > 0
        and 2 <= bits <= 16
        and 0 <= zy < 2**bits
        and 0 <= shift <= 40
        and 2 <= cfg["accumulator_bits"] <= 52,
        "invalid integer output configuration",
    )
    bias = q.effective_bias()
    if bias is None:
        bias = torch.zeros_like(beta)
    unit = alpha * sx * sw
    m = (unit / sy * (1 << shift)).round()
    require(
        bool(((unit == 0) | (m != 0)).all()) and m.abs().max() < 2**52,
        "requantization multiplier lost a nonzero channel or exceeded exact range",
    )
    c = unit == 0
    b = torch.where(
        c, 0.0, (alpha * bias.double() + beta) / torch.where(c, 1.0, unit)
    ).round()
    require(
        bool(torch.isfinite(b).all() and b.abs().max() < 2**52),
        "accumulator bias overflow",
    )
    raw = {
        "multiplier": m.long(),
        "acc_bias": b.long(),
        "constant_code": (beta / sy + zy).round().clamp(0, 2**bits - 1).long(),
        "is_constant": c,
        "sx": sx,
        "sw": sw,
    }
    for k, v in (
        ("output_scale", sy),
        ("output_zero", zy),
        ("output_bits", bits),
        ("shift", shift),
        ("acc_bits", cfg["accumulator_bits"]),
    ):
        raw[k] = torch.tensor(
            v,
            dtype=torch.float64 if k == "output_scale" else torch.int64,
            device=device,
        )
    return raw


def integer_forward(q, x):
    p = q.ptq_output_correction
    q.calls += 1
    q.last_input_shape = list(x.shape)
    conv = q.layer["kind"] == "conv2d"
    flatshape = (-1,) + (1,) * (q.decoded_weight.ndim - 1)
    a = (q.calibration_input(x).double() / p.sx).round()
    w = (q.decoded_weight.double() / p.sw.reshape(flatshape)).round()
    bound = float(a.abs().max()) * float(w.abs().max()) * w[0].numel()
    require(
        bound < 2**52,
        "FP64 reference cannot represent this integer dot product exactly",
    )
    u = (
        F.conv2d(a, w, None, q.stride, q.padding, q.dilation, q.groups)
        if conv
        else F.linear(a, w)
    ).long()
    outshape = (1, -1, 1, 1) if conv else (-1,)
    accum = u + p.acc_bias.reshape(outshape)
    signed_limit = 1 << (int(p.acc_bits) - 1)
    require(
        accum.min() >= -signed_limit and accum.max() < signed_limit,
        "signed accumulator overflow",
    )
    require(
        float(accum.abs().max()) * float(p.multiplier.abs().max()) < 2**62,
        "fixed-point product overflow",
    )
    prod = accum * p.multiplier.reshape(outshape)
    divisor = 1 << int(p.shift)
    whole = prod // divisor
    twice = 2 * prod.remainder(divisor)
    rounded = (
        whole
        + ((twice > divisor) | ((twice == divisor) & whole.remainder(2).bool())).long()
    )
    code = (rounded + p.output_zero).clamp(0, (1 << int(p.output_bits)) - 1)
    code = torch.where(
        p.is_constant.reshape(outshape), p.constant_code.reshape(outshape), code
    )
    return ((code - p.output_zero) * p.output_scale).to(x)


def bind(module, state):
    require(set(state) == FIELDS, "integer affine state fields differ")
    n = module.decoded_weight.size(0)
    require(
        all(
            state[k].numel() == n
            for k in ("multiplier", "acc_bias", "constant_code", "is_constant", "sw")
        ),
        "integer affine channel layout changed",
    )
    module.forward = MethodType(integer_forward, module)
