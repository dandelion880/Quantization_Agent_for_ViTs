"""Calibrated integer operator reference routes; PyTorch integer intermediates with explicit range checks."""

import math
from copy import deepcopy
from contextlib import ExitStack
import torch
from torch import nn
from torch.nn import functional as F
from ..schemas import require
from .model import forward
from .sites import install_attention


def checked(value, maximum, message):
    require(bool((value.abs() <= maximum).all()), message)
    return value


def iptq_metric(energy, error, cost):
    require(
        cost > 0 and energy >= 0 and error >= 0,
        "invalid signal, squared perturbation or operation count",
    )
    if error == 0:
        inverse_quality = 0.0
    elif energy == 0:
        return 0.0
    else:
        quality = 20 * math.log10(energy / error)
        normalized = float(F.softplus(torch.tensor(quality, dtype=torch.float64)))
        if not normalized:
            return 0.0
        inverse_quality = 1 / normalized
    return 3 / (
        inverse_quality
        + float(F.softplus(torch.tensor(error, dtype=torch.float64)))
        + float(F.softplus(torch.tensor(cost, dtype=torch.float64)))
    )


def poly_coefficients(x, input_scale, cfg):
    # Eq. 4 minimizes erf error; the grid of candidate b is part of the declared adapter.
    u = (x.detach().double() / math.sqrt(2.0)).abs()
    if "coefficients" in cfg:
        a, b = cfg["coefficients"]
    else:
        best = None
        for b in cfg["offsets"]:
            require(b < 0, "offset must cap the positive erf branch")
            v = (u.clamp_max(-b) + b).pow(4)
            require(bool(v.square().sum() > 0), "degenerate erf basis")
            a = float(((u.erf() - 1) * v).sum() / v.square().sum())
            err = float((u.erf() - (1 + a * v)).square().sum())
            if best is None or err < best[0]:
                best = (err, a, b)
        _, a, b = best
    require(a < 0 and b < 0, "invalid saturating quartic")
    sc = input_scale / math.sqrt(2.0)
    # Scale for y is signed, as in the released high-order polynomial; both constants are prequantized.
    return {
        "b_integer": math.floor(b / sc),
        "c_integer": math.floor(1 / (a * sc**4)),
        "poly_scale": a * sc**4 * (1 << cfg["right_shift"]),
        "a": a,
        "b": b,
    }


class IntegerFunction(nn.Module):
    def __init__(self, config, state):
        super().__init__()
        self.config = config
        for k, v in state.items():
            self.register_buffer(k, v.detach().clone())
        self.eval()

    def forward(self, x):
        if hasattr(self, "quantizer"):
            x = self.quantizer(x)
        cfg = self.config
        scale = self.input_scale
        n = cfg["input_bits"]
        q = (x / scale).round().clamp(-(1 << (n - 1)), (1 << (n - 1)) - 1).long()
        op = cfg["function"]
        if op == "quartic_gelu":
            u = q.abs().clamp_max(-self.b_integer) + self.b_integer
            # Guard before each multiplication (int64 overflow otherwise wraps).
            checked(
                u, math.isqrt(math.isqrt((1 << 62) - 1)), "quartic accumulator overflow"
            )
            v = (u.square().square() + self.c_integer) * q.sign()
            v = v >> cfg["right_shift"]
            add = v + math.floor(1 / float(self.poly_scale))
            checked(
                add, max(1, ((1 << 62) - 1) // (1 << (n - 1))), "GELU product overflow"
            )
            return (q * add).double().mul(scale * self.poly_scale / 2).to(x)
        d = q - q.amax(-1, keepdim=True)
        if op == "efficient_bit_softmax":
            unit = math.floor(1 / float(scale))
            require(
                unit > 0 and 1 <= cfg["fraction_shift"] <= 30,
                "invalid fixed-point softmax shift",
            )
            p = d + (d >> 1) - (d >> 4)
            shift = (-p) // unit
            remainder = p + shift * unit
            frac = unit + (remainder >> 1) + (remainder >> 3) + (remainder >> 4)
            require(
                bool((frac >= 0).all())
                and unit * (1 << cfg["fraction_shift"]) < (1 << 61) // x.size(-1),
                "softmax exponent accumulator overflow",
            )
            exps = (frac << cfg["fraction_shift"]) >> shift.clamp_max(62)
            den = exps.sum(-1, keepdim=True)
            m = cfg["reciprocal_bits"]
            bits = cfg["output_bits"]
            require(
                1 < bits <= m <= 60 and bool((den > 0).all()),
                "softmax reciprocal domain",
            )
            reciprocal = (1 << m) // den
            y = (exps * reciprocal) >> (m - (bits - 1))
            return (y.double() / (1 << (bits - 1))).to(x)
        require(op == "edge_softmax", "unknown integer softmax route")
        ln2 = round(math.log(2.0) / float(scale))
        require(ln2 > 0, "scale too coarse for quantized ln2")
        power = (-d) // ln2
        rem = d + power * ln2
        poly = rem * (rem + self.linear_integer) + self.constant_integer
        checked(
            poly,
            (1 << 61) // (1 << cfg["exp_shift"]) // x.size(-1),
            "quadratic exp accumulator overflow",
        )
        exps = (poly.clamp_min(0) << cfg["exp_shift"]) >> power.clamp_max(62)
        num = (exps + self.requant_factor // 2) // self.requant_factor
        num = num.clamp(0, (1 << (cfg["bound_bits"] - 1)) - 1)
        den = num.sum(-1, keepdim=True)
        require(
            bool((den > 0).all()), "denominator vanished after exponent requantization"
        )
        shift = 31 - cfg["bound_bits"]
        return (((num << shift) // den).double() / (1 << shift)).to(x)


def state_for(x, cfg):
    require(2 <= cfg["input_bits"] <= 16, "invalid operator input width")
    s = cfg.get(
        "input_scale", float(x.abs().amax()) / ((1 << (cfg["input_bits"] - 1)) - 1)
    )
    require(math.isfinite(s) and s > 0, "nonpositive input scale")
    values = {"input_scale": s}
    if cfg["function"] == "quartic_gelu":
        p = poly_coefficients(x, s, cfg)
        values.update({k: p[k] for k in ("b_integer", "c_integer", "poly_scale")})
    if cfg["function"] == "edge_softmax":
        a, b, c = cfg["exp_coefficients"]
        require(
            a > 0 and 1 < cfg["bound_bits"] < 31 and 0 <= cfg["exp_shift"] < 31,
            "invalid range-reduced polynomial",
        )
        values.update(
            linear_integer=round(b / (s * a)),
            constant_integer=round(c / (s * s * a)),
            requant_factor=cfg["requant_factor"],
        )
        require(values["requant_factor"] >= 1, "invalid exponent requantization")
    return {
        k: torch.tensor(v, dtype=torch.long if isinstance(v, int) else torch.float64)
        for k, v in values.items()
    }


def target(model, stage):
    name = stage["module"]
    if stage["operator"] == "softmax":
        install_attention(model, [name])
        return model.get_submodule(name), "_ptq_softmax"
    require(stage["operator"] == "gelu", "unknown function selection scope")
    parent, _, key = name.rpartition(".")
    require(
        isinstance(model.get_submodule(name), nn.GELU),
        "GELU operator target is not a GELU",
    )
    return model.get_submodule(parent), key


def fit(backend, data, stage, cache_mib):
    teacher = deepcopy(backend.model).eval()
    parent, key = target(teacher, stage)
    values = []

    def hook(mod, args):
        values.append(args[0].detach().cpu())
        require(
            sum(v.numel() * v.element_size() for v in values) <= cache_mib * (1 << 20),
            "operator selection cache exhausted",
        )

    with ExitStack() as hooks:
        hooks.callback(getattr(parent, key).register_forward_pre_hook(hook).remove)
        for x, _ in data.batches("calibration"):
            forward(backend, teacher, x)
    x = torch.cat(values)
    ref = x.softmax(-1) if stage["operator"] == "softmax" else F.gelu(x)
    ranking = []
    winner = None
    for i, cfg in enumerate(stage["candidates"]):
        require(
            (cfg["function"] == "quartic_gelu") == (stage["operator"] == "gelu"),
            "operator mismatch",
        )
        state = state_for(x, cfg)
        decoded = IntegerFunction(cfg, state)(x)
        err = float((decoded - ref).double().square().sum())
        score = iptq_metric(
            float(ref.double().square().sum()), err, cfg["operation_count"]
        )
        ranking.append(
            {
                "index": i,
                "unified_score": score,
                "squared_l2": err,
                "calibration_elements": x.numel(),
                "operation_count": cfg["operation_count"],
            }
        )
        if winner is None or score > winner[0]:
            winner = (score, i, state)
    require(winner is not None, "empty integer operator candidates")
    return {**winner[2], "choice": torch.tensor(winner[1])}, ranking


def install(model, stage, state):
    parent, key = target(model, stage)
    i = int(state["choice"])
    require(0 <= i < len(stage["candidates"]), "invalid operator route")
    op = IntegerFunction(stage["candidates"][i], state)
    source = getattr(parent, key)
    if hasattr(source, "quantizer"):
        op.quantizer = source.quantizer
    device = next(model.parameters(), next(model.buffers(), torch.empty(0))).device
    setattr(parent, key, op.to(device))


def installed(model, stage):
    mod = model.get_submodule(stage["module"])
    fn = mod._ptq_softmax if stage["operator"] == "softmax" else mod
    require(isinstance(fn, IntegerFunction), "integer route absent from deployed graph")
    return {k: v.detach().cpu() for k, v in fn.named_buffers(recurse=False)}
