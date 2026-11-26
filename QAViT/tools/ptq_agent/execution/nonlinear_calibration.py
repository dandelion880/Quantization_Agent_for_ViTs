"""H-ViT PoT channel alignment and PracticalEdge bounded integer GELU/LayerNorm reference."""

from types import MethodType
import math
import torch
from torch import nn
from ..schemas import require
from .model import forward


def uniform_range(x, b):
    lo = x.amin().clamp_max(0)
    hi = x.amax().clamp_min(0)
    s = (hi - lo) / ((1 << b) - 1)
    s = torch.where(s > 0, s, torch.ones_like(s))
    return s, (-lo / s).round().clamp(0, (1 << b) - 1).long()


def pot_exponents(samples, bits, axis, factors):
    low = -(1 << (bits - 1))
    high = (1 << (bits - 1)) - 1
    y = samples.movedim(axis, -1).reshape(-1, samples.size(axis)).double()
    maximum = y.abs().amax(0)
    require(
        factors and all(f > 0 for f in factors),
        "explicit positive clipping candidates required",
    )
    losses = []
    exps = []
    for f in factors:
        e = (maximum.clamp_min(torch.finfo(torch.float32).tiny) * f / high).log2()
        for candidate in (e.floor(), e.ceil()):
            s = torch.exp2(candidate)
            rec = (y / s).round().clamp(low, high) * s
            losses.append((rec - y).square().sum(0))
            exps.append(candidate)
    selected = torch.stack(losses).argmin(0, keepdim=True)
    return torch.stack(exps).gather(0, selected).squeeze(0).long()


def fit(backend, data, stage, cache_mib):
    module = backend.model.get_submodule(stage["module"])
    cfg = stage["settings"]
    method = cfg["function"]
    xs = []
    ys = []
    used = 0

    def capture(m, args, out):
        nonlocal used
        used += (
            args[0].numel() * args[0].element_size() + out.numel() * out.element_size()
        )
        require(used <= cache_mib * (1 << 20), "operator calibration cache exceeded")
        xs.append(args[0].detach().cpu())
        ys.append(out.detach().cpu())

    h = module.register_forward_hook(capture)
    try:
        for x, sid in data.batches("calibration"):
            forward(backend, backend.model, x)
    finally:
        h.remove()
    require(
        xs and len(xs) == len(data.ids("calibration")), "operator missing or repeated"
    )
    X, Y = torch.cat(xs), torch.cat(ys)
    ib, ob = cfg["input_bits"], cfg["output_bits"]
    require(2 <= ib <= 16 and 2 <= ob <= 16, "unsupported operator width")
    if method == "hvit_layernorm":
        require(
            isinstance(module, nn.LayerNorm)
            and len(module.normalized_shape) == 1
            and X.ndim == 3,
            "H-ViT needs B,T,C LayerNorm",
        )
        require(
            cfg["output_layout"] in ("frozen_positions", "instance_tokens"),
            "declare token-scaling semantics",
        )
        state = {"exponent": pot_exponents(X, ib, -1, cfg["clip_factors"])}
        if cfg["output_layout"] == "frozen_positions":
            state["token_exponent"] = pot_exponents(Y, ob, 1, cfg["clip_factors"])
    else:
        require(
            method in ("edge_layernorm", "edge_gelu")
            and isinstance(
                module, nn.LayerNorm if method == "edge_layernorm" else nn.GELU
            ),
            "unexpected PracticalEdge operator",
        )
        s, z = uniform_range(X, ib)
        so, zo = uniform_range(Y, ob)
        state = {"scale": s, "zero": z, "out_scale": so, "out_zero": zo}
        if method == "edge_layernorm":
            require(
                cfg["left_shift"] <= 31 - ib and cfg["left_shift"] >= 0,
                "integer numerator bound exceeded",
            )
        else:
            a, b, c = cfg["erf_coefficients"]
            unit = s / math.sqrt(2.0)
            require(a < 0 and b < 0 and unit > 0, "invalid signed polynomial")
            state.update(
                b_int=(b / unit).round().long(),
                c_int=(c / (a * unit.square())).round().long(),
                erf_scale=a * unit.square(),
            )
    return state, {
        "function": method,
        "input_shape": list(X.shape),
        "output_shape": list(Y.shape),
        "input_bits": ib,
        "output_bits": ob,
        "arithmetic": "software reference; exact integer products and reductions where specified",
    }


def compute(self, x):
    p = self.ptq_nonlinear
    opt = self.ptq_nonlinear_options
    op = opt["function"]
    ib, ob = opt["input_bits"], opt["output_bits"]
    if op == "hvit_layernorm":
        shifts = p.exponent - p.exponent.min()
        require(
            shifts.max() + ib <= 29,
            "aligned activation exceeds safe integer square range",
        )
        s = torch.exp2(p.exponent.to(torch.float64))
        q = (x.double() / s).round().clamp(-(1 << (ib - 1)), (1 << (ib - 1)) - 1).long()
        aligned = q << shifts
        require(
            float(aligned.abs().max()) ** 2 * x.size(-1) < 2**62,
            "overflow in aligned variance",
        )
        n = x.size(-1)
        mean = aligned.sum(-1, keepdim=True).double() / n
        var = (
            aligned.square().sum(-1, keepdim=True).double() / n - mean.square()
        ).clamp_min(0)
        common = torch.exp2(p.exponent.min().double())
        y = (aligned - mean) * torch.rsqrt(var + self.eps / common.square())
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        if opt["output_layout"] == "instance_tokens":
            maximum = (
                y.abs()
                .amax(-1, keepdim=True)
                .clamp_min(torch.finfo(torch.float32).tiny)
            )
            cont = (maximum / ((1 << (ob - 1)) - 1)).log2()
            scales = torch.stack((cont.floor().exp2(), cont.ceil().exp2()))
            candidates = (y / scales).round().clamp(
                -(1 << (ob - 1)), (1 << (ob - 1)) - 1
            ) * scales
            return torch.where(
                (candidates[0] - y).square().sum(-1, keepdim=True)
                <= (candidates[1] - y).square().sum(-1, keepdim=True),
                candidates[0],
                candidates[1],
            ).to(x)
        require(
            x.size(1) == p.token_exponent.numel(), "frozen H-ViT token layout mismatch"
        )
        scale = p.token_exponent.double().exp2().reshape(1, -1, 1)
        return (
            (y / scale).round().clamp(-(1 << (ob - 1)), (1 << (ob - 1)) - 1) * scale
        ).to(x)
    q = (x / p.scale + p.zero).round().clamp(0, (1 << ib) - 1).long() - p.zero
    if op == "edge_layernorm":
        n = x.size(-1)
        mean = (q.sum(-1, keepdim=True).double() / n).round().long()
        ex2 = (q.square().sum(-1, keepdim=True).double() / n).round()
        root = (ex2 - mean.square() + 1).clamp_min(1).sqrt().round().clamp_min(1).long()
        shifted = (q - mean) << opt["left_shift"]
        require(shifted.abs().amax() < 2**31, "int32 shifted numerator overflow")
        quotient = (shifted.double() / root).round()
        gamma = self.weight if self.weight is not None else 1.0
        beta = self.bias if self.bias is not None else 0.0
        code = (
            (quotient * gamma / ((1 << opt["left_shift"]) * p.out_scale)).round()
            + (beta / p.out_scale).round()
            + p.out_zero
        )
    else:
        mag = q.abs().clamp_max(-p.b_int)
        erf = ((mag + p.b_int).square() + p.c_int) * q.sign()
        prod = q * (erf + (1 / p.erf_scale).round().long())
        require(prod.abs().amax() < 2**31, "integer GELU product exceeded int32")
        code = (
            prod.double() * p.scale * p.erf_scale / (2 * p.out_scale)
        ).round() + p.out_zero
    return ((code.clamp(0, (1 << ob) - 1) - p.out_zero) * p.out_scale).to(x)


def bind(model, stage, state):
    m = model.get_submodule(stage["module"])
    cfg = stage["settings"]
    expected = (
        {"exponent"}
        | ({"token_exponent"} if cfg["output_layout"] == "frozen_positions" else set())
        if cfg["function"] == "hvit_layernorm"
        else {"scale", "zero", "out_scale", "out_zero"}
        | ({"erf_scale", "b_int", "c_int"} if cfg["function"] == "edge_gelu" else set())
    )
    require(
        set(state) == expected and all(torch.isfinite(v).all() for v in state.values()),
        "invalid frozen nonlinear parameters",
    )
    device = next(model.parameters()).device
    holder = nn.Module()
    for k, v in state.items():
        holder.register_buffer(k, v.detach().clone().to(device))
    m.ptq_nonlinear = holder.eval()
    m.ptq_nonlinear_options = dict(cfg)
    m.forward = MethodType(compute, m)


def installed(m):
    require(
        getattr(m.forward, "__func__", None) is compute,
        "calibrated operator missing from graph",
    )
    return {k: v.cpu().detach() for k, v in m.ptq_nonlinear.named_buffers()}
