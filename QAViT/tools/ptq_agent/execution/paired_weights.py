"""Paired linear-input correction. X is [observations,input], W is [output,input]."""

import torch
from ..schemas import require
from . import codecs


def solve_right(cross, gram, damping):
    require(damping >= 0, "ridge must be nonnegative")
    system = gram + torch.eye(len(gram), dtype=gram.dtype, device=gram.device) * damping
    return torch.linalg.solve(system, cross.T).T


def correct_weight(weight, x_fp, x_q, *, alpha, damping, normalization):
    w = weight.double()
    x, q = (v.reshape(-1, w.size(1)).to(w) for v in (x_fp, x_q))
    require(x.shape == q.shape and q.size(0) > 0, "paired linear inputs differ")
    factor = {"mean": 1 / len(q), "sum": 1.0}[normalization]
    h = q.T @ q * factor
    delta = solve_right((w @ ((x - q).T @ q)) * factor, h, damping)
    return (w + alpha * delta).to(weight), h


def tail_partition(w, version):
    require(version in ("paper", "author_code"), "unknown DUQ partition")
    counts = torch.zeros(w.size(1), device=w.device, dtype=torch.long)
    for row in w.double():
        low = row[row < 0] if version == "author_code" else row
        high = row[row > 0] if version == "author_code" else row
        if low.numel():
            counts += row < torch.quantile(low, 0.01)
        if high.numel():
            counts += row > torch.quantile(high, 0.99)
    # Author count is based on output dimension. Stable index order resolves tied frequencies.
    n = min(w.size(0) // 20, w.size(1))
    group = torch.zeros_like(counts)
    if 0 < n < w.size(1):
        group[counts.argsort(descending=True, stable=True)[:n]] = 1
    return group


def erq(weight, x_fp, x_q, spec, options, bias=None):
    cfg = options["erq"]
    width = weight.shape[1]
    augmented = bool(cfg.get("augment_bias") and bias is not None)
    if augmented:
        # The constant column participates in both ridge solves and is never rounded.
        fitted = torch.cat((weight, bias.reshape(-1, 1).to(weight)), dim=1)
        x_fp = torch.cat((x_fp, torch.ones_like(x_fp[..., :1])), dim=-1)
        x_q = torch.cat((x_q, torch.ones_like(x_q[..., :1])), dim=-1)
    else:
        fitted = weight
    w, h = correct_weight(
        fitted,
        x_fp,
        x_q,
        alpha=1.0,
        damping=cfg["lambda_a"],
        normalization=cfg["moments"],
    )
    w = w.double()
    partition_source = (
        w[:, :width] if cfg.get("partition_reference") == "post_aqer" else weight
    )
    groups = (
        tail_partition(partition_source, cfg["tail_rule"])
        if "tail_rule" in cfg
        else torch.zeros(width, dtype=torch.long, device=w.device)
    )
    levels = (1 << spec["bits"]) - 1
    scales, zeros = [], []
    for group in range(int(groups.max()) + 1):
        values = w[:, :width][:, groups == group]
        lo, hi = values.amin(1).clamp_max(0), values.amax(1).clamp_min(0)
        scale = (hi - lo) / levels
        scale = torch.where(scale > 0, scale, 1.0)
        scales.append(scale)
        zeros.append((-lo / scale).round().clamp(0, levels))
    sg, zg = torch.stack(scales, 1), torch.stack(zeros, 1)
    scale, zero = sg[:, groups], zg[:, groups]
    codes = torch.empty_like(w[:, :width], dtype=codecs.code_dtype(spec["bits"]))
    trace = []
    start = 0
    while start < width:
        stop = start + max(1, (width - start) // 2)
        sl = slice(start, stop)
        sc, zp, target = scale[:, sl], zero[:, sl], w[:, sl]
        quant = (target / sc + zp).round().clamp(0, levels)
        hs = h[sl, sl]
        for _ in range(cfg["flip_steps"]):
            error = (quant - zp) * sc - target
            grad = error @ hs
            mask = error * grad > 0
            size = min(cfg["flip_k"], quant.shape[1])
            index = grad.abs().masked_fill(~mask, -1).topk(size, dim=1).indices
            trial = quant.clone()
            change = (-error.sign() * mask).gather(1, index)
            trial.scatter_add_(1, index, change)
            trial.clamp_(0, levels)
            updated = (trial - zp) * sc - target
            improve = ((updated @ hs) * updated).sum(1) < (grad * error).sum(1)
            if not improve.any():
                break
            quant = torch.where(improve[:, None], trial, quant)
        error = (quant - zp) * sc - target
        codes[:, sl] = quant.to(codes.dtype)
        if stop < w.size(1):
            w[:, stop:] -= solve_right(
                error @ h[sl, stop:], h[stop:, stop:], cfg["lambda_w"]
            )
        trace.append([start, stop])
        start = stop
    record = {
        "method": "erq",
        "spec": spec,
        "shape": list(weight.shape),
        "data": codecs.pack(codes.cpu(), spec["bits"]),
        "params": {"scale": sg.float().cpu(), "zero": zg.cpu(), "group": groups.cpu()},
    }
    if augmented:
        record["params"]["bias_delta"] = (w[:, -1] - bias.to(w)).to(weight).cpu()
    return record, {
        "augmented_constant": augmented,
        "partition_reference": cfg.get("partition_reference", "transformed_source"),
        "column_stages": trace,
        "groups": len(scales),
        "moments": cfg["moments"],
    }


def decode(record, device):
    p = record["params"]
    codes = codecs.unpack(
        record["data"], record["spec"]["bits"], record["shape"], device
    )
    group = p["group"].to(device)
    return (codes.float() - p["zero"].to(device)[:, group]) * p["scale"].to(device)[
        :, group
    ]


def validate(record, spec, shape):
    p = record["params"]
    require(
        record["shape"] == shape
        and record["spec"] == spec
        and record["data"].dtype == torch.uint8
        and record["data"].numel() == (shape[0] * shape[1] * spec["bits"] + 7) // 8,
        "invalid ERQ packed weight",
    )
    if "bias_delta" in p:
        b = p["bias_delta"]
        require(
            b.shape == (shape[0],) and bool(torch.isfinite(b).all()),
            "invalid unrounded ERQ bias",
        )
    group = p["group"]
    require(
        group.shape == (shape[1],)
        and group.dtype == torch.long
        and int(group.min()) == 0
        and int(group.max()) in (0, 1),
        "invalid DUQ membership",
    )
    require(
        p["scale"].shape == p["zero"].shape == (shape[0], int(group.max()) + 1)
        and bool(torch.isfinite(p["scale"]).all() and (p["scale"] > 0).all()),
        "invalid ERQ scales",
    )
    z = p["zero"]
    require(
        bool(((z == z.round()) & (z >= 0) & (z < 1 << spec["bits"])).all()),
        "invalid ERQ zero points",
    )
