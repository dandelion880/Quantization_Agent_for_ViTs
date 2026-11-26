"""Decoded values and calibration for region payloads and runtime precision routes."""

import math
import torch
from ..schemas import require

NAMES = ("three_region", "two_channel_scales", "patch_precision", "focus_interval")


def three_fields(x, bits, p):
    small = (1 << (bits - 2)) - 1
    large = (1 << (bits - 1)) - 1
    u = (x / p["scale"]).round().long()
    m0, m1 = int(p["mid_shift"]), int(p["high_shift"])
    # The low two regions have region=0, followed by sign and the magnitude field.
    neg = (-u).clamp(0, small)
    mid = (u.clamp_min(0) >> m0).clamp_max(small)
    high = (u.clamp_min(0) >> m1).clamp_max(large)
    threshold = 1 << (bits - 3 + m0)
    payload = torch.where(
        u < 0, small + 1 + neg, torch.where(u < threshold, mid, large + 1 + high)
    )
    code = (
        payload  # explicitly decode bits, including both zero magnitude representations
    )
    y = torch.where(
        code >= large + 1,
        ((code - large - 1) << m1),
        torch.where(code > small, -(code - small - 1), code << m0),
    )
    return payload, y * p["scale"]


def patch_widths(p, cfg, bits):
    require(
        p.ndim == 4 and p.shape[-2] == p.shape[-1],
        "patch routing requires a square multihead attention map",
    )
    left = cfg["prefix"]
    require(0 <= left < p.size(-1), "unbound special-token extent")
    m = p[:, :, left:, left:].double()
    axis = {"literal_outgoing": -1, "incoming_adapter": -2}[cfg["importance"]]
    score = m.mean(1).mean(axis if axis == -1 else -2)
    mean = score.mean(-1, keepdim=True)
    std = score.std(-1, unbiased=cfg["unbiased"], keepdim=True)
    widths = torch.where(
        score > mean + std, bits + 2, torch.where(score < mean - std, bits - 2, bits)
    )
    return torch.cat((torch.full((p.shape[0], left), bits, device=p.device), widths), 1)


def apply(x, spec, method, p, cfg):
    bits = spec["bits"]
    if method == "three_region":
        return three_fields(x, bits, p)[1].to(x)
    if method == "two_channel_scales":
        axis = cfg["channel_axis"] % x.ndim
        require(
            x.size(axis) == p["groups"].numel(),
            "static channel partition shape changed",
        )
        shape = [1] * x.ndim
        shape[axis] = x.size(axis)
        s = (p["scale"] * torch.exp2(p["groups"].to(x) * p["shift"])).reshape(shape)
        return (
            (x / s).round().clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1) * s
        ).to(x)
    if method == "patch_precision":
        bound = p["maximum"]
        widths = patch_widths(x, cfg, bits)[:, None, :, None]
        top = torch.exp2(widths.to(x)) - 1
        return ((x / bound * top).round().clamp_min(0).minimum(top) / top * bound).to(x)
    lo, hi = p["low"], p["high"]
    require(bool((hi > lo).all()), "focus interval collapsed")
    levels = (1 << bits) - 1
    code = ((x - lo) / (hi - lo) * levels).round().clamp(0, levels)
    return code / levels * (hi - lo) + lo


def fit(x, spec, method, cfg, *, candidate_loss=None):
    bits = spec["bits"]
    require(bits >= 3, "insufficient region/sign payload")
    best = None
    if method == "three_region":
        neg = -x.flatten(1).amin(1).mean()
        upper = torch.quantile(x.flatten(1), cfg["upper_percentile"], dim=-1).mean()
        initial = neg / ((1 << (bits - 2)) - 1)
        require(
            initial > 0 and upper > 0,
            "three-region fit needs signed GELU and a positive tail",
        )
        high = round(math.log2(float(upper / ((1 << (bits - 1)) - 1) / initial)))
        if "high_shift_floor" in cfg:
            high = max(high, cfg["high_shift_floor"])
        require(high > 0, "insufficient range for distinct paper AutoScale shifts")
        for mid in range(high):
            for factor in cfg["base_factors"]:
                p = {
                    "scale": initial * factor,
                    "mid_shift": x.new_tensor(mid, dtype=torch.long),
                    "high_shift": x.new_tensor(high, dtype=torch.long),
                }
                err = (
                    candidate_loss(p)
                    if candidate_loss is not None
                    else (apply(x, spec, method, p, cfg) - x).double().square().mean()
                )
                if best is None or err < best[0]:
                    best = float(err), p
        return best[1]
    if method == "two_channel_scales":
        axis = cfg["channel_axis"] % x.ndim
        rows = x.movedim(axis, 0).flatten(1)
        maxima = rows.abs().amax(1)
        centers = torch.stack((maxima.min(), maxima.max()))
        for _ in range(cfg["kmeans_steps"]):
            groups = (maxima[:, None] - centers).abs().argmin(1)
            centers = torch.stack(
                [
                    maxima[groups == i].mean() if (groups == i).any() else centers[i]
                    for i in range(2)
                ]
            )
        initial = (centers[0] / ((1 << (bits - 1)) - 1)).clamp_min(
            torch.finfo(x.dtype).eps
        )
        for shift in cfg["shifts"]:
            for f in cfg["base_factors"]:
                p = {
                    "groups": groups,
                    "scale": initial * f,
                    "shift": x.new_tensor(shift, dtype=torch.long),
                }
                err = (apply(x, spec, method, p, cfg) - x).double().square().mean()
                if best is None or err < best[0]:
                    best = float(err), p
        return best[1]
    if method == "patch_precision":
        require(
            bits - 2 >= 1 and bits + 2 <= 8 and x.amin() >= 0,
            "invalid patch bit alternatives or attention domain",
        )
        return {
            "maximum": torch.quantile(x.flatten(), cfg["percentile"]).clamp_min(
                torch.finfo(x.dtype).eps
            )
        }
    require(
        bool((x >= 0).all() and (x <= 1 + 1e-6).all()),
        "focus interval is for post-softmax probabilities",
    )
    p = {
        "low": x.new_tensor(cfg["initial_bounds"][0]),
        "high": x.new_tensor(cfg["initial_bounds"][1]),
    }
    require(0 <= p["low"] < p["high"] <= 1, "invalid initial focus interval")
    return p


def validate(p, layer, shape):
    name = layer["activation_method"]
    keys = {
        "three_region": {"scale", "mid_shift", "high_shift"},
        "two_channel_scales": {"groups", "scale", "shift"},
        "patch_precision": {"maximum"},
        "focus_interval": {"low", "high"},
    }[name]
    require(
        set(p) == keys and all(torch.isfinite(v).all() for v in p.values()),
        "invalid region grid state",
    )
    require(
        all(v.numel() == 1 for k, v in p.items() if k != "groups"),
        "expected scalar region fields",
    )
    if name == "two_channel_scales":
        groups = p["groups"]
        require(
            groups.dtype == torch.long
            and groups.shape == (shape[layer["options"]["encoding"]["channel_axis"]],)
            and bool(((groups == 0) | (groups == 1)).all()),
            "invalid binary channel partition",
        )
    if "scale" in p:
        require(p["scale"] > 0, "invalid region scale")
    if "low" in p:
        require(0 <= p["low"] < p["high"] <= 1, "invalid focus endpoints")
