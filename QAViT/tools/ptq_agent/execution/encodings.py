"""Frozen nonlinear codecs and instance-routed range groups used by the PTQ recipes."""

import math
import torch
from ..schemas import require

from . import region_grids

NAMES = (
    *region_grids.NAMES,
    "domain_clip",
    "tanq",
    "sulq",
    "shift_log",
    "bit_window",
    "pot_error",
    "igq_ranges",
    "adfq_outliers",
)


def range_params(lo, hi, n):
    scale = ((hi - lo) / (n - 1)).clamp_min(torch.finfo(lo.dtype).eps)
    return scale, (-lo / scale).round()


def uniform(x, scale, zero, n):
    return ((x / scale).round() + zero).clamp(0, n - 1).sub(zero) * scale


def group_features(x, mode):
    if mode == "attention_row":
        return torch.stack((torch.zeros_like(x[..., 0]), x.amax(-1)), -1)
    require(x.ndim == 3, "instance channel ranges require B,T,C")
    return torch.stack((x.amin(1), x.amax(1)), -1)


def nearest(bounds, centers):
    return (bounds.unsqueeze(-2) - centers).square().sum(-1).argmin(-1)


def fit_groups(features, count, steps):
    points = features.reshape(-1, 2)
    require(
        1 <= count <= len(points) and steps > 0,
        "invalid IGQ group count or update budget",
    )
    order = (points[:, 1] - points[:, 0]).argsort(stable=True)
    centers = points[
        order[
            torch.linspace(0, len(order) - 1, count, device=points.device)
            .round()
            .long()
        ]
    ]
    for _ in range(steps):
        labels = nearest(points, centers)
        sums = torch.zeros_like(centers).index_add_(0, labels, points)
        counts = torch.bincount(labels, minlength=count).unsqueeze(-1)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    return centers


def shift_decode(x, n, scale, shift, cfg):
    v = (x + shift).clamp_min(0)
    base = cfg["steps_per_octave"]
    e = (-base * torch.log2(v / scale)).round()
    limit = n - int(cfg["reserve_underflow"])
    return (
        torch.where(e >= limit, 0.0, torch.exp2(-e.clamp(0, limit - 1) / base) * scale)
        - shift
    )


def bit_window(x, n, scale, cfg):
    payload = n // 2
    magnitude = (x / scale).round()
    if cfg["domain"] == "probability":
        small = magnitude.clamp(0, payload - 1)
        boundary = payload
    else:
        boundary = payload // 2
        small = magnitude.clamp(-(boundary - 1), boundary - 1)
    ratio = 2 ** cfg["shift_bits"]
    large = (magnitude / ratio).round().clamp(0, payload - 1) * ratio
    return torch.where(magnitude < boundary, small, large) * scale


def fit(x, spec, *, method, options):
    cfg = options["encoding"]
    n = 2 ** spec["bits"]
    require(
        x.is_floating_point() and bool(torch.isfinite(x).all()),
        "codec needs finite observations",
    )
    if method == "domain_clip":
        from .domain_fit import initial_clip

        return initial_clip(x, cfg)
    if method in region_grids.NAMES:
        return region_grids.fit(x, spec, method, cfg)
    if method == "igq_ranges":
        centers = fit_groups(
            group_features(x, cfg["mode"]), cfg["groups"], cfg["steps"]
        )
        return {"centers": centers}
    if method == "adfq_outliers":
        require(
            x.ndim == 3 and cfg["threshold"] > 0,
            "ADFQ needs B,T,C and a positive effective threshold",
        )
        ordinary = (
            x.masked_fill(x.abs() > cfg["threshold"], 0).transpose(0, 1).flatten(1)
        )
        qs = ordinary.new_tensor(cfg["percentiles"]).reshape(-1)
        require(
            len(qs) > 0 and bool(((qs > 0.5) & (qs <= 1)).all()),
            "invalid percentile seeds",
        )
        lo, hi = (
            torch.quantile(ordinary, 1 - qs, dim=1),
            torch.quantile(ordinary, qs, dim=1),
        )
        s, z = range_params(lo, hi, n)
        loss = torch.stack(
            [
                (uniform(ordinary, ss[:, None], zz[:, None], n) - ordinary)
                .square()
                .mean(1)
                for ss, zz in zip(s, z)
            ]
        )
        best = loss.argmin(0, keepdim=True)
        return {
            "scale": s.gather(0, best).unsqueeze(-1),
            "zero": z.gather(0, best).unsqueeze(-1),
        }
    if method == "pot_error":
        require(spec["symmetric"], "PoT adapter uses a signed integer grid")
        maxcode = n // 2 - 1
        base = (x.abs().max() / maxcode).clamp_min(torch.finfo(x.dtype).eps).log2()
        best = None
        for clip in cfg["clip_factors"]:
            require(0 < clip <= 1, "clipping must be in (0,1]")
            center = float(base) + math.log2(clip)
            for e in range(
                math.floor(center) - cfg["neighbors"],
                math.ceil(center) + cfg["neighbors"] + 1,
            ):
                scale = x.new_tensor(2.0**e)
                err = (
                    ((x / scale).round().clamp(-n // 2, maxcode) * scale - x)
                    .double()
                    .square()
                    .mean()
                    .item()
                )
                if best is None or err < best[0]:
                    best = err, scale
        require(best is not None, "empty clipping search")
        return {"scale": best[1]}
    if method == "bit_window":
        require(
            spec["bits"] >= 3
            and cfg["domain"] in ("probability", "gelu")
            and cfg["shift_bits"] >= 0,
            "invalid region-bit layout",
        )
        return {
            "scale": (x.amax() / ((n // 2 - 1) * 2 ** cfg["shift_bits"])).clamp_min(
                torch.finfo(x.dtype).eps
            )
        }
    if method == "shift_log":
        require(
            cfg["shift_mode"] in ("fixed_calibration", "runtime_min")
            and cfg["steps_per_octave"] in (1, 2)
            and cfg["epsilon"] > 0
            and type(cfg["reserve_underflow"]) is bool,
            "invalid shifted log convention",
        )
        shift = -x.min() + cfg["epsilon"]
        flat = (x + shift).reshape(-1)
        best = None
        for pct in cfg["percentiles"]:
            require(0 < pct <= 1, "invalid shifted log percentile")
            scale = torch.quantile(flat, pct).clamp_min(torch.finfo(x.dtype).eps)
            error = (
                (shift_decode(x, n, scale, shift, cfg) - x)
                .double()
                .square()
                .mean()
                .item()
            )
            if best is None or error < best[0]:
                best = error, scale
        require(best is not None, "empty shifted log clipping search")
        return {"scale": best[1], "shift": shift}
    require(
        bool(((x >= 0) & (x <= 1 + 1e-6)).all()),
        "probability-domain codec on a non-probability tensor",
    )
    candidates = cfg["pairs"] if method == "tanq" else cfg["eta"]
    best = None
    for candidate in candidates:
        if method == "tanq":
            a, c = candidate
            require(
                a > 0
                and max(abs(a * c), abs(a * (1 - c))) < math.pi / 2
                and cfg["inverse_round"] in ("paper_v4", "analytic"),
                "tangent pole inside the deployed interval or unknown inverse",
            )
            p = {"a": x.new_tensor(a), "c": x.new_tensor(c)}
            v = torch.tan(p["a"] * (x - p["c"]))
        else:
            require(candidate > 0 and method == "sulq", "SULQ eta must be positive")
            p = {"eta": x.new_tensor(candidate)}
            v = -(x + p["eta"]).log2()
        p["scale"], p["zero"] = range_params(v.amin(), v.amax(), n)
        error = (
            (apply(x, spec, method=method, params=p, options=options) - x)
            .double()
            .square()
            .mean()
            .item()
        )
        if best is None or error < best[0]:
            best = error, p
    require(best is not None, "empty companding grid")
    return best[1]


def apply(x, spec, *, method, params, options):
    cfg = options["encoding"]
    p = {k: v.to(x.device) for k, v in params.items()}
    n = 2 ** spec["bits"]
    if method == "domain_clip":
        from .domain_fit import activation

        return activation(x, p, spec["bits"], cfg)
    if method in region_grids.NAMES:
        return region_grids.apply(x, spec, method, p, cfg)
    if method == "igq_ranges":
        centers = p["centers"]
        idx = nearest(group_features(x, cfg["mode"]), centers)
        lo, hi = centers[idx].unbind(-1)
        scale, zero = range_params(lo, hi, n)
        dim = -1 if cfg["mode"] == "attention_row" else 1
        return uniform(x, scale.unsqueeze(dim), zero.unsqueeze(dim), n).to(x.dtype)
    if method == "adfq_outliers":
        require(
            x.ndim == 3 and x.size(1) == p["scale"].size(1), "ADFQ token layout changed"
        )
        return torch.where(
            x.abs() > cfg["threshold"], x, uniform(x, p["scale"], p["zero"], n)
        ).to(x.dtype)
    if method == "pot_error":
        return (
            (x / p["scale"])
            .round()
            .clamp(-n // 2, n // 2 - 1)
            .mul(p["scale"])
            .to(x.dtype)
        )
    if method == "bit_window":
        return bit_window(x, n, p["scale"], cfg).to(x.dtype)
    if method == "shift_log":
        shift = (
            -x.min() + cfg["epsilon"]
            if cfg["shift_mode"] == "runtime_min"
            else p["shift"]
        )
        return shift_decode(x, n, p["scale"], shift, cfg).to(x.dtype)
    if method == "sulq":
        v = uniform(-(x + p["eta"]).log2(), p["scale"], p["zero"], n)
        return (torch.exp2(-v.round()) - p["eta"]).to(x.dtype)
    v = uniform(torch.tan(p["a"] * (x - p["c"])), p["scale"], p["zero"], n)
    if cfg["inverse_round"] == "paper_v4":
        v = v.round()
    return (torch.atan(v) / p["a"] + p["c"]).to(x.dtype)


def validate(params, layer, shape):
    name = layer["activation_method"]
    if name == "domain_clip":
        require(
            len(shape) == 3
            and set(params) == {"lower", "upper", "low_gate", "high_gate"}
            and all(
                v.shape == (1, 1, shape[-1]) and torch.isfinite(v).all()
                for v in params.values()
            ),
            "invalid domain channel state",
        )
        require(
            torch.all(params["upper"] >= params["lower"])
            and all(
                torch.all((params[k] >= 0) & (params[k] <= 1))
                for k in ("low_gate", "high_gate")
            ),
            "invalid domain bounds or gates",
        )
        return
    if name in region_grids.NAMES:
        return region_grids.validate(params, layer, shape)
    fields = {
        "igq_ranges": {"centers"},
        "adfq_outliers": {"scale", "zero"},
        "sulq": {"scale", "zero", "eta"},
        "tanq": {"a", "c", "scale", "zero"},
        "shift_log": {"scale", "shift"},
    }.get(name, {"scale"})
    require(
        set(params) == fields
        and all(bool(torch.isfinite(t).all()) for t in params.values()),
        "invalid frozen encoding state",
    )
    if name == "igq_ranges":
        c = params["centers"]
        require(
            c.shape == (layer["options"]["encoding"]["groups"], 2)
            and bool((c[:, 1] >= c[:, 0]).all()),
            "invalid group ranges",
        )
    else:
        require(bool((params["scale"] > 0).all()), "invalid codec scale")
        if name == "adfq_outliers":
            require(
                len(shape) == 3
                and all(list(v.shape) == [1, shape[1], 1] for v in params.values()),
                "invalid fixed token states",
            )
        else:
            require(
                all(v.numel() == 1 for v in params.values()),
                "scalar encoding state has unexpected axes",
            )


def levels(params, layer):
    """Enumerate deployed decoder outputs for a fixed group, including its underflow convention."""
    p = params
    name = layer["activation_method"]
    n = 2 ** layer["activation"]["bits"]
    cfg = layer["options"]["encoding"]
    j = torch.arange(n, device=p["scale"].device, dtype=p["scale"].dtype)
    if name == "sulq":
        return torch.exp2(-((j - p["zero"]) * p["scale"]).round()) - p["eta"]
    if name == "tanq":
        v = (j - p["zero"]) * p["scale"]
        if cfg["inverse_round"] == "paper_v4":
            v = v.round()
        return torch.atan(v) / p["a"] + p["c"]
    limit = n - int(cfg["reserve_underflow"])
    return torch.cat(
        [
            torch.exp2(-j[:limit] / cfg["steps_per_octave"]) * p["scale"] - p["shift"],
            -p["shift"].reshape(1),
        ]
    )
