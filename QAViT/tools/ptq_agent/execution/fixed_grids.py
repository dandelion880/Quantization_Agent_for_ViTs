"""Frozen positions or fixed contiguous feature groups. Batch ordinals never select parameters."""

import torch
from ..schemas import require


def shape_for(shape, spec):
    axis = spec["axis"]
    require(0 < axis < len(shape), "fixed feature axis must exclude batch")
    moved = [n for i, n in enumerate(shape) if i not in (0, axis)] + [shape[axis]]
    if spec["granularity"] == "per_token":
        return (1, *moved[:-1], 1)
    size = spec["group_size"]
    require(
        size > 0 and moved[-1] % size == 0,
        "fixed groups must divide the declared feature axis",
    )
    return (*([1] * (len(shape) - 1)), moved[-1] // size, 1)


def layout(x, spec):
    value = x.movedim(spec["axis"], -1)
    return (
        value
        if spec["granularity"] == "per_token"
        else value.unflatten(-1, (-1, spec["group_size"]))
    )


def fit(x, spec):
    require(
        x.numel() and bool(torch.isfinite(x).all()),
        "empty/nonfinite fixed-grid calibration",
    )
    shape_for(x.shape, spec)
    value = layout(x.detach().double(), spec)
    dims = (
        (0, value.ndim - 1)
        if spec["granularity"] == "per_token"
        else tuple(range(value.ndim - 2)) + (-1,)
    )
    lo, hi = (
        value.amin(dims, keepdim=True).clamp_max(0),
        value.amax(dims, keepdim=True).clamp_min(0),
    )
    levels = (1 << spec["bits"]) - 1
    if spec["symmetric"]:
        scale = torch.maximum(-lo, hi) / (levels // 2)
        zero = torch.full_like(scale, levels // 2)
    else:
        scale = (hi - lo) / levels
        zero = (-lo / torch.where(scale > 0, scale, 1.0)).round().clamp(0, levels)
    return {
        "scale": torch.where(scale > 0, scale, 1.0),
        "zero": zero,
        "layout_shape": torch.tensor(x.shape[1:], dtype=torch.long),
    }


def validate(params, spec, shape):
    require(
        set(params) == {"scale", "zero", "layout_shape"}
        and list(shape[1:]) == params["layout_shape"].tolist(),
        "frozen activation layout changed",
    )
    size = shape_for(shape, spec)
    scale, zero = params["scale"], params["zero"]
    require(
        scale.shape == zero.shape == size
        and bool(torch.isfinite(scale).all() and (scale > 0).all()),
        "invalid fixed activation scales",
    )
    require(
        bool(
            ((zero == zero.round()) & (zero >= 0) & (zero < (1 << spec["bits"]))).all()
        ),
        "invalid fixed zero",
    )


def quantize(x, spec, params):
    require(
        tuple(x.shape[1:]) == tuple(params["layout_shape"].tolist()),
        "frozen token/window shape mismatch",
    )
    value = layout(x, spec)
    scale, zero = (params[k].to(value) for k in ("scale", "zero"))
    codes = (
        (value / scale)
        .round()
        .add(zero)
        .clamp(0, (1 << spec["bits"]) - 1 - int(spec["symmetric"]))
    )
    result = (codes - zero) * scale
    if spec["granularity"] == "per_group":
        result = result.flatten(-2)
    return result.movedim(-1, spec["axis"])


def first_batch_rows(x, spec, percentiles):
    """Choose row-wise clipping by reconstruction error on the named first calibration batch."""
    p = fit(x, spec)
    value = layout(x.double(), spec)
    flat = value.movedim(0, -2).flatten(-2)
    n = (1 << spec["bits"]) - 1
    best = None
    for percentile in percentiles:
        lo = torch.quantile(flat, 1 - percentile, dim=-1).clamp_max(0)
        hi = torch.quantile(flat, percentile, dim=-1).clamp_min(0)
        scale = (
            torch.maximum(-lo, hi) / (n // 2) if spec["symmetric"] else (hi - lo) / n
        )
        scale = torch.where(scale > 0, scale, 1.0)
        zero = (
            torch.full_like(scale, n // 2)
            if spec["symmetric"]
            else (-lo / scale).round().clamp(0, n)
        )
        error = (
            (
                ((flat / scale[..., None]).round() + zero[..., None])
                .clamp(0, n - int(spec["symmetric"]))
                .sub(zero[..., None])
                * scale[..., None]
                - flat
            )
            .square()
            .mean(-1)
        )
        if best is None:
            best = (error, scale, zero)
        else:
            improved = error < best[0]
            best = tuple(
                torch.where(improved, a, b) for a, b in zip((error, scale, zero), best)
            )
    require(best is not None, "no first-batch clipping candidates")
    p.update(scale=best[1].reshape_as(p["scale"]), zero=best[2].reshape_as(p["zero"]))
    return p
