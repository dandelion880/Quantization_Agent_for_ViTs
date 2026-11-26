"""PTQ-only, shared-weight Swin window paths. Input layout, shifts, mask and stage grouping are explicit."""

from contextvars import ContextVar
from copy import deepcopy
from types import MethodType
import torch
from torch.nn import functional as F
from ..schemas import require
from . import grids

branch = ContextVar("qavit_window_branch", default=None)
rank_cache = ContextVar("qavit_stage_window_rank", default={})


def pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def partition(x, window):
    b, h, w, c = x.shape
    wh, ww = window
    return (
        x.reshape(b, h // wh, wh, w // ww, ww, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, wh * ww, c)
    )


def restore(windows, shape, window):
    b, h, w, c = shape
    wh, ww = window
    return (
        windows.reshape(b, h // wh, w // ww, wh, ww, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(shape)
    )


def low_grid(q, x, bits):
    p = q.params()
    spec = q.layer["activation"]
    method = q.layer["activation_method"]
    require(
        method in grids.METHODS
        and spec["granularity"] in ("per_tensor", "per_channel"),
        "window branch needs the shared full-map uniform initialization",
    )
    n = (1 << spec["bits"]) - 1
    m = (1 << bits) - 1
    params = {**p, "scale": p["scale"] * (n / m)}
    if "zero" in p:
        params["zero"] = (p["zero"] * (m / n)).round()
    lower = {
        **spec,
        "bits": bits,
        "format": ("INT" if spec["symmetric"] else "UINT") + str(bits),
    }
    return grids.apply(x, lower, method, params)


def routed(self, x, *args, **kwargs):
    if not self.ptq_routing_active:
        return self.ptq_original_forward(x, *args, **kwargs)
    cfg = self.ptq_window
    require(
        not self.training and x.ndim in (3, 4),
        "window routing expects frozen NLC or NHWC Swin features",
    )
    hw = (
        x.shape[1:3]
        if x.ndim == 4
        else getattr(self, "input_resolution", None) or (self.H, self.W)
    )
    h, w = hw
    b = len(x)
    c = x.shape[-1]
    window = pair(self.window_size)
    shift = pair(self.shift_size)
    original_shape = x.shape
    grid = x.reshape(b, h, w, c)
    wh, ww = window
    hp, wp = h + (-h) % wh, w + (-w) % ww
    padded = F.pad(grid, (0, 0, 0, wp - w, 0, hp - h))
    valid = None
    if hp != h or wp != w:
        valid = F.pad(torch.ones_like(grid[..., :1]), (0, 0, 0, wp - w, 0, hp - h))
        valid = partition(valid.roll(tuple(-s for s in shift), (1, 2)), window)
    if cfg["first"]:
        strengths = (
            partition(padded, window).flatten(1).double().norm(dim=1).reshape(b, -1)
        )
        rank_cache.set({**rank_cache.get(), cfg["stage"]: strengths})
    scores = rank_cache.get()[cfg["stage"]]
    nw = hp // wh * (wp // ww)
    require(
        scores.shape == (b, nw),
        "stage geometry changed before window routing completed",
    )
    # Shifted blocks use the mean score of physical member patches from the original stage grid.
    field = restore(
        scores.reshape(-1, 1, 1).expand(-1, wh * ww, 1), (b, hp, wp, 1), window
    )
    rolls = tuple(-s for s in shift)
    order = (
        partition(field.roll(rolls, (1, 2)), window)
        .mean((1, 2))
        .reshape(b, nw)
        .argsort(descending=True, stable=True)
    )
    order = order + torch.arange(b, device=x.device)[:, None] * nw
    src = partition(padded.roll(rolls, (1, 2)), window)
    out = torch.empty_like(src)
    mask = (
        kwargs.get("mask_matrix", args[0] if args else getattr(self, "attn_mask", None))
        if any(shift)
        else None
    )
    if any(shift):
        require(mask is not None, "shifted window mask must be bound")
    n = nw - round(nw * cfg["low_fraction"])
    for low, ids in ((False, order[:, :n].flatten()), (True, order[:, n:].flatten())):
        if ids.numel() == 0:
            continue
        token = branch.set(cfg["low_bits"] if low else None)
        try:
            z = src.index_select(0, ids)
            ln1, ln2 = (
                (self.ptq_low_norm1, self.ptq_low_norm2)
                if low
                else (self.norm1, self.norm2)
            )
            args1 = {"mask": mask.index_select(0, ids % nw)} if mask is not None else {}
            normalized = ln1(z)
            if valid is not None and x.ndim == 3:
                normalized = normalized * valid.index_select(0, ids)
            z = z + self.attn(normalized, **args1)
            z = z + self.mlp(ln2(z))
            out.index_copy_(0, ids, z)
        finally:
            branch.reset(token)
    if cfg["last"]:
        rank_cache.set({k: v for k, v in rank_cache.get().items() if k != cfg["stage"]})
    return (
        restore(out, (b, hp, wp, c), window)
        .roll(shift, (1, 2))[:, :h, :w]
        .reshape(original_shape)
    )


def bind(model, stage, frozen=False):
    m = model.get_submodule(stage["module"])
    cfg = stage["settings"]
    require(
        0 <= cfg["low_fraction"] <= 1
        and cfg["low_bits"] > 0
        and all(
            hasattr(m, n)
            for n in ("norm1", "norm2", "attn", "mlp", "window_size", "shift_size")
        ),
        "window stage needs a verified Swin residual block",
    )
    m.ptq_low_norm1 = deepcopy(m.norm1)
    m.ptq_low_norm2 = deepcopy(m.norm2)
    m.ptq_window = deepcopy(cfg)
    m.ptq_original_forward = m.forward
    m.ptq_routing_active = frozen
    m.forward = MethodType(routed, m)


def activate(model, stages):
    for row in stages:
        if row["kind"] == "window_branches":
            m = model.get_submodule(row["module"])
            m.ptq_routing_active = True
            for child in m.modules():
                if hasattr(child, "ptq_site_id"):
                    child.ptq_window_member = True
