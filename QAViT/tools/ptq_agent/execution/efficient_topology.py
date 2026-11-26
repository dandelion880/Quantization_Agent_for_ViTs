"""Trio-ViT MBConv reparameterization and explicit normalized linear-attention denominator."""

import torch
from torch.nn import functional as F
from ..schemas import require


def channel_migration(x, w, *, settings):
    require(
        x.ndim == 4 and w.ndim == 4 and w.shape[:2] == (x.size(1), 1),
        "migration needs depthwise NCHW convolution",
    )
    maxima = x.amax((0, 2, 3))
    threshold = torch.maximum(
        maxima.mean(), torch.quantile(x.flatten(), settings["percentile"])
    )
    require(threshold > 0, "nonpositive migration threshold")
    scale = (maxima / threshold).clamp(*settings["ratio_bounds"])
    return {"scale": scale, "shift": torch.zeros_like(scale)}, None


def filter_shift(x, w, *, settings):
    require(
        x.ndim == 4 and w.shape[1:] == (x.size(1), 1, 1),
        "shifting requires a 1x1 dense pointwise edge",
    )
    if settings["center"] == "mean":
        z = x.mean((0, 2, 3))
    else:
        require(settings["center"] == "extrema_midpoint", "unknown shift statistic")
        z = (x.amax((0, 2, 3)) + x.amin((0, 2, 3))) / 2
    return {"scale": torch.ones_like(z), "shift": z}, None


def denominator(x, cfg):
    require(
        torch.isfinite(x).all() and x.amin() >= 0,
        "positive-kernel attention denominator required",
    )
    low, high = cfg["exponent_bounds"]
    require(low <= high, "unsupported exponent interval")
    if cfg["version"] == "author_5628eb2":
        require(
            [low, high] == [-12, 11],
            "Trio public code fixes exponent clamp at [-12,11]",
        )
        exp = torch.ceil(x.log2())
    else:
        require(
            cfg["version"] == "bounded_nearest" and high - low + 1 <= 1 << cfg["bits"],
            "denominator grid exceeds declared capacity",
        )
        exp = x.log2().floor().clamp_min(low)
        exp = exp + (x > 1.5 * torch.exp2(exp))
    return torch.exp2(exp.clamp(low, high))


def lite_forward(self, x):
    """EfficientViT LiteMLA layout [B,H,N,D], pad V by one before computing K^T V."""
    n, _, h, w = x.shape
    first = self.qkv(x)
    qkv = torch.cat([first, *[branch(first) for branch in self.aggreg]], 1)
    qkv = qkv.reshape(n, -1, 3 * self.dim, h * w).transpose(-1, -2)
    q, k, v = qkv.split(self.dim, -1)
    q, k = self.kernel_func(q), self.kernel_func(k)
    products = q @ (k.transpose(-1, -2) @ F.pad(v, (0, 1), value=1.0))
    y = products[..., :-1] / denominator(products[..., -1:], self.ptq_denominator)
    return self.proj(y.transpose(-1, -2).reshape(n, -1, h, w))


def bind_lite(module, cfg):
    from types import MethodType

    require(
        all(hasattr(module, a) for a in ("qkv", "aggreg", "proj", "kernel_func", "dim"))
        and module.dim > 0,
        "Trio denominator requires an actual LiteMLA QKV/aggregation/projection",
    )
    module.ptq_denominator = dict(cfg)
    module.forward = MethodType(lite_forward, module)
