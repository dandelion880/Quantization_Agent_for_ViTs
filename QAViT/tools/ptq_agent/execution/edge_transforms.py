"""Calibration-only equivalent Linear input transformations, effective bias uses the bound decoded W."""

import torch
from ..schemas import require


def shifted_noise(x, weight, *, spec, method, settings):
    from .methods import get_target_method

    op = get_target_method("activation", method)
    frozen = op.fit(x, spec, options={})
    rng = torch.Generator(device=x.device).manual_seed(settings["seed"])
    distribution = settings["distribution"]
    require(
        distribution in ("gaussian", "uniform")
        and settings["objective"] in ("input", "linear_output"),
        "declare the fixed noise distribution and scoring point",
    )
    shape = (x.size(-1),)
    direction = (
        torch.randn(shape, generator=rng, device=x.device)
        if distribution == "gaussian"
        else torch.rand(shape, generator=rng, device=x.device) * 2 - 1
    )
    best = None
    for mean in settings["means"]:
        for amplitude in settings["amplitudes"]:
            require(amplitude >= 0, "negative noise amplitude")
            noise = direction * amplitude + mean
            err = op.apply(x + noise, spec, params=frozen, options={}) - noise - x
            if settings["objective"] == "linear_output":
                err = err @ weight.T
            loss = float(err.double().square().mean())
            if best is None or loss < best[0]:
                best = loss, noise
    require(best is not None, "empty noise candidate set")
    return {"scale": torch.ones_like(best[1]), "shift": -best[1]}, frozen


def centered_smoothing(x, weight, *, settings):
    center = x.reshape(-1, x.shape[-1]).mean(0)
    maximum = (x - center).reshape(-1, x.shape[-1]).abs().amax(0).clamp_min(1e-12)
    alpha = settings["alpha"]
    require(0 <= alpha <= 1, "smoothing alpha outside [0,1]")
    ratio = maximum.pow(alpha) / weight.abs().amax(0).clamp_min(1e-12).pow(1 - alpha)
    if settings.get("pot_scale"):
        ratio = torch.exp2(ratio.log2().round())
    return {"scale": ratio, "shift": center / ratio}, None


def bilateral_clip(x, bits, settings):
    """Independent upper/lower sigmoid shrinkage, affine zero point and STE on calibration rows."""
    from torch import nn

    rows = x.detach().reshape(-1, x.shape[-1]).float()
    lo0 = rows.amin(0).clamp_max(0)
    hi0 = rows.amax(0).clamp_min(0)
    levels = (1 << bits) - 1
    lo_param = nn.Parameter(torch.full_like(lo0, settings["initial_logit"]))
    hi_param = nn.Parameter(torch.full_like(hi0, settings["initial_logit"]))
    optimizer = torch.optim.Adam((lo_param, hi_param), lr=settings["lr"])
    history = []
    with torch.enable_grad():
        for index in range(settings["steps"]):
            lo = lo0 * lo_param.sigmoid()
            hi = hi0 * hi_param.sigmoid()
            scale = ((hi - lo) / levels).clamp_min(1e-12)
            z = (-lo / scale).round().clamp(0, levels).detach()
            raw = rows / scale
            q = (raw + (raw.round() - raw).detach() + z).clamp(0, levels)
            loss = ((q - z) * scale - rows).square().mean(0).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index in (0, settings["steps"] - 1):
                history.append(float(loss.detach()))
    with torch.no_grad():
        lo = lo0 * lo_param.sigmoid()
        hi = hi0 * hi_param.sigmoid()
        scale = ((hi - lo) / levels).clamp_min(1e-12)
        return {"scale": scale, "zero": (-lo / scale).round().clamp(0, levels)}, {
            "initial_final_loss": history,
            "saturation_per_channel": ((rows < lo) | (rows > hi))
            .float()
            .mean(0)
            .tolist(),
            "min_per_channel": lo.tolist(),
            "max_per_channel": hi.tolist(),
            "steps": settings["steps"],
        }


def domain_affine(x, weight, *, settings):
    from .domain_fit import affine_init

    return affine_init(x, weight, settings=settings)
