"""Continuous calibration resolution, Eq. 7 adjacent sharpness and bounded bisection. Final grid is integral."""

import torch
from ..schemas import require
from . import grids
from .reconstruction import paired_block


def next_precision(current, target, sharpness, ceiling, tolerance, tries):
    base = sharpness(current)
    remaining = sharpness(target) - base
    if remaining <= ceiling + tolerance:
        return target, [(target, remaining)]
    high_precision, low_precision = current, target
    trials = []
    for _ in range(tries):
        mid = (high_precision + low_precision) / 2
        require(
            target < mid < current, "continuous precision stagnated before the target"
        )
        delta = sharpness(mid) - base
        trials.append((mid, delta))
        if abs(delta - ceiling) <= tolerance:
            return mid, trials
        if delta > ceiling:
            low_precision = mid
        else:
            high_precision = mid
    feasible = [b for b, v in trials if v <= ceiling + tolerance]
    require(
        feasible,
        "sharpness discontinuity: no feasible progress inside the declared bisection budget",
    )
    return min(feasible), trials


def continuous(x, spec, params, width):
    b = spec["bits"]
    factor = 2 ** (b - width)
    s = params["scale"].to(x) * factor
    z = params.get("zero", torch.zeros_like(s)).to(x)
    z = (z / factor).round()
    lo = -(2 ** (width - 1)) if spec["symmetric"] else 0.0
    hi = lo + 2**width - 1
    v = (x / s + z).round().clamp(lo, hi)
    grid = (v - z) * s
    return x + (grid - x).detach()


def fit(backend, model, data, stage, cache_mib):
    from .pipeline import quantizers

    path = stage["module"]
    cfg = stage["settings"]
    block = model.get_submodule(path)
    qs = {k: q for k, q in quantizers(model).items() if q in set(block.modules())}
    weights = {k: q for k, q in qs.items() if q.layer["weight"]}
    require(
        weights
        and all(q.weight_record["method"] in grids.METHODS for q in weights.values()),
        "bit shrinking needs uniform W",
    )
    activations = {k: q for k, q in qs.items() if q.layer["activation"]}
    require(
        all(
            q.layer["activation_method"] in grids.METHODS for q in activations.values()
        ),
        "temporary bit schedule requires uniform A",
    )
    wbits = {q.layer["weight"]["bits"] for q in weights.values()}
    abits = {q.layer["activation"]["bits"] for q in activations.values()}
    require(len(wbits) == len(abits) == 1, "bind per-block W/A target widths")
    wt, at = wbits.pop(), abits.pop()
    target = min(wt, at)
    current = cfg["start_bits"]
    require(current >= max(wt, at), "start precision below target")
    rows = paired_block(backend, model, data, path, cache_mib)
    required = cfg["sharpness_images"]
    require(
        0 < required <= len(rows), "sharpness pool exceeds permitted calibration set"
    )
    gen = torch.Generator().manual_seed(cfg["seed"])
    wvars = {
        k: torch.nn.Parameter(q.reference_weight.clone()) for k, q in weights.items()
    }

    def set_width(b):
        for k, q in weights.items():
            v = wvars[k].view_as(wvars[k])
            q.decoded_weight = (
                v
                if b is None
                else continuous(
                    v, q.layer["weight"], q.weight_record["params"], max(b, wt)
                )
            )
        for q in activations.values():
            q._continuous_width = None if b is None else max(b, at)

    def loss(row):
        _, _, x, y, args, kwargs = row
        return (block(x, *args, **kwargs) - y).square().mean()

    def sharp(b):
        with torch.no_grad():
            set_width(None)
            fp = torch.stack([loss(r) for r in rows[:required]]).mean()
            set_width(b)
            value = torch.stack([loss(r) for r in rows[:required]]).mean()
            return float(value - fp)

    trajectory = []
    try:
        total = sharp(target)
        require(
            total > 0, "target perturbation has no positive loss excess for this block"
        )
        ceiling = total * cfg["ceiling_fraction"]
        tolerance = total * cfg["tolerance_fraction"]
        while True:
            opt = torch.optim.Adam(wvars.values(), lr=cfg["lr"])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **cfg["plateau"])
            with torch.enable_grad():
                for step in range(cfg["steps_per_width"]):
                    set_width(current)
                    batch = torch.randperm(len(rows), generator=gen)[
                        : cfg["batch_images"]
                    ].tolist()
                    value = torch.stack([loss(rows[j]) for j in batch]).mean()
                    opt.zero_grad()
                    value.backward()
                    opt.step()
                    sched.step(float(value.detach()))
            receipt = {"width": current, "objective": float(value.detach())}
            trajectory.append(receipt)
            if current == target:
                break
            require(
                len(trajectory) < cfg["max_stages"],
                "stage budget exhausted before final width",
            )
            current, receipt["bisection"] = next_precision(
                current, target, sharp, ceiling, tolerance, cfg["bisection_steps"]
            )
        for k, q in weights.items():
            r = q.weight_record
            q.weight_record = grids.encode(
                wvars[k].detach(), r["spec"], r["params"], method=r["method"]
            )
            q.decoded_weight = grids.decode(q.weight_record, q.reference_weight.device)
    finally:
        for q in activations.values():
            if hasattr(q, "_continuous_width"):
                del q._continuous_width
    return {
        "kind": "bit_shrinking",
        "module": path,
        "total_sharpness": total,
        "ceiling": ceiling,
        "tolerance": tolerance,
        "sharpness_samples": [r[0] for r in rows[:required]],
        "stages": trajectory,
        "final_weight_bits": wt,
        "final_activation_bits": at,
    }
