"""Finite block fitting on paired calibration examples; hard state is exported by the native executor."""

from copy import deepcopy
from contextlib import ExitStack
import math
import torch
from torch import nn
from ..schemas import require
from . import adaround, codecs, grids
from .model import forward, tensor_digest


def paired_block(backend, model, data, path, cache_mib, end=None):
    """Capture one invocation per image. Preserve additional arguments (e.g. positional state)."""
    rows, current, used = [], {}, 0

    def capture(key):
        def hook(_m, args, kwargs, y):
            require(
                key not in current and torch.is_tensor(y) and torch.is_tensor(args[0]),
                "unsupported repeated or structured block",
            )
            current[key] = (args, kwargs, y)

        return hook

    with ExitStack() as stack, torch.no_grad():
        for key, net in (("fp", backend.model), ("q", model)):
            stack.callback(
                net.get_submodule(path)
                .register_forward_hook(capture(key), with_kwargs=True)
                .remove
            )
        if end is not None and end != path:

            def end_capture(_m, args, output):
                require(
                    "target" not in current and torch.is_tensor(output),
                    "slice output is repeated or structured",
                )
                current["target"] = output

            stack.callback(
                backend.model.get_submodule(end)
                .register_forward_hook(end_capture)
                .remove
            )
        for image, sid in data.batches("calibration"):
            current.clear()
            forward(backend, backend.model, image)
            forward(backend, model, image)
            fp, _, y = current["fp"]
            y = current.get("target", y)
            q, kwargs, _ = current["q"]
            require(q[0].shape == fp[0].shape, "FP and Q prefix layouts differ")
            used += sum(t.numel() * t.element_size() for t in (q[0], fp[0], y))
            require(
                used <= cache_mib * (1 << 20),
                "block cache exceeds calibration_cache_mib",
            )
            rows.append(
                (
                    sid,
                    fp[0].detach().clone(),
                    q[0].detach().clone(),
                    y.detach().clone(),
                    q[1:],
                    kwargs,
                )
            )
    require(
        [r[0] for r in rows] == list(data.ids("calibration")) and rows,
        "missing block calibration records",
    )
    return rows


def mixed(fp, q, drop, generator):
    if drop == 0:
        return q
    if drop == 1:
        return fp
    return torch.where(
        torch.rand(fp.shape, device=fp.device, generator=generator) < drop, fp, q
    )


def reconstruct_activation(x, quantizer):
    """STE through rounding only. Learned scales receive gradients from dequantization and clipping."""
    scale, drop, generator = quantizer._fitting
    params = quantizer.params()
    sc = scale.clamp_min(torch.finfo(scale.dtype).tiny)
    spec = quantizer.layer["activation"]
    zero = params.get("zero", 0)
    value = x
    if "layout_shape" in params:
        from .fixed_grids import layout

        value = layout(x, spec)
    signed = spec["symmetric"] and "layout_shape" not in params
    lo = -(1 << (spec["bits"] - 1)) if signed else 0
    hi = (
        lo
        + (1 << spec["bits"])
        - 1
        - int("layout_shape" in params and spec["symmetric"])
    )
    v = value / sc
    straight = v + (v.round() - v).detach()
    out = ((straight + zero).clamp(lo, hi) - zero) * sc
    if "layout_shape" in params:
        if spec["granularity"] == "per_group":
            out = out.flatten(-2)
        out = out.movedim(-1, spec["axis"])
    return mixed(x, out.to(x.dtype), drop, generator)


def fit_stage(backend, model, data, stage, cache_mib):
    from .pipeline import quantizers

    name, cfg = stage["module"], stage["settings"]
    if stage["kind"] == "block_coordinate_search":
        from .block_search import search

        return search(backend, model, data, stage, cache_mib)
    if stage["kind"] == "bit_shrinking":
        from .progressive_bits import fit

        return fit(backend, model, data, stage, cache_mib)
    if stage["kind"] == "domain_affine_fit":
        from .domain_fit import fit

        return fit(backend, model, data, stage, cache_mib)
    if stage["kind"] == "merged_weight_fit":
        from .block_objectives import merged_fit

        return merged_fit(backend, model, data, stage, cache_mib)
    if stage["kind"] == "scale_search":
        from .research_search import search_stage

        return search_stage(backend, model, data, stage)
    steps, batch, seed = cfg["steps"], cfg["batch_images"], cfg["seed"]
    require(
        steps >= 1 and batch >= 1 and 0 <= cfg["drop_probability"] <= 1,
        "invalid block budget",
    )
    block = model.get_submodule(name)
    subset = cfg.get("slice")
    all_sites = quantizers(model)
    members = (
        {s: all_sites[s] for s in subset["sites"]}
        if subset
        else {
            s: q
            for s, q in all_sites.items()
            if q is block or q in set(block.modules())
        }
    )
    require(
        members and cfg["update"] in ("scale", "rounding_and_scale"),
        "no supported block fit target",
    )
    rows = paired_block(
        backend, model, data, name, cache_mib, subset["end"] if subset else None
    )
    rng = torch.Generator(device=rows[0][1].device).manual_seed(seed)
    rounds, scales = {}, {}
    require(
        stage["reference"] in ("fp_model", "local_fp_candidate"), "unknown block target"
    )
    fp_block = backend.model.get_submodule(name)
    objective_cfg = cfg.get("objective")
    images = (
        dict((sid, x) for x, sid in data.batches("calibration"))
        if objective_cfg or subset
        else {}
    )
    if cfg.get("bn_dc"):
        from .block_objectives import bn_correct

        corrected = []
        for sid, xf, xq, _, args, kwargs in rows:
            dc = bn_correct(fp_block, xf, args, kwargs, cfg["bn_dc"])
            with torch.no_grad():
                target = fp_block(dc, *args, **kwargs)
            corrected.append((sid, xf, xq, target, args, kwargs))
        rows = corrected
    curvature = None
    if objective_cfg:
        from .research_metrics import SENSITIVITY

        if objective_cfg["kind"] in SENSITIVITY:
            from .curvature_fit import BlockObjective

            curvature = BlockObjective(
                backend,
                backend.model,
                images,
                data.labels,
                name,
                objective_cfg,
                cache_mib,
            )

    def slice_value(image, first):
        from .calibration import differentiable_forward

        result = []

        def start(_m, args):
            return (first, *args[1:])

        with ExitStack() as scope:
            scope.callback(block.register_forward_pre_hook(start).remove)
            scope.callback(
                model.get_submodule(subset["end"])
                .register_forward_hook(lambda m, a, y: result.append(y))
                .remove
            )
            differentiable_forward(backend, model, image)
        require(len(result) == 1, "slice output must be observed exactly once")
        return result[0]

    history = []
    try:
        for sid, q in members.items():
            if q.layer.get("weight") and cfg["update"] == "rounding_and_scale":
                record = q.weight_record
                require(
                    record.get("method") in grids.METHODS
                    and q.layer["weight"]["granularity"]
                    in ("per_channel", "per_tensor"),
                    "block rounding requires an explicit native uniform weight grid",
                )
                sc = record["params"]["scale"].to(q.reference_weight)
                rounds[sid] = nn.Parameter(adaround.initialize(q.reference_weight, sc))
            if q.layer["activation"]:
                p = q.params()
                require(
                    q.layer["activation_method"] in grids.METHODS
                    or "layout_shape" in p,
                    "scale reconstruction needs declared affine uniform grids",
                )
                require(
                    "repq_affine" not in p,
                    "fractional RepQ endpoints require their own backward rule",
                )
                scales[sid] = nn.Parameter(p["scale"].detach().float().clone())
                q._fitting = (scales[sid], cfg["drop_probability"], rng)
        require(scales or rounds, "no learnable quantization state")
        optim_w = (
            torch.optim.Adam(list(rounds.values()), lr=cfg["round_lr"])
            if rounds
            else None
        )
        optim_a = (
            torch.optim.Adam(list(scales.values()), lr=cfg["scale_lr"])
            if scales
            else None
        )
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optim_a, T_max=steps)
            if optim_a
            else None
        )

        def weights(hard):
            for sid, alpha in rounds.items():
                q = members[sid]
                record = q.weight_record
                spec = q.layer["weight"]
                w = q.reference_weight
                sc = record["params"]["scale"].to(w)
                zp = record["params"].get("zero", 0)
                if torch.is_tensor(zp):
                    zp = zp.to(w)
                offset = (1 << (spec["bits"] - 1)) if spec["symmetric"] else 0
                frac = (alpha >= 0).to(w) if hard else adaround.soft_round(alpha)
                code = ((w / sc).floor() + frac + zp + offset).clamp(
                    0, (1 << spec["bits"]) - 1
                )
                q.decoded_weight = ((code - offset - zp) * sc).to(w)
                if hard:
                    q.weight_record = {
                        **record,
                        "data": codecs.pack(
                            code.to(codecs.code_dtype(spec["bits"])), spec["bits"]
                        ),
                    }

        with torch.enable_grad():
            for step in range(steps):
                for opt in (optim_a, optim_w):
                    if opt:
                        opt.zero_grad(set_to_none=True)
                weights(False)
                if curvature:
                    curvature.refresh(model, step)
                losses = []
                for i in torch.randperm(
                    len(rows), device=rows[0][1].device, generator=rng
                )[:batch].tolist():
                    _, xf, xq, y, args, kwargs = rows[i]
                    x = mixed(xf, xq, cfg["drop_probability"], rng)
                    if stage["reference"] == "local_fp_candidate":
                        with torch.no_grad():
                            y = fp_block(x, *args, **kwargs)
                    if curvature:
                        losses.append(
                            curvature.evaluate(y, block(x, *args, **kwargs), i)
                        )
                    elif objective_cfg:
                        from .block_objectives import objective

                        losses.append(
                            objective(
                                backend,
                                model,
                                data,
                                name,
                                fp_block,
                                block,
                                rows[i],
                                images[rows[i][0]],
                                objective_cfg,
                                x,
                                y,
                            )
                        )
                    else:
                        out = (
                            slice_value(images[rows[i][0]], x)
                            if subset
                            else block(x, *args, **kwargs)
                        )
                        losses.append((out - y).square().sum(-1).mean())
                loss = torch.stack(losses).mean()
                reg = sum(
                    adaround.rounding_loss(
                        a,
                        step,
                        steps,
                        cfg.get("round_reg", 0.01),
                        cfg.get("warmup", 0.2),
                        tuple(cfg.get("beta_range", [20.0, 2.0])),
                        schedule=cfg.get("beta_schedule", "cosine"),
                    )
                    for a in rounds.values()
                )
                require(
                    bool(torch.isfinite(loss + reg)), "nonfinite block reconstruction"
                )
                (loss + reg).backward()
                for opt in (optim_w, optim_a):
                    if opt:
                        opt.step()
                if sched:
                    sched.step()
                with torch.no_grad():
                    for sc in scales.values():
                        sc.clamp_min_(torch.finfo(sc.dtype).eps)
                if step in (0, steps - 1):
                    history.append(float(loss.detach()))
        with torch.no_grad():
            weights(True)
            for sid, sc in scales.items():
                q = members[sid]
                p = q.params()
                q.set_params({**p, "scale": sc.detach().to(p["scale"])})
        return {
            "kind": "block_reconstruction",
            "module": name,
            "reference": stage["reference"],
            "update": cfg["update"],
            "calibration_ids": [r[0] for r in rows],
            "iterations": steps,
            "optimizer_steps": steps * (bool(optim_w) + bool(optim_a)),
            "drop_probability": cfg["drop_probability"],
            "initial_final_reconstruction": history,
            "slice": subset,
            "rounding_sites": list(rounds),
            "scale_sites": list(scales),
            "curvature_refreshes": curvature.refreshes if curvature else [],
            "objective": objective_cfg or {"kind": "block_output_squared_error"},
            "bn_dc": cfg.get("bn_dc"),
            "final_state": "hard weight codes and fixed scales; drop disabled",
        }
    finally:
        for q in members.values():
            if hasattr(q, "_fitting"):
                del q._fitting
