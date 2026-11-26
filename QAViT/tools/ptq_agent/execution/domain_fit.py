"""DAQuant code-derived per-channel clipping and four finite parameter groups, with an explicit deployment mode."""

from contextlib import ExitStack
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from ..schemas import require
from . import grids
from .calibration import differentiable_forward
from .reconstruction import paired_block
from .task_sensitivity import substitute


def affine_init(x, weight, *, settings):
    require(
        x.ndim == 3 and x.shape[1] > 1,
        "DGA initialization needs token rows for each image",
    )
    act = x.detach().abs().std(1).amax(0).clamp_min(settings["min_scale"])
    std = weight.std(0).clamp_min(settings["min_scale"])
    if settings["weight_statistic"] == "author_first_coordinate":
        std = std[0]
    else:
        require(
            settings["weight_statistic"] == "per_channel_adapter",
            "unbound DGA statistic",
        )
    scale = (act / std).sqrt().clamp_min(settings["min_scale"])
    means = x.detach().mean(1)
    shift = means[0].clone()
    for m in means[1:]:
        shift.lerp_(m, 0.01)
    return {
        "scale": scale,
        "shift": shift / scale if settings["use_shift"] else torch.zeros_like(scale),
    }, None


def initial_clip(x, cfg):
    require(
        x.ndim == 3 and cfg["range_mode"] in ("instance_tokens", "frozen_channels"),
        "DA activation needs BTC and a declared range mode",
    )
    lo = x.amin((0, 1), keepdim=True)
    hi = x.amax((0, 1), keepdim=True)
    gate = torch.full_like(lo, cfg["initial_logit"]).sigmoid()
    return {"lower": lo, "upper": hi, "low_gate": gate, "high_gate": gate.clone()}


def grid(lo, hi, low_gate, high_gate, bits, cfg):
    scale = ((hi * high_gate - lo * low_gate) / ((1 << bits) - 1)).clamp(
        cfg["min_scale"], cfg["max_scale"]
    )
    zero = (
        (-lo * low_gate / scale)
        .clamp(-cfg["max_zero"], cfg["max_zero"])
        .round()
        .detach()
    )
    return {"scale": scale, "zero": zero}


def quant(x, p, bits):
    scaled = x / p["scale"]
    rounded = scaled + (scaled.round() - scaled).detach()
    return ((rounded + p["zero"]).clamp(0, (1 << bits) - 1) - p["zero"]) * p["scale"]


def activation(x, state, bits, cfg):
    lo, hi = (
        (x.amin(1, keepdim=True), x.amax(1, keepdim=True))
        if cfg["range_mode"] == "instance_tokens"
        else (state["lower"], state["upper"])
    )
    return quant(
        x, grid(lo, hi, state["low_gate"], state["high_gate"], bits, cfg), bits
    )


def fit(backend, model, data, stage, cache_mib):
    from .pipeline import quantizers

    name, cfg = stage["module"], stage["settings"]
    rows = paired_block(backend, model, data, name, cache_mib)
    pairs = cfg["target_pairs"]
    images = {sid: x for x, sid in data.batches("calibration")}
    require(
        pairs and set(pairs).union(pairs.values()) <= images.keys(),
        "domain pairs outside permitted calibration",
    )
    rows = [r for r in rows if r[0] in pairs]
    block = model.get_submodule(name)
    fp = backend.model.get_submodule(name)
    members = {
        s: q
        for s, q in quantizers(model).items()
        if q in set(block.modules()) and q.layer["activation_method"] == "domain_clip"
    }
    require(
        members
        and all(q.layer["weight_method"] == "uniform_affine" for q in members.values()),
        "DA fit requires packed affine W and declared LAC sites",
    )
    variables = {}
    groups = {n: [] for n in ("DGA", "LWC", "WRC", "LAC")}
    for site, q in members.items():
        w = backend.model.get_submodule(site).weight.detach()
        dtype = w.dtype

        def variable(group, value):
            param = nn.Parameter(value.detach().clone().to(w))
            groups[group].append(param)
            return param

        e = q.layer["options"]["encoding"]
        start = e["initial_logit"]
        gate_w = torch.full_like(w[:, :1], start)
        gate_a = q.params()["low_gate"].new_full((1, 1, w.shape[-1]), start)
        variables[site] = (
            w,
            variable("DGA", q.input_scale.log()),
            variable("DGA", q.input_scale * q.input_shift),
            variable("LWC", gate_w),
            variable("LWC", gate_w),
            variable("WRC", torch.zeros_like(w)),
            variable("LAC", gate_a),
            variable("LAC", gate_a),
        )

    def current(site, q):
        w, logs, shift, wl, wh, wrc, al, ah = variables[site]
        e = q.layer["options"]["encoding"]
        scale = logs.exp().clamp_min(e["min_scale"])
        base = w * scale
        p = grid(
            base.amin(1, keepdim=True),
            base.amax(1, keepdim=True),
            wl.sigmoid(),
            wh.sigmoid(),
            q.layer["weight"]["bits"],
            e,
        )
        return scale, shift, base + wrc, p, al.sigmoid(), ah.sigmoid()

    targets = {}
    with torch.no_grad():
        for sid in dict.fromkeys(pairs.values()):
            values = []
            with substitute(fp, lambda y: values.append(y.detach())):
                differentiable_forward(backend, backend.model, images[sid])
            require(len(values) == 1, "target domain block must execute once")
            targets[sid] = values[0]

    def forward_for(site):
        def run(q, x):
            scale, shift, v, p, al, ah = current(site, q)
            local = (x - shift) / scale
            e = q.layer["options"]["encoding"]
            a = activation(
                local,
                {**q.params(), "low_gate": al, "high_gate": ah},
                q.layer["activation"]["bits"],
                e,
            )
            bias = variables[site][0] @ shift
            if q.bias is not None:
                bias = bias + (q.bias if q.bias_mask is None else q.bias * q.bias_mask)
            return F.linear(a, quant(v, p, q.layer["weight"]["bits"]), bias)

        return run

    def refresh():
        with ExitStack() as stack, torch.no_grad():
            limits = {}
            for site, q in members.items():

                def observe(m, args, site=site):
                    scale, shift, *_ = current(site, m)
                    v = (args[0] - shift) / scale
                    lo, hi = v.amin((0, 1), keepdim=True), v.amax((0, 1), keepdim=True)
                    if site in limits:
                        lo, hi = (
                            torch.minimum(lo, limits[site][0]),
                            torch.maximum(hi, limits[site][1]),
                        )
                    limits[site] = (lo, hi)

                stack.callback(q.register_forward_pre_hook(observe).remove)
            for _, _, x, _, args, kwargs in rows:
                block(x, *args, **kwargs)
            for site, (lo, hi) in limits.items():
                q = members[site]
                q.set_params({**q.params(), "lower": lo, "upper": hi})

    optimizer = torch.optim.AdamW(
        [{"params": v, "lr": cfg["lr"][k]} for k, v in groups.items()],
        weight_decay=cfg["weight_decay"],
    )
    rng = torch.Generator().manual_seed(cfg["seed"])
    history = []
    with ExitStack() as undo, torch.enable_grad():
        for site, q in members.items():
            q.forward = MethodType(forward_for(site), q)
            undo.callback(delattr, q, "forward")
        for phase in cfg["phases"]:
            require(
                phase["steps"] > 0 and phase["kind"] in ("source", "alignment"),
                "unknown DA phase",
            )
            losses = []
            for _ in range(phase["steps"]):
                refresh()
                terms = []
                for i in torch.randperm(len(rows), generator=rng)[
                    : cfg["batch_images"]
                ].tolist():
                    sid, _, x, y, args, kwargs = rows[i]
                    out = block(x, *args, **kwargs)
                    loss = F.smooth_l1_loss(out, y)
                    if phase["kind"] == "alignment":
                        loss = (
                            loss
                            + cfg["alignment_weight"]
                            * F.kl_div(
                                out.log_softmax(2),
                                targets[pairs[sid]].softmax(2),
                                reduction=cfg["kl_reduction"],
                            ).abs()
                        )
                    terms.append(loss)
                loss = torch.stack(terms).mean()
                require(torch.isfinite(loss), "nonfinite DA loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append({"phase": phase, "loss": losses})
        refresh()
        with torch.no_grad():
            for site, q in members.items():
                scale, shift, v, p, al, ah = current(site, q)
                q.input_scale.copy_(scale)
                q.input_shift.copy_(shift / scale)
                q.reference_weight = variables[site][0] * scale
                r = q.weight_record
                q.weight_record = grids.encode(v, r["spec"], p, method=r["method"])
                q.decoded_weight = grids.decode(q.weight_record, v.device)
                q.set_params({**q.params(), "low_gate": al, "high_gate": ah})
    return {
        "kind": "domain_affine_fit",
        "source_ids": list(pairs),
        "target_pairs": pairs,
        "updates": {k: len(v) for k, v in groups.items()},
        "phases": history,
        "deployment": {
            k: q.layer["options"]["encoding"]["range_mode"] for k, q in members.items()
        },
        "representation": "one packed W, affine input, fixed bound gates",
    }
