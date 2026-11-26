"""Finite calibration on actual residual/linear/block boundaries; export only final hard weights."""

from contextlib import ExitStack
from copy import copy, deepcopy
import math
import torch
from ..schemas import require
from . import grids
from .calibration import differentiable_forward
from .model import forward
from .operators import install_flows
from .plugin_contracts import site_id


def mark_residuals(net, assignments):
    targets = {
        g["observe"].rsplit(".", 1)[0]
        for g in assignments
        if g["observe"].endswith("._ptq_attention_residual")
    }
    install_flows(net, [{"module": n, "kind": "attention_residual"} for n in targets])
    net.eval()


def output(backend, net, path, x, side):
    values = []
    module = net.get_submodule(path)
    handle = (
        module.register_forward_pre_hook(lambda m, a: values.append(a[0]))
        if side == "input"
        else module.register_forward_hook(lambda m, a, y: values.append(y))
    )
    try:
        differentiable_forward(backend, net, x)
    finally:
        handle.remove()
    require(
        len(values) == 1 and torch.is_tensor(values[0]),
        "reconstruction boundary is absent, repeated or structured",
    )
    return values[0]


def tree(blocks, sites, initial_steps, initial_lr, *, activation_only=False):
    """PFCR Eq. 15, 18, 19. Each leaf includes its shortcut in the observed tensor."""
    leaves = []
    for block in blocks:
        for part, observe in (
            ("attn", block + "._ptq_attention_residual"),
            ("mlp", block),
        ):
            members = [
                name for name in sites if name.startswith(block + "." + part + ".")
            ]
            require(members, "missing mapped attention or MLP branch")
            leaves.append((members, observe))
    n = len(leaves)
    require(n, "no residual units")
    largest = int(math.log2(n)) if n & (n - 1) == 0 else round(math.log2(n)) - 1
    result = []
    for level in range(min(1, largest) + 1 if activation_only else largest + 1):
        width = 1 << level
        for start in range(0, n - width + 1, width):
            group = leaves[start : start + width]
            result.append(
                {
                    "observe": group[-1][1],
                    "side": "output",
                    "weight_modules": [v for g, _ in group for v in g],
                    "level": level,
                    "steps": max(1, round(initial_steps * (1 + 0.2 * level))),
                    "lr": initial_lr * (1 - 0.2 * level),
                }
            )
    return result


def optimize(backend, model, data, settings, *, fp_phase=False):
    from .pipeline import quantizers

    teacher = deepcopy(backend.model).eval().requires_grad_(False)
    groups = settings["groups"]
    require(
        groups and settings["batch_images"] > 0, "empty graph reconstruction schedule"
    )
    mark_residuals(model, groups)
    mark_residuals(teacher, groups)
    qmap = quantizers(model)
    rows = list(data.batches("calibration"))
    require(
        len(rows) == len(data.ids("calibration")) and rows,
        "finite calibration split required",
    )
    generator = torch.Generator().manual_seed(settings["seed"])
    values = {}
    log = []
    for group in groups:
        sites = group["weight_modules"]
        require(
            sites and len(set(sites)) == len(sites) and all(k in qmap for k in sites),
            "unbound or duplicate W",
        )
        observe = group["observe"]
        side = group.get("side", "output")
        require(side in ("input", "output"), "unknown observed boundary")
        with torch.no_grad():
            targets = [
                output(backend, teacher, observe, x, side).detach() for x, _ in rows
            ]
        require(
            sum(v.numel() * v.element_size() for v in targets)
            <= settings["cache_mib"] * (1 << 20),
            "calibration target cache exhausted",
        )
        members = {k: qmap[k] for k in sites}
        parameters = {}
        for k, q in members.items():
            require(
                hasattr(q, "reference_weight")
                and (fp_phase or q.weight_record["method"] in grids.METHODS),
                "unsupported W fit grid",
            )
            start = values.get(k, q.reference_weight)
            parameters[k] = torch.nn.Parameter(start.detach().clone())
        opt = torch.optim.Adam(
            list(parameters.values()), lr=group["lr"], weight_decay=0.0
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=group["steps"])
        scores = []
        try:
            for q in qmap.values():
                q.gradient_probe = True
            with torch.enable_grad():
                for i in range(group["steps"]):
                    for k, q in members.items():
                        w = parameters[k]
                        if fp_phase:
                            decoded = w
                        else:
                            record = q.weight_record
                            rounded = grids.apply(
                                w, record["spec"], record["method"], record["params"]
                            )
                            decoded = w + (rounded - w).detach()
                        # Parameter stored in the optimizer, not registered on the frozen model.
                        q.decoded_weight = decoded.view_as(w)
                        q.weight_enabled = True
                    batch = torch.randperm(len(rows), generator=generator)[
                        : settings["batch_images"]
                    ].tolist()
                    loss = torch.stack(
                        [
                            (
                                output(backend, model, observe, rows[j][0], side)
                                - targets[j]
                            )
                            .square()
                            .mean()
                            for j in batch
                        ]
                    ).mean()
                    require(
                        bool(torch.isfinite(loss)) and loss.requires_grad,
                        "no differentiable connection between selected W and target",
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                    if i in (0, group["steps"] - 1):
                        scores.append(float(loss.detach()))
            with torch.no_grad():
                for k, q in members.items():
                    values[k] = parameters[k].detach()
                    if fp_phase:
                        q.decoded_weight = values[k]
                    else:
                        r = q.weight_record
                        q.weight_record = grids.encode(
                            values[k], r["spec"], r["params"], method=r["method"]
                        )
                        q.decoded_weight = grids.decode(
                            q.weight_record, q.reference_weight.device
                        )
        finally:
            for q in qmap.values():
                q.gradient_probe = False
        log.append(
            {
                "boundary": observe,
                "side": side,
                "level": group.get("level"),
                "updated": sites,
                "steps": group["steps"],
                "lr": group["lr"],
                "loss": scores,
                "prefix": "current candidate full forward",
                "reference": "original FP",
            }
        )
    return values, log


def activation_phase(backend, cfg, data, plan):
    from .pipeline import QuantizationExecutor
    from .sites import site_catalog

    phase = plan["activation_first"]
    catalog = site_catalog(backend)
    temporary = deepcopy(plan)
    for k in ("activation_first", "postfit", "preprocess", "structure", "operators"):
        temporary.pop(k, None)
    temporary["preprocess"] = []
    temporary["execution"].update(order="joint", within_stage="sequential")
    for row in temporary["layers"]:
        row["weight"] = row["weight_method"] = None
        row.pop("weight_calibration", None)
        if (
            phase["post_ln"] == "per_channel"
            and row["activation"]
            and catalog[site_id(row)]["semantic"] == "post_layernorm"
        ):
            row["activation"].update(granularity="per_channel", axis=2, group_size=None)
    temporary["layers"] = [r for r in temporary["layers"] if r["activation"]]
    with torch.no_grad():
        model, *_ = QuantizationExecutor(backend, cfg, data, diagnostic=True)._prepare(
            temporary, {}
        )
    values, receipt = optimize(backend, model, data, phase, fp_phase=True)
    proxy = copy(backend)
    proxy.model = deepcopy(backend.model)
    covered = {r["module"] for r in plan["layers"] if r.get("weight")}
    require(
        set(values) <= covered,
        "every optimized W must be frozen into the final package",
    )
    with torch.no_grad():
        for name, w in values.items():
            proxy.model.get_submodule(name).weight.copy_(w)
    from ..backends.server import state_sha256

    proxy.original_state = state_sha256(proxy.model)
    return proxy, receipt


def restore_source(model, backend):
    """Fold the change to a transformed reference W into its constant output bias."""
    from .pipeline import quantizers, PipelineConv2d

    for name, q in quantizers(model).items():
        if not hasattr(q, "reference_weight"):
            continue
        source = backend.model.get_submodule(name).weight.detach() * q.input_scale
        if not isinstance(q, PipelineConv2d) and not q.bias_uses_decoded_weight:
            before = q.reference_weight @ q.input_shift
            delta = before - source @ q.input_shift
            if bool((delta != 0).any()):
                q.register_buffer("fitted_input_bias", delta)
        q.reference_weight = source
