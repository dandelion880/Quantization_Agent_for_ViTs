"""Differentiable local, intra-block and suffix losses used during bounded calibration."""

from contextlib import ExitStack
import math
import torch
import torch.nn.functional as F
from ..schemas import require
from .calibration import differentiable_forward
from .task_sensitivity import substitute
from .research_metrics import task_loss


def run_taps(block, x, args, kwargs, modules):
    seen = {}
    with ExitStack() as stack:
        for path in modules:

            def capture(m, a, y, name=path):
                require(
                    name not in seen and torch.is_tensor(y),
                    "repeated or structured within-block tap",
                )
                seen[name] = y

            stack.callback(
                block.get_submodule(path).register_forward_hook(capture).remove
            )
        result = block(x, *args, **kwargs)
    require(set(seen) == set(modules), "missing intra-block output")
    return result, seen


def bn_correct(block, fp, args, kwargs, cfg):
    modules = [
        m
        for m in block.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        and m.track_running_stats
    ]
    require(
        modules
        and all(
            m.running_mean is not None and m.running_var is not None for m in modules
        ),
        "PD DC needs pretrained BN running moments",
    )
    variable = torch.nn.Parameter(fp.detach().clone())
    opt = torch.optim.Adam([variable], lr=cfg["lr"])
    for _ in range(cfg["steps"]):
        moment = []

        def collect(m, a):
            x = a[0]
            axes = (0, *range(2, x.ndim))
            moment.append(
                (x.mean(axes) - m.running_mean).norm()
                + (
                    (x.var(axes, unbiased=False) + m.eps).sqrt()
                    - (m.running_var + m.eps).sqrt()
                ).norm()
            )

        with ExitStack() as stack:
            for m in modules:
                stack.callback(m.register_forward_pre_hook(collect).remove)
            block(variable, *args, **kwargs)
        loss = (variable - fp).square().sum() + cfg["lambda_c"] * sum(moment)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return variable.detach()


def objective(
    backend,
    model,
    data,
    block_name,
    fp_block,
    block,
    row,
    image,
    cfg,
    local_input,
    reference,
    domain_target=None,
):
    kind = cfg["kind"]
    sid, xf, xq, fp_output, args, kwargs = row
    if kind == "mgrq":
        with torch.no_grad():
            target, fp_inner = run_taps(
                fp_block, xf, args, kwargs, cfg["inner_modules"]
            )
            ft = differentiable_forward(backend, backend.model, image).detach()
        out, q_inner = run_taps(block, xf, args, kwargs, cfg["inner_modules"])
        qt = differentiable_forward(backend, model, image)
        inner = torch.stack(
            [(fp_inner[k] - q_inner[k]).square().mean() for k in q_inner]
        ).mean()
        return (
            (out - target).square().mean()
            + cfg["alpha"] * (qt - ft).square().mean()
            + cfg["beta"] * inner
        )
    if kind in ("pd_code", "pd_paper"):
        with torch.no_grad():
            teacher = differentiable_forward(backend, backend.model, image).detach()
        local = block(local_input, *args, **kwargs)
        predicted = block(xq, *args, **kwargs)
        with substitute(fp_block, lambda _: predicted):
            logits = differentiable_forward(backend, backend.model, image)
        kl = task_loss(logits, teacher=teacher, temperature=cfg["temperature"])
        mse = (local - reference).square().mean()
        return (
            mse + kl / cfg["lambda_r"]
            if kind == "pd_code"
            else kl + cfg["lambda_r"] * mse
        )
    require(
        kind == "paired_domain_features",
        "unknown differentiable reconstruction objective",
    )
    target_id = cfg["target_pairs"][sid]
    require(
        target_id in data.ids("calibration"),
        "target image not in the permitted paired calibration view",
    )
    target_image = (
        domain_target
        if domain_target is not None
        else next(v for v, i in data.batches("calibration") if i == target_id)
    )
    target = []
    with torch.no_grad(), substitute(fp_block, lambda y: target.append(y.detach())):
        differentiable_forward(backend, backend.model, target_image)
    out = block(local_input, *args, **kwargs)
    require(
        len(target) == 1 and out.shape == target[0].shape,
        "domain feature axis or shape mismatch",
    )
    return (
        F.smooth_l1_loss(out, reference)
        + cfg["alignment_weight"]
        * F.kl_div(
            out.log_softmax(cfg["feature_axis"]),
            target[0].softmax(cfg["feature_axis"]),
            reduction=cfg["kl_reduction"],
        ).abs()
    )


def merged_fit(backend, model, data, stage, cache_mib):
    from .pipeline import quantizers
    from . import grids
    from .reconstruction import paired_block

    path, cfg = stage["module"], stage["settings"]
    rows = paired_block(backend, model, data, path, cache_mib)
    block = model.get_submodule(path)
    members = {
        k: q
        for k, q in quantizers(model).items()
        if q in set(block.modules()) and q.layer.get("weight")
    }
    require(
        members
        and all(
            q.weight_record["method"] in grids.METHODS and q.reference_weight.ndim == 2
            for q in members.values()
        ),
        "merged fit needs specified uniform Linear weights",
    )
    seed = cfg["seed"]
    gen = torch.Generator(device=rows[0][1].device).manual_seed(seed)
    lowrank = cfg["parameterization"] == "low_rank"
    require(
        lowrank or cfg["parameterization"] == "full_correction",
        "unknown merged parameterization",
    )
    images = (
        {sid: x for x, sid in data.batches("calibration")}
        if cfg.get("objective")
        else {}
    )
    domain = cfg.get("objective")
    if domain:
        require(
            domain["kind"] == "paired_domain_features",
            "unsupported merged reconstruction objective",
        )
        pairs = domain["target_pairs"]
        require(
            pairs and set(pairs).union(pairs.values()) <= images.keys(),
            "both domains must belong to permitted calibration",
        )
        rows = [row for row in rows if row[0] in pairs]
        require(len(rows) == len(pairs), "missing or repeated source observations")
    variables = {}
    search_receipt = None
    if cfg.get("rank_search"):
        from .rank_search import search

        require(lowrank, "bilevel rank selection applies only to a low-rank correction")
        with torch.enable_grad():
            variables, search_receipt = search(
                block, members, rows, cfg["rank_search"], gen
            )
    for site, q in members.items():
        if site in variables:
            continue
        w = q.reference_weight
        if lowrank:
            rank = cfg["ranks"][site]
            require(0 < rank <= min(w.shape), "invalid correction rank")
            a = (
                torch.randn((rank, w.size(1)), generator=gen, device=w.device)
                * cfg["init_std"]
            )
            variables[site] = [
                torch.nn.Parameter(a),
                torch.nn.Parameter(w.new_zeros(w.size(0), rank)),
            ]
        else:
            variables[site] = [torch.nn.Parameter(torch.zeros_like(w))]

    def merge(site, q):
        v = variables[site]
        return q.reference_weight + (v[1] @ v[0] if lowrank else v[0])

    focus = {}
    if cfg.get("learn_intervals"):
        for k, q in quantizers(model).items():
            if (
                q in set(block.modules())
                and q.layer["activation_method"] == "focus_interval"
            ):
                low, hi = (q.params()[name] for name in ("low", "high"))
                require(
                    0 < low < hi < 1,
                    "learnable endpoints need a strictly interior initialization",
                )
                variable = torch.nn.Parameter(
                    torch.stack([low, (hi - low) / (1 - low)]).logit()
                )
                # Keep transient optimizer variables out of the deployed module.
                object.__setattr__(q, "_focus_bounds", variable)
                focus[k] = (q, variable)
        require(focus, "no post-softmax focus edge in the selected block")
    optimizer = torch.optim.Adam(
        [v for group in variables.values() for v in group]
        + [v for _, v in focus.values()],
        lr=cfg["lr"],
    )
    loss_trace = []
    steps = cfg["steps"]
    require(
        steps > 0 and cfg["batch_images"] > 0,
        "unbounded or empty reconstruction budget",
    )
    with torch.enable_grad():
        for step in range(steps):
            for site, q in members.items():
                w = merge(site, q)
                record = q.weight_record
                grid = grids.apply(
                    w, record["spec"], record["method"], record["params"]
                )
                q.decoded_weight = w + (grid - w).detach()

            def diff(row):
                _, xf, xq, y, args, kwargs = row
                source = xq if cfg["prefix"] == "quantized" else xf
                if domain:
                    sid = row[0]
                    return objective(
                        backend,
                        model,
                        data,
                        path,
                        backend.model.get_submodule(path),
                        block,
                        row,
                        images[sid],
                        domain,
                        source,
                        y,
                        domain_target=images[domain["target_pairs"][sid]],
                    )
                err = block(source, *args, **kwargs) - y
                return (
                    err.norm()
                    if cfg.get("loss") == "frobenius"
                    else err.square().mean()
                )

            if cfg.get("easy_to_hard"):
                with torch.no_grad():
                    order = sorted(range(len(rows)), key=lambda i: float(diff(rows[i])))
                pool = order[
                    : max(1, math.ceil(len(rows) * min(1, 0.5 + 0.5 * step / steps)))
                ]
            else:
                pool = list(range(len(rows)))
            ids = torch.randperm(len(pool), generator=gen, device=rows[0][1].device)[
                : cfg["batch_images"]
            ].tolist()
            loss = torch.stack([diff(rows[pool[i]]) for i in ids]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_trace.append(float(loss.detach()))
    with torch.no_grad():
        from .rank_search import bounds

        for q, variable in focus.values():
            lo, hi = bounds(variable)
            q.set_params({"low": lo, "high": hi})
            del q._focus_bounds
        for site, q in members.items():
            r = q.weight_record
            q.weight_record = grids.encode(
                merge(site, q), r["spec"], r["params"], method=r["method"]
            )
            q.decoded_weight = grids.decode(q.weight_record, q.reference_weight.device)
    return {
        "kind": "merged_weight_fit",
        "parameterization": cfg["parameterization"],
        "sites": list(members),
        "rank": search_receipt["chosen_ranks"] if search_receipt else cfg.get("ranks"),
        "rank_search": search_receipt,
        "objective": domain or {"kind": cfg.get("loss", "mse")},
        "fit_sample_ids": [row[0] for row in rows],
        "focus_intervals": {
            k: {n: float(t) for n, t in q.params().items()}
            for k, (q, _) in focus.items()
        },
        "frozen_state": "one re-encoded W per module",
        "loss": loss_trace,
        "steps": steps,
    }
