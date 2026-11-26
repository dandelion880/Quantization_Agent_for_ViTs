"""Budgeted research searches. Objective and feasible variables come from the bound experiment."""

from itertools import product
import math
import torch
from ..schemas import require
from .research_metrics import score


def fpcs(axes, objective, *, beam, refinements, subdivisions, integer_axes=()):
    """Retain multiple coarse regions and refine each; report every distinct evaluated point."""
    require(
        axes and beam > 0 and subdivisions >= 2 and refinements >= 0,
        "invalid FPCS budget",
    )
    values = [sorted(set(v)) for v in axes]
    require(all(len(v) > 1 for v in values), "FPCS needs an interval per axis")
    widths = [min(b - a for a, b in zip(v, v[1:])) for v in values]
    candidates = product(*values)
    scores, traces = {}, []
    for step in range(refinements + 1):
        pool = set()
        for x in candidates:
            point = tuple(
                int(round(v)) if i in integer_axes else float(v)
                for i, v in enumerate(x)
            )
            if point not in scores:
                loss = float(objective(*point))
                require(math.isfinite(loss), "nonfinite FPCS objective")
                scores[point] = loss
            pool.add(point)
        retained = sorted(pool, key=lambda x: (scores[x], x))[:beam]
        traces.append(
            {
                "evaluated": len(pool),
                "retained": [[list(x), scores[x]] for x in retained],
            }
        )
        if step < refinements:
            regions = []
            for x in retained:
                sets = [
                    torch.linspace(
                        max(v[0], xi - w),
                        min(v[-1], xi + w),
                        subdivisions,
                        dtype=torch.float64,
                    ).tolist()
                    for xi, w, v in zip(x, widths, values)
                ]
                regions.extend(product(*sets))
            candidates = [*retained, *regions]
            widths = [2 * w / (subdivisions - 1) for w in widths]
    return retained[0], {
        "rounds": traces,
        "evaluations": [[list(x), v] for x, v in scores.items()],
    }


def allocate(cost_loss, budget):
    """Exact multiple-choice knapsack with dominated partial states removed; works with integer byte/BOP costs."""
    require(
        type(budget) is int and budget >= 0,
        "allocation requires explicit integer resource budget",
    )
    states = {0: (0.0, {})}
    for site, options in cost_loss.items():
        following = {}
        for old, (base, choice) in states.items():
            for value, cost, loss in options:
                require(
                    type(cost) is int and cost >= 0 and math.isfinite(loss),
                    "invalid allocation candidate",
                )
                total = old + cost
                if total <= budget and (
                    total not in following or base + loss < following[total][0]
                ):
                    following[total] = (base + loss, {**choice, site: value})
        states = {}
        best = math.inf
        for total, (loss, choice) in sorted(following.items()):
            if loss < best:
                states[total] = (loss, choice)
                best = loss
        require(states, "infeasible resource budget")
    cost, (loss, choice) = min(states.items(), key=lambda kv: (kv[1][0], kv[0]))
    return {"assignment": choice, "cost": cost, "objective": loss}


def pareto(rows):
    """All objectives minimized. Preserve ties as distinct measured configurations."""
    require(
        rows
        and len({len(v) for v in rows.values()}) == 1
        and all(all(math.isfinite(x) for x in v) for v in rows.values()),
        "incomparable objective vectors",
    )
    return [
        key
        for key, v in rows.items()
        if not any(
            all(x <= y for x, y in zip(w, v)) and any(x < y for x, y in zip(w, v))
            for w in rows.values()
        )
    ]


@torch.no_grad()
def search_stage(backend, model, data, stage):
    from .pipeline import quantizers
    from .model import forward
    from .methods import get_target_method
    from .plugin_math import decode_weight

    cfg = stage["settings"]
    kind = cfg["variant"]
    block = model.get_submodule(stage["module"])
    require(
        kind in ("cpt_paper", "evol_paper", "evol_qavit_fixed_final"),
        "unknown evolutionary profile",
    )
    candidates = quantizers(model)
    selected = set(cfg.get("site_ids", candidates))
    require(selected <= set(candidates), "unknown evolutionary scale site")
    targets = cfg.get(
        "targets", ["weight"] if kind == "evol_qavit_fixed_final" else ["activation"]
    )
    require(
        bool(targets) and set(targets) <= {"weight", "activation"},
        "unknown evolutionary operand",
    )
    qs = []
    for sid, q in candidates.items():
        if sid not in selected or q not in set(block.modules()):
            continue
        if (
            "activation" in targets
            and q.layer.get("activation")
            and "scale" in q.params()
        ):
            qs.append((q, "activation", q.params()))
        if (
            "weight" in targets
            and q.layer.get("kind") == "linear"
            and q.layer.get("weight")
        ):
            require(
                q.layer["weight_method"]
                in ("rtn", "uniform_minmax", "uniform_full", "uniform_affine"),
                "evolutionary W edit requires an independent uniform grid",
            )
            qs.append((q, "weight", q.weight_record["params"]))
    require(
        qs and len(data.ids("calibration")) > 1,
        "contrastive search needs multiple fixed negatives",
    )
    seed = cfg["seed"]
    gen = torch.Generator().manual_seed(seed)
    rows = list(data.batches("calibration"))
    bank = [(x, forward(backend, backend.model, x)) for x, _ in rows]
    initial = torch.cat([p["scale"].detach().cpu().reshape(-1) for _, _, p in qs])

    def install(vector):
        start = 0
        for q, target, p in qs:
            base = p["scale"]
            end = start + base.numel()
            params = {**p, "scale": vector[start:end].reshape_as(base).to(base)}
            if target == "activation":
                q.set_params(params)
            else:
                record = get_target_method("weight", q.layer["weight_method"]).fit(
                    q.reference_weight, q.layer["weight"], params
                )
                q.weight_record = record
                q.decoded_weight = decode_weight(record, q.reference_weight.device)
            start = end

    opts = {
        "temperature": cfg["temperature"],
        "normalize": cfg["normalize"],
        "multiplier": cfg.get("multiplier", 1.0),
    }

    def fitness(v):
        install(v)
        return float(
            score(
                "contrastive_teacher_logits",
                torch.cat([f for _, f in bank]),
                torch.cat([forward(backend, model, x) for x, _ in bank]),
                options=opts,
            )
        )

    def mutate(v):
        a = v + (2 * torch.rand(v.shape, generator=gen) - 1) * cfg["radius"]
        return (
            a.abs().clamp_min(torch.finfo(a.dtype).eps)
            if kind == "evol_qavit_fixed_final"
            else torch.where(a > 0, a, v)
        )

    pop = cfg["population"]
    tour = cfg["tournament"]
    require(1 <= tour <= pop and cfg["cycles"] > 0, "invalid tournament")
    pool = [initial] + [
        initial.clone() if kind == "evol_qavit_fixed_final" else mutate(initial)
        for _ in range(pop - 1)
    ]
    losses = [fitness(v) for v in pool]
    trajectory = [min(losses)]
    for _ in range(cfg["cycles"]):
        subset = (
            torch.randint(pop, (tour,), generator=gen)
            if kind == "evol_qavit_fixed_final"
            else torch.randperm(pop, generator=gen)[:tour]
        ).tolist()
        trial = mutate(pool[min(subset, key=losses.__getitem__)])
        i = max(range(pop), key=losses.__getitem__)
        pool[i], losses[i] = trial, fitness(trial)
        trajectory.append(min(losses))
    install(pool[min(range(pop), key=losses.__getitem__)])
    return {
        "kind": kind,
        "module": stage["module"],
        "fixed_negative_ids": [sid for _, sid in rows],
        "best_trajectory": trajectory,
        "candidate_forwards": len(rows) * (pop + cfg["cycles"]),
        "scale_sites": [q.ptq_site_id + "/" + t for q, t, _ in qs],
    }
