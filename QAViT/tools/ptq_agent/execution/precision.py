"""Measured bit alternatives and resource-constrained allocation with bound inputs and graph costs."""

from copy import deepcopy
import math
import torch
from ..schemas import require
from .model import forward
from .plugin_contracts import site_id
from .research_metrics import task_loss
from .research_search import pareto


def budget_search(options, budgets):
    """Exact additive multiple-choice allocation. Each option contains a joint (W,A) and its own W*A cost."""
    require(
        budgets and all(type(x) is int and x >= 0 for x in budgets.values()),
        "invalid resource limits",
    )
    axes = tuple(budgets)
    states = {tuple(0 for _ in axes): (0.0, {})}
    for site, choices in options.items():
        nxt = {}
        for used, (old, assignment) in states.items():
            for opt in choices:
                costs = opt["cost"]
                require(
                    set(costs) == set(axes)
                    and all(type(x) is int and x >= 0 for x in costs.values()),
                    "incomparable budget dimensions",
                )
                new = tuple(u + costs[k] for u, k in zip(used, axes))
                loss = old + opt["loss"]
                require(math.isfinite(loss), "nonfinite measured cost")
                if any(x > budgets[k] for x, k in zip(new, axes)):
                    continue
                if new not in nxt or loss < nxt[new][0]:
                    nxt[new] = (loss, {**assignment, site: opt["index"]})
        require(nxt, "no feasible assignment after " + site)
        surviving = pareto({k: (*k, v[0]) for k, v in nxt.items()})
        states = {k: nxt[k] for k in surviving}
    cost, (loss, assignment) = min(states.items(), key=lambda p: (p[1][0], p[0]))
    return {
        "assignment": assignment,
        "resource_totals": dict(zip(axes, cost)),
        "predicted_objective": loss,
    }


def greedy_width(scores, numel, target):
    """AMP/MPTQ lowering with SQNR(b-1)*log(numel), preserving the element-weighted bit target."""
    chosen = {k: max(v) for k, v in scores.items()}
    denominator = sum(numel.values())
    trace = []
    while sum(numel[s] * b for s, b in chosen.items()) > denominator * target:
        candidates = [
            (row[chosen[s] - 1] * math.log(numel[s]), s)
            for s, row in scores.items()
            if chosen[s] - 1 in row
        ]
        require(candidates, "greedy precision target is infeasible")
        _, s = max(candidates)
        chosen[s] -= 1
        trace.append((s, chosen[s]))
    return chosen, trace


def pair_exchange(assignment, numel, up, down, limit):
    """Expose the true parameter-weighted change even for the unprojected LampQ paper update."""
    trial = dict(assignment)
    require(up != down and up in trial and down in trial, "unbound paired allocation")
    trial[up] += 1
    trial[down] -= 1
    cost = sum(numel[k] * v for k, v in trial.items())
    return {
        "assignment": trial,
        "cost": cost,
        "cost_change": numel[up] - numel[down],
        "feasible": cost <= limit,
    }


def select(backend, cfg, data, base, settings):
    from .pipeline import QuantizationExecutor
    from .contracts import FORMATS

    if settings.get("allocation_rule") == "hessian_genetic":
        from .source_allocation import genetic_weights

        return genetic_weights(backend, cfg, data, base, settings)
    if settings.get("allocation_rule") == "group_count_schedule":
        from .source_allocation import group_schedule

        return group_schedule(backend, cfg, data, base, settings)
    if settings.get("allocation_rule") == "greedy_operand_sqnr":
        from .allocation_feedback import greedy_operands

        return greedy_operands(backend, cfg, data, base, settings)
    if settings.get("configuration_groups"):
        from .configuration_search import configuration_grid

        return configuration_grid(backend, cfg, data, base, settings)
    alternatives = settings["alternatives"]
    budgets = settings["budgets"]
    require(
        bool(alternatives)
        and set(budgets) <= {"weight_bits", "bitops", "measured_latency_ticks"},
        "unknown resource dimension",
    )
    layer_map = {site_id(l): l for l in base["layers"]}
    require(set(alternatives) <= set(layer_map), "unknown precision site")
    rows = list(data.batches("calibration"))
    require(rows, "allocation needs a permitted finite calibration set")
    teachers = [forward(backend, backend.model, x) for x, _ in rows]
    importance = dict(settings.get("importance", {}))
    stats = {}
    if settings.get("fisher"):
        from .layer_importance import fisher

        stats = fisher(backend, rows, data.labels, alternatives, settings["fisher"])
        importance = stats[settings["fisher"]["statistic"]]
    if settings.get("relevance"):
        from .relevance import prepare

        r = settings["relevance"]
        stats["relevance"] = prepare(backend, rows, data.labels, r)
        importance = {
            s: stats["relevance"]["scores"][n] for s, n in r["site_blocks"].items()
        }
        require(
            set(importance) == set(alternatives),
            "bit alternatives need bound relevance scores",
        )

    def change(plan, site, choice):
        layer = next(l for l in plan["layers"] if site_id(l) == site)
        for t in ("weight", "activation"):
            if t not in choice:
                continue
            fmt = choice[t]
            spec = layer[t]
            require(
                spec and fmt in FORMATS and fmt.startswith(("INT", "UINT")),
                "invalid precision alternative",
            )
            spec.update(format=fmt, bits=FORMATS[fmt], symmetric=fmt.startswith("INT"))

    def measure(plan):
        executor = QuantizationExecutor(backend, cfg, data)
        model, *_ = executor.prepare(plan, {})
        by_image = []
        for (x, sid), t in zip(rows, teachers):
            q = forward(backend, model, x)
            metric = settings["loss"]
            if metric == "teacher_kl":
                val = task_loss(q, teacher=t, temperature=settings["temperature"])
            elif metric == "top1_error":
                require(sid in data.labels, "top1 requires permitted labels")
                val = (q.argmax(-1) != data.labels[sid]).double().mean()
            elif metric == "labels_ce":
                val = task_loss(
                    q, labels=torch.as_tensor([data.labels[sid]], device=q.device)
                )
            else:
                require(metric == "output_mse", "unknown allocation target")
                val = (q - t).double().square().mean()
            by_image.append(float(val))
        return sum(by_image) / len(by_image), by_image

    initial, _ = measure(base)
    if settings.get("type_calibration"):
        from .layer_importance import type_adjust

        tc = settings["type_calibration"]
        require(
            stats and settings["loss"] == "top1_error",
            "type calibration needs actual accuracy drops and Fisher statistics",
        )
        probes = {}
        for typ, items in tc["probes"].items():
            probes[typ] = {}
            for sid, formats in items.items():
                unit = deepcopy(base)
                change(unit, sid, formats)
                probes[typ][sid] = measure(unit)[0] - initial
        importance, stats["type_coefficients"] = type_adjust(
            stats["trace"], tc["types"], probes
        )
        stats["type_probes"] = probes
    options = {}
    for site, choices in alternatives.items():
        info = layer_map[site]
        module = backend.model.get_submodule(info["module"])
        measured = []
        for i, choice in enumerate(choices):
            plan = deepcopy(base)
            change(plan, site, choice)
            row = next(l for l in plan["layers"] if site_id(l) == site)
            w = row.get("weight")
            a = row.get("activation")
            cost = {}
            if "weight_bits" in budgets:
                cost["weight_bits"] = module.weight.numel() * w["bits"] if w else 0
            if "bitops" in budgets:
                require(
                    w and a and settings["mac_counts"][site] > 0,
                    "joint bitops needs actual operand MAC count",
                )
                cost["bitops"] = settings["mac_counts"][site] * w["bits"] * a["bits"]
            if "measured_latency_ticks" in budgets:
                require(
                    settings.get("device_measurement_ref")
                    and "latency_ticks" in choice,
                    "missing bound device measurement",
                )
                cost["measured_latency_ticks"] = choice["latency_ticks"]
            val, by_image = measure(plan)
            risk = (val - initial) * importance.get(site, 1.0)
            if settings.get("allocation_rule") == "lampq_initial":
                require(
                    w and a and a["bits"] == w["bits"] and settings["gamma"] > 1,
                    "LampQ initial requires a shared layer width and gamma>1",
                )
                risk = importance[site] * settings["gamma"] ** (-w["bits"])
            measured.append(
                {
                    "index": i,
                    "formats": choice,
                    "delta": val - initial,
                    "loss": risk,
                    "per_image": by_image,
                    "cost": cost,
                }
            )
        options[site] = measured
    if settings.get("allocation_rule") == "normalized_relative_loss":
        values = [r["delta"] for pool in options.values() for r in pool]
        shifted = [v - min(values) for v in values]
        normalizer = sum(shifted)
        require(normalizer > 0, "constant sensitivity table")
        for sid, pool in options.items():
            for opt in pool:
                opt["loss"] = (
                    (opt["delta"] - min(values)) / normalizer * importance.get(sid, 1.0)
                )
    winner = budget_search(options, budgets)
    final = deepcopy(base)
    for site, i in winner["assignment"].items():
        change(final, site, alternatives[site][i])
    feedback = []
    if settings.get("feedback"):
        from .allocation_feedback import lamp_update

        require(
            settings.get("allocation_rule") == "lampq_initial",
            "initial allocation for iterative Fisher updates is missing",
        )
        final, feedback = lamp_update(
            backend, cfg, data, final, options, settings, measure
        )
        current = {site_id(r): r for r in final["layers"]}
        applied = {
            sid: next(
                i
                for i, choice in enumerate(alternatives[sid])
                if choice["weight"] == current[sid]["weight"]["format"]
            )
            for sid in options
        }
        winner = {
            **winner,
            "initial_assignment": winner["assignment"],
            "assignment": applied,
            "resource_totals": {
                k: sum(options[s][i]["cost"][k] for s, i in applied.items())
                for k in budgets
            },
        }
    return final, {
        **winner,
        "bit_update_history": feedback,
        "individual_measurements": options,
        "baseline": initial,
        "importance": importance,
        "importance_preparation": stats,
        "fixed_complement": "base plan",
        "sample_ids": [sid for _, sid in rows],
        "loss": settings["loss"],
        "budgets": budgets,
        "cost_scope": "selected sites only; fixed costs must be subtracted from whole-model limits",
        "final_state": "selected plan must be prepared again after all replacements",
    }
