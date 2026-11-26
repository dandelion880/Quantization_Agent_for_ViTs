"""Recompute layer reconstruction errors after changes to a complete precision assignment."""

import math
import torch
from ..schemas import require
from .calibration import capture
from .configuration_search import set_width


def local_error(backend, model, rows, site):
    from contextlib import ExitStack
    from .model import forward

    q = model.get_submodule(site)
    original = backend.model.get_submodule(site)
    actual, teacher = [], []
    with torch.no_grad(), ExitStack() as hooks:
        hooks.callback(
            original.register_forward_hook(
                lambda m, a, y: teacher.append(
                    (y - m.bias if m.bias is not None else y).detach()
                )
            ).remove
        )
        hooks.callback(
            q.register_forward_hook(
                lambda m, a, y: actual.append(
                    (
                        y - m.effective_bias() if m.effective_bias() is not None else y
                    ).detach()
                )
            ).remove
        )
        for x, _ in rows:
            forward(backend, backend.model, x)
            forward(backend, model, x)
    require(
        len(actual) == len(teacher) == len(rows),
        "incomplete reconstruction observation",
    )
    noise = sum((a - b).double().square().sum() for a, b in zip(actual, teacher))
    power = sum(b.double().square().sum() for b in teacher)
    require(power > 0, "zero layer reference power")
    return float(noise / power)


def lamp_update(backend, cfg, data, plan, options, settings, assess):
    from .pipeline import QuantizationExecutor
    from .plugin_contracts import site_id
    from copy import deepcopy

    rules = settings["feedback"]
    rows = list(data.batches("calibration"))
    require(
        settings["loss"] == "top1_error"
        and rules["formula"] in ("neighbor_difference", "printed_gain_labels"),
        "declare LampQ accuracy and gain convention",
    )
    table = {
        1: 1.0,
        2: 1.209e-2,
        3: 1.835e-3,
        4: 3.903e-4,
        5: 9.093e-5,
        6: 2.199e-5,
        7: 5.339e-6,
        8: 1.324e-6,
    }
    limit = settings["budgets"]["weight_bits"]
    sizes = {s: backend.model.get_submodule(s).weight.numel() for s in options}
    pool = {
        s: {int(row["formats"]["weight"].lstrip("UINT")) for row in choices}
        for s, choices in options.items()
    }
    measured, _ = assess(plan)
    history = []
    visited = set()
    for _ in range(rules["max_updates"]):
        layers = {site_id(r): r for r in plan["layers"]}
        widths = {s: layers[s]["weight"]["bits"] for s in pool}
        key = tuple(widths.values())
        if key in visited:
            break
        visited.add(key)
        executor = QuantizationExecutor(backend, cfg, data)
        model, *_ = executor.prepare(plan, {})
        error = {s: local_error(backend, model, rows, s) for s in widths}
        up = {
            s: e * (1 - table[b + 1] / table[b])
            for s, e in error.items()
            if (b := widths[s]) + 1 in pool[s]
        }
        down = {
            s: e * (table[b - 1] / table[b] - 1)
            for s, e in error.items()
            if (b := widths[s]) - 1 in pool[s]
        }
        if rules["formula"] == "printed_gain_labels":
            up = {
                s: error[s] * (table[widths[s] - 1] / table[widths[s]] - 1)
                for s in up
                if widths[s] > 1
            }
            down = {
                s: error[s] * (1 - table[widths[s] + 1] / table[widths[s]])
                for s in down
                if widths[s] < 8
            }
        pairs = sorted(
            ((u, d) for u in up for d in down if u != d),
            key=lambda pair: (-up[pair[0]], down[pair[1]], pair),
        )
        current = sum(widths[s] * sizes[s] for s in widths)
        policy = rules["budget_rule"]
        require(
            policy in ("record_paper_exchange", "require_feasible"),
            "unknown exchange budget rule",
        )
        if policy == "require_feasible":
            pairs = [(u, d) for u, d in pairs if current + sizes[u] - sizes[d] <= limit]
        if not pairs:
            break
        u, d = pairs[0]
        trial = deepcopy(plan)
        next_layers = {site_id(r): r for r in trial["layers"]}
        for s, w in ((u, widths[u] + 1), (d, widths[d] - 1)):
            for target in ("weight", "activation"):
                set_width(next_layers[s], target, w)
        accuracy_error, _ = assess(trial)
        cost = current + sizes[u] - sizes[d]
        accepted = accuracy_error <= measured
        history.append(
            {
                "from_bits": widths,
                "relative_frobenius": error,
                "up": u,
                "down": d,
                "cost": cost,
                "within_budget": cost <= limit,
                "top1_error": accuracy_error,
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        plan, measured = trial, accuracy_error
    return plan, history


def greedy_operands(backend, cfg, data, base, settings):
    from .configuration_search import set_width
    from .pipeline import QuantizationExecutor, quantizers
    from .plugin_contracts import site_id
    from .model import forward
    from copy import deepcopy

    current = deepcopy(base)
    samples = list(data.batches("calibration"))
    trace = []

    def candidate(plan, site, target):
        exe = QuantizationExecutor(backend, cfg, data)
        m, *_ = exe.prepare(plan, {})
        q = quantizers(m)[site]
        if target == "weight":
            x, y = q.reference_weight, q.decoded_weight
            return (
                float(x.double().square().sum()),
                float((x - y).double().square().sum()),
                x.numel(),
            )
        power = error = elements = 0

        def observe(module, args):
            nonlocal power, error, elements
            x = args[0]
            if hasattr(q, "input_scale"):
                require(
                    x.ndim in (2, 3),
                    "SQNR activation transform requires declared Linear axes",
                )
                x = x / q.input_scale - q.input_shift
            y = q.quantize(x)
            power += float(x.double().square().sum())
            error += float((x - y).double().square().sum())
            elements += x.numel()

        handle = q.register_forward_pre_hook(observe)
        try:
            for x, _ in samples:
                forward(backend, m, x)
        finally:
            handle.remove()
        require(
            elements % len(samples) == 0,
            "variable layout requires an explicit workload distribution",
        )
        return power, error, elements // len(samples)

    for target in settings["operand_order"]:
        ids = settings["sites"][target]
        minimum = settings["minimum"][target]
        goal = settings["average_bits"][target]
        selected = []
        while True:
            layers = {site_id(r): r for r in current["layers"]}
            rows = []
            total_bits = total_elements = 0
            for sid in ids:
                width = layers[sid][target]["bits"]
                next_plan = deepcopy(current)
                row = next(r for r in next_plan["layers"] if site_id(r) == sid)
                if width > minimum:
                    set_width(row, target, width - 1)
                signal, noise, count = candidate(next_plan, sid, target)
                require(count > 0 and signal > 0, "SQNR needs nonempty positive signal")
                total_elements += count
                total_bits += count * width
                if width > minimum:
                    # Infinite SQNR ties are compared as exact reconstructions, never serialized as nonfinite JSON.
                    score = (
                        math.inf
                        if noise == 0
                        else 10 * math.log10(signal / noise) * math.log(count)
                    )
                    rows.append((score, sid, width - 1, signal, noise, count))
            if total_bits <= goal * total_elements:
                break
            require(rows, "infeasible average-bit objective")
            score, sid, width, signal, noise, count = max(rows, key=lambda v: v[0])
            set_width(layers[sid], target, width)
            selected.append(
                {
                    "site": sid,
                    "new_bits": width,
                    "signal": signal,
                    "noise": noise,
                    "elements": count,
                    "score": None if noise == 0 else score,
                }
            )
        trace.append(
            {
                "operand": target,
                "steps": selected,
                "weighted_bits": total_bits / total_elements,
                "elements": total_elements,
            }
        )
    return current, {
        "search": "greedy_operand_sqnr",
        "trace": trace,
        "calibration_ids": [sid for _, sid in samples],
        "reprepare_on_every_width_change": True,
    }
