"""P2-ViT Hessian/coarse-to-fine search and IGQ alternating group-count assignment."""

from copy import deepcopy
import math
import random
import torch
from torch.nn import functional as F
from ..schemas import require
from .calibration import differentiable_forward
from .configuration_search import set_width
from .model import forward
from .plugin_contracts import site_id
from .research_metrics import task_loss


def weight_hessian_trace(backend, samples, labels, sites, config):
    teacher = deepcopy(backend.model).eval().requires_grad_(False)
    weights = [teacher.get_submodule(n).weight.requires_grad_(True) for n in sites]
    gen = torch.Generator(device=weights[0].device).manual_seed(config["seed"])
    totals = dict.fromkeys(sites, 0.0)
    require(config["vectors"] > 0, "empty Hessian trace probe")
    with torch.inference_mode(False), torch.enable_grad():
        for image, sid in samples:
            scores = differentiable_forward(backend, teacher, image)
            if config["target"] == "teacher_class":
                target = scores.detach().argmax(-1)
            else:
                require(
                    config["target"] == "labels" and sid in labels,
                    "missing Hessian calibration class",
                )
                target = torch.as_tensor([labels[sid]], device=scores.device)
            gradients = torch.autograd.grad(
                F.cross_entropy(scores, target), weights, create_graph=True
            )
            for name, w, g in zip(sites, weights, gradients):
                for _ in range(config["vectors"]):
                    v = (
                        torch.randint(2, w.shape, device=w.device, generator=gen)
                        .to(w)
                        .mul_(2)
                        .sub_(1)
                    )
                    hv = torch.autograd.grad((g * v).sum(), w, retain_graph=True)[0]
                    totals[name] += float((v.double() * hv.detach()).sum())
    return {k: v / (len(samples) * config["vectors"]) for k, v in totals.items()}


def genetic_weights(backend, cfg, data, base, settings):
    from .pipeline import QuantizationExecutor, quantizers
    from .precision import budget_search

    sites = list(settings["sites"])
    allowed = settings["widths"]
    limit = settings["weight_bits"]
    ga = settings["genetic"]
    require(
        len(allowed) >= 2
        and len(set(allowed)) == len(allowed)
        and sites
        and len(set(sites)) == len(sites),
        "invalid mixed W candidates",
    )
    require(
        ga["population"] >= 1
        and ga["generations"] >= 1
        and 0 <= ga["probability"] <= 1
        and all(ga[k] >= 0 for k in ("mutations", "crossovers")),
        "invalid genetic budget",
    )
    samples = list(data.batches("calibration"))
    classes = data.labels
    traces = weight_hessian_trace(backend, samples, classes, sites, settings["hessian"])
    sizes = [backend.model.get_submodule(n).weight.numel() for n in sites]
    rng = random.Random(ga["seed"])

    def plan_for(bits):
        plan = deepcopy(base)
        rows = {site_id(r): r for r in plan["layers"]}
        for n, b in zip(sites, bits):
            set_width(rows[n], "weight", b)
        return plan

    exe = QuantizationExecutor(backend, cfg, data)
    options = {n: [] for n in sites}
    for i, b in enumerate(allowed):
        model, *_ = exe.prepare(plan_for([b] * len(sites)), {})
        qs = quantizers(model)
        for n, size in zip(sites, sizes):
            distance = (
                (qs[n].decoded_weight - qs[n].reference_weight).double().norm().item()
            )
            options[n].append(
                {
                    "index": i,
                    "cost": {"bits": size * b},
                    "loss": traces[n] * distance,
                    "distance": distance,
                }
            )

    def cost(x):
        return sum(b * n for b, n in zip(x, sizes))

    lowest = [min(allowed)] * len(sites)
    require(cost(lowest) <= limit, "infeasible mixed-weight budget")
    pool = set()
    for i in range(ga["population"]):
        cap = limit if i == 0 else rng.randint(cost(lowest), limit)
        sol = budget_search(options, {"bits": cap})
        pool.add(tuple(allowed[sol["assignment"][n]] for n in sites))
    labels = []
    for image, sid in samples:
        if settings["fitness"] == "labels_top1":
            require(sid in classes, "missing classification label for genetic fitness")
            labels.append(classes[sid])
        else:
            require(settings["fitness"] == "teacher_top1", "unknown genetic fitness")
            labels.append(int(forward(backend, backend.model, image).argmax(-1)))

    def coarse(x):
        return sum(options[n][allowed.index(b)]["loss"] for n, b in zip(sites, x))

    cache = {}
    timeline = []

    def measure(x):
        if x not in cache:
            m, *_ = exe.prepare(plan_for(x), {})
            cache[x] = sum(
                int(forward(backend, m, image).argmax(-1)) != c
                for (image, _), c in zip(samples, labels)
            ) / len(samples)
        return cache[x]

    for _ in range(ga["generations"]):
        parents = sorted(pool)
        for crossover, count in ((False, ga["mutations"]), (True, ga["crossovers"])):
            for i in range(count):
                a = rng.choice(parents)
                b = rng.choice(parents)
                child = tuple(
                    (b[j] if crossover else rng.choice([v for v in allowed if v != s]))
                    if rng.random() < ga["probability"]
                    else s
                    for j, s in enumerate(a)
                )
                if cost(child) <= limit:
                    pool.add(child)
        pool = set(
            sorted(pool, key=lambda x: (measure(x), coarse(x), x))[: ga["population"]]
        )
        timeline.append([list(x) for x in sorted(pool)])
    best = min(pool, key=lambda x: (cache[x], coarse(x), x))
    return plan_for(best), {
        "search": "Hutchinson, additive Hessian Pareto, measured mutation/crossover",
        "sites": sites,
        "bound_weight_bits": limit,
        "sample_ids": [sid for _, sid in samples],
        "traces": traces,
        "coarse_options": options,
        "generations": timeline,
        "measured": [
            {"bits": list(x), "weight_bits": cost(x), "error": v}
            for x, v in sorted(cache.items())
        ],
        "selected": list(best),
        "fitness": settings["fitness"],
    }


def group_schedule(backend, cfg, data, base, settings):
    from .pipeline import QuantizationExecutor, quantizers
    from .precision import budget_search

    choices = settings["groups"]
    cadence = settings["reallocate_every"]
    total = settings["alternating_steps"]
    require(
        total >= cadence > 0
        and choices
        and all(len(set(v)) == len(v) and min(v) >= 1 for v in choices.values()),
        "invalid IGQ schedule",
    )
    samples = list(data.batches("calibration"))
    scope = settings["cost_scope"]
    require(
        scope
        and all(k in scope for k in ("minmax", "assignment", "scaled_sum", "matmul")),
        "declare group-routing and GEMM cost accounting",
    )
    exe = QuantizationExecutor(backend, cfg, data)
    result = deepcopy(base)
    history = []
    for stop in [*range(cadence, total, cadence), total]:
        layers = {site_id(r): r for r in result["layers"]}
        for r in layers.values():
            if r.get("activation_method") == "igq_ranges":
                r["options"]["encoding"]["steps"] = stop
        options = {}
        for name, counts in choices.items():
            row = layers[name]
            require(
                row["activation_method"] == "igq_ranges",
                "count needs an IGQ activation site",
            )
            options[name] = []
            for count in counts:
                plan = deepcopy(result)
                next(r for r in plan["layers"] if site_id(r) == name)["options"][
                    "encoding"
                ]["groups"] = count
                m, *_ = exe.prepare(plan, {})
                q = quantizers(m)[name]
                q.enabled = False
                reference = [forward(backend, m, x) for x, _ in samples]
                q.enabled = True
                errors = [
                    float(task_loss(forward(backend, m, x), teacher=y, temperature=1.0))
                    for (x, _), y in zip(samples, reference)
                ]
                parts = settings["costs"][name][str(count)]
                require(
                    set(parts) == {"minmax", "assignment", "scaled_sum", "matmul"}
                    and all(type(v) is int and v >= 0 for v in parts.values()),
                    "invalid declared BOP components",
                )
                options[name].append(
                    {
                        "index": count,
                        "loss": sum(errors) / len(errors),
                        "cost": {"bops": sum(parts.values())},
                        "components": parts,
                    }
                )
        allocation = budget_search(
            options, {"bops": settings["budget"] - settings["fixed_bops"]}
        )
        for name, count in allocation["assignment"].items():
            layers[name]["options"]["encoding"]["groups"] = count
        history.append(
            {"alternating_updates": stop, "options": options, "allocation": allocation}
        )
    return result, {
        "search": "periodic finite-range fitting and multiple-choice ILP equivalent",
        "reference": "same other quantized edges; selected activation disabled",
        "sample_ids": [sid for _, sid in samples],
        "cost_scope": scope,
        "fixed_bops": settings["fixed_bops"],
        "budget": settings["budget"],
        "history": history,
    }
