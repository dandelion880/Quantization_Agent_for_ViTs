"""Measure complete configurations with an explicit resource scope and bound graph observations."""

from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from itertools import product
import math
from statistics import median
import time
import torch
from ..schemas import require
from .plugin_contracts import site_id
from .research_search import pareto
from .sites import install_attention
from .model import forward


def set_width(row, target, width):
    from .contracts import FORMATS

    spec = row[target]
    require(
        spec and spec["format"].startswith(("UINT", "INT")),
        "integer allocation needs an integer edge",
    )
    name = ("INT" if spec["symmetric"] else "UINT") + str(width)
    require(name in FORMATS, "unsupported width")
    spec.update(bits=width, format=name)


def change_plan(base, groups, assignment):
    plan = deepcopy(base)
    layers = {site_id(r): r for r in plan["layers"]}
    stages = {s["module"]: s for s in plan.get("structure", [])}
    for group, ix in zip(groups, assignment):
        value = group["choices"][ix]
        for site, targets in group.get("sites", {}).items():
            row = layers[site]
            for target in targets:
                if target == "scheme":
                    for operand, choice in value.items():
                        require(
                            operand in ("weight", "activation")
                            and row[operand]
                            and choice["method"]
                            in ("uniform_affine", "uniform_full", "uniform_minmax"),
                            "unsupported bridge scheme",
                        )
                        row[operand].update(choice["spec"])
                        row[operand + "_method"] = choice["method"]
                elif target == "group_count":
                    require(
                        row["activation_method"] == "igq_ranges",
                        "group count applies to fitted IGQ",
                    )
                    row["options"]["encoding"]["groups"] = value
                else:
                    set_width(row, target, value)
        for module in group.get("window_blocks", []):
            require(
                stages[module]["kind"] == "window_branches" and 0 <= value <= 1,
                "unbound window fraction",
            )
            stages[module]["settings"]["low_fraction"] = value
    return plan


def collect(backend, model, samples, observations):
    result = defaultdict(list)

    def point(key):
        def save(module, args, y):
            if key[1] == "input":
                val = args[0]
            elif isinstance(key[1], int):
                idx = key[1]
                q = getattr(module, "quant_a" if idx == 0 else "quant_b", None)
                val = args[idx] if q is None or not q.enabled else q.quantize(args[idx])
            else:
                val = y
            require(torch.is_tensor(val), "metric tap requires a tensor")
            result[key].append(val.detach().cpu())

        return save

    with torch.no_grad(), ExitStack() as stack:
        for path, side in dict.fromkeys(observations):
            stack.callback(
                model.get_submodule(path)
                .register_forward_hook(point((path, side)))
                .remove
            )
        outputs = [forward(backend, model, x).detach().cpu() for x, _ in samples]
    require(
        all(len(v) == len(samples) for v in result.values())
        and set(observations) == set(result),
        "missing/repeated candidate tap",
    )
    return outputs, result


def configuration_grid(backend, cfg, data, base, settings):
    from .pipeline import QuantizationExecutor

    groups = settings["configuration_groups"]
    samples = list(data.batches("calibration"))
    require(
        groups and samples and all(g["choices"] for g in groups),
        "empty candidate or sample pool",
    )
    fields = [
        (s, t) for g in groups for s, tg in g.get("sites", {}).items() for t in tg
    ]
    windows = [s for g in groups for s in g.get("window_blocks", [])]
    require(
        len(set(fields)) == len(fields) and len(set(windows)) == len(windows),
        "overlapping configuration variables",
    )
    pool_size = math.prod(len(g["choices"]) for g in groups)
    require(
        pool_size <= settings["max_configurations"],
        "explicit search budget smaller than grid",
    )
    units = settings.get("encoder_units", [])
    keys = []
    for u in units:
        att = u["attention"]
        keys.extend(
            [
                (att + "._ptq_qk", 0),
                (att + "._ptq_qk", 1),
                (att + "._ptq_probability", 1),
                (att + "._ptq_probability", 0),
                (u["output"], "output"),
            ]
        )
    if settings.get("nuclear_taps"):
        keys.extend((t, "output") for t in settings["nuclear_taps"])
    teacher = deepcopy(backend.model)
    install_attention(teacher, [u["attention"] for u in units])
    if settings.get("nuclear_attention"):
        install_attention(teacher, settings["nuclear_attention"])
    bridge = settings.get("bridge")
    curvature = None
    if bridge:
        from .curvature_fit import BlockObjective

        keys.append((bridge["module"], "output"))
        opts = {"kind": "squared_gradient_error", "target": bridge["target"]}
        curvature = BlockObjective(
            backend,
            teacher,
            {sid: x for x, sid in samples},
            data.labels,
            bridge["module"],
            opts,
            bridge["cache_mib"],
        )
        curvature.refresh(teacher, 0)
    outputs, reference = collect(backend, teacher, samples, keys)
    fixed_grid_model = None
    if windows and not fields:
        fixed_grid_model, *_ = QuantizationExecutor(backend, cfg, data).prepare(
            base, {}
        )
    rows = []
    vectors = {}
    for ids in product(*(range(len(g["choices"])) for g in groups)):
        plan = change_plan(base, groups, ids)
        if fixed_grid_model is None:
            m, *_ = QuantizationExecutor(backend, cfg, data).prepare(plan, {})
        else:
            m = fixed_grid_model
            for g, i in zip(groups, ids):
                for path in g["window_blocks"]:
                    m.get_submodule(path).ptq_window["low_fraction"] = g["choices"][i]
        out, quantized = collect(backend, m, samples, keys)
        metrics = {
            "output_mse": sum(
                float((q - y).double().square().mean()) for q, y in zip(out, outputs)
            )
            / len(out)
        }
        if bridge:
            key = (bridge["module"], "output")
            metrics["squared_gradient_error"] = sum(
                float(curvature.evaluate(y, q, i))
                for i, (y, q) in enumerate(zip(reference[key], quantized[key]))
            ) / len(samples)
        if settings.get("nuclear_taps"):
            metrics["nuclear_error"] = sum(
                float(
                    torch.linalg.matrix_norm(y.double(), ord="nuc").mean()
                    * (q - y).double().square().sum()
                )
                for tap in settings["nuclear_taps"]
                for y, q in zip(reference[(tap, "output")], quantized[(tap, "output")])
            ) / len(samples)
        if units:
            for label, idxs in (
                ("E1_qkv", (0, 1, 2)),
                ("E2_attention", (3,)),
                ("E3_encoder", (4,)),
            ):
                norm = settings["error_norm"]
                require(norm in (1, 2), "declare source norm convention")
                metrics[label] = sum(
                    float(torch.linalg.vector_norm((q - y).double(), ord=norm))
                    for i in range(len(units))
                    for j in idxs
                    for q, y in zip(
                        quantized[keys[5 * i + j]], reference[keys[5 * i + j]]
                    )
                ) / len(samples)
        total_bits = sum(
            backend.model.get_submodule(r["module"]).weight.numel()
            * r["weight"]["bits"]
            for r in plan["layers"]
            if r.get("weight")
        )
        metrics["quantized_weight_bits"] = total_bits
        if settings.get("latency"):
            options = settings["latency"]
            key = ",".join(map(str, ids))
            if options["kind"] == "measured_lookup":
                require(
                    options["platform"] and options["measurement_ref"],
                    "missing measured platform identity",
                )
                metrics["latency"] = options["by_assignment"][key]
            else:
                require(
                    options["kind"] == "software_qdq"
                    and options["warmup"] >= 0
                    and options["repeats"] >= 1,
                    "invalid software latency conditions",
                )

                def sync():
                    if torch.device(backend.device).type == "cuda":
                        torch.cuda.synchronize(backend.device)

                times = []
                for index in range(options["warmup"] + options["repeats"]):
                    sync()
                    start = time.perf_counter()
                    forward(backend, m, samples[0][0])
                    sync()
                    if index >= options["warmup"]:
                        times.append(time.perf_counter() - start)
                metrics["latency"] = median(times)
        index = len(rows)
        rows.append(
            {
                "assignment": list(ids),
                "values": [g["choices"][i] for i, g in zip(ids, groups)],
                "metrics": metrics,
            }
        )
        if all(metrics[k] <= v for k, v in settings["limits"].items()):
            vectors[index] = tuple(metrics[k] for k in settings["objectives"])
    front = pareto(vectors)
    priority = settings["priority"]
    require(
        priority and set(priority) <= set(settings["objectives"]),
        "declare selection from the measured Pareto front",
    )
    selected = min(front, key=lambda i: tuple(rows[i]["metrics"][k] for k in priority))
    return change_plan(base, groups, rows[selected]["assignment"]), {
        "search": "complete_configuration_grid",
        "objectives": settings["objectives"],
        "priority": priority,
        "limits": settings["limits"],
        "latency": settings.get("latency"),
        "sample_ids": [i for _, i in samples],
        "candidates": rows,
        "pareto": front,
        "selected": selected,
        "shared_window_grids": fixed_grid_model is not None,
        "measurement_kind": "bounded software; device lookup only when explicitly bound",
    }
