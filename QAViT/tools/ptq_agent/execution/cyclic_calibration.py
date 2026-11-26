"""Alternate CLAMP image and quantizer updates. Persist the winning ordinary plan and final calibration pixels."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from ..schemas import fingerprint, require
from ..storage import atomic_json
from ..backends.server import file_sha256
from .calibration import differentiable_forward
from .model import tensor_digest
from .pipeline import QuantizationExecutor, quantizers
from .plugin_contracts import site_id
from .sites import install_attention


class PixelPool:
    def __init__(self, source, images, targets):
        self.images = images.detach()
        self.names = ["synthetic/" + tensor_digest(x[None]) for x in self.images]
        self.labels = dict(zip(self.names, targets.argmax(-1).tolist()))
        self.identity = {
            **source.identity,
            "split_ids": {"calibration": self.names},
            "counts": {"calibration": len(images)},
        }
        self.shape_observation_splits = ("calibration",)

    def batches(self, split):
        require(
            split == "calibration",
            "CLAMP search may use only its finite synthetic pool",
        )
        for x, sid in zip(self.images, self.names):
            yield x[None], sid

    def ids(self, split):
        require(split == "calibration", "unbound split")
        return list(self.names)


def contrast(anchor, reference, positive, eligible, temperature):
    logits = (
        F.normalize(anchor, dim=-1)
        @ F.normalize(reference, dim=-1).transpose(-1, -2)
        / temperature
    )
    return (
        logits.masked_fill(~eligible, -torch.inf).logsumexp(-1)
        - logits.masked_fill(~positive, -torch.inf).logsumexp(-1)
    ).mean()


def patch_loss(query, teacher, layout, settings):
    start, h, w = layout["prefix"], *layout["grid"]
    q, p = (v[:, :, start : start + h * w] for v in (query, teacher))
    require(
        q.shape == p.shape and q.size(2) == h * w,
        "bound attention patch layout mismatch",
    )
    coord = torch.stack(
        torch.meshgrid(
            torch.arange(h, device=p.device),
            torch.arange(w, device=p.device),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 2)
    distance = (coord[:, None] - coord).abs().amax(-1)
    eligible = (distance > 0) & (distance <= settings["window"] // 2)
    k = settings["positives"]
    require((eligible.sum(-1) >= k).all(), "CLAMP positive neighborhood too small")
    order = (
        (F.normalize(p, dim=-1) @ F.normalize(p, dim=-1).transpose(-1, -2))
        .masked_fill(~eligible, -torch.inf)
        .topk(k, dim=-1)
        .indices
    )
    pos = torch.zeros_like(q @ p.transpose(-1, -2), dtype=torch.bool).scatter_(
        -1, order, True
    )
    return contrast(q, p, pos, eligible, settings["temperature"])


def run(backend, cfg, source, recipe, settings, directory):
    from .baselines import build_baseline

    base = recipe["base"]
    base = (
        deepcopy(base)
        if base.get("schema_version") == 3
        else build_baseline(
            backend, base["baseline"], quantize_head=base["quantize_head"]
        )
    )
    groups = settings["groups"]
    layout = settings["attention_layout"]
    device = torch.device(backend.device)
    rng = random.Random(settings["seed"])
    noise = torch.Generator(device=device).manual_seed(settings["seed"])
    images = torch.randn(
        (settings["images"], *settings["input_chw"]), generator=noise, device=device
    )
    targets = images.new_tensor(settings["target_vectors"])
    require(
        len(images) == len(targets) > 1
        and settings["temperature"] > 0
        and settings["population"] >= 2,
        "invalid cyclic calibration pool",
    )
    teacher = deepcopy(backend.model).eval().requires_grad_(False)
    install_attention(teacher, list(layout))
    rows = {site_id(r): r for r in base["layers"]}
    assigned = [k for group in groups for k in group["sites"]]
    require(
        len(set(assigned)) == len(assigned) and set(assigned) <= set(rows),
        "invalid CLAMP layer grouping",
    )

    def plan_for(genome):
        plan = deepcopy(base)
        by_site = {site_id(r): r for r in plan["layers"]}
        prefix = settings["initial_activation_scale"]
        for group, (width, scale) in zip(groups, genome):
            require(width in group["bits"] and scale > 0, "illegal CLAMP candidate")
            act = (
                min(8, width * 2)
                if settings["activation_rule"] == "paper"
                else settings["fixed_activation_bits"]
            )
            prefix += scale
            for sid in group["sites"]:
                row = by_site[sid]
                for target, bits, value in (
                    ("weight", width, scale),
                    ("activation", act, prefix),
                ):
                    if not row[target]:
                        continue
                    row[target].update(
                        format="INT" + str(bits),
                        bits=bits,
                        symmetric=True,
                        granularity="per_tensor",
                        axis=None,
                        group_size=None,
                    )
                    row[target + "_method"] = "rtn"
                    row[target + "_calibration"] = {
                        "strategy": "manual",
                        "scale_values": [value],
                    }
        return plan

    def build(genome):
        exe = QuantizationExecutor(backend, cfg, PixelPool(source, images, targets))
        return exe.prepare(plan_for(genome), {})[0]

    def observations(net, x, taps):
        seen = {}
        with ExitStack() as stack:
            for path in taps:

                def save(m, args, y, n=path):
                    seen[n] = y

                stack.callback(
                    net.get_submodule(path).register_forward_hook(save).remove
                )
            out = differentiable_forward(backend, net, x)
        require(
            set(seen) == set(taps) and out.shape == targets.shape,
            "CLAMP class target or observation mismatch",
        )
        return (
            out.softmax(-1) if settings["target_space"] == "probability" else out
        ), seen

    taps = [g["output"] for g in groups]
    htaps = [k + "._ptq_probability" for k in layout]
    lo, hi = settings["scale_perturbation"]

    def perturb(scale):
        return max(settings["min_scale"], scale + rng.uniform(lo, hi))

    def start():
        return tuple(
            (rng.choice(g["bits"]), perturb(g["initial_scale"])) for g in groups
        )

    def cross(a, b, i):
        child = list(a)
        bits = groups[i]["bits"]
        al, bl = a[i], b[i]
        allowed = [
            w for w in bits if min(al[0], bl[0]) - 1 <= w <= max(al[0], bl[0]) + 1
        ]
        child[i] = (rng.choice(allowed), perturb((al[1] + bl[1]) / 2))
        return tuple(child)

    genotypes = [start() for _ in range(settings["population"])]
    history = []
    cache = {}
    generation = 0

    def fitness(g):
        if g not in cache:
            with torch.no_grad():
                m = build(g)
                q, qb = observations(m, images, taps)
                f, fb = observations(teacher, images, taps)
                loss = (q - targets).abs().mean() + (f - targets).abs().mean()
                for name in taps:
                    a, b = (v[name].mean(1) for v in (qb, fb))
                    identity = torch.eye(len(images), device=device, dtype=torch.bool)
                    loss = loss + contrast(
                        a,
                        b,
                        identity,
                        torch.ones_like(identity),
                        settings["temperature"],
                    )
                cache[g] = float(loss)
        return cache[g]

    refreshes = 0
    for pass_index in range(settings["passes"]):
        for j, group in enumerate(groups):
            for cycle in range(settings["cycles"]):
                if generation % settings["refresh_every"] == 0:
                    g = min(genotypes, key=fitness)
                    student = build(g)
                    for q in quantizers(student).values():
                        q.gradient_probe = True
                    initial = tensor_digest(images)
                    images = images.detach().requires_grad_(True)
                    optim = torch.optim.Adam([images], lr=settings["lr"])
                    losses = []
                    for _ in range(
                        settings["initial_steps"]
                        if refreshes == 0
                        else settings["refresh_steps"]
                    ):
                        q, qs = observations(student, images, htaps)
                        f, fs = observations(teacher, images, htaps)
                        loss = (q - targets).abs().mean() + (f - targets).abs().mean()
                        for name, h in zip(layout, htaps):
                            loss = loss + patch_loss(
                                qs[h], fs[h], layout[name], settings
                            )
                        require(torch.isfinite(loss), "nonfinite synthetic objective")
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        losses.append(float(loss.detach()))
                    images = images.detach()
                    history.append(
                        {
                            "stage": "pixels",
                            "before": initial,
                            "after": tensor_digest(images),
                            "steps": len(losses),
                            "loss": losses,
                        }
                    )
                    cache.clear()
                    refreshes += 1
                    del student
                ranked = sorted(genotypes, key=fitness)
                child = cross(ranked[0], ranked[1], j)
                diverse = [
                    cross(child, start(), j)
                    for _ in range(settings["diversity_parents"])
                ]
                genotypes = sorted(
                    [*ranked, child, min(diverse, key=fitness)], key=fitness
                )[: settings["population"]]
                history.append(
                    {
                        "stage": "scales_and_bits",
                        "unit": group["output"],
                        "pass": pass_index,
                        "cycle": cycle,
                        "fitness": fitness(genotypes[0]),
                        "winner": genotypes[0],
                        "evaluated_in_current_pool": len(cache),
                    }
                )
                generation += 1
    final = plan_for(genotypes[0])
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=False)
    torch.save(images.cpu(), d / "images.pt")
    ids = [tensor_digest(x[None]) for x in images]
    result = {
        "schema": 1,
        "generator": settings,
        "teacher": backend.identity(),
        "labels": targets.argmax(-1).tolist(),
        "ids": ids,
        "image_sha256": file_sha256(d / "images.pt"),
        "winning_plan": final,
        "trajectory": history,
        "steps": sum(h["steps"] for h in history if h["stage"] == "pixels"),
        "loss": [x["fitness"] for x in history if "fitness" in x],
    }
    atomic_json(d / "manifest.json", result)
    return result
