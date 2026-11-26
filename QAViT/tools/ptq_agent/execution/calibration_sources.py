"""Image-only PTQ synthesis and immutable, content-addressed calibration views."""

from contextlib import ExitStack
from copy import deepcopy
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from ..schemas import require, fingerprint
from ..storage import atomic_json, read_json
from ..backends.server import file_sha256
from .calibration import differentiable_forward
from .model import tensor_digest
from .sites import install_attention
from ..profiling.adapters import ProfileAdapter


def patch_entropy(features, bandwidth, samples, epsilon):
    require(
        features.ndim == 3 and bandwidth > 0 and samples > 1 and epsilon > 0,
        "invalid patch KDE",
    )
    normalized = F.normalize(features, dim=-1)
    pairs = (normalized @ normalized.transpose(-1, -2)).flatten(1)
    grid = torch.linspace(
        float(pairs.detach().min()),
        float(pairs.detach().max()),
        samples,
        device=pairs.device,
    )
    require(bool(grid[-1] > grid[0]), "constant patch similarity")
    density = torch.stack(
        [torch.exp(-0.5 * ((pairs - k) / bandwidth).square()).mean(1) for k in grid], -1
    ) / (bandwidth * math.sqrt(2 * math.pi))
    p = density + epsilon
    return -torch.trapz(p * p.log(), grid, dim=-1).mean()


def local_patch_loss(features, hw, k, window, temperature):
    h, w = hw
    require(
        h * w == features.shape[1] and k > 0 and window % 2 and temperature > 0,
        "invalid local patch layout",
    )
    feats = F.normalize(features, dim=-1)
    sims = feats @ feats.transpose(1, 2)
    y, x = torch.meshgrid(
        torch.arange(h, device=feats.device),
        torch.arange(w, device=feats.device),
        indexing="ij",
    )
    positions = torch.stack((y.flatten(), x.flatten()), -1)
    allowed = (positions[:, None] - positions[None, :]).abs().amax(-1) <= window // 2
    eye = torch.eye(h * w, dtype=torch.bool, device=feats.device)
    allowed = allowed & ~eye
    require(bool((allowed.sum(-1) >= k).all()), "insufficient valid neighbor positives")
    positives = sims.masked_fill(~allowed, -torch.inf).topk(k, dim=-1).values
    den = (sims / temperature).masked_fill(eye, -torch.inf).logsumexp(-1)
    return (den - positives.mean(-1) / temperature).mean()


def total_variation(x):
    return sum(
        t.norm()
        for t in (
            x[:, :, :, 1:] - x[:, :, :, :-1],
            x[:, :, 1:] - x[:, :, :-1],
            x[:, :, 1:, 1:] - x[:, :, :-1, :-1],
            x[:, :, 1:, :-1] - x[:, :, :-1, 1:],
        )
    )


def synthesize(backend, cfg, directory):
    """Generate from noise only. No original image/validation loader is opened."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=False)
    device = torch.device(backend.device)
    rng = torch.Generator(device=device).manual_seed(cfg["seed"])
    count, shape = cfg["images"], cfg["input_chw"]
    require(
        0 < count and len(shape) == 3 and all(v > 0 for v in shape),
        "invalid synthetic source shape",
    )
    teacher = deepcopy(backend.model).eval().requires_grad_(False)
    names = cfg.get("attention_modules", [])
    install_attention(teacher, names)
    image = torch.randn((count, *shape), generator=rng, device=device).requires_grad_(
        True
    )
    optim = torch.optim.Adam(
        [image], lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["adam_epsilon"]
    )
    labels = torch.randint(cfg["classes"], (count,), generator=rng, device=device)
    parts, history = {}, []
    variant = cfg["variant"]
    require(
        variant
        in (
            "psaq_kde",
            "e2h_crop",
            "clamp_local_adapter",
            "structured_prior_adapter",
            "sardfq_gmm",
        ),
        "unknown image objective",
    )
    priors = {}
    if variant == "sardfq_gmm":
        from .semantic_priors import attention_prior, semantic_targets, semantic_loss

        targets = semantic_targets(image, cfg["classes"], cfg, rng)
    lo, hi = [image.new_tensor(v).view(1, -1, 1, 1) for v in cfg["channel_limits"]]
    target_tv = cfg.get("tv_target")
    if isinstance(target_tv, list):
        target_tv = float(
            torch.empty((), device=device).uniform_(*target_tv, generator=rng)
        )
    steps = sum(s["steps"] for s in cfg["phases"])
    iteration = 0
    require(
        steps > 0 and (variant == "e2h_crop" or bool(names)),
        "missing synthesis phases or attention taps",
    )
    collecting = True

    def tap(name):
        def capture(_m, a, y):
            if not collecting:
                return
            effective_batch = (
                image.shape
                if y.size(0) == len(image) or backend.spec.get("family") == "Swin"
                else (y.size(0), *image.shape[1:])
            )
            layout = ProfileAdapter().token_layout(
                backend,
                name + ".qkv",
                [y.size(0), y.size(2), y.size(3)],
                effective_batch,
            )
            left, right = layout["patch_slice"]
            parts[name] = (y, a[0], left, right)

        return capture

    with ExitStack() as stack, torch.enable_grad():
        for name in names:
            stack.callback(
                teacher.get_submodule(name + "._ptq_probability")
                .register_forward_hook(tap(name))
                .remove
            )
        for phase in cfg["phases"]:
            warm = phase["warmup"]
            lim = phase["jitter"]
            for step in range(phase["steps"]):
                multiplier = (
                    (step + 1) / warm
                    if step < warm
                    else (
                        1
                        + math.cos(
                            math.pi * (step - warm) / max(1, phase["steps"] - warm)
                        )
                    )
                    / 2
                )
                optim.param_groups[0]["lr"] = cfg["lr"] * multiplier
                optim.zero_grad(set_to_none=True)
                parts.clear()
                offset = int(
                    torch.randint(-lim, lim + 1, (), generator=rng, device=device)
                )
                view = image.roll((offset, offset), (-2, -1))
                if (
                    cfg.get("flip")
                    and float(torch.rand((), generator=rng, device=device)) > 0.5
                ):
                    view = view.flip(-1)
                if variant == "e2h_crop":
                    lower = 1.0 - iteration / max(1, steps - 1) * (
                        1 - cfg["crop_min_area"]
                    )
                    area = (
                        float(torch.rand((), generator=rng, device=device))
                        * (1 - lower)
                        + lower
                    )
                    h, w = view.shape[-2:]
                    ch, cw = (
                        max(1, round(h * math.sqrt(area))),
                        max(1, round(w * math.sqrt(area))),
                    )
                    iy = int(
                        torch.randint(h - ch + 1, (), generator=rng, device=device)
                    )
                    ix = int(
                        torch.randint(w - cw + 1, (), generator=rng, device=device)
                    )
                    view = F.interpolate(
                        view[:, :, iy : iy + ch, ix : ix + cw],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                pred = differentiable_forward(backend, teacher, view)
                require(
                    torch.is_tensor(pred)
                    and pred.shape == (count, cfg["classes"])
                    and pred.requires_grad,
                    "synthesis needs bound differentiable classifier",
                )
                loss = F.cross_entropy(pred, labels) * cfg["class_weight"]
                if variant == "sardfq_gmm":
                    collecting = False
                    try:
                        loss = cfg["class_weight"] * semantic_loss(
                            backend, teacher, pred, view, targets
                        )
                    finally:
                        collecting = True
                tv = total_variation(view)
                tv = tv if target_tv is None else (tv - target_tv).abs()
                loss = loss + cfg["tv_weight"] * tv
                for name, (feat, prob, left, right) in parts.items():
                    patches = feat.mean(1)[:, left:right]
                    if variant == "psaq_kde":
                        loss = loss - cfg["feature_weight"] * patch_entropy(
                            patches,
                            cfg["bandwidth"],
                            cfg["kde_samples"],
                            cfg["kde_epsilon"],
                        )
                    elif variant == "clamp_local_adapter":
                        loss = loss + cfg["feature_weight"] * local_patch_loss(
                            patches,
                            cfg["patch_grid"],
                            cfg["positives"],
                            cfg["window"],
                            cfg["temperature"],
                        )
                    elif variant == "sardfq_gmm":
                        spec = cfg["prior_sites"][name]
                        require(
                            left in (0, 1) and right == prob.size(-1),
                            "additional special tokens require a bound prior",
                        )
                        actual = prob[:, :, 0, left:right] if left else prob.mean(-2)
                        if name not in priors:
                            priors[name] = attention_prior(
                                prob, spec["grid"], left, cfg, rng
                            )
                        loss = (
                            loss
                            + cfg["feature_weight"]
                            * spec["depth_fraction"]
                            * (actual - priors[name]).square().mean((0, 2)).sum()
                        )
                    elif variant == "structured_prior_adapter":
                        target = image.new_tensor(cfg["attention_prior"][name])
                        require(
                            target.shape[-2:] == prob.shape[-2:], "prior shape differs"
                        )
                        loss = (
                            loss
                            + cfg["feature_weight"] * (prob - target).square().mean()
                        )
                require(
                    bool(torch.isfinite(loss)) and set(parts) == set(names),
                    "invalid synthesis trajectory",
                )
                loss.backward()
                optim.step()
                with torch.no_grad():
                    image.clamp_(lo, hi)
                history.append(float(loss.detach()))
                iteration += 1
    images = image.detach().cpu()
    torch.save(images, d / "images.pt")
    manifest = {
        "schema": 1,
        "generator": cfg,
        "teacher": backend.identity(),
        "ids": [tensor_digest(t.unsqueeze(0)) for t in images],
        "labels": labels.tolist(),
        "loss": history,
        "steps": iteration,
        "tv_target": target_tv,
        "image_sha256": file_sha256(d / "images.pt"),
    }
    if variant == "sardfq_gmm":
        manifest["semantic_targets"] = [
            {
                "boxes": boxes,
                "classes": ids,
                "whole_probabilities": probs.tolist(),
                "crop_probabilities": [t.tolist() for t in locals_],
            }
            for boxes, ids, probs, locals_ in targets
        ]
        torch.save({k: v.cpu() for k, v in priors.items()}, d / "priors.pt")
        manifest["prior_sha256"] = file_sha256(d / "priors.pt")
    atomic_json(d / "manifest.json", manifest)
    return manifest


class CalibrationView:
    """Replace just the calibration split; keep genuine evaluation views and labels bound to the original model."""

    def __init__(self, data, backend, directory):
        self._data = data
        d = Path(directory)
        record = read_json(d / "manifest.json")
        require(
            record["schema"] == 1
            and record["teacher"] == backend.identity()
            and file_sha256(d / "images.pt") == record["image_sha256"],
            "synthetic source differs from the bound teacher or bytes",
        )
        self.images = torch.load(d / "images.pt", map_location="cpu", weights_only=True)
        self.generated = ["synthetic/" + s for s in record["ids"]]
        require(
            [tensor_digest(t[None]) for t in self.images] == record["ids"]
            and len(set(self.generated)) == len(self.images),
            "synthetic image identity mismatch",
        )
        self.labels = {**data.labels, **dict(zip(self.generated, record["labels"]))}
        self.identity = deepcopy(data.identity)
        self.identity["synthetic_source"] = fingerprint(record)
        self.identity["generator_variant"] = record["generator"]["variant"]
        if record["generator"]["variant"] == "clamp_cyclic":
            self.identity["cyclic_result"] = {
                "plan": record["winning_plan"],
                "pool_sha256": record["image_sha256"],
            }
        self.identity["split_ids"]["calibration"] = self.generated
        self.identity.setdefault("counts", {})["calibration"] = len(self.images)
        self.shape_observation_splits = ("calibration",)
        self.backend = backend

    def ids(self, split):
        return list(self.generated) if split == "calibration" else self._data.ids(split)

    def batches(self, split):
        if split != "calibration":
            yield from self._data.batches(split)
            return
        for i, sid in enumerate(self.generated):
            yield self.images[i : i + 1].to(self.backend.device), sid

    def source_rows(self, split):
        if split != "calibration":
            yield from self._data.source_rows(split)
            return
        for sid in self.generated:
            yield {
                "sample_id": sid,
                "input_sha256": sid.split("/", 1)[1],
                "dataset_split": "generated_calibration",
            }
