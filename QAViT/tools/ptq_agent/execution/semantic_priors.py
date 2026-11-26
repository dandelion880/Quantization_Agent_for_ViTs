"""SARDFQ Eq. 6–12. Paper and public generator settings are named independently."""

import torch
from torch.nn import functional as F
from ..schemas import require
from .calibration import differentiable_forward


def attention_prior(prob, grid, cls, cfg, gen):
    h, w = grid
    n, heads = prob.shape[:2]
    limits = cfg["gaussians"]
    centers = cfg["center_bounds"]
    widths = cfg["grid_extents"]
    require(
        h * w == prob.shape[-1] - cls and cls in (0, 1) and limits and widths,
        "prior geometry requires checked patch/window axes",
    )
    values = []
    for _ in range(n * heads):
        count = limits[
            int(torch.randint(len(limits), (), device=prob.device, generator=gen))
        ]
        extent = widths[
            int(torch.randint(len(widths), (), device=prob.device, generator=gen))
        ]
        x, y = torch.meshgrid(
            torch.linspace(-extent, extent, w, device=prob.device, dtype=torch.float64),
            torch.linspace(-extent, extent, h, device=prob.device, dtype=torch.float64),
            indexing="xy",
        )
        points = torch.stack((x, y), -1)
        means = torch.empty((count, 2), device=x.device, dtype=x.dtype).uniform_(
            *centers, generator=gen
        )
        A = (
            torch.rand((count, 2, 2), device=x.device, dtype=x.dtype, generator=gen)
            * cfg["covariance_scale"]
        )
        eig, U = torch.linalg.eigh(A @ A.mT)
        eig = eig.clamp(*cfg["eigen_bounds"])
        local = torch.einsum("khwi,kij->khwj", points[None] - means[:, None, None], U)
        logpdf = -0.5 * (
            (local.square() / eig[:, None, None]).sum(-1)
            + eig.log().sum(-1)[:, None, None]
        )
        density = (logpdf.amax(0) - logpdf.max()).exp().flatten()
        if cfg["normalization"] == "code_softmax_max":
            density = density.softmax(0)
        else:
            require(cfg["normalization"] == "paper_mass", "unknown prior normalization")
        mass = 1 - torch.rand((), device=prob.device, generator=gen) if cls else 1.0
        values.append(density / density.sum() * mass)
    return torch.stack(values).reshape(n, heads, h * w).to(prob)


def semantic_targets(images, classes, cfg, gen):
    """Use a seeded nonoverlapping rectangular partition and save each correspondence."""
    h, w = images.shape[-2:]
    result = []

    def target(ids):
        logits = torch.rand(classes, device=images.device, generator=gen)
        values = torch.empty(len(ids), device=images.device).uniform_(
            *cfg["semantic_logit_range"], generator=gen
        )
        logits[ids] = values
        return logits.softmax(0)

    for _ in images:
        m = int(
            torch.randint(
                1, cfg["max_crops"] + 1, (), device=images.device, generator=gen
            )
        )
        require(m <= classes, "one target per local semantic")
        ids = torch.randperm(classes, generator=gen, device=images.device)[:m].tolist()
        boxes = [(0, 0, h, w)]
        while len(boxes) < m:
            j = max(
                range(len(boxes)),
                key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
            )
            a, b, c, d = boxes.pop(j)
            if c - a >= d - b:
                mid = (a + c) // 2
                require(a < mid < c, "crop partition exhausted")
                boxes.extend(((a, b, mid, d), (mid, b, c, d)))
            else:
                mid = (b + d) // 2
                require(b < mid < d, "crop partition exhausted")
                boxes.extend(((a, b, c, mid), (a, mid, c, d)))
        result.append((boxes, ids, target(ids), [target([i]) for i in ids]))
    return result


def semantic_loss(backend, teacher, global_logits, images, targets):
    value = F.cross_entropy(global_logits, torch.stack([item[2] for item in targets]))
    local = []
    for i, (boxes, ids, _, labels) in enumerate(targets):
        crops = [
            F.interpolate(
                images[i : i + 1, :, a:c, b:d],
                images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for a, b, c, d in boxes
        ]
        local.append(
            F.cross_entropy(
                differentiable_forward(backend, teacher, torch.cat(crops)),
                torch.stack(labels),
            )
        )
    return value + torch.stack(local).mean()
