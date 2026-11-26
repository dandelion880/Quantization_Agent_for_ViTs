"""HC-PTQ reference. Each packed index selects a whole input-feature vector."""

import math
import torch
from ..schemas import require
from . import codecs


def exp0(x, c):
    norm = x.norm(dim=-1, keepdim=True)
    length = math.sqrt(c) * norm
    return (
        torch.where(length > 0, torch.tanh(length) / length.clamp_min(1e-100), 1.0) * x
    )


def log0(x, c):
    length = math.sqrt(c) * x.norm(dim=-1, keepdim=True)
    require(
        bool((length < 1).all()),
        "Poincare radius reached precision boundary; select a documented smaller curvature",
    )
    return (
        torch.where(length > 0, torch.atanh(length) / length.clamp_min(1e-100), 1.0) * x
    )


def fit(w, spec, settings):
    c, steps, seed = settings["curvature"], settings["iterations"], settings["seed"]
    k = 1 << spec["bits"]
    require(
        c > 0 and steps > 0 and k <= len(w) and w.ndim == 2,
        "invalid row-codebook budget or geometry",
    )
    rows = w.detach().cpu().double()
    u = exp0(rows, c)
    tangent = log0(u, c)
    generator = torch.Generator().manual_seed(seed)
    centers = tangent[torch.randperm(len(rows), generator=generator)[:k]]
    previous = None
    for step in range(steps):
        assignment = torch.cdist(tangent, centers).argmin(1)
        if previous is not None and torch.equal(assignment, previous):
            break
        counts = torch.bincount(assignment, minlength=k)[:, None]
        sums = torch.zeros_like(centers).index_add_(0, assignment, tangent)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
        previous = assignment
    ball = exp0(centers, c)
    # acosh is monotone on [1,infinity). Compute the denominator of hyperbolic distance per pair.
    margin = (1 - c * u.square().sum(1, keepdim=True)) * (
        1 - c * ball.square().sum(1)[None, :]
    )
    require(bool((margin > 0).all()), "invalid curvature or center")
    chosen = (torch.cdist(u, ball).square() / margin).argmin(1).byte()
    table = log0(ball, c).float()
    result = {
        "method": "hc_rows",
        "spec": dict(spec),
        "shape": list(w.shape),
        "data": codecs.pack(chosen, spec["bits"]),
        "params": {"codebook": table},
    }
    return result, {
        "unit": "index per output row",
        "codebook_bytes": table.nbytes,
        "index_bytes": result["data"].nbytes,
        "iterations": step + 1,
        "curvature": c,
        "initialization": "seeded distinct rows",
        "empty_centers": "retain",
    }


def decode(record, device):
    table = record["params"]["codebook"].to(device)
    ids = codecs.unpack(
        record["data"], record["spec"]["bits"], [record["shape"][0]], device
    ).long()
    return table.index_select(0, ids)


def validate(record, spec, shape):
    table = record["params"].get("codebook")
    require(
        record["shape"] == shape
        and record["spec"] == spec
        and set(record["params"]) == {"codebook"}
        and table.shape == (1 << spec["bits"], shape[1])
        and table.dtype == torch.float32
        and bool(torch.isfinite(table).all())
        and record["data"].dtype == torch.uint8
        and record["data"].numel() == (shape[0] * spec["bits"] + 7) // 8,
        "invalid vector codebook or packed row indices",
    )
