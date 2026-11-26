"""Read-only Fisher/GSM and conservative positive relevance. Keep data usage distinct from scale fitting."""

from copy import deepcopy
import torch
import torch.nn.functional as F
from ..schemas import require
from .calibration import differentiable_forward


def fisher(backend, rows, labels, sites, settings):
    model = deepcopy(backend.model).eval().requires_grad_(False)
    modes = {"empirical_labels", "model_distribution", "teacher_class"}
    require(settings["target"] in modes, "unknown Fisher supervision")
    params = {n: model.get_submodule(n).weight.requires_grad_(True) for n in sites}
    gen = torch.Generator(device=backend.device).manual_seed(settings["seed"])
    sums = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in params.items()}
    targets = []
    with torch.inference_mode(False), torch.enable_grad():
        for x, sid in rows:
            logits = differentiable_forward(backend, model, x)
            require(
                logits.ndim == 2 and len(logits) == 1,
                "classification Fisher needs one image per identity",
            )
            if settings["target"] == "empirical_labels":
                require(sid in labels, "missing calibration target")
                c = torch.as_tensor([labels[sid]], device=logits.device)
            elif settings["target"] == "teacher_class":
                c = logits.argmax(-1)
            else:
                c = logits.detach().softmax(-1).multinomial(1, generator=gen).flatten()
            gs = torch.autograd.grad(F.cross_entropy(logits, c), list(params.values()))
            for n, g in zip(params, gs):
                sums[n] += g.detach().double().square()
            targets.append(int(c))
    trace = {n: float(v.sum() / len(rows)) for n, v in sums.items()}
    gsm = {
        n: float((0.5 * params[n].detach().double().square() * v / len(rows)).sum())
        for n, v in sums.items()
    }
    return {
        "trace": trace,
        "gsm": gsm,
        "target": settings["target"],
        "classes": targets,
        "sample_ids": [s for _, s in rows],
    }


def positive_linear(x, weight, relevance, *, zero_policy):
    """Eq.20, exact positive contributions without allocating [batch,tokens,out,in]."""
    pos, neg = x.clamp_min(0), x.clamp_max(0)
    wp, wn = weight.clamp_min(0), weight.clamp_max(0)
    den = F.linear(pos, wp) + F.linear(neg, wn)
    dead = (den == 0) & (relevance != 0)
    require(
        zero_policy in ("error", "retain_unassigned")
        and (not dead.any() or zero_policy == "retain_unassigned"),
        "nonzero relevance has no positive contributors",
    )
    normalized = torch.where(
        den > 0, relevance / den.clamp_min(torch.finfo(den.dtype).tiny), 0
    )
    r = pos * (normalized @ wp) + neg * (normalized @ wn)
    return r, torch.where(dead, relevance, 0).sum()


def type_adjust(fisher_trace, types, probes):
    """Each probe contains measured drops of individually quantized layers of one type."""
    coeff = {}
    for t, rows in probes.items():
        require(
            rows and all(types[s] == t for s in rows),
            "type calibration crossed a declared layer type",
        )
        sums = sum(fisher_trace[s] for s in rows)
        require(sums > 0, "type calibration has zero Fisher trace")
        coeff[t] = sum(rows.values()) / sums
    return {s: coeff[types[s]] * v for s, v in fisher_trace.items()}, coeff
