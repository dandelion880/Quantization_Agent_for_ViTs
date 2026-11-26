"""Prepared sensitivities and candidate objectives. Gradients are taken only in prepare."""

import math
import torch
import torch.nn.functional as F
from ..schemas import fingerprint, require

SENSITIVITY = {
    "squared_gradient_error",
    "apq_masked_gradient_error",
    "aph_signed_directional",
    "aph_official_abs",
    "fim_secant_paper",
    "fima_official_abs_normalized",
}
PAIR = {
    "mse",
    "pearson_error",
    "sqnr_power_db",
    "attention_pairwise_hinge",
    "attention_rank_agreement",
    "attention_topk_overlap",
    "teacher_kl",
    "contrastive_teacher_logits",
    "residual_bias_profile",
    "residual_variance_profile",
    "nuclear_norm_proxy",
}
VERSIONS = dict.fromkeys(SENSITIVITY | PAIR, 1)


def task_loss(
    pred, *, teacher=None, labels=None, mask=None, temperature=1.0, ignore_index=None
):
    """Bound classification [B,C] or dense [B,C,H,W]. Mean over valid class vectors."""
    require(
        pred.ndim in (2, 4)
        and temperature > 0
        and (teacher is None) != (labels is None),
        "task objective needs logits and exactly one compatible target",
    )
    if labels is not None:
        require(
            labels.shape == pred.shape[:1] + pred.shape[2:],
            "label geometry differs from logits",
        )
        if ignore_index is not None:
            valid = labels != ignore_index
            mask = valid if mask is None else mask & valid
        loss = F.cross_entropy(
            pred,
            labels.masked_fill(~mask, 0) if mask is not None else labels,
            reduction="none",
        )
    else:
        require(pred.shape == teacher.shape, "teacher geometry differs from candidate")
        a, b = teacher.detach() / temperature, pred / temperature
        loss = (a.softmax(1) * (a.log_softmax(1) - b.log_softmax(1))).sum(1)
    if mask is not None:
        require(
            mask.shape == loss.shape and mask.dtype == torch.bool and bool(mask.any()),
            "no valid task positions",
        )
        loss = loss[mask]
    return loss.mean()


def cache_key(name, binding, options):
    required = {
        "reference",
        "input_state",
        "suffix_state",
        "head",
        "task",
        "view",
        "sample_ids",
        "observation",
        "axes",
        "reduction",
    }
    require(
        name in VERSIONS and required <= set(binding),
        "incomplete metric cache identity",
    )
    return fingerprint(
        {
            "metric": name,
            "version": VERSIONS[name],
            "binding": binding,
            "estimator": options,
        }
    )


def prepare(
    name,
    z,
    loss_at,
    *,
    binding,
    options,
    perturbations=None,
    initial_error=None,
    defer_factorization=False,
):
    """z is [samples,...features]. loss_at must execute the bound differentiable head/suffix."""
    require(name in SENSITIVITY, "this objective does not require a backward prepare")
    state = {
        "key": cache_key(name, binding, options),
        "metric": name,
        "options": dict(options),
    }
    with torch.inference_mode(False), torch.enable_grad():
        z = z.detach().clone()

        def gradient(value):
            value = value.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(loss_at(value), value)
            require(bool(torch.isfinite(g).all()), "nonfinite read-only gradient")
            return g.detach()

        if name.startswith("aph_"):
            delta = options["delta"]
            require(math.isfinite(delta) and delta > 0, "invalid APH perturbation")
            diff = gradient(z + delta) - gradient(z - delta)
            state["weight"] = (
                diff.abs() if name == "aph_official_abs" else diff / (2 * delta)
            ).mean(0)
        elif name in ("squared_gradient_error", "apq_masked_gradient_error"):
            state["weight"] = gradient(z).square()
        else:
            require(
                perturbations is not None
                and perturbations.shape[1:] == z.shape
                and len(perturbations) == options["rank"],
                "FIMA needs aligned progressive perturbations [rank,B,...]",
            )
            D = perturbations.detach().to(z)
            G = torch.stack([gradient(z + d) for d in D])
            if name == "fima_official_abs_normalized":
                D, G = D.abs(), G.abs()
            D, G = (v.mean(1).flatten(1).double() for v in (D, G))
            state.update(D=D, G=G)
            if initial_error is not None:
                state["initial_error"] = initial_error.detach()
            if not defer_factorization:
                finalize_fima(state)
    return state


def finalize_fima(state):
    D = state["D"]
    system = D @ D.T
    state["effective_rank"] = int(torch.linalg.matrix_rank(D))
    if state["options"].get("solver", "solve") == "pinv":
        state["inverse"] = torch.linalg.pinv(system, hermitian=True)
    else:
        require(
            state["effective_rank"] == len(D),
            "rank-deficient FIMA directions; specify pinv as an adaptation",
        )
        state["inverse"] = torch.linalg.solve(
            system, torch.eye(len(D), device=D.device, dtype=D.dtype)
        )
    initial = state.pop("initial_error", None)
    if state["metric"] == "fima_official_abs_normalized":
        require(initial is not None, "missing initial error for this refresh")
        normalizers = _fima_terms(state, initial)
        require(
            all(bool(v != 0) for v in normalizers),
            "undefined FIMA initial normalization",
        )
        state["normalizers"] = [v.detach() for v in normalizers]


def _fima_terms(state, e):
    D, G, B = (
        state[k].to(device=e.device, dtype=torch.float64) for k in ("D", "G", "inverse")
    )
    e = e.reshape(-1, D.shape[1]).double()
    if state["metric"] == "fima_official_abs_normalized":
        u = e.abs() @ G.T
        return ((u @ B) * u).sum(-1).mean(), (e.square() * G.mean(0)).mean()
    u = e @ G.T
    require(
        bool((D[-1] != 0).all()), "undefined paper diagonal secant on zero perturbation"
    )
    return ((u @ B) * (e @ D.T)).sum(-1).mean(), (e.square() * (G[-1] / D[-1])).sum(
        -1
    ).mean()


def score(name, reference, candidate, *, options=None, prepared=None, binding=None):
    require(
        reference.shape == candidate.shape and reference.numel() > 0,
        "paired metric shape mismatch",
    )
    opt = options or {}
    a, b = reference.double(), candidate.double()
    e = b - a
    if name in SENSITIVITY:
        require(
            prepared and prepared["key"] == cache_key(name, binding, opt),
            "stale sensitivity preparation",
        )
        if name == "squared_gradient_error" or name.startswith(("aph_", "apq_")):
            if name in ("squared_gradient_error", "apq_masked_gradient_error"):
                require(
                    prepared["weight"].shape == e.shape,
                    "per-observation gradient and output error must have identical axes",
                )
            weighted = e.square() * prepared["weight"].to(e)
            if name.startswith("apq_"):
                require(0 <= opt["gamma"] <= 1, "invalid APQ quantile")
                cutoff = torch.quantile(e.abs(), opt["gamma"])
                keep = (
                    e.abs() >= cutoff
                    if opt["mask_rule"] == "text_top_error"
                    else e.abs() < cutoff
                )
                require(
                    opt["mask_rule"] in ("text_top_error", "bottom_magnitude_adapter"),
                    "declare APQ mask convention",
                )
                weighted = weighted * keep
            return weighted.flatten(1).sum(1).mean()
        lo, diag = _fima_terms(prepared, e)
        if name == "fim_secant_paper":
            alpha = opt["alpha"]
            return alpha * lo + (1 - alpha) * diag
        n1, n2 = prepared["normalizers"]
        return lo / n1 + diag / n2
    if name == "mse":
        return e.square().mean()
    if name == "teacher_kl":
        return task_loss(b, teacher=a, temperature=opt.get("temperature", 1.0))
    if name == "contrastive_teacher_logits":
        require(
            a.ndim == 2 and len(a) > 1 and opt["temperature"] > 0,
            "contrastive fitness requires the same batch of negatives",
        )
        if opt["normalize"]:
            a, b = F.normalize(a), F.normalize(b)
        logits = b @ a.T / opt["temperature"]
        return F.cross_entropy(logits, torch.arange(len(a), device=a.device)) * opt.get(
            "multiplier", 1.0
        )
    if name == "pearson_error":
        axes = tuple(opt["axes"])
        ca, cb = a - a.mean(axes, keepdim=True), b - b.mean(axes, keepdim=True)
        denom = (ca.square().sum(axes) * cb.square().sum(axes)).sqrt()
        require(bool((denom > 0).all()), "undefined Pearson for a constant slice")
        return 1 - ((ca * cb).sum(axes) / denom).mean()
    if name == "sqnr_power_db":
        return 10 * (a.square().sum() / e.square().sum()).log10()
    if name == "nuclear_norm_proxy":
        return (
            torch.linalg.matrix_norm(a, ord="nuc", dim=(-2, -1)).mean()
            * e.square().sum()
        )
    if name in ("residual_bias_profile", "residual_variance_profile"):
        axes = tuple(opt["axes"])
        return (
            e.mean(axes)
            if name == "residual_bias_profile"
            else e.var(axes, unbiased=False)
        )
    a, b = (x.movedim(opt["key_axis"], -1) for x in (a, b))
    if name == "attention_topk_overlap":
        k = opt["k"]
        require(0 < k <= a.size(-1), "invalid top-k")
        ai, bi = (
            x.argsort(dim=-1, descending=True, stable=True)[..., :k] for x in (a, b)
        )
        return (ai.unsqueeze(-1) == bi.unsqueeze(-2)).any(-1).double().mean()
    require(
        name in ("attention_pairwise_hinge", "attention_rank_agreement"),
        "unknown metric",
    )
    total, count = a.new_zeros(()), 0
    for i in range(a.size(-1) - 1):
        sign = (a[..., i : i + 1] - a[..., i + 1 :]).sign()
        tie = opt.get("ties", "exclude")
        require(tie in ("exclude", "paper_zero_sign"), "unknown ranking tie rule")
        valid = (
            torch.ones_like(sign, dtype=torch.bool)
            if tie == "paper_zero_sign"
            else sign != 0
        )
        margin = ((b[..., i : i + 1] - b[..., i + 1 :]) * sign)[valid]
        total = total + (
            (opt["margin"] - margin).clamp_min(0).sum()
            if name == "attention_pairwise_hinge"
            else (margin > 0).sum()
        )
        count += int(valid.sum())
    require(count > 0, "no nontied reference key pairs")
    reduction = opt.get("pair_reduction", "mean")
    require(reduction in ("sum", "mean"), "unknown pair reduction")
    return total if reduction == "sum" else total / count


def selection_quality(scores, utility, *, baseline, k=1):
    require(
        scores and scores.keys() == utility.keys() and 0 < k <= len(scores),
        "selection pool mismatch",
    )
    require(
        all(math.isfinite(x) for x in [*scores.values(), *utility.values(), baseline]),
        "nonfinite selection values",
    )
    proxy = sorted(scores, key=lambda i: (scores[i], i))
    truth = sorted(scores, key=lambda i: (-utility[i], i))
    selected, best = proxy[0], truth[0]
    pairs = [
        (a, b)
        for i, a in enumerate(proxy)
        for b in proxy[i + 1 :]
        if utility[a] != utility[b] and scores[a] != scores[b]
    ]
    return {
        "selected_id": selected,
        "pool_best_id": best,
        "selected_regret": utility[best] - utility[selected],
        "candidate_pool_coverage": utility[best] - baseline,
        "topk_recall": len(set(proxy[:k]) & set(truth[:k])) / k,
        "pairwise_accuracy": sum(utility[a] > utility[b] for a, b in pairs) / len(pairs)
        if pairs
        else None,
        "utilities": dict(utility),
        "scores": dict(scores),
        "tie_policy": "ID for winner; exclude tied pairs",
    }
