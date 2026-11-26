"""Compare a measured calibration selector with the same complete candidates on unseen task images."""

import math
from random import Random
from statistics import median
from .common import EvalError


def average_ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for ix in ordered[i:j]:
            ranks[ix] = (i + j - 1) / 2
        i = j
    return ranks


def correlations(proxy, utility):
    left, right = average_ranks([-v for v in proxy]), average_ranks(utility)

    def centered(vec):
        return [x - sum(vec) / len(vec) for x in vec]

    a, b = centered(left), centered(right)

    def safe(num, den):
        return num / den if den else None

    pairs = [
        ((x > y) - (x < y), (u > v) - (u < v))
        for i, (x, u) in enumerate(zip(left, right))
        for y, v in zip(left[i + 1 :], right[i + 1 :])
    ]
    return {
        "spearman": safe(
            sum(x * y for x, y in zip(a, b)),
            math.sqrt(sum(x * x for x in a) * sum(y * y for y in b)),
        ),
        "kendall_tau_b": safe(
            sum(x * y for x, y in pairs),
            math.sqrt(sum(x != 0 for x, _ in pairs) * sum(y != 0 for _, y in pairs)),
        ),
    }


def mean_utility(rows, ids, field, direction):
    result = {}
    for name, r in rows.items():
        values = [r["per_image"][sid][field] for sid in ids]
        if direction == "confusion_miou":
            import numpy as np

            if isinstance(values[0], dict):
                size = values[0]["classes"]
                m = np.zeros((size, size), dtype=np.int64)
                for item in values:
                    if item["classes"] != size:
                        raise EvalError("class axes changed")
                    np.add.at(m.reshape(-1), item["indices"], item["counts"])
            else:
                m = np.array(values).sum(axis=0)
            if m.ndim != 2 or m.shape[0] != m.shape[1] or (m < 0).any():
                raise EvalError("invalid per-original class confusion")
            denom = m.sum(axis=0) + m.sum(axis=1) - m.diagonal()
            if not (denom > 0).any():
                raise EvalError("no class support")
            result[name] = float((m.diagonal()[denom > 0] / denom[denom > 0]).mean())
        else:
            result[name] = sum(values) / len(values) * (-1 if direction == "min" else 1)
    return result


def compare(
    calibration, heldout, *, proxy, task, direction, baseline, k=1, resamples=0, seed=0
):
    from ptq_agent.execution.research_metrics import selection_quality

    p, t = calibration, heldout
    if not (
        p["split"] == "calibration"
        and t["split"] == "search"
        and p["parent_state_sha256"] == t["parent_state_sha256"]
        and p["parent_model_ref"] == t["parent_model_ref"]
    ):
        raise EvalError(
            "selection assessment needs matched parent and independently identified views"
        )
    pi, ti = p["sample_ids"], t["sample_ids"]
    if (
        not pi
        or not ti
        or set(pi) & set(ti)
        or len(set(pi)) != len(pi)
        or len(set(ti)) != len(ti)
    ):
        raise EvalError(
            "use disjoint original-image IDs; no repeated image/window units"
        )
    prows, trows = [{r["id"]: r for r in d["candidates"]} for d in (p, t)]
    if (
        prows.keys() != trows.keys()
        or baseline not in prows
        or not (0 < k <= len(prows))
        or resamples < 0
    ):
        raise EvalError("candidate pool, baseline or bootstrap scope mismatch")
    for n in prows:
        identity = (
            "candidate_sha256"
            if "candidate_sha256" in prows[n]["receipt"]
            else "model_ref"
        )
        if prows[n]["receipt"][identity] != trows[n]["receipt"][identity]:
            raise EvalError("candidate state changed: " + n)

    def evaluate(left, right):
        scores = mean_utility(prows, left, proxy, "max")
        util = mean_utility(trows, right, task, direction)
        return selection_quality(scores, util, baseline=util[baseline], k=k)

    report = evaluate(pi, ti)
    ids = sorted(prows)
    report.update(
        correlations(
            [report["scores"][i] for i in ids], [report["utilities"][i] for i in ids]
        )
    )
    report["baseline_id"] = baseline
    report["task_best_in_proxy_topk"] = (
        report["pool_best_id"]
        in sorted(ids, key=lambda i: (report["scores"][i], i))[:k]
    )
    report["measurement_cost"] = {
        label: d.get("measurement_cost")
        for label, d in (("calibration", p), ("heldout", t))
    }
    if resamples:
        import numpy as np

        rng = Random(seed)
        draws = [
            evaluate(rng.choices(pi, k=len(pi)), rng.choices(ti, k=len(ti)))
            for _ in range(resamples)
        ]
        report["bootstrap"] = {
            "resamples": resamples,
            "seed": seed,
            "unit": "original image; candidates paired within each split; choose again on each resample",
            "regret_interval_95": np.quantile(
                [r["selected_regret"] for r in draws], [0.025, 0.975]
            ).tolist(),
            "wins": {n: sum(r["selected_id"] == n for r in draws) for n in ids},
        }
    return {
        "assessment": "independent search evaluation of a predeclared pool",
        "proxy_field": proxy,
        "task_field": task,
        "task_direction": direction,
        "proxy_direction": "min",
        "calibration_ids": pi,
        "heldout_ids": ti,
        "parent_state_sha256": p["parent_state_sha256"],
        "result": report,
    }
