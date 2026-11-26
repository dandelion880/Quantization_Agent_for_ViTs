"""Alternating operand searches at a bound block/bridge, with reusable FP derivatives."""

from copy import deepcopy
import torch
from ..schemas import require
from .calibration_search import apply_grid_changes
from .graph_fit import output
from .curvature_fit import BlockObjective
from .research_metrics import SENSITIVITY, score


def search(backend, model, data, stage, cache_mib):
    from .pipeline import quantizers

    name, cfg = stage["module"], stage["settings"]
    samples = {sid: x for x, sid in data.batches("calibration")}
    args = cfg["metric"]
    metric = args["kind"]
    qs = quantizers(model)
    require(
        cfg["rounds"] > 0 and all(v > 0 for v in cfg["factors"]),
        "scale candidates must be positive",
    )
    sensitivity = (
        BlockObjective(
            backend, backend.model, samples, data.labels, name, args, cache_mib
        )
        if metric in SENSITIVITY
        else None
    )
    if sensitivity:
        sensitivity.refresh(model, 0)
    with torch.no_grad():
        targets = [
            output(backend, backend.model, name, x, "output").detach()
            for x in samples.values()
        ]
    ranking = cfg.get("attention_ranking")
    if ranking:
        from .sites import install_attention

        teacher = deepcopy(backend.model)
        install_attention(teacher, [ranking["module"]])
        tap = ranking["module"] + "._ptq_softmax"
        with torch.no_grad():
            rank_refs = [
                output(backend, teacher, tap, x, "input").detach()
                for x in samples.values()
            ]
    trace = []
    bias_history = []
    for _ in range(cfg["rounds"]):
        for sid, target in cfg["operand_order"]:
            q = qs[sid]
            require(
                q.layer[target]
                and (
                    q in set(model.get_submodule(name).modules())
                    or cfg.get("upstream_bridge")
                ),
                "operand outside declared scope",
            )
            old = (
                q.params()
                if target == "activation"
                else (q.weight_record, q.decoded_weight)
            )

            def reset():
                if target == "activation":
                    q.set_params(old)
                else:
                    q.weight_record, q.decoded_weight = old

            results = []
            with torch.no_grad():
                for factor in cfg["factors"]:
                    reset()
                    apply_grid_changes(
                        model, {sid: {target: factor}}, {sid: q.last_input_shape}
                    )
                    vals = []
                    for i, x in enumerate(samples.values()):
                        y = output(backend, model, name, x, "output")
                        val = (
                            sensitivity.evaluate(targets[i], y, i)
                            if sensitivity
                            else score(metric, targets[i], y, options=args)
                        )
                        if ranking:
                            logits = output(backend, model, tap, x, "input")
                            val = val + ranking["gamma"] * score(
                                "attention_pairwise_hinge",
                                rank_refs[i],
                                logits,
                                options=ranking,
                            )
                        vals.append(val)
                    results.append(float(torch.stack(vals).mean()))
                idx = min(range(len(results)), key=results.__getitem__)
                reset()
                apply_grid_changes(
                    model,
                    {sid: {target: cfg["factors"][idx]}},
                    {sid: q.last_input_shape},
                )
            trace.append(
                {
                    "site": sid,
                    "operand": target,
                    "scores": results,
                    "chosen": cfg["factors"][idx],
                }
            )
        with torch.no_grad():
            for site in cfg.get("mean_bias_sites", []):
                q = qs[site]
                require(
                    q.layer["kind"] == "linear",
                    "per-round bias correction needs a Linear output",
                )
                residuals = [
                    output(backend, backend.model, site, x, "output")
                    - output(backend, model, site, x, "output")
                    for x in samples.values()
                ]
                delta = torch.cat([d.reshape(-1, d.shape[-1]) for d in residuals]).mean(
                    0
                )
                if hasattr(q, "fitted_input_bias"):
                    q.fitted_input_bias.add_(delta)
                else:
                    q.register_buffer("fitted_input_bias", delta.clone())
                bias_history.append({"site": site, "delta_norm": float(delta.norm())})
    return {
        "kind": "block_coordinate_search",
        "rank_objective": ranking,
        "folded_bias": bias_history,
        "module": name,
        "metric": args,
        "sample_ids": list(samples),
        "sensitivity_preparation": sensitivity.refreshes if sensitivity else [],
        "trace": trace,
    }
