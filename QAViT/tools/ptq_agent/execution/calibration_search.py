"""Discrete, training-free selection on an already loaded frozen native plan.

This is a custom-experiment numerical API, not a final evaluation or an automatic
research policy. It consumes only the caller's calibration view and scores on a private
copy unless winner installation was explicitly requested. Export/reload/dual-task evaluation retain
all their existing contracts.
"""
from copy import deepcopy
from itertools import islice
import math

import torch
from torch.nn import functional as F

from ..schemas import require, fingerprint
from ..profiling.adapters import ProfileAdapter, boundary_tensor
from .model import forward, tensor_digest
from .calibration import parameter_evidence
from .methods import get_target_method
from .plugin_math import adjusted_params, decode_weight, score, validate_activation, validate_weight

METRIC_DEFINITIONS = {
    "mse": "mean squared error, averaged over actual samples and selected output elements",
    "cosine": "1 - cosine of flattened selected output per sample; zero norm uses MSE",
    "teacher_kl": "KL(softmax(original FP logits) || softmax(candidate logits)), temperature=1, sample mean",
    "task_loss": "classification cross entropy against actual calibration labels, sample mean",
    "top1_error": "fraction of calibration examples whose argmax differs from the actual label",
    "hessian_diag_ce": "mean((grad_logits CE(baseline logits,label) * (candidate-reference))^2); diagonal squared-gradient proxy, not a Hessian",
}


def _weight_site(module):
    return module.layer.get("kind") in ("linear", "conv2d") and all(
        hasattr(module, key) for key in ("weight_record", "decoded_weight", "reference_weight", "weight_enabled"))


def _quantizers(model):
    # load_frozen imports the audited historical source into its own namespace.
    # Its classes are deliberately not isinstance(current PipelineLinear).
    # Match the explicit numerical interface, not Python module identity.
    found = {}
    for module in model.modules():
        if not hasattr(module, "ptq_site_id"):
            continue
        require(all(hasattr(module, key) for key in ("layer", "execution", "enabled"))
                and callable(getattr(module, "params", None)) and callable(getattr(module, "set_params", None)),
                "frozen quantizer lacks the supported static numerical interface")
        require(module.ptq_site_id not in found, "duplicate frozen quantizer site")
        found[module.ptq_site_id] = module
    return found


def _digest(value):
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_digest(value), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(k): _digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_digest(v) for v in value]
    return value


def state_identity(model):
    """Include nonpersistent buffers, packed codes, execution flags and bound recipes."""
    from .evaluation import model_digest
    qmap = _quantizers(model)
    return fingerprint({**({"structure":model._qavit_structure} if getattr(model,"_qavit_structure",None) else {}), "parameters_and_buffers": model_digest(model), "sites": {
        sid: {"layer": q.layer, "execution": q.execution, "record": _digest(getattr(q, "weight_record", None)),
              "flags": {key: getattr(q, key) for key in ("enabled", "weight_enabled", "gradient_probe",
                                                         "bias_uses_decoded_weight","grid_edit_from_packed_state") if hasattr(q, key)}}
        for sid, q in qmap.items()}, "operators": {
        name: {k: getattr(m, k) for k in ("integer_enabled", "_ptq_integer_enabled") if hasattr(m, k)}
        for name, m in model.named_modules() if hasattr(m, "integer_enabled") or hasattr(m, "_ptq_integer_enabled")}})


def _candidate_params(params, edit, method):
    if isinstance(edit, dict):
        require(edit and set(edit) <= {"factor", "indices", "zero"} and ("factor" in edit or "zero" in edit),
                "grid edit needs a factor or zero")
        factor, indices, zero = edit.get("factor", 1), edit.get("indices"), edit.get("zero")
    else:
        factor, indices, zero = edit, None, None
    require(type(factor) in (float, int) and math.isfinite(factor) and factor > 0, "invalid scale factor")
    if indices is None and zero is None:
        return adjusted_params(params, factor, method)
    require(method in ("uniform_affine", "uniform_full", "uniform_minmax", "rtn") and
            "scale" in params and "repq_affine" not in params, "local edits require independent uniform grid parameters")
    p = {k: v.clone() for k, v in params.items()}
    if indices is not None:
        require(isinstance(indices, list) and indices and all(type(i) is int and 0 <= i < p['scale'].numel()
                for i in indices) and len(indices) == len(set(indices)), "indices must identify existing grid elements")
    index = slice(None) if indices is None else indices
    p["scale"].view(-1)[index] *= factor
    if zero is not None:
        require(type(zero) is int and "zero" in p and p["zero"].shape == p["scale"].shape,
                "zero requires an integer on an existing affine grid")
        # uniform_affine permits out-of-code-range offsets by construction. The
        # method validator below enforces each codec's own signed/minmax rules.
        p["zero"].view(-1)[index] = zero
    return p


def apply_grid_changes(model, scales, input_shapes):
    """Validate the complete edit before installing it. An empty edit calls no fitter or setter."""
    qmap = _quantizers(model)
    require(isinstance(scales, dict) and scales.keys() <= qmap.keys(), "unknown frozen grid site")
    if scales:
        require(not any(getattr(m, "integer_enabled", False) or getattr(m, "_ptq_integer_enabled", False)
                        for m in model.modules()), "integer compound operators need explicit rebinding")
    pending, changes = [], {}
    for sid, targets in scales.items():
        q = qmap[sid]
        require(isinstance(targets, dict) and targets and targets.keys() <= {"weight", "activation"}, "invalid grid target")
        require(q.execution["activation_calibration"] == "static", "requires static state")
        for target, edit in targets.items():
            spec, method = q.layer[target], q.layer[target + "_method"]
            require(spec is not None, "cannot enable an absent target")
            if target == "weight":
                require(_weight_site(q) and q.weight_enabled and method in ("rtn", "uniform_full", "uniform_affine", "uniform_minmax"),
                        "weight edits require an active independent RTN/uniform grid")
                old = q.weight_record["params"]
            else:
                require(q.enabled and q.quantize.__func__.__qualname__ == "ActivationQuantizer.quantize",
                        "custom activation arithmetic needs its own approved grid edit")
                old = q.params()
            new = _candidate_params(old, edit, method)
            if all(torch.equal(old[k], new[k]) for k in old):
                continue
            require(not hasattr(getattr(q,"ptq_output_correction",None),"multiplier"),"changing a folded accumulator requires a complete refit to update the signed multiplier and bias")
            if target == "activation":
                shape = input_shapes.get(sid, getattr(q, "last_input_shape", None))
                require(shape is not None, "missing bound input shape for " + sid)
                validate_activation(new, q.layer, shape)
                pending.append((q, target, new))
                before, after = old, new
            else:
                source=q.decoded_weight if getattr(q,"grid_edit_from_packed_state",False) else q.reference_weight
                record = get_target_method("weight", method).fit(source, spec, new)
                validate_weight(record, spec, list(q.reference_weight.shape))
                pending.append((q, target, (record, decode_weight(record, q.reference_weight.device))))
                before, after = q.weight_record, record
            changes.setdefault(sid, {})[target] = {"before": fingerprint(_digest(before)), "after": fingerprint(_digest(after))}
    for q, target, value in pending:
        if target == "activation":
            q.set_params(value)
        else:
            q.weight_record, q.decoded_weight = value
    return changes


def _select_scales(backend, model, data, candidates, *, metric="mse", block=None,
                         reference="fp_model", context="quantized", token_scope="all",
                         count=None, cache_mib=256):
    """Evaluate explicit scale-factor candidates and install the minimum-score one.

    candidates: [{"id": str, "scales": {site_id: {"activation": positive_factor,
                                                "weight": positive_factor}}}].
    All factors are relative to the initial frozen state, never compounded.
    block=None selects whole-model logits/features; block="blocks.3" captures the
    actual block output. reference=local_fp compares with the original FP block
    on the baseline quantized block input (block-only intervention required).
    context=float disables quantizers outside the intervention during fitting;
    the final selected state returns to its original full quantization background.
    No optimizer, weight training, calibration on search/final, or head fitting.
    """
    require(metric in METRIC_DEFINITIONS, "unsupported calibration selection metric")
    require(context in ("quantized", "float") and reference in ("fp_model", "local_fp"), "unknown calibration context/reference")
    require(token_scope in ("all", "cls", "special", "patch"), "unknown token scope")
    require(block is not None or (reference == "fp_model" and token_scope == "all"),
            "local_fp and CLS/patch selection require an explicit block")
    require(block is None or metric in ("mse", "cosine"), "logit/task metrics require whole-model outputs")
    require(reference == "fp_model" or context == "quantized", "local_fp reference uses the actual quantized block input")
    require(candidates and all(set(c) == {"id", "scales"} and isinstance(c["id"], str) and c["id"] and isinstance(c["scales"], dict) for c in candidates),
            "each candidate needs id and scales ({} is a no-op)")
    require(len({c["id"] for c in candidates}) == len(candidates), "candidate ids must be unique")
    require(count is None or type(count) is int and count > 0, "invalid calibration count")
    ids = data.ids("calibration")[:count]
    require(ids and (count is None or len(ids) == count), "calibration range unavailable; no silent shrinking")
    qmap = _quantizers(model)
    require(qmap and not model.training and all(not p.requires_grad for p in model.parameters()), "requires a frozen native quantized model")
    require(all(q.execution["activation_calibration"] == "static" for q in qmap.values()), "scale selection requires frozen static activation state")
    # Integer compound operators cache dependent scale state. Their discrete
    # replacement needs an explicit custom implementation, not stale bindings.
    require(not any(getattr(m, "integer_enabled", False) for m in model.modules()), "integer compound operators require explicit rebinding; unsupported here")
    changed = set().union(*(set(c["scales"]) for c in candidates))
    require(changed <= set(qmap), "candidate references an unknown quantizer")
    if block:
        model.get_submodule(block)
        backend.model.get_submodule(block)
        require(all(qmap[s].layer["module"] == block or qmap[s].layer["module"].startswith(block + ".") for s in changed),
                "block reconstruction only permits interventions within that block")
    baseline = {s: {"params": deepcopy(q.params()), "weight": getattr(q, "weight_record", None),
                    "decoded": getattr(q, "decoded_weight", None), "enabled": q.enabled,
                    "weight_enabled": getattr(q, "weight_enabled", None)} for s, q in qmap.items()}
    before = fingerprint(_digest({s: {"params": r["params"], "weight": r["weight"]} for s, r in baseline.items()}))
    selected, history, references, gradients = None, [], {}, {}
    handles, current = [], {}

    def restore():
        for sid, q in qmap.items():
            row = baseline[sid]
            q.set_params(row["params"])
            q.enabled = row["enabled"]
            if _weight_site(q):
                q.weight_record, q.decoded_weight, q.weight_enabled = row["weight"], row["decoded"], row["weight_enabled"]

    def install(candidate, fitting=True):
        restore()
        if fitting and context == "float":
            for sid, q in qmap.items():
                if sid not in changed:
                    q.enabled = False
                    if _weight_site(q):
                        q.weight_enabled = False
        apply_grid_changes(model, candidate["scales"], {})

    def selected_tensor(t, x):
        if block:
            t = boundary_tensor(t)
        require(torch.is_tensor(t) and t.numel() and bool(torch.isfinite(t).all()), "objective requires finite tensor output")
        if token_scope != "all":
            layout = ProfileAdapter().token_layout(backend, block, t.shape, x.shape)
            start, stop = layout[token_scope + "_slice"]
            require(stop > start, "requested token group is empty: " + token_scope)
            t = t.narrow(layout["token_axis"], start, stop - start)
        return t.detach().float().cpu().clone()

    def batches():
        for x, sid in islice(data.batches("calibration"), len(ids)):
            yield x, sid

    def measured(x):
        current.clear()
        out = forward(backend, model, x)
        return current["q"] if block else out

    if block:
        def q_hook(_m, args, kwargs, out):
            current["q"] = out
            current["input"] = args, kwargs
        handles.append(model.get_submodule(block).register_forward_hook(q_hook, with_kwargs=True))
        handles.append(backend.model.get_submodule(block).register_forward_hook(lambda _m, _a, out: current.update(fp=out)))
    try:
        # Warmup establishes actual shapes even after a package reload.
        x, _ = next(batches())
        measured(x)
        used = 0
        for x, sid in batches():
            base_output = measured(x)
            if reference == "local_fp":
                with torch.inference_mode():
                    args, kwargs = current["input"]
                    teacher = backend.model.get_submodule(block)(*args, **kwargs)
            else:
                teacher = forward(backend, backend.model, x)
                if block:
                    teacher = current["fp"]
            references[sid] = selected_tensor(teacher, x)
            used += references[sid].numel() * references[sid].element_size()
            if metric == "hessian_diag_ce":
                logits = selected_tensor(base_output, x).requires_grad_(True)
                require(logits.ndim == 2 and logits.shape[0] == 1 and sid in data.labels,
                        "CE proxy requires batch=1 classifier logits and actual calibration labels")
                gradient = torch.autograd.grad(F.cross_entropy(logits, torch.tensor([data.labels[sid]])), logits)[0]
                gradients[sid] = gradient.detach()
                used += gradient.numel() * gradient.element_size()
            require(used <= cache_mib * 1024 ** 2, "objective reference cache exceeds budget", code="memory_budget")
        require(list(references) == ids, "actual calibration sample order differs from declared view")
        for candidate in candidates:
            install(candidate)
            total, observed, per_sample_scores = 0., [], []
            for x, sid in batches():
                output, teacher = selected_tensor(measured(x), x), references[sid]
                require(output.shape == teacher.shape, "candidate/reference output shape differs")
                if metric in ("mse", "cosine", "hessian_diag_ce"):
                    value = score(teacher, output, "hessian" if metric == "hessian_diag_ce" else metric, gradients.get(sid))
                else:
                    require(output.ndim == 2 and output.shape[0] == 1 and output.shape[1] > 1, "task metrics require batch=1 classifier logits")
                    if metric == "teacher_kl":
                        value = float(F.kl_div(F.log_softmax(output.double(), -1), F.softmax(teacher.double(), -1), reduction="batchmean"))
                    else:
                        require(sid in data.labels, "task metric requires actual calibration label")
                        label = torch.tensor([data.labels[sid]])
                        value = float(F.cross_entropy(output.double(), label)) if metric == "task_loss" else float(output.argmax(-1).item() != label.item())
                require(math.isfinite(value), "nonfinite candidate objective")
                total += value
                observed.append(sid)
                per_sample_scores.append(value)
            require(observed == ids, "candidate did not consume identical calibration view")
            history.append({"id": candidate["id"], "scales": deepcopy(candidate["scales"]), "score": total / len(ids), "sample_count": len(ids),
                            "per_sample_scores": per_sample_scores})
            if selected is None or history[-1]["score"] < selected[0]:
                selected = history[-1]["score"], candidate
        install(selected[1], fitting=False)
        # Compare hashes over the same complete state as `before`. Hashing only
        # intervention sites made even an identity candidate look different.
        installed = {s: {"params": q.params(), "weight": getattr(q, "weight_record", None)}
                     for s, q in qmap.items()}
        values = {s: {"activation": parameter_evidence(qmap[s].params()),
                      "weight": parameter_evidence(qmap[s].weight_record["params"]) if getattr(qmap[s], "weight_record", None) else None}
                  for s in changed}
        return {"api": "frozen_scale_search/1", "purpose": "internal_calibration_selection", "metric": metric,
                "metric_definition": METRIC_DEFINITIONS[metric], "unit": "block" if block else "network", "block": block,
                "reference": reference, "context": context, "token_scope": token_scope,
                "split": "calibration", "sample_ids": ids, "sample_count": len(ids), "cache_bytes": used,
                "candidates": history, "selected_id": selected[1]["id"], "selected_score": selected[0],
                "selected_scales": deepcopy(selected[1]["scales"]), "selected_parameters": values,
                "baseline_active_targets": {s: {"activation": r["enabled"], "weight": r["weight_enabled"]} for s, r in baseline.items()},
                "baseline_configuration_sha256": fingerprint({s: {"layer": q.layer, "execution": q.execution} for s, q in qmap.items()}),
                "baseline_state_sha256": before, "selected_state_sha256": fingerprint(_digest(installed)),
                "parameters_installed": True, "final_evaluation": False,
                "selection_rule": "minimum score; first declared candidate wins exact ties"}
    except BaseException:
        restore()
        raise
    finally:
        for handle in handles:
            handle.remove()
        backend.reset()


def search_frozen_scales(backend, model, data, candidates, *, install_selected=False, **options):
    """Single-objective compatibility API. Explicitly opt in to legacy winner installation."""
    trial = deepcopy(model)
    report = _select_scales(backend, trial, data, candidates, **options)
    if install_selected:
        shapes = {s: q.last_input_shape for s, q in _quantizers(trial).items() if hasattr(q, "last_input_shape")}
        apply_grid_changes(model, report["selected_scales"], shapes)
    report["parameters_installed"] = install_selected
    return report
