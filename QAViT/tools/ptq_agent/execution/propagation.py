"""Paired, frozen-model propagation measurements for approved custom experiments.

No fitting, ranking policy, or automatic site selection is performed here. Boundary
replay requires tensor-to-tensor blocks with aligned FP/Q residual coordinates.
"""
from contextlib import contextmanager

import torch

from ..schemas import require
from ..profiling.adapters import ProfileAdapter, boundary_tensor
from .evaluation import model_digest
from .metrics import OutputErrors, tensors
from .model import forward


def _groups(value, token_layout):
    result = {"all": value}
    if token_layout is not None and value.ndim == 3:
        axis = token_layout["axis"]
        cls = token_layout.get("cls_index")
        start = token_layout["patch_start"]
        require(axis in (1, 2) and 0 <= start < value.shape[axis], "invalid token layout")
        if cls is not None:
            require(0 <= cls < start, "CLS must precede the declared patch range")
            result["cls"] = value.narrow(axis, cls, 1)
        special_start = 0 if cls is None else cls + 1
        if start > special_start:
            result["special"] = value.narrow(axis, special_start, start - special_start)
        result["patch"] = value.narrow(axis, start, value.shape[axis] - start)
    return result


def _add(errors, key, reference, candidate, token_layout):
    errors.setdefault(key, OutputErrors()).add(_groups(reference, token_layout),
                                             _groups(candidate, token_layout))


def _capture(model, inputs, names, run):
    captured, handles = {}, []

    def hook(name):
        def save(module, args, kwargs, output):
            require(name not in captured, "boundary executed more than once: " + name)
            from torch.utils._pytree import tree_map
            require(bool(args), "block input is missing: " + name)
            call = tree_map(lambda t: t.detach().clone() if torch.is_tensor(t) else t, (args, kwargs))
            captured[name] = (boundary_tensor(call[0][0]), boundary_tensor(output).detach().clone(), call)
        return save

    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(hook(name), with_kwargs=True))
        output = run(model, inputs)
        require(set(captured) == set(names), "forward did not visit all declared boundaries")
        return output, captured
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _intervene(reference, candidate, item):
    """Temporary frozen FP-block substitution or a declared quantizer flag bypass."""
    handle, module, field, old = None, None, None, None
    try:
        if item["kind"] == "fp_block":
            name = item["block"]
            fp = reference.get_submodule(name)
            handle = candidate.get_submodule(name).register_forward_hook(
                lambda _m, args, kwargs, _out: fp(*args, **kwargs), with_kwargs=True)
        elif item["kind"] == "bypass":
            matches = [m for m in candidate.modules() if getattr(m, "ptq_site_id", None) == item["site"]]
            require(len(matches) == 1, "bypass requires one exact frozen quantizer site")
            module = matches[0]
            require(item["target"] in ("weight", "activation"), "invalid bypass target")
            field = "weight_enabled" if item["target"] == "weight" else "enabled"
            require(getattr(module, "layer", {}).get(item["target"]) is not None
                    and getattr(module, field, None) is True, "target is not actively quantized")
            old = getattr(module, field)
            setattr(module, field, False)
        else:
            raise ValueError("unknown frozen intervention kind: " + str(item["kind"]))
        yield
    finally:
        if handle is not None:
            handle.remove()
        if module is not None and old is not None:
            setattr(module, field, old)


def profile_frozen_propagation(backend, model, data, *, blocks, split="calibration",
                               interventions=(), token_layout=None, classification_output=None,
                               forward_fn=None, source=None):
    """Measure selected block boundaries and optional interventions on frozen state.

    `model` must be obtained via the host's context['load_frozen'] for historical
    research. `blocks` are explicit module paths, not inferred quantizer names.
    `data` is the host restricted view; no final data are accepted. `forward_fn`
    optionally runs an explicitly frozen task head, returning tensors/dictionaries.
    classification_output names a rank-2 logits tensor (usually 'output'); omission
    avoids treating arbitrary embeddings as logits. Known block layouts come from the model adapter.
    Other boundaries use an explicit layout or only the complete tensor.
    """
    require(split in ("calibration", "search"), "propagation diagnostics cannot consume final data")
    require(blocks and len(set(blocks)) == len(blocks), "provide unique block boundaries")
    stochastic = (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm)
    require(not backend.model.training and not model.training
            and all(not m.training for root in (backend.model, model)
                    for m in root.modules() if isinstance(m, stochastic)),
            "frozen diagnostics require eval mode for models and stochastic modules")
    require(model is not backend.model, "candidate must be distinct from the FP reference")
    for name in blocks:
        backend.model.get_submodule(name)
        model.get_submodule(name)
    names = [item["id"] for item in interventions]
    require(len(set(names)) == len(names) and "baseline" not in names, "interventions need unique ids")
    run = forward_fn or (lambda m, x: forward(backend, m, x))
    before = (model_digest(backend.model), model_digest(model))
    flag_state = [(m, key, getattr(m, key)) for root in (backend.model, model)
                  for m in root.modules() for key in ("enabled", "weight_enabled", "gradient_probe")
                  if hasattr(m, key)]
    rows, cross, endpoint = {}, {}, {}
    layouts = {}
    arm_boundaries, task, seen = {}, {}, []

    def output_error(arm, fp, q):
        endpoint.setdefault(arm, OutputErrors()).add(fp, q)

    def classification(arm, fp, q, sid):
        if classification_output is None:
            return
        require(sid in data.labels, "classification diagnostic requires actual sample labels")
        a, b = tensors(fp)[classification_output], tensors(q)[classification_output]
        require(a.ndim == 2 and a.shape[0] == 1, "classification currently requires one labeled image per batch")
        totals = task.setdefault(arm, {"correct": 0, "reference_correct": 0, "teacher_kl_sum": 0.0,
                                      "task_loss_sum": 0.0, "samples": 0})
        label = int(data.labels[sid])
        totals["correct"] += int(b.argmax(-1).item() == label)
        totals["reference_correct"] += int(a.argmax(-1).item() == label)
        totals["teacher_kl_sum"] += torch.nn.functional.kl_div(
            b.double().log_softmax(-1), a.double().softmax(-1), reduction="batchmean").item()
        totals["task_loss_sum"] += torch.nn.functional.cross_entropy(
            b.double(), torch.tensor([label], device=b.device)).item()
        totals["samples"] += 1

    try:
        with torch.inference_mode():
            for inputs, sid in data.batches(split):
                require(sid not in seen, "duplicate diagnostic sample id")
                seen.append(sid)
                fp_out, fp = _capture(backend.model, inputs, blocks, run)
                q_out, q = _capture(model, inputs, blocks, run)
                output_error("baseline", fp_out, q_out)
                classification("baseline", fp_out, q_out, sid)
                for name in blocks:
                    fx, fy, _ = fp[name]
                    qx, qy, (args, kwargs) = q[name]
                    layout = token_layout
                    if name in getattr(backend, "spec", {}).get("block_paths", ()):
                        actual = ProfileAdapter().token_layout(backend, name, qx.shape, inputs.shape)
                        expected = {"axis": actual["token_axis"],
                                    "cls_index": 0 if actual["cls_slice"][1] else None,
                                    "patch_start": actual["patch_slice"][0]}
                        require(layout is None or layout == expected, "declared token layout conflicts with bound model")
                        layout = expected
                    layouts[name] = layout
                    # F on the exact Q input isolates the block from its prefix.
                    conditional = backend.model.get_submodule(name)(*args, **kwargs)
                    conditional = boundary_tensor(conditional)
                    require(conditional.shape == fy.shape == qy.shape,
                            "FP/Q boundaries are not in aligned coordinates: " + name)
                    errors = rows.setdefault(name, {})
                    for key, a, b in (("input_total", fx, qx), ("output_total", fy, qy),
                                      ("local_added", conditional, qy), ("upstream_mapped", fy, conditional)):
                        _add(errors, key, a, b, layout)
                    # total SSE = local SSE + mapped upstream SSE + signed cross term.
                    local, upstream = qy.double() - conditional.double(), conditional.double() - fy.double()
                    for group, value in _groups(local, layout).items():
                        row = cross.setdefault(name, {}).setdefault(group, {"twice_dot": 0.0, "elements": 0})
                        row["twice_dot"] += 2 * (value * _groups(upstream, layout)[group]).sum().item()
                        row["elements"] += value.numel()
                for item in interventions:
                    arm = item["id"]
                    with _intervene(backend.model, model, item):
                        out, captured = _capture(model, inputs, blocks, run)
                    output_error(arm, fp_out, out)
                    classification(arm, fp_out, out, sid)
                    for name in blocks:
                        errors = arm_boundaries.setdefault(arm, {}).setdefault(name, {})
                        _add(errors, "error_to_fp", fp[name][1], captured[name][1], layouts[name])
                        _add(errors, "change_from_frozen", q[name][1], captured[name][1], layouts[name])
        require(seen == list(data.ids(split)) and seen, "diagnostic data view was not fully consumed")
    finally:
        require(all(getattr(m, key) == value for m, key, value in flag_state),
                "diagnostic quantizer flags changed", code="protection_failed")
        require(before == (model_digest(backend.model), model_digest(model)),
                "diagnostic numerical state changed", code="protection_failed")
        backend.reset()

    boundaries = {name: {key: error.result() for key, error in errors.items()}
                  for name, errors in rows.items()}
    arms = {}
    base = endpoint["baseline"].result()
    for item in interventions:
        arm = item["id"]
        result = endpoint[arm].result()
        # Positive = improvement. Keep every named task representation separately.
        delta = {key: base["outputs"][key]["mse"] - row["mse"]
                 for key, row in result["outputs"].items() if row["elements"]}
        arms[arm] = {"intervention": dict(item), "output_error": result,
                     "mse_reduction_to_fp": delta,
                     "boundaries": {name: {key: err.result() for key, err in errors.items()}
                                    for name, errors in arm_boundaries[arm].items()}}
    classification_results = {arm: {**row, "top1": 100 * row["correct"] / row["samples"],
                                    "reference_top1": 100 * row["reference_correct"] / row["samples"],
                                    "teacher_kl": row["teacher_kl_sum"] / row["samples"],
                                    "task_loss": row["task_loss_sum"] / row["samples"]}
                              for arm, row in task.items()}
    return {"schema_version": 1, "source": source, "split": split, "sample_ids": seen,
            "sample_count": len(seen), "token_layout": token_layout, "block_token_layouts": layouts, "boundaries": boundaries,
            "decomposition_cross_term": cross, "baseline_output_error": base, "interventions": arms,
            "classification_diagnostic": classification_results,
            "numeric_state_sha256": {"reference": before[0], "candidate": before[1]},
            "state_unchanged": True, "no_recalibration": True,
            "semantics": {"local_added": "Q_block(x_Q) - F_block(x_Q)",
                          "upstream_mapped": "F_block(x_Q) - F_block(x_FP)",
                          "output_total": "Q_block(x_Q) - F_block(x_FP)",
                          "cross_term": "signed 2*dot(local,upstream); SSE components are not additive without it",
                          "fp_block": "frozen FP substitution at current Q input; diagnostic headroom, not achievable W4A4 gain",
                          "scope": "selected aligned tensor block boundaries; task outputs only from the supplied forward; no causal ranking or formal dual-task evaluation"}}
