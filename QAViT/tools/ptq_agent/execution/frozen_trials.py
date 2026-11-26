"""Paired observations on explicit frozen grid candidates. Dataset and package selection stay with the caller."""
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
import math
import time

import torch
from ..schemas import fingerprint, require
from ..profiling.adapters import ProfileAdapter, boundary_tensor
from .calibration_search import _quantizers, apply_grid_changes, state_identity
from .metrics import OutputErrors
from .research_metrics import SENSITIVITY
from .model import forward, tensor_digest


def image_views(backend, data, split, views, dense_task):
    require(split in ("calibration", "search"), "only calibration or a bound diagnostic search view is permitted")
    ids = list(data.ids(split))
    require(ids and len(set(ids)) == len(ids), "empty or repeated image identities")
    rows = iter(data.source_rows(split)) if hasattr(data, "source_rows") else None
    observed = []
    for image, sid in data.batches(split):
        row = next(rows) if rows is not None else {}
        require(row.get("sample_id", sid) == sid, "source row differs from decoded image")
        source = {k: row[k] for k in ("path", "file_sha256", "dataset_split") if k in row}
        if "input_sha256" in row:
            source["manifest_classification_input_sha256"] = row["input_sha256"]
        if "path" in source and "file_sha256" in source:
            from PIL import Image
            with Image.open(source["path"]) as original:
                source["original_hw"] = [original.height, original.width]
        observed.append(sid)
        if "classification" in views:
            yield image, {}, {"sample_id": sid, "source": source, "view": "classification",
                              "index": 0, "shape": list(image.shape), "input_sha256": tensor_digest(image)}
        if "dense" in views:
            require("file_sha256" in source and "path" in source, "dense geometry requires a verified original image")
            for i, (window, meta, geometry) in enumerate(dense_task.windows(row)):
                yield window, meta, {"sample_id": sid, "source": source, "view": "dense", "index": i,
                    "shape": list(window.shape), "input_sha256": tensor_digest(window), **geometry}
    require(observed == ids and (rows is None or next(rows, None) is None), "actual input order differs from bound view")


def _observe(backend, model, x, meta, specs, dense, *, conditional=None, geometry=None):
    blocks = {s["block"] for s in specs if s["output"] == "block"}
    captured = {}
    def hook(name):
        def record(_m, args, kwargs, out):
            require(name not in captured, "block repeated within a single task window: " + name)
            if conditional is not None:
                out = conditional.get_submodule(name)(*args, **kwargs)
            captured[name] = boundary_tensor(out)
        return record
    with ExitStack() as stack:
        for name in blocks:
            stack.callback(model.get_submodule(name).register_forward_hook(hook(name), with_kwargs=True).remove)
        if specs[0]["view"] == "dense":
            outputs = dense.observe(model, x, meta, logits=any(s["output"] == "segmentation_logits" for s in specs))
        else:
            outputs = {"logits": forward(backend, model, x)}
    require(set(captured) == blocks, "task path did not execute all selected blocks")
    result = {}
    for s in specs:
        key, kind = s["id"], s["output"]
        if kind == "block":
            t = captured[s["block"]]
            scope = s.get("tokens", "all")
            layout = None
            if scope != "all" or (s["block"] in getattr(backend, "spec", {}).get("block_paths", ())
                                  and hasattr(backend.model, "blocks") and hasattr(backend.model, "patch_embed")):
                layout = ProfileAdapter().token_layout(backend, s["block"], t.shape, x.shape)
            if geometry is not None:
                geometry[key] = {"full_shape": list(t.shape), "scope": scope, "layout": layout}
            if scope != "all":
                lo, hi = layout[scope + "_slice"]
                require(hi > lo, "empty requested token scope")
                t = t.narrow(layout["token_axis"], lo, hi-lo)
            result[key] = {s["block"]: t}
        elif kind == "head_features":
            indices = s.get("layers", dense.indices)
            require(indices and len(set(indices)) == len(indices) and set(indices) <= set(dense.indices),
                    "layers must be actual bound segmentation feature indices")
            result[key] = {str(i): outputs[str(i)] for i in indices}
        else:
            logits = outputs["logits"]
            require(torch.is_tensor(logits) and logits.ndim == (2 if kind == "classification_logits" else 4)
                    and logits.shape[0] == 1 and logits.shape[1] > 1, "expected task logits from the bound head")
            result[key] = {"logits": logits}
        if geometry is not None:
            geometry.setdefault(key, {})["observed_shapes"] = {k: list(v.shape) for k, v in result[key].items()}
    return result


def observations(backend, model, x, meta, specs, dense_task, *, parent=None, candidate=None, geometry=None):
    """Conditional FP is the original block on the fixed parent's input for this image/window."""
    result = {}
    for mode in ("fp_model", "local_fp", "local_fp_parent", "local_fp_candidate"):
        group = [s for s in specs if s.get("reference", "fp_model") == mode]
        if group:
            driver = candidate if mode == "local_fp_candidate" else parent if mode in ("local_fp","local_fp_parent") else None
            result.update(_observe(backend, driver if driver is not None else model, x, meta, group, dense_task,
                                   conditional=model if driver is not None else None, geometry=geometry))
    return result


def _metrics(reference, candidate, names, estimators=None, sensitivities=None, label=None):
    err = OutputErrors()
    require(reference.shape == candidate.shape, "candidate observation shape differs from its reference")
    if reference.numel() > (1 << 20):
        # Bound FP64 temporaries; sums/dot/energies still describe the whole output.
        for a, b in zip(reference.reshape(-1).split(1 << 20), candidate.reshape(-1).split(1 << 20)):
            err.add(a, b)
    else:
        err.add(reference, candidate)
    stats = err.result()
    values = {"mse": stats["mse"], "nmse": stats["relative_mse"], "cosine": stats["cosine_error"]}
    if "teacher_kl" in names:
        # Keep temporary double tensors bounded for a dense C x H x W logit map.
        a, b = (v.flatten(2) if v.ndim == 4 else v.unsqueeze(-1) for v in (reference, candidate))
        total = 0.
        for fp, q in zip(a.split(2048, -1), b.split(2048, -1)):
            total += torch.nn.functional.kl_div(q.double().log_softmax(1), fp.double().softmax(1),
                                                reduction="sum").item()
        values["teacher_kl"] = total / (a.shape[0] * a.shape[-1])
    if set(names)&{"task_nll","top1_error"}:
        require(label is not None and candidate.ndim==2 and len(candidate)==1,"missing true classification target")
        values["task_nll"]=float(torch.nn.functional.cross_entropy(candidate.double(),torch.tensor([label],device=candidate.device)))
        values["top1_error"]=float(candidate.argmax(1).item()!=label)
    from . import research_metrics
    for name in set(names) - {"mse","nmse","cosine"}:
        opt = (estimators or {}).get(name, {})
        if name not in values or opt:
            state,binding=(sensitivities or {}).get(name,(None,None))
            values[name] = float(research_metrics.score(name,reference,candidate,options=opt,prepared=state,binding=binding))
    require(all(values[n] is not None and math.isfinite(values[n]) for n in names),
            "undefined or non-finite objective on the requested observation")
    return {n: values[n] for n in names}


def _weighted(values, weights):
    require(weights and weights.keys() <= values.keys() and
            all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in weights.values())
            and 0 < sum(weights.values()) < math.inf, "weights must name measured components and have a positive finite sum")
    return math.fsum(values[k] * w for k, w in weights.items()) / math.fsum(weights.values())


def _validate(specs):
    require(specs and len({s["id"] for s in specs}) == len(specs), "observation ids must be unique")
    for s in specs:
        kind, view, metrics = s["output"], s["view"], s["metrics"]
        require(isinstance(s["id"], str) and s["id"] and set(s) <= {
            "id", "view", "output", "block", "tokens", "reference", "metrics", "layers", "layer_weights","estimators"},
            "invalid observation")
        require(view in ("classification", "dense") and kind in
                ("block", "head_features", "classification_logits", "segmentation_logits"), "unknown view or output")
        require(kind == "block" or (view == "classification") == (kind == "classification_logits"), "output/view mismatch")
        require(metrics and len(set(metrics)) == len(metrics) and (not ({'top1_error','task_nll'} & set(metrics)) or kind=='classification_logits'), 'task classification metric requires native logits')
        require(set(metrics) <= SENSITIVITY | {"top1_error","task_nll","mse", "nmse", "cosine", "teacher_kl","pearson_error","sqnr_power_db","nuclear_norm_proxy",
                    "attention_pairwise_hinge","attention_rank_agreement","attention_topk_overlap"}
                and ("teacher_kl" not in metrics or kind.endswith("_logits")), "unsupported metric for this observation")
        require(s.get("reference", "fp_model") in (("fp_model", "local_fp", "local_fp_parent","local_fp_candidate") if kind == "block" else ("fp_model",)),
                "conditional FP requires a block")
        require((kind == "block") == ("block" in s) and s.get("tokens", "all") in ("all", "cls", "special", "patch")
                and (kind == "block" or "tokens" not in s), "invalid block/token selection")
        require(kind == "head_features" or not {"layers", "layer_weights"}.intersection(s), "layer weights require head features")


def candidate_receipt(parent_sha, source_ref, candidate, changes, shapes=None):
    body = {"parent_state_sha256": parent_sha, "parent_model_ref": source_ref,
            "id": candidate["id"], "scales": deepcopy(candidate["scales"]), "changes": changes}
    if shapes is not None:
        body["validation_input_shapes"] = {s: list(shapes[s]) for s,t in changes.items() if "activation" in t}
    return {**body, "candidate_sha256": fingerprint(body)}


def accepted_copy(model, candidate, *, source_ref, input_shapes):
    """Install only on a new instance. A stale or altered receipt cannot mutate the parent."""
    parent = state_identity(model)
    require(candidate["parent_state_sha256"] == parent and candidate["parent_model_ref"] == source_ref,
            "candidate belongs to another frozen parent")
    child = deepcopy(model)
    recorded = candidate.get("validation_input_shapes")
    shapes = dict(input_shapes or {})
    for site, shape in (recorded or {}).items():
        require(site not in shapes or list(shapes[site]) == shape, "candidate input shape differs from bound parent")
        shapes[site] = shape
    changes = apply_grid_changes(child, candidate["scales"], shapes)
    receipt = candidate_receipt(parent, source_ref, candidate, changes, recorded)
    require(candidate["candidate_sha256"] == receipt["candidate_sha256"] and changes == candidate["changes"],
            "accepted grid state differs from scored candidate")
    return child, {**receipt, "child_state_sha256": state_identity(child)}


def score_frozen_candidates(backend, model, data, candidates, *, observations_spec, objectives=None,
                            split="calibration", source_ref=None, input_shapes=None, dense_task=None, load_candidate=None):
    """Re-evaluate the explicit pool on identical inputs. Return scores and reproducible edit receipts."""
    _validate(observations_spec)
    require(candidates and all(isinstance(c["id"], str) and c["id"] and (set(c) == {"id","scales"} and isinstance(c["scales"],dict)
                     or set(c) == {"id","source_ref"} and isinstance(c["source_ref"],str) and load_candidate) for c in candidates)
            and len({c["id"] for c in candidates}) == len(candidates), "provide unique explicit candidates")
    require(split in ("calibration", "search") and (split == "calibration" or not objectives),
            "search only confirms declared candidates; calibration selects parameters")
    require(model is not backend.model and all(not m.training for m in model.modules())
            and all(not p.requires_grad for p in model.parameters()) and _quantizers(model), "requires frozen quantizers")
    groups = {v: [s for s in observations_spec if s["view"] == v] for v in {s["view"] for s in observations_spec}}
    if "dense" in groups and dense_task is None:
        from tools.segmentation import DenseTask
        dense_task = DenseTask(backend)
    from .task_sensitivity import prepare_pool
    def wall():
        if torch.device(backend.device).type=="cuda":torch.cuda.synchronize(backend.device)
        return time.perf_counter()
    begin=wall()
    prepared, receipts = prepare_pool(backend,data,observations_spec,split,dense_task,load_candidate)
    return _score(backend,model,data,candidates,observations_spec,objectives,split,source_ref,input_shapes,dense_task,
                  load_candidate,groups,prepared,receipts,wall,begin)


@torch.inference_mode()
def _score(backend,model,data,candidates,observations_spec,objectives,split,source_ref,input_shapes,dense_task,
           load_candidate,groups,prepared,receipts,wall,begin):
    parent = state_identity(model)
    prepared_at=wall()
    rows, expected_views = [], None
    # Use a private copy for counters, shape-dependent caches and arbitrary approved
    # parent modules as well as grid state. Only one arm and one window live at once.
    try:
        for candidate in candidates:
            if "source_ref" in candidate:
                trial, origin = load_candidate(candidate["source_ref"])
                require(trial is not model and all(not m.training for m in trial.modules())
                        and all(not p.requires_grad for p in trial.parameters()), "complete candidate must be separately cold-loaded and frozen")
            else:
                trial = deepcopy(model)
            shapes = dict(input_shapes or {})
            if any("activation" in targets and site not in shapes for site, targets in candidate.get("scales",{}).items()):
                # Bare Python callers may have a newly loaded model without observed shapes.
                x, _ = next(data.batches(split))
                forward(backend, trial, x)
                shapes.update({s:q.last_input_shape for s,q in _quantizers(trial).items()
                               if s not in shapes and hasattr(q, "last_input_shape")})
            if "source_ref" in candidate:
                receipt = {"id":candidate["id"], "source_ref":candidate["source_ref"],"model_ref":origin["model_ref"],
                           "state_sha256":state_identity(trial),"origin":"complete_frozen_package"}
            else:
                changes = apply_grid_changes(trial, candidate["scales"], shapes or {})
                receipt = candidate_receipt(parent, source_ref, candidate, changes, shapes)
            references = deepcopy(model) if any(s.get("reference") in ("local_fp","local_fp_parent") for s in observations_spec) else None
            measured, samples, layers = [], {}, {}
            for x, meta, view in image_views(backend, data, split, groups, dense_task):
                specs = groups[view["view"]]
                view["output_geometry"] = {}
                teacher = observations(backend, backend.model, x, meta, specs, dense_task,
                                       parent=references, candidate=trial, geometry=view["output_geometry"])
                outputs = observations(backend, trial, x, meta, specs, dense_task)
                per_image = samples.setdefault(view["sample_id"], defaultdict(list))
                for s in specs:
                    key = s["id"]
                    estimates={n:prepared[(view["input_sha256"],key,n)] for n in set(s["metrics"]) & SENSITIVITY}
                    by_layer = {k: _metrics(t, outputs[key][k], s["metrics"], s.get("estimators"),estimates,data.labels.get(view["sample_id"]) if set(s["metrics"])&{"task_nll","top1_error"} else None) for k,t in teacher[key].items()}
                    weights = s.get("layer_weights", dict.fromkeys(by_layer, 1))
                    require(set(weights) == set(by_layer), "layer_weights must cover exactly the observed layers")
                    for metric in s["metrics"]:
                        values = {k: stats[metric] for k,stats in by_layer.items()}
                        per_image[key + "." + metric].append(_weighted(values, weights))
                        for layer, val in values.items():
                            layers.setdefault(view["sample_id"], {}).setdefault(key, {}).setdefault(layer, {}).setdefault(metric, []).append(val)
                measured.append(view)
                del teacher, outputs
            require(expected_views is None or measured == expected_views, "candidate pool saw different inputs/windows")
            expected_views = measured
            per_image = {sid: {k: math.fsum(v)/len(v) for k,v in row.items()} for sid,row in samples.items()}
            scores = {k: math.fsum(row[k] for row in per_image.values())/len(per_image) for k in next(iter(per_image.values()))}
            rows.append({"id": candidate["id"], "receipt": receipt, "scores": scores, "per_image": per_image,
                         "per_window_layers": layers,
                         "objectives": {k: _weighted(scores,w) for k,w in (objectives or {}).items()}})
            del trial, references
    finally:
        backend.reset()
        require(state_identity(model) == parent, "scoring changed parent state")
    return {"measurement_cost":{"prepare_and_parent_binding_seconds":prepared_at-begin,"frozen_score_seconds":wall()-prepared_at,"candidates":len(rows),"original_images":len(data.ids(split)),"view_units_per_candidate":len(expected_views),"device":str(backend.device),"clock":"synchronized observed wall time"},
            "api": "frozen_candidate_scores/1", "parent_state_sha256": parent, "parent_model_ref": source_ref,
            "sensitivity_preparations":receipts, "parameters_installed": False, "split": split, "sample_ids": data.ids(split), "views": expected_views,
            "view_transforms": {**({"classification": data.identity.get("preprocessing")} if "classification" in groups else {}),
                                **({"dense": dense_task.identity} if "dense" in groups else {})},
            "observations": observations_spec, "objectives": objectives or {}, "candidates": rows,
            "winners": {k: min(rows, key=lambda r: (r["objectives"][k], r["id"]))["id"] for k in (objectives or {})},
            "reduction": "per output/window: MSE=mean squared, NMSE=SSE/FP energy, cosine=1-cos, KL=teacher to candidate per class vector; declared layer weights, equal windows per original then equal images; objective weighted mean, ties by id",
            "background": "complete frozen quantized prefix and suffix; local_fp uses FP block on the identical parent input",
            "training": False, "final_evaluation": False}
