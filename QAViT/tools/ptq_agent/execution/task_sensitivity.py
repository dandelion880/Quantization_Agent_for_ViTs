"""Read-only suffix differentiation on an explicitly bound classification image or dense window."""

from contextlib import contextmanager
import torch
from ..schemas import require
from . import research_metrics as metrics
from .calibration import differentiable_forward
from .model import tensor_digest


@contextmanager
def substitute(module, callback):
    handle = module.register_forward_hook(lambda m, args, y: callback(y))
    try:
        yield
    finally:
        handle.remove()


def prepare_window(
    backend,
    model,
    x,
    meta,
    spec,
    name,
    options,
    binding,
    *,
    dense=None,
    label=None,
    perturbed_models=(),
    defer_factorization=False,
):
    require(
        spec["output"] == "block"
        and spec.get("tokens", "all") == "all"
        and spec.get("reference", "fp_model") == "fp_model",
        "sensitivity tap requires full block FP coordinates",
    )
    block = spec["block"]

    def run(net):
        return (
            dense.observe(net, x, meta, logits=True)["logits"]
            if spec["view"] == "dense"
            else differentiable_forward(backend, net, x)
        )

    observed = []

    def capture(y):
        require(torch.is_tensor(y), "sensitivity tap needs tensor output")
        observed.append(y.detach())

    with torch.inference_mode(False), torch.no_grad():
        x = x.detach().clone()
        with substitute(model.get_submodule(block), capture):
            teacher = run(model).detach().clone()
        require(len(observed) == 1, "tap was repeated or absent from bound task")
        z = observed.pop().clone()
        differences = []
        for net in perturbed_models:
            with substitute(net.get_submodule(block), capture):
                run(net)
            require(len(observed) == 1, "unpaired perturbation block")
            differences.append(observed.pop().to(z) - z)
    target = options["target"]
    require(
        target in ("labels", "teacher_kl")
        and (target != "labels" or label is not None),
        "missing bound task supervision",
    )
    if target == "labels":
        label = torch.as_tensor(label, device=z.device, dtype=torch.long).reshape(
            teacher.shape[:1] + teacher.shape[2:]
        )

    def loss_at(value):
        with substitute(model.get_submodule(block), lambda _: value):
            logits = run(model)
        require(
            logits.requires_grad,
            "task adapter detached the suffix; a differentiable head binding is required",
        )
        return metrics.task_loss(
            logits,
            labels=label if target == "labels" else None,
            teacher=teacher if target == "teacher_kl" else None,
            temperature=options.get("temperature", 1.0),
            ignore_index=options.get("ignore_index"),
        )

    state = metrics.prepare(
        name,
        z,
        loss_at,
        binding=binding,
        options=options,
        perturbations=torch.stack(differences) if differences else None,
        initial_error=differences[0] if differences else None,
        defer_factorization=defer_factorization,
    )
    return state, {
        "key": state["key"],
        "task": spec["view"],
        "raw_head_sha256": tensor_digest(teacher),
        "target_sha256": tensor_digest(label)
        if target == "labels"
        else tensor_digest(teacher),
        "rank": state.get("effective_rank"),
        "shape": list(z.shape),
        "reference": binding["reference"],
        "grad_norm": float(state["weight"].norm())
        if "weight" in state
        else float(state["G"].norm()),
        "actual_abs_perturbation": float(
            ((z + options["delta"]) - (z - options["delta"])).abs().mean()
        )
        if "delta" in options
        else None,
    }


def prepare_pool(backend, data, specs, split, dense, load_candidate):
    from .frozen_trials import image_views
    from .calibration_search import state_identity

    selected = [s for s in specs if set(s["metrics"]) & metrics.SENSITIVITY]
    states, receipts = {}, []
    pooled = {}
    if not selected:
        return states, receipts
    require(
        split == "calibration",
        "backward sensitivity preparation requires a calibration input view",
    )
    original = state_identity(backend.model)
    for x, meta, view in image_views(
        backend, data, split, {s["view"] for s in selected}, dense
    ):
        for s in selected:
            if s["view"] != view["view"]:
                continue
            for name in set(s["metrics"]) & metrics.SENSITIVITY:
                opts = s["estimators"][name]
                label = (
                    data.labels.get(view["sample_id"])
                    if s["view"] == "classification"
                    else meta.get("_qavit_labels")
                )
                binding = {
                    "reference": "fp_model",
                    "input_state": view["input_sha256"],
                    "suffix_state": original,
                    "head": dense.identity
                    if s["view"] == "dense"
                    else getattr(backend, "task_head", None) or original,
                    "task": s["view"],
                    "view": view,
                    "sample_ids": [view["sample_id"]],
                    "observation": s["block"],
                    "axes": "all block output nonbatch coordinates",
                    "reduction": "independent window, equal image",
                    "label_sha256": None
                    if label is None or opts["target"] != "labels"
                    else tensor_digest(torch.as_tensor(label)),
                    "perturbation_states": [],
                }
                members = []
                for source in opts.get("perturbation_sources", []):
                    m, origin = load_candidate(source)
                    binding["perturbation_states"].append(
                        [source, origin["model_ref"], state_identity(m)]
                    )
                    members.append(m)
                head_meta = {k: v for k, v in meta.items() if k != "_qavit_labels"}
                state, receipt = prepare_window(
                    backend,
                    backend.model,
                    x,
                    head_meta,
                    s,
                    name,
                    opts,
                    binding,
                    dense=dense,
                    label=label,
                    perturbed_models=members,
                    defer_factorization=True,
                )
                states[(view["input_sha256"], s["id"], name)] = (state, binding)
                if name != "apq_masked_gradient_error":
                    key = (s["id"], name, s["view"], tuple(receipt["shape"][1:]))
                    pooled.setdefault(key, []).append((state, binding, receipt))
                receipts.append(receipt)
    for (_, name, _, _), items in pooled.items():
        if name.startswith("aph_"):
            shared = {
                "weight": torch.stack([state["weight"] for state, _, _ in items]).mean(
                    0
                )
            }
        else:
            shared = {
                k: torch.stack([state[k] for state, _, _ in items]).mean(0)
                for k in ("D", "G")
            }
            shared.update(
                metric=name,
                options=items[0][0]["options"],
                initial_error=torch.cat(
                    [state["initial_error"] for state, _, _ in items]
                ),
            )
            metrics.finalize_fima(shared)
        ids = [b["view"] for _, b, _ in items]
        for state, binding, receipt in items:
            state.pop("initial_error", None)
            state.update(shared)
            binding["preparation_pool"] = ids
            state["key"] = metrics.cache_key(name, binding, state["options"])
            receipt.update(
                key=state["key"],
                calibration_windows=len(items),
                rank=state.get("effective_rank"),
                grad_norm=float(state["weight"].norm())
                if "weight" in state
                else float(state["G"].norm()),
            )
    require(
        original == state_identity(backend.model),
        "read-only prepare modified the teacher",
    )
    return states, receipts
