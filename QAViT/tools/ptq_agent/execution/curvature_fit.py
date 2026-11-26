"""Prepare/reuse actual FP-suffix derivatives for bounded block optimization and MLP replacement."""

from contextlib import ExitStack
from copy import deepcopy
import torch
from ..schemas import require
from .calibration import differentiable_forward
from .model import tensor_digest
from .research_metrics import prepare, score, task_loss, SENSITIVITY
from .task_sensitivity import substitute


class BlockObjective:
    def __init__(self, backend, teacher, images, labels, name, options, cache_mib):
        self.backend, self.teacher, self.name = backend, teacher, name
        self.options = dict(options)
        self.method = options["kind"]
        require(self.method in SENSITIVITY, "unknown derivative estimator")
        self.module = teacher.get_submodule(name)
        self.images = list(images.items())
        self.labels = labels
        ids, targets, outputs = [], [], []
        with torch.no_grad():
            for sid, x in self.images:
                ids.append(sid)
                with substitute(self.module, lambda y: outputs.append(y.detach())):
                    targets.append(differentiable_forward(backend, teacher, x).detach())
        require(
            len(outputs) == len(ids)
            and all(y.shape == outputs[0].shape for y in outputs),
            "block output geometry changed",
        )
        self.z = torch.cat(outputs)
        self.targets = targets
        self.history = []
        self.refreshes = []
        self.limit = cache_mib * (1 << 20)
        self.binding = {
            "reference": "fp_model",
            "input_state": [tensor_digest(x) for _, x in self.images],
            "suffix_state": backend.identity(),
            "head": [tensor_digest(t) for t in targets],
            "task": "classification",
            "view": "calibration",
            "sample_ids": ids,
            "observation": name,
            "axes": "all retained block coordinates",
            "reduction": "equal image / calibration mean",
        }
        self.state = None

    def loss_at(self, z):
        losses = []
        for j, (sid, x) in enumerate(self.images):
            with substitute(self.module, lambda _y: value):
                value = z[j : j + 1]
                pred = differentiable_forward(self.backend, self.teacher, x)
            if self.options.get("target", "teacher_kl") == "teacher_kl":
                loss = task_loss(
                    pred,
                    teacher=self.targets[j],
                    temperature=self.options["temperature"],
                )
            elif self.options.get("target") == "teacher_class":
                loss = task_loss(pred, labels=self.targets[j].argmax(-1))
            else:
                require(
                    self.options.get("target") == "labels" and sid in self.labels,
                    "missing permitted classification label for " + sid,
                )
                loss = task_loss(
                    pred, labels=torch.as_tensor([self.labels[sid]], device=z.device)
                )
            losses.append(loss)
        return torch.stack(losses).sum()

    def refresh(self, model, step):
        is_fima = self.method.startswith(("fim_", "fima_"))
        D = initial = None
        if is_fima:
            if step and (
                step % self.options["refresh_interval"]
                or len(self.history) >= self.options["rank"]
            ):
                return
            outputs = []
            with torch.no_grad():
                for _, x in self.images:
                    with substitute(
                        model.get_submodule(self.name),
                        lambda y: outputs.append(y.detach()),
                    ):
                        differentiable_forward(self.backend, model, x)
            delta = torch.cat(outputs) - self.z
            self.history.append(delta)
            require(
                delta.numel() * delta.element_size() * (2 + len(self.history))
                <= self.limit,
                "progressive curvature cache exceeds bound",
            )
            D = torch.stack(self.history)
            initial = delta
            opt = {**self.options, "rank": len(self.history)}
        else:
            if self.state is not None:
                return
            opt = self.options
        self.actual_options = opt
        self.binding["perturbation_states"] = [tensor_digest(d) for d in self.history]
        self.state = prepare(
            self.method,
            self.z,
            self.loss_at,
            binding=self.binding,
            options=opt,
            perturbations=D,
            initial_error=initial,
        )
        self.refreshes.append(
            {
                "step": step,
                "rank": len(self.history),
                "key": self.state["key"],
                "effective_rank": self.state.get("effective_rank"),
            }
        )

    def evaluate(self, ref, pred, index):
        if self.method in ("squared_gradient_error", "apq_masked_gradient_error"):
            state = {**self.state, "weight": self.state["weight"][index : index + 1]}
        else:
            state = self.state
        return score(
            self.method,
            ref,
            pred,
            options=self.actual_options,
            prepared=state,
            binding=self.binding,
        )


def mlp_replacement(backend, proxy, data, stage):
    """MR on the original GELU teacher, with a separate clamped branch feeding the same FC2."""
    config = stage["fit"]
    path = stage["module"]
    root = config["mlp"]
    teacher = backend.model
    from .reconstruction import paired_block

    rows = paired_block(backend, proxy.model, data, root, config["cache_mib"])
    images = {sid: x for x, sid in data.batches("calibration")}
    objective = BlockObjective(
        backend,
        teacher,
        images,
        data.labels,
        root,
        config["objective"],
        config["cache_mib"],
    )
    objective.refresh(proxy.model, 0)
    mlp = proxy.model.get_submodule(root)
    activation = proxy.model.get_submodule(path)
    trainable = [proxy.model.get_submodule(n).weight for n in config["weight_modules"]]
    allowed = {id(p) for p in mlp.parameters()}
    require(
        trainable and all(id(p) in allowed for p in trainable),
        "MR can update only selected MLP W",
    )
    rng = torch.Generator().manual_seed(config["seed"])
    opt = torch.optim.Adam(trainable, lr=config["lr"])
    losses = []
    try:
        for w in trainable:
            w.requires_grad_(True)
        with torch.enable_grad():
            for step in range(config["steps"]):
                parts = []
                for j in torch.randperm(len(rows), generator=rng)[
                    : config["batch_images"]
                ].tolist():
                    _, xf, _, target, args, kwargs = rows[j]
                    direct = mlp(xf, *args, **kwargs)

                    def clip(v):
                        return v.clamp_max(
                            torch.quantile(
                                v.detach().flatten().float(), config["percentile"]
                            )
                        )

                    with substitute(activation, clip):
                        capped = mlp(xf, *args, **kwargs)
                    parts.append(
                        objective.evaluate(target, direct, j)
                        + 2 * objective.evaluate(target, capped, j)
                    )
                loss = torch.stack(parts).mean()
                require(bool(torch.isfinite(loss)), "nonfinite MR objective")
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step in (0, config["steps"] - 1):
                    losses.append(float(loss.detach()))
    finally:
        for w in trainable:
            w.requires_grad_(False)
    return {
        "module": root,
        "teacher": "original GELU",
        "objective": config["objective"],
        "updated": config["weight_modules"],
        "calibration_ids": list(images),
        "steps": config["steps"],
        "direct_plus_twice_clamped": losses,
    }
