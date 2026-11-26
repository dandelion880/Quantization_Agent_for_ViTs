"""Sequential frozen-output compensation on the candidate's actual forward path."""

from contextlib import ExitStack
import torch
from torch import nn
from ..schemas import require
from .model import forward


class Correction(nn.Module):
    def __init__(self, kind, state):
        super().__init__()
        self.kind = kind
        for key, value in state.items():
            self.register_buffer(key, value.detach().clone())
        self.eval()

    def forward(self, x, y):
        if self.kind == "qwt":
            require(
                x.shape[-1] == self.weight.size(1)
                and y.shape[-1] == self.weight.size(0),
                "QwT feature mismatch",
            )
            if x.device.type == "cpu":
                correction = torch.nn.functional.linear(
                    x.half().float(), self.weight.float(), self.bias.float()
                ).half()
            else:
                correction = torch.nn.functional.linear(
                    x.to(self.weight.dtype), self.weight, self.bias
                )
            return y + correction.to(y)
        require(
            y.ndim == self.bias.ndim
            and all(a == 1 or a == b for a, b in zip(self.bias.shape, y.shape)),
            "correction position/grid mismatch",
        )
        if self.kind == "channel_affine":
            return y * self.alpha.to(y) + self.bias.to(y)
        return y + self.bias.to(y)


def corrected_output(module, args, output):
    return module.ptq_output_correction(args[0], output)


def install(model, name, kind, state):
    module = model.get_submodule(name)
    require(
        not hasattr(module, "ptq_output_correction"),
        "duplicate compensation at " + name,
    )
    device = next(model.parameters(), next(model.buffers(), torch.empty(0))).device
    module.add_module("ptq_output_correction", Correction(kind, state).to(device))
    if "multiplier" in state:
        from .integer_affine import bind

        require(
            kind == "channel_affine",
            "folded channel multiplier cannot represent a matrix correction",
        )
        bind(module, state)
    else:
        module.register_forward_hook(corrected_output)


def fit(backend, model, data, stage):
    """Stream sufficient statistics, re-running the installed prefix for each named destination."""
    name, kind = stage["module"], stage["kind"]
    matmul = name.endswith(("._ptq_qk", "._ptq_probability"))
    teacher = backend.model
    if matmul:
        from copy import deepcopy
        from .sites import install_attention

        teacher = deepcopy(teacher)
        install_attention(teacher, [name.rsplit(".", 1)[0]])
    original = teacher.get_submodule(name)
    candidate = model.get_submodule(name)
    stats, seen = {}, []

    def accumulate(key, value):
        value = value.detach().cpu()
        stats[key] = value if key not in stats else stats[key] + value

    calls = 0

    def pair(module, args, kwargs, y):
        nonlocal calls
        ref = (
            original(*args, **kwargs)
            if stage["reference"] == "local_fp_candidate"
            else full_outputs.pop(0)
        )
        require(
            torch.is_tensor(y) and torch.is_tensor(ref) and y.shape == ref.shape,
            "paired output shape mismatch",
        )
        q, fp = y.double(), ref.double()
        require(
            bool(torch.isfinite(q).all() and torch.isfinite(fp).all()),
            "nonfinite compensation pair",
        )
        if kind == "qwt":
            x = args[0].reshape(-1, args[0].shape[-1]).double()
            e = (fp - q).reshape(-1, fp.shape[-1])
            require(x.size(0) == e.size(0), "QwT needs paired input/output token rows")
            x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
            accumulate("xx", x.T @ x)
            accumulate("xe", x.T @ e)
            accumulate("ee", e.square().sum())
            accumulate("sum_e", e.sum(0))
            accumulate("n", x.new_tensor(len(x)))
        else:
            axes = tuple(stage["reduce_axes"])
            require(
                0 in axes
                and len(set(axes)) == len(axes)
                and all(0 <= a < q.ndim for a in axes),
                "declare reduced axes including batch; preserve all other coordinates",
            )
            shape = fp.sum(axes, keepdim=True)
            require(
                "fp" not in stats or shape.shape == stats["fp"].shape,
                "position layout changed during calibration",
            )
            accumulate("fp", shape)
            accumulate("q", q.sum(axes, keepdim=True))
            accumulate("n", q.new_tensor(q.numel() // shape.numel()))
            if kind == "channel_affine":
                accumulate("qq", q.square().sum(axes, keepdim=True))
                accumulate("qf", (q * fp).sum(axes, keepdim=True))
        calls += 1

    with ExitStack() as stack, torch.no_grad():
        full_outputs = []
        if stage["reference"] == "fp_model":
            stack.callback(
                original.register_forward_hook(
                    lambda m, a, y: full_outputs.append(y)
                ).remove
            )
        stack.callback(candidate.register_forward_hook(pair, with_kwargs=True).remove)
        for x, sid in data.batches("calibration"):
            before = calls
            if stage["reference"] == "fp_model":
                forward(backend, teacher, x)
            forward(backend, model, x)
            require(
                calls > before and not full_outputs,
                "unpaired or missing forward for " + name,
            )
            seen.append(sid)
    require(
        seen == list(data.ids("calibration")) and calls, "calibration stream mismatch"
    )
    n = stats["n"]
    report = {
        "module": name,
        "kind": kind,
        "reference": stage["reference"],
        "calibration_ids": seen,
        "observations": int(n),
    }
    if kind == "qwt":
        xx, xe = stats["xx"], stats["xe"]
        coeff = torch.linalg.pinv(xx, hermitian=True) @ xe
        sse = (
            stats["ee"] - 2 * (coeff * xe).sum() + (coeff * (xx @ coeff)).sum()
        ).clamp_min(0)
        total = (stats["ee"] - stats["sum_e"].square().sum() / n).clamp_min(0)
        r2 = float(1 - sse / total) if total > 0 else None
        if r2 is None or r2 <= 0:
            coeff.zero_()
        state = {"weight": coeff[:-1].T.half(), "bias": coeff[-1].half()}
        report.update(
            r2=r2,
            rank=int(torch.linalg.matrix_rank(xx, hermitian=True)),
            accepted=r2 is not None and r2 > 0,
        )
    elif kind == "channel_affine":
        var = (stats["qq"] - stats["q"].square() / n).clamp_min(0)
        cov = stats["qf"] - stats["q"] * stats["fp"] / n
        alpha = torch.where(var > 0, cov / torch.where(var > 0, var, 1.0), 0.0)
        state = {
            "alpha": alpha.float(),
            "bias": ((stats["fp"] - alpha * stats["q"]) / n).float(),
        }
        report.update(
            constant_channels=int((var == 0).sum()),
            zero_variance_policy="constant reference mean",
        )
    else:
        state = {"bias": ((stats["fp"] - stats["q"]) / n).float()}
    require(
        all(bool(torch.isfinite(x).all()) for x in state.values()),
        "nonfinite frozen output correction",
    )
    fold = stage.get("settings", {}).get("integer_fold")
    if fold:
        from .integer_affine import compile_state

        require(
            kind == "channel_affine", "folded multiplier needs a channel regression"
        )
        state = compile_state(candidate, state["alpha"], state["bias"], fold)
        report["deployment"] = {
            "operator": "integer_affine",
            "output": fold,
            "zero_slope_channels": int(state["is_constant"].sum()),
            "negative_multiplier_channels": int((state["multiplier"] < 0).sum()),
        }
    install(model, name, kind, state)
    return {
        **report,
        "tensor_bytes": sum(x.numel() * x.element_size() for x in state.values()),
    }


def audit(model, plan):
    expected = {
        s["module"]: s["kind"]
        for s in plan.get("postfit", [])
        if s["kind"] in CORRECTIONS
    }
    actual = {
        n: m.ptq_output_correction
        for n, m in model.named_modules()
        if hasattr(m, "ptq_output_correction")
    }
    require(set(actual) == set(expected), "output correction coverage mismatch")
    for name, module in actual.items():
        from .integer_affine import integer_forward

        node = model.get_submodule(name)
        fold = next(
            s.get("settings", {}).get("integer_fold")
            for s in plan["postfit"]
            if s["module"] == name and s["kind"] in CORRECTIONS
        )
        active = (
            (getattr(node.forward, "__func__", None) is integer_forward)
            if fold
            else corrected_output in node._forward_hooks.values()
        )
        require(
            module.kind == expected[name] and active,
            "declared correction absent from forward graph",
        )
    return {
        n: {k: v.detach().cpu() for k, v in m.named_buffers()}
        for n, m in actual.items()
    }


CORRECTIONS = {"bias_compensation", "channel_affine", "qwt"}
