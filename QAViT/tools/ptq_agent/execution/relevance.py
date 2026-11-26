"""Conservative positive Taylor propagation over a bound attention/residual graph."""

from contextlib import ExitStack
from copy import deepcopy
import torch
from ..schemas import require
from .calibration import differentiable_forward
from .sites import install_attention
from .task_sensitivity import substitute


def positive_pull(outputs, inputs, relevance, chunk):
    """Positive x_j * dy_i/dx_j, normalized separately at every nonzero output relevance."""
    flat = outputs.reshape(-1)
    ids = relevance.flatten().nonzero().flatten()
    result = [torch.zeros_like(x) for x in inputs]
    lost = relevance.new_zeros(())
    require(chunk > 0, "empty relevance Jacobian chunk")
    for start in range(0, len(ids), chunk):
        which = ids[start : start + chunk]
        selected = flat.index_select(0, which)
        eye = torch.eye(len(which), device=flat.device, dtype=flat.dtype)
        deriv = torch.autograd.grad(
            selected,
            inputs,
            grad_outputs=eye,
            is_grads_batched=True,
            retain_graph=True,
            allow_unused=True,
        )
        terms = [
            (g * x.detach()).clamp_min(0).reshape(len(which), -1)
            if g is not None
            else x.new_zeros(len(which), x.numel())
            for x, g in zip(inputs, deriv)
        ]
        denom = sum(t.sum(1) for t in terms)
        r = relevance.reshape(-1)[which]
        lost += r[denom == 0].sum()
        scale = torch.where(
            denom > 0, r / denom.clamp_min(torch.finfo(denom.dtype).tiny), 0
        )
        for v, t in zip(result, terms):
            v.add_((scale[:, None] * t).sum(0).reshape_as(v))
    return result, lost


def prepare(backend, rows, labels, settings):
    net = deepcopy(backend.model).eval().requires_grad_(False)
    blocks = settings["blocks"]
    require(
        blocks and len({u["block"] for u in blocks}) == len(blocks),
        "empty or repeated relevance chain",
    )
    install_attention(net, [u["attention"] for u in blocks])
    chunk = settings["jacobian_chunk"]
    result = []
    importance = {u["block"]: 0.0 for u in blocks}
    with torch.inference_mode(False), torch.enable_grad():
        for image, sid in rows:
            image = image.detach().clone().requires_grad_(True)
            seen = {}
            maps = {}
            with ExitStack() as stack:
                for u in blocks:
                    n = u["block"]

                    def watch(m, args, kwargs, y, n=n):
                        require(n not in seen, "duplicate relevance block execution")
                        seen[n] = (args[0].detach(), args[1:], kwargs, y.detach())

                    stack.callback(
                        net.get_submodule(n)
                        .register_forward_hook(watch, with_kwargs=True)
                        .remove
                    )

                    def tap(y, n=n):
                        maps[n] = y

                    stack.enter_context(
                        substitute(
                            net.get_submodule(u["attention"] + "._ptq_softmax"), tap
                        )
                    )
                logits = differentiable_forward(backend, net, image)
            require(
                list(seen) == [u["block"] for u in blocks],
                "relevance chain differs from actual execution order",
            )
            require(
                all(
                    torch.equal(seen[a["block"]][3], seen[b["block"]][0])
                    for a, b in zip(blocks, blocks[1:])
                ),
                "unbound operation between relevance blocks",
            )
            require(
                all(
                    torch.equal(seen[a["block"]][3], seen[b["block"]][0])
                    for a, b in zip(blocks, blocks[1:])
                ),
                "relevance chain needs contiguous residual outputs",
            )
            if settings["target"] == "labels":
                require(sid in labels, "missing calibration label")
                cls = int(labels[sid])
            else:
                require(
                    settings["target"] == "teacher_class", "unknown relevance target"
                )
                cls = int(logits.argmax(-1))
            grads = torch.autograd.grad(logits[0, cls], list(maps.values()))
            last = blocks[-1]["block"]
            z = seen[last][3].requires_grad_(True)
            with substitute(net.get_submodule(last), lambda _: z):
                score = differentiable_forward(backend, net, image)[0, cls]
            (relevance,), lost = positive_pull(
                score, (z,), torch.ones_like(score), chunk
            )
            losses = [float(lost)]
            scores = {}
            for u, grad in reversed(list(zip(blocks, grads))):
                n = u["block"]
                unit = net.get_submodule(n)
                tap = net.get_submodule(u["attention"] + "._ptq_softmax")
                x, args, kwargs, y = seen[n]
                x = x.requires_grad_(True)
                A = maps[n].detach().requires_grad_(True)
                with substitute(tap, lambda _: A):
                    out = unit(x, *args, **kwargs)
                require(
                    torch.allclose(out.detach(), y) and relevance.shape == out.shape,
                    "broken relevance boundary or noncontiguous stage",
                )
                (direct, rA), lost = positive_pull(out, (x, A), relevance, chunk)
                captured = []
                with substitute(tap, lambda y: captured.append(y)):
                    unit(x, *args, **kwargs)
                (via,), attention_lost = positive_pull(captured[0], (x,), rA, chunk)
                relevance = direct + via
                val = float((grad * rA).clamp_min(0).mean(1).mean())
                scores[n] = val
                importance[n] += val / len(rows)
                losses.append(float(lost + attention_lost))
            require(
                abs(float(relevance.sum()) + sum(losses) - 1) < 1e-3,
                "positive relevance failed the conservation audit",
            )
            result.append(
                {
                    "id": sid,
                    "class": cls,
                    "scores": scores,
                    "input_mass": float(relevance.sum()),
                    "unassigned_mass": sum(losses),
                }
            )
    return {
        "scores": importance,
        "samples": result,
        "rule": "positive Taylor normalized per output; separate direct and attention paths at each residual block",
        "target": settings["target"],
    }
