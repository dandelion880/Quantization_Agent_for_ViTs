"""Structural PTQ state. Token coordinates and module bindings come from the selected graph."""

from contextlib import ExitStack
from copy import deepcopy
import torch
from torch import nn
from ..schemas import require
from .sites import install_attention, attention_kind
from .model import forward


def token_bank(backend, data, stage):
    """Stream top-K (source image, patch position, norm, K, V) with a bounded heap."""
    import heapq
    from ..profiling.adapters import ProfileAdapter

    name = stage["module"]
    limit = stage["candidates"]
    teacher = deepcopy(backend.model)
    install_attention(teacher, [name])
    m = teacher.get_submodule(name)
    require(
        attention_kind(m) in ("timm", "dinov2"),
        "KV prefix needs the mask-free global attention adapter",
    )
    state, heap = {}, []

    def features(module, args, y):
        state["x"] = args[0]

    def key(module, args):
        state["key"] = args[1].transpose(-1, -2)

    def value(module, args):
        state["value"] = args[1]

    with ExitStack() as hooks:
        for hook in (
            m.qkv.register_forward_hook(features),
            m._ptq_qk.register_forward_pre_hook(key),
            m._ptq_probability.register_forward_pre_hook(value),
        ):
            hooks.callback(hook.remove)
        for seq, (image, sid) in enumerate(data.batches("calibration")):
            state.clear()
            forward(backend, teacher, image)
            x = state["x"]
            require(x.size(0) == 1, "candidate source must identify individual images")
            layout = ProfileAdapter().token_layout(
                backend, name + ".qkv", x.shape, image.shape
            )
            left, right = layout["patch_slice"]
            norm = x[0, left:right].double().norm(dim=-1)
            for j in norm.argsort(descending=True, stable=True)[:limit].tolist():
                pos = left + j
                entry = (
                    float(norm[j]),
                    -seq,
                    -pos,
                    sid,
                    state["key"][0, :, pos].cpu().clone(),
                    state["value"][0, :, pos].cpu().clone(),
                )
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                elif entry[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, entry)
    require(
        len(heap) == limit and stage["copies"] >= 1,
        "insufficient candidate tokens or invalid register copies",
    )
    ranked = sorted(heap, key=lambda v: v[:3], reverse=True)
    return {
        "key": torch.stack([v[4] for v in ranked]).mean(0),
        "value": torch.stack([v[5] for v in ranked]).mean(0),
    }, [{"sample_id": v[3], "position": -v[2], "norm": v[0]} for v in ranked]


def prepend_key(module, args):
    q, k = args
    v = module.register_key.to(k)[None, :, :, None].expand(
        k.size(0), -1, -1, module.register_copies
    )
    return q, torch.cat((v, k), -1)


def prepend_value(module, args):
    p, v = args
    prefix = module.register_value.to(v)[None, :, None, :].expand(
        v.size(0), -1, module.register_copies, -1
    )
    return p, torch.cat((prefix, v), -2)


def select_tokens(x, prefix, count):
    require(
        x.ndim == 3 and 0 <= prefix < x.size(1) and 0 < count < x.size(1) - prefix,
        "invalid token deletion extent",
    )
    keep = (
        x[:, prefix:]
        .double()
        .norm(dim=-1)
        .argsort(stable=True)[:, : x.size(1) - prefix - count]
        .sort(1)
        .values
        + prefix
    )
    special = torch.arange(prefix, device=x.device).expand(len(x), -1)
    indices = torch.cat((special, keep), 1)
    return x.gather(1, indices[:, :, None].expand(-1, -1, x.size(-1))), indices


def recover_neighbors(short, indices, hw, prefix):
    h, w = hw
    spatial = indices[:, prefix:] - prefix
    pos = torch.arange(h * w, device=short.device)
    rc = torch.stack((pos // w, pos % w), 1).float()
    distance = torch.cdist(rc.expand(len(short), -1, -1), rc[spatial], p=2)
    nearest = (distance == distance.amin(-1, keepdim=True)).to(short)
    restored = (nearest / nearest.sum(-1, keepdim=True)) @ short[:, prefix:]
    return torch.cat((short[:, :prefix], restored), 1)


def route_windows(x, fraction, high, low):
    """A branch is evaluated only for its assigned windows; restore original order and geometry."""
    require(x.ndim >= 3 and 0 <= fraction <= 1, "invalid window features")
    order = x.flatten(1).double().norm(dim=1).argsort(descending=True, stable=True)
    nlow = round(len(x) * fraction)
    groups = (order[: len(x) - nlow], order[len(x) - nlow :])
    results = [
        (ids, fn(x.index_select(0, ids)))
        for ids, fn in zip(groups, (high, low))
        if len(ids)
    ]
    shape = results[0][1].shape[1:]
    out = results[0][1].new_empty((len(x), *shape))
    for ids, values in results:
        out.index_copy_(0, ids, values)
    return out, {"high_indices": groups[0], "low_indices": groups[1]}


class TokenRoute:
    """Shared hook pair; per-forward indices remain local to the execution context."""

    def __init__(self, prefix, grid, count):
        self.prefix, self.grid, self.count = prefix, tuple(grid), count

    def enter(self, module, args, kwargs):
        x = args[0]
        require(
            x.ndim == 3 and x.size(1) == self.prefix + self.grid[0] * self.grid[1],
            "token geometry differs from the declared prefix and patch grid",
        )
        require(
            len(args) == 1 and not kwargs,
            "positional, mask or window operands require an aligned index adapter",
        )
        short, indices = select_tokens(x, self.prefix, self.count)
        token_indices.set({**token_indices.get(), id(self): indices})
        return (short,), kwargs

    def leave(self, module, args, y):
        active = token_indices.get()
        require(id(self) in active, "token recovery has no matching deletion")
        indices = active[id(self)]
        token_indices.set({k: v for k, v in active.items() if k != id(self)})
        return recover_neighbors(y, indices, self.grid, self.prefix)


from contextvars import ContextVar

token_indices = ContextVar("qavit_patch_routes", default={})


def configure(model, stages, states):
    model._qavit_structure = [
        *getattr(model, "_qavit_structure", []),
        *deepcopy(stages),
    ]
    for s in stages:
        name = s["module"]
        kind = s["kind"]
        if kind == "calibrated_operator":
            from .nonlinear_calibration import bind

            bind(model, s, states[name])
        elif kind == "window_branches":
            from .window_routes import bind

            bind(model, s)
        elif kind == "lite_denominator":
            from .efficient_topology import bind_lite

            bind_lite(model.get_submodule(name), s["settings"])
        elif kind == "integer_function":
            from .integer_functions import install

            install(model, s, states[name])
        elif kind == "outlier_tokens":
            module = model.get_submodule(name)
            end = model.get_submodule(s["end_module"])
            require(not hasattr(module, "ptq_token_route"), "duplicate token route")
            route = TokenRoute(s["prefix_tokens"], s["grid_hw"], s["delete_count"])
            module.ptq_token_route = route
            module.register_forward_pre_hook(route.enter, with_kwargs=True)
            end.register_forward_hook(route.leave)
        elif kind == "gelu_to_relu":
            original = model.get_submodule(name)
            require(
                isinstance(original, nn.GELU),
                "GELU substitution needs an actual GELU leaf",
            )
            parent, _, key = name.rpartition(".")
            setattr(model.get_submodule(parent), key, nn.ReLU().eval())
        else:
            require(kind == "cached_kv" and name in states, "missing KV bank")
            install_attention(model, [name])
            m = model.get_submodule(name)
            state = states[name]
            device = next(m.parameters(), next(m.buffers(), torch.empty(0))).device
            for key, child, hook in (
                ("key", m._ptq_qk, prepend_key),
                ("value", m._ptq_probability, prepend_value),
            ):
                v = state[key].to(device)
                require(
                    v.ndim == 2
                    and v.size(0) == m.num_heads
                    and v.numel() == m.qkv.in_features
                    and bool(torch.isfinite(v).all()),
                    "cached KV shape differs from attention heads",
                )
                child.register_buffer("register_" + key, v)
                child.register_copies = s["copies"]
                child.register_forward_pre_hook(hook)


def fitted_state(model, stages):
    state = {}
    for s in stages:
        m = model.get_submodule(s["module"])
        if s["kind"] == "window_branches":
            from .window_routes import routed

            require(
                m.forward.__func__ is routed and m.ptq_window == s["settings"],
                "window branch disconnected",
            )
        elif s["kind"] == "lite_denominator":
            from .efficient_topology import lite_forward

            require(
                m.ptq_denominator == s["settings"]
                and m.forward.__func__ is lite_forward,
                "denominator binding differs",
            )
        elif s["kind"] == "calibrated_operator":
            from .nonlinear_calibration import installed

            state[s["module"]] = installed(m)
        elif s["kind"] == "integer_function":
            from .integer_functions import installed

            state[s["module"]] = installed(model, s)
        elif s["kind"] == "outlier_tokens":
            route = m.ptq_token_route
            end = model.get_submodule(s["end_module"])
            require(
                route.enter in m._forward_pre_hooks.values()
                and route.leave in end._forward_hooks.values(),
                "token deletion/recovery disconnected",
            )
        elif s["kind"] == "gelu_to_relu":
            require(isinstance(m, nn.ReLU), "declared activation rewrite missing")
        else:
            require(
                prepend_key in m._ptq_qk._forward_pre_hooks.values()
                and prepend_value in m._ptq_probability._forward_pre_hooks.values(),
                "cached KV not connected to both operands",
            )
            state[s["module"]] = {
                "key": m._ptq_qk.register_key.detach().cpu(),
                "value": m._ptq_probability.register_value.detach().cpu(),
            }
    return state
