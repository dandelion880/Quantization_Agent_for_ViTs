"""局部输出驱动的坐标尺度搜索，与网格插件、模型 adapter 分离。

顺序：初始化全部目标 → 每轮依次搜索 W/A（或 MatMul 左/右操作数）。
候选始终相对初始尺度，平分保留首项；其它目标使用已选尺度。
这复用已核对的 PTQ4ViT 搜索规则，但不强制选择 twin 或某一论文配方。
"""

import torch
from torch.nn import functional as F

from ..schemas import require
from . import grids
from .calibration import capture, combined_input, local_output
from .methods import get_target_method
from .plugin_contracts import calibration, site_id, WEIGHT_KINDS


def operator_id(layer):
    kind = layer["kind"]
    suffix = "qk" if kind in ("attention_query", "attention_key") else "av" if kind in ("post_softmax", "attention_value") else kind
    return layer["module"], suffix


def validate_group(layers):
    targets = [(l, t, calibration(l, t)) for l in layers for t in ("weight", "activation") if l[t]]
    searches = [c for _, _, c in targets if c["strategy"] == "coordinate"]
    require(searches, "coordinate 组没有搜索目标")
    require(all(c["strategy"] in ("minmax", "coordinate") for _, _, c in targets), "同一算子的 coordinate 搜索只与明确 minmax 目标组合")
    require(len({(c["rounds"], c["metric"], c["loss"]) for c in searches}) == 1, "同一算子的搜索轮数/metric/loss 必须相同")
    for l, target, c in targets:
        require(l[target + "_method"] in ("uniform_full", "twin_softmax", "twin_gelu"), "coordinate 组包含未接入的网格")
        if target == "activation" and l[target]["granularity"] != "per_tensor":
            require(l["kind"] in ("attention_query", "attention_key", "attention_value") and l[target]["axis"] == 1,
                    "coordinate 的多尺度激活只支持独立 attention heads")
    return searches[0]


def _loss(reference, candidate, metric, gradient=None):
    if metric == "cosine":
        return -F.cosine_similarity(reference, candidate, dim=-1).mean()
    error = reference - candidate
    if metric == "hessian":
        require(gradient is not None, "coordinate Hessian 缺少真实梯度")
        error = error * gradient
    return error.square().mean()


def _scores(row, result, layer, target, count, metric):
    y, grad = row["y"], row.get("gradient")
    if count == 1:
        return _loss(y, result, metric, grad).reshape(1)
    # 这些分组在输出上相互独立，允许一次 forward 同时比较同一倍率。
    axis = (1 if layer["kind"] == "conv2d" else y.ndim - 1) if target == "weight" else 1
    require(y.shape[axis] % count == 0, "coordinate 输出分组不能整除")
    ys, qs = y.chunk(count, axis), result.chunk(count, axis)
    gs = grad.chunk(count, axis) if grad is not None else [None] * count
    return torch.stack([_loss(a, b, metric, g) for a, b, g in zip(ys, qs, gs)])


def fit_operator(agent, model, layers):
    from .plugin_math import fp32_statistics
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        with fp32_statistics(agent.backend.device):
            return _fit_operator(agent, model, layers)
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _fit_operator(agent, model, layers):
    settings = validate_group(layers)
    ordered = sorted(layers, key=lambda l: {"attention_query": 0, "post_softmax": 0,
                                          "attention_key": 1, "attention_value": 1}.get(l["kind"], 0))
    anchor = ordered[0]
    records, report = capture(agent.backend, model, anchor, agent.data, agent.cache_mib,
                              settings["loss"] if settings["metric"] == "hessian" else None)
    module = model.get_submodule(anchor["module"])
    weighted = anchor["kind"] in WEIGHT_KINDS
    right_anchor = anchor["kind"] in ("attention_key", "attention_value")
    states, base, targets = {}, {}, []
    for layer in ordered:
        sid = site_id(layer)
        for target in ("weight", "activation"):
            if layer[target] is None:
                continue
            if target == "weight":
                x = module.reference_weight.detach().cpu()
            else:
                right = layer["kind"] in ("attention_key", "attention_value")
                key = "x" if weighted or right == right_anchor else "rhs"
                x = torch.cat([r[key] for r in records], 0)
                agent.input_shapes[sid] = list(records[0][key].shape)
            method, spec = layer[target + "_method"], layer[target]
            params = (grids.fit(x, spec, method) if target == "weight" else
                      get_target_method("activation", method).fit(x, spec, options=layer.get("options", {})))
            states[sid, target] = params
            base[sid, target] = {k: v.clone() for k, v in params.items()}
            targets.append((layer, target))

    # observer 的初始尺度仍按原路径拟合；只把重建计算放到视觉设备。
    # 保持 FP32、候选顺序、逐样本 reduction 和平分规则，不缩减搜索预算。
    device = agent.backend.device
    records = [{k: v.to(device) for k, v in row.items()} for row in records]
    states = {key: {k: v.to(device) for k, v in params.items()} for key, params in states.items()}
    base = {key: {k: v.to(device) for k, v in params.items()} for key, params in base.items()}
    reference_weight = module.reference_weight.detach().to(device) if weighted else None

    def quantize(value, layer, target):
        spec, method = layer[target], layer[target + "_method"]
        p = states[site_id(layer), target]
        return grids.apply(value, spec, method, p) if target == "weight" else get_target_method("activation", method).apply(value, spec, params=p)

    def output(row, current, float_peers, weight_value=None):
        if weighted:
            x, w = row["x"], weight_value
            for layer, target in targets:
                if float_peers and (site_id(layer), target) != current:
                    continue
                if target == "weight":
                    continue  # 同一尺度候选的权重只 Q/DQ 一次，复用于全部样本。
                else:
                    x = quantize(x, layer, target)
            return local_output(anchor, module, x, row, w)
        a, b = (row["rhs"], row["x"]) if right_anchor else (row["x"], row["rhs"])
        for layer, target in targets:
            if float_peers and (site_id(layer), target) != current:
                continue
            if layer["kind"] in ("attention_key", "attention_value"):
                b = quantize(b, layer, target)
            else:
                a = quantize(a, layer, target)
        return a @ b

    history = []
    for round_index in range(settings["rounds"]):
        for layer, target in targets:
            cfg = calibration(layer, target)
            if cfg["strategy"] != "coordinate":
                continue
            key = site_id(layer), target
            method = layer[target + "_method"]
            param = "split" if method == "twin_softmax" else "scale"
            initial = base[key][param]
            factors = cfg["factors"]
            if method == "twin_softmax" and not {"factors", "factor_grid"}.intersection(layer.get(target + "_calibration", {})):
                factors = [2. ** -i / initial.item() for i in range(20)]
            best, best_value, best_index = None, initial.clone(), None
            for index, factor in enumerate(factors):
                trial = initial * factor
                require(bool((trial > 0).all()) and (param != "split" or trial.item() <= 1), "搜索候选尺度越界")
                states[key][param] = trial
                float_peers = cfg["peer_precision"] == "float"
                weight_value = reference_weight
                if weighted:
                    for peer, peer_target in targets:
                        if peer_target == "weight" and (not float_peers or (site_id(peer), peer_target) == key):
                            weight_value = quantize(reference_weight, peer, peer_target)
                losses = [_scores(r, output(r, key, float_peers, weight_value), layer, target,
                                  initial.numel(), cfg["metric"]) * r["y"].shape[0] for r in records]
                loss = torch.stack(losses).sum(0)
                require(bool(torch.isfinite(loss).all()), "coordinate 搜索 loss 非有限")
                if best is None:
                    best, best_value = loss, trial.clone()
                    best_index = torch.zeros_like(loss, dtype=torch.long)
                else:
                    improve = loss < best
                    best = torch.where(improve, loss, best)
                    best_value = torch.where(improve.reshape(initial.shape), trial, best_value)
                    best_index = torch.where(improve, index, best_index)
            states[key][param] = best_value
            history.append({"round": round_index + 1, "site": key[0], "target": target,
                            "candidates": len(factors), "selected_indices": best_index.tolist()})
    weights, activations = {}, {}
    for layer, target in targets:
        sid = site_id(layer)
        params = {k: v.cpu() for k, v in states[sid, target].items()}
        if target == "weight":
            weights[layer["module"]] = get_target_method("weight", layer["weight_method"]).fit(
                module.reference_weight.detach().cpu(), layer["weight"], params)
        else:
            activations[sid] = params
    return weights, activations, {**report, "strategy": "coordinate", "rounds": settings["rounds"],
                                  "metric": settings["metric"], "history": history,
                                  "reconstruction_device": str(device), "reconstruction_dtype": "float32"}
