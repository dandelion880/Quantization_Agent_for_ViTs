"""固定网格上的 AdaRound；仅 alpha 可训练，推理只保存离散编码。

hard sigmoid、初始化、余弦 beta 和正则项适配 AIMET 1.35.1/7b60b525。
BSD-3-Clause 完整版权声明保存在 shared/reference_sources/aimet_adaround_7b60b525。
项目显式使用 token-row Linear 局部目标，预算由配置给出，不隐式采用论文配方。
"""

from copy import deepcopy
import math
import torch
from torch.nn import functional as F

from ..schemas import require
from .model import tensor_digest
from . import codecs, grids


def soft_round(alpha):
    return (alpha.sigmoid() * 1.2 - 0.1).clamp(0, 1)


def initialize(weight, scale):
    fraction = weight / scale - (weight / scale).floor()
    return -torch.log(1.2 / (fraction + 0.1) - 1)


def rounding_loss(alpha, iteration, iterations, reg=0.01, warmup=0.2, beta_range=(20., 2.)):
    if iteration < iterations * warmup:
        return alpha.new_tensor(0.)
    fraction = (iteration - iterations * warmup) / (iterations * (1 - warmup))
    beta = beta_range[1] + 0.5 * (beta_range[0] - beta_range[1]) * (1 + math.cos(math.pi * fraction))
    return reg * (1 - (2 * soft_round(alpha) - 1).abs().pow(beta)).sum()


def fit(weight, spec, params, records, teachers, output, options):
    """output(x, decoded_w) 必须可微，但 teacher/网格/原权重均只读。"""
    grid_method = options["rounding_grid"]
    # 手动scale/zero也须先满足网格定义，不能优化后才在导出时发现非法。
    grids.validate_params(params, spec, list(weight.shape), grid_method)
    iterations = options["rounding_iterations"]
    device = weight.device
    w = weight.detach().float()
    p = {k: v.detach().to(device).clone() for k, v in params.items()}
    require(not any(t.requires_grad for t in p.values()), "AdaRound 网格必须冻结")
    grid_before = {k: tensor_digest(v) for k, v in p.items()}
    scale = p["scale"]
    zero = p.get("zero", torch.zeros_like(scale))
    signed = spec["format"].startswith("INT")
    offset = 2 ** (spec["bits"] - 1) if signed else 0
    low, high = -offset, 2 ** spec["bits"] - 1 - offset
    floor = (w / scale).floor()
    alpha = torch.nn.Parameter(initialize(w, scale))
    optimizer = torch.optim.Adam([alpha], lr=options.get("rounding_lr", 1e-3))
    generator = torch.Generator().manual_seed(options.get("rounding_seed", 0))
    require(len(records) == len(teachers) and records, "AdaRound teacher 样本未对齐")
    require(all(r["x"].shape[:-1] == t["y"].shape[:-1] for r, t in zip(records, teachers)), "AdaRound token 对齐失败")
    batch = min(options.get("rounding_batch_size", 32), len(records))
    history = []
    try:
        with torch.enable_grad():
            for step in range(iterations):
                indices = torch.randperm(len(records), generator=generator)[:batch].tolist()
                x = torch.cat([records[i]["x"].reshape(-1, w.shape[1]) for i in indices]).to(device)
                target = torch.cat([teachers[i]["y"].reshape(-1, w.shape[0]) for i in indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                rounded = (floor + soft_round(alpha) + zero).clamp(low, high)
                decoded = (rounded - zero) * scale
                y = output(x, decoded)
                # Linear 的 feature 轴求和，然后对 token-row 求均值。
                reconstruction = (y - target).square().sum(-1).mean()
                regularizer = rounding_loss(alpha, step, iterations,
                                           options.get("rounding_reg", .01), options.get("rounding_warmup", .2))
                loss = reconstruction + regularizer
                require(bool(torch.isfinite(loss)), "AdaRound loss 非有限", code="numerical_error")
                loss.backward()
                require(alpha.grad is not None and bool(torch.isfinite(alpha.grad).all()), "AdaRound 梯度非法")
                optimizer.step()
                if step in (0, iterations - 1):
                    history.append({"iteration": step + 1, "reconstruction": float(reconstruction.detach()),
                                    "rounding_regularizer": float(regularizer.detach())})
                del x, target, rounded, decoded, y, reconstruction, regularizer, loss
        with torch.no_grad():
            hard = (floor + (alpha >= 0).to(w.dtype) + zero).clamp(low, high)
            codes = (hard + offset).to(torch.uint8)
            require(all(tensor_digest(p[k]) == v for k, v in grid_before.items()), "AdaRound 修改了依赖网格")
            record = {"method": "adaround", "grid_method": grid_method, "spec": deepcopy(spec),
                      "shape": list(w.shape), "data": codecs.pack(codes, spec["bits"]),
                      "params": {k: v.cpu() for k, v in p.items()}}
        return record, {"iterations_requested": iterations, "iterations_completed": iterations,
                        "optimizer": "Adam(alpha only)", "batch_images": batch,
                        "grid_digest": grid_before, "inference": "hard alpha>=0; no alpha retained",
                        "teacher": "original FP module output", "history": history}
    finally:
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        alpha.grad = None


def base_record(record):
    require(record.get("grid_method") in grids.METHODS, "AdaRound 缺少合法冻结网格")
    return dict(record, method=record["grid_method"])


def decode(record, device):
    return grids.decode(base_record(record), device)


def validate_record(record, spec, shape):
    grids.validate_record(base_record(record), spec, shape)
