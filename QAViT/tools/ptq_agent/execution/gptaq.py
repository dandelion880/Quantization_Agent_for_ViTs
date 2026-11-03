"""GPTAQ v1 (d6bd2e4) 的原始 FP 对齐与三角输入补偿更新。

仅适配只读输入、分块统计和编码导出，不调用上游的模型原地赋值。
与 GPTQ 共用存储编码，不能把此方法退化为仅启用 quantized-prefix。
"""

from copy import deepcopy
import torch

from ..schemas import require
from . import codecs


def statistics(records, teachers, device):
    require(len(records) == len(teachers) and records, "GPTAQ FP/量化输入数量不一致")
    columns = records[0]["x"].shape[-1]
    h = torch.zeros(columns, columns, dtype=torch.float32, device=device)
    difference = torch.zeros_like(h)
    samples = 0
    for row, teacher in zip(records, teachers):
        x, fp = row["x"].to(device).float(), teacher["x"].to(device).float()
        require(x.shape == fp.shape and row["y"].shape == teacher["y"].shape,
                "GPTAQ FP/量化 token 维度或输出不对齐")
        count = x.shape[0] if x.ndim == 3 else 1
        h *= samples / (samples + count)
        difference *= samples / (samples + count)
        samples += count
        x, fp = (t.reshape(-1, columns).t() * (2.0 / samples) ** 0.5 for t in (x, fp))
        h += x @ x.t()
        difference += (fp - x) @ x.t()
    return h, difference


def grid(weight, bits, symmetric, clip=False):
    """官方 WeightQuantizer 的 per-channel 网格（不同于旧 GPTQ 的中点网格）。"""
    zero = weight.new_zeros(weight.shape[0])
    lo, hi = torch.minimum(weight.amin(1), zero), torch.maximum(weight.amax(1), zero)
    maximum = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
    if symmetric:
        hi = torch.maximum(lo.abs(), hi).clamp_min(1e-5)
        lo = -hi
    else:
        dead = (lo == 0) & (hi == 0)
        lo, hi = lo.masked_fill(dead, -1), hi.masked_fill(dead, 1)
    def params(factor, initial=False):
        interval = hi * factor - lo * factor
        scale = (hi * factor / maximum if symmetric else
                 (interval.clamp_min(1e-5) if initial else interval) / maximum)
        return scale, zero if symmetric else (-lo * factor / scale).round()
    scale, offset = params(1.0, initial=True)
    if clip:
        best = weight.new_full((weight.shape[0],), float("inf"))
        for i in range(80):
            s, z = params(1 - i / 100)
            q = (weight / s[:, None]).round() + z[:, None]
            q = q.clamp(-maximum - 1 if symmetric else 0, maximum)
            error = ((q - z[:, None]) * s[:, None] - weight).abs().pow(2.4).sum(1)
            better = error < best
            best[better], scale[better], offset[better] = error[better], s[better], z[better]
    # Packed unsigned code 用固定偏移表示 signed 网格；数学舍入仍是原始 signed。
    return scale, offset + (2 ** (bits - 1) if symmetric else 0)


def fit(weight, hessian, difference, spec, options):
    w, h, dx = (t.detach().float().clone() for t in (weight, hessian, difference))
    rows, columns = w.shape
    scale, zero = grid(w, spec["bits"], spec["symmetric"], options.get("gptaq_clip", False))
    dead = torch.diag(h) == 0
    h[dead, dead], w[:, dead], dx[:, dead] = 1, 0, 0
    order = torch.argsort(torch.diag(h), descending=True) if options.get("actorder", False) else torch.arange(columns, device=w.device)
    inverse = torch.argsort(order)
    w, h, dx = w[:, order], h[order][:, order], dx[order][:, order]
    h.diagonal().add_(options.get("damping", 0.01) * h.diagonal().mean())
    h = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    correction = 0.25 * ((dx @ h.t()).triu(diagonal=1)) @ h
    codes, result = torch.zeros_like(w, dtype=torch.uint8), torch.zeros_like(w)
    blocksize = options.get("blocksize", 128)
    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        local, error = w[:, start:end].clone(), torch.zeros_like(w[:, start:end])
        for j in range(end - start):
            column = local[:, j]
            q = ((column / scale).round() + zero).clamp(0, 2 ** spec["bits"] - 1)
            decoded = scale * (q - zero)
            result[:, start+j], codes[:, start+j] = decoded, q.to(torch.uint8)
            err = (column - decoded) / h[start+j, start+j]
            # RHS 在原地更新前求值；跨 block 补偿使用更新后的 local（与官方一致）。
            local[:, j:] -= err[:, None] @ h[start+j, start+j:end][None, :] - column[:, None] @ correction[start+j, start+j:end][None, :]
            error[:, j] = err
        w[:, end:] -= error @ h[start:end, end:] - local @ correction[start:end, end:]
    result, codes = result[:, inverse], codes[:, inverse]
    require(bool(torch.isfinite(result).all()), "GPTAQ 更新产生非有限值")
    record = {"method": "gptaq", "spec": deepcopy(spec), "shape": list(weight.shape),
              "data": codecs.pack(codes, spec["bits"]),
              "params": {"scale": scale[:, None].cpu(), "zero": zero[:, None].cpu()}}
    return record, result
