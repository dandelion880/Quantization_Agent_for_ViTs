"""有预算的逐位点校准；只读 calibration 分区，梯度仅用于测量，不更新参数。"""

import torch
from torch.nn import functional as F

from ..schemas import require
from .model import forward
from .sites import observe, target_tensor
from .plugin_math import score
from .plugin_contracts import ATTENTION_KINDS, WEIGHT_KINDS


def differentiable_forward(backend, model, x):
    if hasattr(backend, "forward_model"):
        return backend.forward_model(model, x)
    return model.forward_features(x) if backend.spec["family"].startswith(("DINOv2", "DINOv3")) else model(x)


def capture(backend, model, layer, data, limit_mib, gradient_loss=None, *, observer_batch_size=None):
    """缓存单个位点的完整指定样本；超限即拒绝，不偷偷减少样本/token。"""
    records, current, record_ids = [], [], []
    used = 0
    last_batch_records = 0
    count = len(data.ids("calibration"))
    last_batch_size = (count - 1) % observer_batch_size + 1 if observer_batch_size else count
    # 最后一批之前保持 FP 算子。最后一批内按图拓扑启用已冻结输入尺度的算子，
    # 对应 running observer 的 finalize 语义；没有丢弃或抽减校准样本。
    from .operators import SoftmaxSite
    progressive = [(m, m.integer_enabled) for m in model.modules() if isinstance(m, SoftmaxSite)]
    gradient_flags = [(m, m.gradient_probe) for m in model.modules() if hasattr(m, "gradient_probe")]
    active = {"weight": [], "activation": []}
    for module in model.modules():
        if not hasattr(module, "ptq_site_id"):
            continue
        if getattr(module, "weight_enabled", False):
            active["weight"].append(module.ptq_site_id)
        if module.enabled:
            active["activation"].append(module.ptq_site_id)

    def hook(args, output):
        module = model.get_submodule(layer["module"])
        x = target_tensor(layer, args, output)
        if layer["kind"] in WEIGHT_KINDS and hasattr(module, "calibration_input"):
            x = module.calibration_input(args[0])
        row = {"x": x, "y": output}
        if layer["kind"] in ATTENTION_KINDS:
            row["rhs"] = args[0] if layer["kind"] in ("attention_key", "attention_value") else args[1]
        current.append(row)

    handle = observe(model, layer, hook)
    try:
        for module in model.modules():
            if hasattr(module, "gradient_probe"):
                module.gradient_probe = gradient_loss is not None
        for index, (x, sample_id) in enumerate(data.batches("calibration")):
            current.clear()
            last_batch = index >= count - last_batch_size
            if observer_batch_size:
                for module, enabled in progressive:
                    module.integer_enabled = enabled and last_batch
            if gradient_loss:
                with torch.enable_grad():
                    pred = differentiable_forward(backend, model, x.detach().requires_grad_(True))
                    require(torch.is_tensor(pred) and pred.ndim == 2 and pred.shape[1] == 1000,
                            "Hessian-guided 当前需要已验证 ImageNet 分类 head")
                    if gradient_loss == "classification_ce":
                        require(sample_id in data.labels, "缺少校准样本真实标签")
                        loss = F.cross_entropy(pred, torch.tensor([data.labels[sample_id]], device=pred.device))
                    else:
                        require(gradient_loss == "teacher_kl", "未知 Hessian loss")
                        teacher = forward(backend, backend.model, x).clone()
                        loss = F.kl_div(F.log_softmax(pred, -1), F.softmax(teacher, -1), reduction="batchmean")
                    require(current and all(r["y"].requires_grad for r in current), "位点输出不可微，不能伪造 Hessian 权重")
                    gradients = torch.autograd.grad(loss, [r["y"] for r in current])
                    for row, grad in zip(current, gradients):
                        row["gradient"] = grad
                    del loss, pred, gradients
            else:
                forward(backend, model, x)
            require(current, "校准 forward 未经过目标位点", code="coverage_mismatch")
            for row in current:
                size = sum(t.numel() * t.element_size() for t in row.values())
                used += size
                require(used <= limit_mib * 1024 ** 2, "逐位点校准缓存超过 calibration_cache_mib；不会截断样本", code="memory_budget")
                require(all(bool(torch.isfinite(t).all()) for t in row.values()), "校准数据/梯度非有限")
                records.append({k: v.detach().cpu().clone() for k, v in row.items()})
                record_ids.append(sample_id)
                last_batch_records += int(last_batch)
            current.clear()
        if gradient_loss:
            require(any(bool((r["gradient"] != 0).any()) for r in records),
                    "Hessian 梯度全零；请明确改用有标签 CE 或其它 metric，不会退回 MSE")
    finally:
        handle.remove()
        current.clear()
        for module, flag in gradient_flags:
            module.gradient_probe = flag
        for module, enabled in progressive:
            module.integer_enabled = enabled
        backend.reset()
    require(records, "校准没有样本")
    gradient_statistics = None
    if gradient_loss:
        # 只报告测量信号，不改 loss、梯度、候选顺序或搜索目标。
        elements = sum(r["gradient"].numel() for r in records)
        rms = (sum(r["gradient"].double().square().sum().item() for r in records) / elements) ** 0.5
        gradient_statistics = {"elements": elements, "rms": rms,
                               "max_abs": max(r["gradient"].abs().max().item() for r in records),
                               "note": "very small gradient: inspect score separation; this is not an accuracy guarantee"
                               if rms < 1e-10 else "measured gradient; no optimizer update"}
    return records, {"samples": count, "calls": len(records),
                     "record_sample_ids": record_ids,
                     "input_path": "quantized" if any(active.values()) else "FP",
                     "active_quantization_targets": active,
                     "observer_batch_size": observer_batch_size or count,
                     "last_batch_records": last_batch_records,
                     "cache_bytes": used, "gradient_loss": gradient_loss,
                     "gradient_statistics": gradient_statistics}


def combined_input(records):
    require(all(r["x"].shape[1:] == records[0]["x"].shape[1:] for r in records),
            "此静态校准器要求固定输入形状；不能静默合并不同 token 布局")
    return torch.cat([r["x"] for r in records], dim=0)


def gram(records, device):
    columns = records[0]["x"].shape[-1]
    h = torch.zeros(columns, columns, device=device, dtype=torch.float32)
    samples = 0
    for row in records:
        x = row["x"].to(device).float()
        count = x.shape[0] if x.ndim == 3 else 1
        h *= samples / (samples + count)
        samples += count
        x = x.reshape(-1, columns).t() * (2.0 / samples) ** 0.5
        h += x @ x.t()
    return h


def local_output(layer, module, x, row, weight=None):
    if layer["kind"] in WEIGHT_KINDS:
        w = (module.decoded_weight if module.weight_enabled else module.reference_weight) if weight is None else weight
        bias = module.effective_bias(w)
        # 局部搜索跟随输入设备；不能把 CUDA 校准强制拉回 CPU 做大矩阵乘。
        args = (x.float(), w.detach().to(x.device).float(), None if bias is None else bias.detach().to(x.device).float())
        return F.linear(*args) if layer["kind"] == "linear" else F.conv2d(*args, module.stride, module.padding, module.dilation, module.groups)
    if layer["kind"] in ("post_softmax", "attention_query"):
        return x @ row["rhs"]
    if layer["kind"] in ("attention_key", "attention_value"):
        return row["rhs"] @ x
    if layer["kind"] == "pre_layernorm":
        return F.layer_norm(x, module.normalized_shape,
                            None if module.weight is None else module.weight.detach().to(x.device),
                            None if module.bias is None else module.bias.detach().to(x.device), module.eps)
    return x


def reconstruction_score(layer, module, records, transform, metric, *, weight=False):
    total, elements = 0.0, 0
    for row in records:
        result = local_output(layer, module, row["x"] if weight else transform(row["x"]), row,
                              weight=transform(None) if weight else None)
        # 按元素数加权，避免不同 token/窗口大小改变样本权重；与原文局部 metric 同定义。
        count = row["y"].numel()
        total += score(row["y"], result, metric, row.get("gradient")) * count
        elements += count
    return total / elements
