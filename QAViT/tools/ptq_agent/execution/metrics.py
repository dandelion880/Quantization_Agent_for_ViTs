"""模型输出误差的统一归约；评价、候选诊断及 FP 等价门禁共用。"""

import math
import torch

from ..schemas import require

ERROR_REDUCTION = "max_per_output_relative_mse_v1"


def tensors(output, prefix="output"):
    """保留输出名字，不能把字典按插入顺序配对或把 CLS 混入 patch。"""
    if torch.is_tensor(output):
        return {prefix: output} if output.is_floating_point() else {}
    result = {}
    if isinstance(output, dict):
        for key, value in sorted(output.items()):
            result.update(tensors(value, prefix + "." + key))
    elif isinstance(output, (tuple, list)):
        for i, value in enumerate(output):
            result.update(tensors(value, prefix + "." + str(i)))
    return result


def relative_error(sse, energy):
    # 零参考且精确相等应为0；零参考但有误差无法定义相对值，必须门禁失败。
    value = sse / energy if energy else (0.0 if sse == 0 else None)
    return value if value is None or math.isfinite(value) else None


class OutputErrors:
    def __init__(self):
        self.sums = {}

    def add(self, reference, candidate):
        a, b = tensors(reference), tensors(candidate)
        require(a and set(a) == set(b), "候选输出结构不同", code="unhealthy_model")
        require(not self.sums or set(a) == set(self.sums), "输出字段在样本间改变", code="unhealthy_model")
        for key, ref in a.items():
            value = b[key]
            require(ref.shape == value.shape and bool(torch.isfinite(ref).all())
                    and bool(torch.isfinite(value).all()),
                    f"输出形状/有限值检查失败：{key}", code="unhealthy_model")
            row = self.sums.setdefault(key, dict(sse=0.0, energy=0.0, candidate_energy=0.0,
                                                 dot=0.0, elements=0, max_abs_error=0.0))
            # DINOv2 的空 register-token 输出仍检查结构，但不参与数值归约。
            if ref.numel() == 0:
                continue
            ref, value = ref.detach().double(), value.detach().double()
            delta = value - ref
            row["sse"] += delta.square().sum().item()
            row["energy"] += ref.square().sum().item()
            row["candidate_energy"] += value.square().sum().item()
            row["dot"] += (ref * value).sum().item()
            row["elements"] += ref.numel()
            row["max_abs_error"] = max(row["max_abs_error"], delta.abs().max().item())
        return a, b

    def result(self):
        outputs = {}
        for key, row in self.sums.items():
            require(all(math.isfinite(v) for v in row.values()), "误差统计溢出", code="unhealthy_model")
            norm = math.sqrt(row["energy"]) * math.sqrt(row["candidate_energy"])
            outputs[key] = {
                "mse": row["sse"] / row["elements"] if row["elements"] else None,
                "relative_mse": relative_error(row["sse"], row["energy"]) if row["elements"] else None,
                "cosine": max(-1.0, min(1.0, row["dot"] / norm)) if norm else None,
                "max_abs_error": row["max_abs_error"] if row["elements"] else None,
                "elements": row["elements"],
            }
        active = {k: v for k, v in outputs.items() if v["elements"]}
        require(active, "模型没有可评价的非空浮点输出", code="unhealthy_model")
        undefined = [k for k, v in active.items() if v["relative_mse"] is None]
        worst = undefined[0] if undefined else max(active, key=lambda k: active[k]["relative_mse"])
        sse, energy, candidate_energy, dot, elements = (
            sum(row[k] for row in self.sums.values())
            for k in ("sse", "energy", "candidate_energy", "dot", "elements"))
        require(all(math.isfinite(v) for v in (sse, energy, candidate_energy, dot)),
                "总误差统计溢出", code="unhealthy_model")
        norm = math.sqrt(energy) * math.sqrt(candidate_energy)
        return {
            # 每个输出内按整个固定分区累计 SSE/能量；输出之间取最大值。
            # x_prenorm 的高能量因此不能掩盖 x_norm_clstoken/patchtokens 的退化。
            "relative_mse": active[worst]["relative_mse"],
            "relative_mse_reduction": ERROR_REDUCTION,
            "worst_output": worst,
            "undefined_relative_mse_outputs": undefined,
            "aggregate_relative_mse": relative_error(sse, energy),
            "mse": sse / elements,
            "cosine_error": 1 - max(-1.0, min(1.0, dot / norm)) if norm else (0.0 if sse == 0 else 1.0),
            "outputs": outputs,
        }
