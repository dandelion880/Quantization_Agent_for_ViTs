"""校准诊断的量化器输入误差；只累计标量，不保留激活张量或拟合新参数。"""

import torch

from .pipeline import PipelineLinear, quantizers


class ActivationDiagnostics:
    def __init__(self, model):
        self.rows, self.handles = {}, []
        for sid, module in quantizers(model).items():
            if not module.enabled:
                continue
            self.rows[sid] = {"elements": 0, "sse": 0.0, "energy": 0.0,
                              "negative_elements": 0, "negative_zeroed": 0,
                              "negative_sse": 0.0, "negative_energy": 0.0,
                              "nonzero_elements": 0, "nonzero_zeroed": 0}
            self.handles.append(module.register_forward_pre_hook(self._hook(sid)))

    def _hook(self, sid):
        def collect(module, args):
            # PipelineLinear 的 observer 输入是显式边仿射之后；Conv 使用标量1/0。
            x = args[0].detach()
            if isinstance(module, PipelineLinear):
                x = x / module.input_scale - module.input_shift
            with torch.no_grad():
                q = module.quantize(x)
                negative, nonzero = x < 0, x != 0
                error, energy = (q.double() - x.double()).square(), x.double().square()
                row = self.rows[sid]
                row["elements"] += x.numel()
                row["sse"] += error.sum().item()
                row["energy"] += energy.sum().item()
                row["negative_elements"] += negative.sum().item()
                row["negative_zeroed"] += (negative & (q == 0)).sum().item()
                row["negative_sse"] += error[negative].sum().item()
                row["negative_energy"] += energy[negative].sum().item()
                row["nonzero_elements"] += nonzero.sum().item()
                row["nonzero_zeroed"] += (nonzero & (q == 0)).sum().item()
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self):
        return [{"site": sid, **row,
                 "relative_mse": row["sse"] / max(row["energy"], 1e-30),
                 "negative_relative_mse": row["negative_sse"] / max(row["negative_energy"], 1e-30)
                 if row["negative_elements"] else None,
                 "negative_zero_fraction": row["negative_zeroed"] / row["negative_elements"]
                 if row["negative_elements"] else None,
                 "nonzero_zero_fraction": row["nonzero_zeroed"] / max(row["nonzero_elements"], 1)}
                for sid, row in self.rows.items()]


def frozen_ablation(backend, model, data, variants):
    """单因素旁路，不重新拟合；其它数值状态/开关必须保持冻结。"""
    from .model import forward
    from .evaluation import model_digest
    from .metrics import OutputErrors
    from ..schemas import require

    modules = quantizers(model)
    original, frozen = model_digest(backend.model), model_digest(model)
    flags = {sid: (m.enabled, getattr(m, "weight_enabled", None)) for sid, m in modules.items()}
    require(len({(v["site"], v["target"]) for v in variants}) == len(variants), "重复旁路项")
    for item in variants:
        require(item["site"] in modules, "旁路位点不在冻结候选中")
        module = modules[item["site"]]
        require(module.layer.get(item["target"]) is not None, "所选目标未量化")
        require(item["target"] == "activation" or isinstance(module, PipelineLinear), "此位点没有权重量化开关")
    summaries = []
    try:
        for item in [None, *variants]:
            module = modules[item["site"]] if item else None
            field = ("enabled" if item["target"] == "activation" else "weight_enabled") if item else None
            old = getattr(module, field) if module is not None else None
            if module is not None:
                require(old is True, "只能旁路正在启用的量化目标")
                setattr(module, field, False)
            errors = OutputErrors()
            totals = {"baseline_correct1": 0, "candidate_correct1": 0, "teacher_kl_sum": 0.0}
            count = 0
            try:
                for x, sample_id in data.batches("calibration"):
                    reference, candidate = errors.add(forward(backend, backend.model, x), forward(backend, model, x))
                    if data.labels:
                        a, b = reference["output"], candidate["output"]
                        totals["baseline_correct1"] += int(a.argmax(-1).item() == data.labels[sample_id])
                        totals["candidate_correct1"] += int(b.argmax(-1).item() == data.labels[sample_id])
                        totals["teacher_kl_sum"] += torch.nn.functional.kl_div(
                            b.log_softmax(-1), a.softmax(-1), reduction="batchmean").item()
                    count += 1
                require(count == len(data.ids("calibration")) and count > 0,
                        "旁路诊断采样未完整结束", code="evaluation_incomplete")
            finally:
                if module is not None:
                    setattr(module, field, old)
            require(model_digest(model) == frozen and model_digest(backend.model) == original,
                    "旁路诊断改变了数值状态", code="protection_failed")
            require(flags == {sid: (m.enabled, getattr(m, "weight_enabled", None)) for sid, m in modules.items()},
                    "旁路诊断未恢复量化开关", code="protection_failed")
            task = ({**totals, "samples": count, "top1": 100 * totals["candidate_correct1"] / count,
                     "baseline_top1": 100 * totals["baseline_correct1"] / count,
                     "teacher_kl": totals["teacher_kl_sum"] / count} if data.labels else None)
            summaries.append({"variant": item or {"target": "none", "site": "frozen_candidate"},
                              **errors.result(), "classification_diagnostic": task})
    finally:
        for sid, (activation, weight) in flags.items():
            modules[sid].enabled = activation
            if weight is not None:
                modules[sid].weight_enabled = weight
        backend.reset()
    return {"rows": summaries, "sample_ids": data.ids("calibration"),
            "all_numeric_state_unchanged": True, "flags_restored": True,
            "state_sha256": frozen, "no_recalibration": True,
            "scope": "one frozen W/A bypass at a time on calibration only; sensitivity bound, not a legal quantized candidate or validation accuracy"}
