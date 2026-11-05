"""固定数据协议与评价门禁；search 可反馈，final 冻结后仅执行一次。"""

from pathlib import Path
import math
import time
import numpy as np
import torch

from ..schemas import fingerprint, require
from .model import forward, tensor_digest, QuantizedLinear
from .sites import coverage
from .plugin_contracts import site_id
from .metrics import ERROR_REDUCTION, OutputErrors, tensors


class VerifiedManifestData:
    """复用已验证图像和预处理，三个分区按样本身份互斥；不改写数据集。"""

    def __init__(self, backend, config):
        self.backend = backend
        counts = {s: config["backend"][s + "_samples"] for s in ("calibration", "search", "final")}
        require(
            sum(counts.values()) <= len(backend.rows),
            "已验证输入池不足以建立互斥的 calibration/search/final 分区",
        )
        rows = backend.rows
        require(len({r["sample_id"] for r in rows}) == len(rows), "输入池样本身份重复")
        self.rows = {}
        declared = getattr(backend.data_adapter, "splits", None)
        if declared is not None:
            require({s: len(v) for s, v in declared.items()} == counts,
                    "配置样本数必须与显式清单完全一致，禁止截断或重分区")
        offset = 0
        for split, count in counts.items():
            pool = declared[split] if declared is not None else rows[offset : offset + count]
            order = np.random.default_rng(config["backend"]["data_seeds"][split]).permutation(count)
            self.rows[split] = [pool[int(i)] for i in order]
            offset += count
        self.labels = {}
        self.label_mapping = None
        head_available = (backend.spec.get("classification_head") == "official pretrained ImageNet-1k head retained"
                          or bool(getattr(backend, "task_head", None)))
        if config["loop"]["evaluation"] == "classification":
            require(
                head_available,
                "当前模型没有已验证分类 head；不能将特征相似度当作分类精度",
                code="evaluation_not_configured",
            )
        if head_available:
            mapping = getattr(backend.data_adapter, "class_to_idx", None)
            if mapping is None:
                roots = {Path(r["path"]).parent.parent for r in rows}
                require(len(roots) == 1, "分类图像目录不一致")
                classes = sorted(p.name for p in next(iter(roots)).iterdir() if p.is_dir())
                require(len(classes) == 1000 and all(n.startswith("n") for n in classes),
                        "缺少 ImageNet 1000 类 ImageFolder 标签映射")
                mapping = {n: i for i, n in enumerate(classes)}
            if getattr(backend, "head_class_mapping", None) is not None:
                require(mapping == backend.head_class_mapping, "冻结分类头与数据标签映射不一致", code="asset_mismatch")
            self.labels = {r["sample_id"]: mapping[Path(r["path"]).parent.name] for r in rows}
            self.label_mapping = fingerprint(mapping)
        self.identity = {
            "split_ids": {s: self.ids(s) for s in counts},
            "counts": counts,
            "ordering_seeds": dict(config["backend"]["data_seeds"]),
            "preprocessing": getattr(backend.data_adapter, "PREPROCESS", "injected provider"),
            "label_mapping_sha256": self.label_mapping,
            "batch_size": 1,
            "scope": ("disjoint train calibration/search; complete ImageNet validation final"
                      if getattr(backend.data_adapter, "full_validation", False)
                      else "disjoint subset of verified manifest; not full dataset evaluation"),
            "full_validation": getattr(backend.data_adapter, "full_validation", False),
            "feature_error_reduction": ERROR_REDUCTION,
        }

    def ids(self, split):
        return [r["sample_id"] for r in self.rows[split]]

    def batches(self, split):
        for row in self.rows[split]:
            yield self.backend.data_adapter.load_image(row).to(self.backend.device), row["sample_id"]


def model_digest(model):
    return fingerprint(
        {
            name: tensor_digest(value)
            for name, value in list(model.named_parameters()) + list(model.named_buffers())
            if value is not None
        }
    )


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def latency(backend, model, x, warmup, repeats):
    for _ in range(warmup):
        forward(backend, model, x)
    synchronize(backend.device)
    values = []
    for _ in range(repeats):
        begin = time.perf_counter()
        forward(backend, model, x)
        synchronize(backend.device)
        values.append((time.perf_counter() - begin) * 1000)
    return {
        "samples_ms": values,
        "mean_ms": sum(values) / len(values),
        "warmup": warmup,
        "repeats": repeats,
        "batch_size": int(x.shape[0]),
        "scope": "synchronized forward; excludes disk read and H2D",
    }


class Evaluator:
    def __init__(self, backend, config, data):
        self.backend, self.options, self.data = backend, config["loop"], data

    def evaluate(self, model, plan, split, *, native=True, sites=()):
        if native:
            return self._evaluate(model, plan, split, native=True)
        # A custom implementation need not install our quantizer wrapper classes.
        # Record observable module visits without claiming they prove its math.
        observed, handles = {}, []
        modules = dict(model.named_modules())
        try:
            for row in sites:
                sid = row["site_id"]
                if row["module"] in modules:
                    observed[sid] = 0
                    def visit(_module, _inputs, sid=sid):
                        observed[sid] += 1
                    handles.append(modules[row["module"]].register_forward_pre_hook(visit))
            return self._evaluate(model, plan, split, native=False, observed=observed)
        finally:
            for handle in handles:
                handle.remove()

    def _evaluate(self, model, plan, split, *, native, observed=None):
        require(split in ("search", "final"), "评价分区无效")
        before = model_digest(model)
        baseline_before = model_digest(self.backend.model)
        for m in model.modules():
            if native and (isinstance(m, QuantizedLinear) or hasattr(m, "ptq_site_id")):
                m.calls = 0
        errors = OutputErrors()
        top = {"baseline_top1": 0, "baseline_top5": 0, "top1": 0, "top5": 0}
        ids = []
        cuda = str(self.backend.device).startswith("cuda")
        memory_before = torch.cuda.memory_allocated(self.backend.device) if cuda else None
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.backend.device)
        first = None
        for x, sample_id in self.data.batches(split):
            if first is None:
                first = x
            reference, candidate = errors.add(forward(self.backend, self.backend.model, x),
                                              forward(self.backend, model, x))
            if self.data.labels:
                require(
                    list(reference) == ["output"] and reference["output"].shape == (1, 1000),
                    "分类输出必须为 [1,1000]",
                )
                label = self.data.labels[sample_id]
                for prefix, values in (("baseline_", reference), ("", candidate)):
                    indices = values["output"].topk(5, dim=-1).indices[0].tolist()
                    top[prefix + "top1"] += label == indices[0]
                    top[prefix + "top5"] += label in indices
            ids.append(sample_id)
            if len(ids) % 100 == 0:
                print(f"evaluation {split}: {len(ids)}/{len(self.data.ids(split))}; "
                      f"FP correct1={top['baseline_top1']}, Q correct1={top['top1']}", flush=True)
        require(ids == self.data.ids(split), "评价样本未完整结束", code="evaluation_incomplete")
        observed_coverage = coverage(model) if native else dict(observed)
        if native:
            require(
                set(observed_coverage) == {site_id(l) for l in plan["layers"]} and all(n > 0 for n in observed_coverage.values()),
                "候选实际 forward 未覆盖量化位点",
                code="coverage_mismatch",
            )
        candidate_timing = latency(
            self.backend, model, first, self.options["warmup"], self.options["repeats"]
        )
        baseline_timing = latency(
            self.backend, self.backend.model, first, self.options["warmup"], self.options["repeats"]
        )
        require(
            before == model_digest(model) and baseline_before == model_digest(self.backend.model),
            "评价修改了权重/量化参数/buffer",
            code="protection_failed",
        )
        self.backend.reset()
        metrics = {
            **errors.result(),
            "native_latency_ms": None,
            # 此 evaluator 同时持有 FP 和 QDQ 模型，不能提供部署显存。
            "deployment_peak_memory_mib": None,
            "latency_ms": candidate_timing["mean_ms"],
            "baseline_latency_ms": baseline_timing["mean_ms"],
            "measured_speedup": baseline_timing["mean_ms"] / candidate_timing["mean_ms"],
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(self.backend.device) / 1024**2 if cuda else None
            ),
            "cuda_increment_mib": (
                (torch.cuda.max_memory_allocated(self.backend.device) - memory_before) / 1024**2
                if cuda
                else None
            ),
        }
        if self.data.labels:
            metrics.update({k: 100 * v / len(ids) for k, v in top.items()})
            metrics["top1_drop_pp"] = metrics["baseline_top1"] - metrics["top1"]
        return {
            "status": "completed",
            "split": split,
            "evaluation": self.options["evaluation"],
            "sample_count": len(ids),
            "classification_counts": top if self.data.labels else None,
            "sample_ids": ids,
            "metrics": metrics,
            "coverage": observed_coverage,
            "coverage_scope": "native_quantizer_calls" if native else "observable_module_calls; custom algorithm reviewed in source",
            "health": {
                "finite_outputs": True,
                "matching_output_shapes": True,
                "quantization_state_unchanged": True,
                "original_state_unchanged": True,
            },
            "timing": {"candidate": candidate_timing, "baseline": baseline_timing},
            "memory_scope": "whole worker including immutable FP baseline and candidate; not isolated deployment memory",
            "execution_mode": plan.get("execution", {}).get("mode") if native else "approved_custom_executor",
            "native_low_precision_kernel": False,
            "latency_interpretation": "reference overhead diagnostic; not a native low-bit acceleration result" if not native or plan["execution"]["mode"] == "reference_qdq" else "floating-point kernel timing",
            "native_float_matmul": native and plan["execution"]["mode"] == "native",
            "protocol": self.data.identity,
            "task_accuracy_measured": bool(self.data.labels),
            "used_for_search": split == "search",
            "full_dataset_evaluation": split == "final" and self.data.identity.get("full_validation", False),
            "scope": ("complete ImageNet validation 50000" if split == "final" and self.data.identity.get("full_validation", False)
                      else "subset; feature fidelity is not task accuracy"),
        }

    def gate(self, result, weight_ratio):
        m = result["metrics"]
        limits = self.options
        metric, limit = (
            ("relative_mse", limits["max_relative_mse"])
            if limits["evaluation"] == "feature_fidelity"
            else ("top1_drop_pp", limits["max_top1_drop_pp"])
        )
        checks = {}
        if limits.get("min_top1") is not None:
            checks["min_top1"] = {"value": m.get("top1"), "minimum": limits["min_top1"]}
        if limit is not None:
            checks[metric] = {"value": m.get(metric), "maximum": limit}
        if limits.get("search_max_top1_drop_pp") is not None and result.get("split") == "search":
            # train搜索绝对分数通常偏高；这是事前声明的选优代理，不是val估计。
            checks["search_top1_drop_pp"] = {"value": m.get("top1_drop_pp"),
                                            "maximum": limits["search_max_top1_drop_pp"]}
        if limits["max_weight_bytes_ratio"] is not None:
            checks["weight_bytes_ratio"] = {"value": weight_ratio, "maximum": limits["max_weight_bytes_ratio"]}
        if limits.get("selection") == "best_accuracy":
            # 无预设降幅时只验证精度确实测得；按 Top-1 排序由 loop 负责。
            checks["top1_measurement"] = {"value": m.get("top1"), "maximum": 100.0}
        outputs = m.get("outputs", {})
        if limits["evaluation"] == "feature_fidelity":
            # 明列各非空输出的门槛结果，失败原因可直接进入 dynamic prompt。
            require(outputs and m.get("relative_mse_reduction") == ERROR_REDUCTION,
                    "缺少逐输出误差或使用了旧归约协议", code="evaluation_protocol_mismatch")
            for key, output in outputs.items():
                if output["elements"]:
                    checks["output_relative_mse:" + key] = {
                        "value": output["relative_mse"], "maximum": limits["max_relative_mse"]}
        # 显式逐输出限制是附加硬门槛，缺少输出/空输出同样失败，不能被总体指标替代。
        for key, maximum in limits.get("max_output_relative_mse", {}).items():
            name = "output_relative_mse:" + key
            previous = checks.get(name, {}).get("maximum", maximum)
            checks[name] = {"value": outputs.get(key, {}).get("relative_mse"),
                            "maximum": min(previous, maximum)}
        for name, field in (
            ("native_latency_ms" if result["execution_mode"] == "reference_qdq" else "latency_ms", "max_latency_ms"),
            ("deployment_peak_memory_mib", "max_peak_memory_mib"),
        ):
            if limits[field] is not None:
                checks[name] = {"value": m.get(name), "maximum": limits[field]}
        for item in checks.values():
            item["passed"] = (item["value"] is not None and math.isfinite(item["value"])
                              and item["value"] <= item.get("maximum", float("inf"))
                              and item["value"] >= item.get("minimum", -float("inf")))
        return {
            "passed": all(item["passed"] for item in checks.values()),
            "checks": checks,
            "criterion_scope": limits["evaluation"],
            "accuracy_threshold_specified": limit is not None or limits.get("min_top1") is not None,
            "full_dataset_evaluation": result.get("full_dataset_evaluation", False),
        }

    def reload_check(self, first, second, split="search"):
        x, sample_id = next(self.data.batches(split))
        a, b = tensors(forward(self.backend, first, x)), tensors(forward(self.backend, second, x))
        require(
            set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a),
            "导出重载输出不一致",
            code="reload_mismatch",
        )
        return {
            "passed": True,
            "comparison": "torch.equal for all floating outputs",
            "samples": 1,
            "split": split,
            "sample_ids": [sample_id],
        }
