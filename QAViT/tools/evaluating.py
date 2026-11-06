"""Evaluating Tools：任务指标、状态对照、文件大小与重载；无native收益推断。"""
from copy import deepcopy
from pathlib import Path

from runtime.files import read, file_hash


class DataView:
    def __init__(self, data, split, count=None, shard=None):
        self.data, self.split, self.labels = data, split, data.labels
        self.shape_observation_splits = getattr(data, "shape_observation_splits", ())
        ids = data.ids(split)
        if count is not None:
            if not 0 < count <= len(ids):
                raise ValueError("评价样本范围越界")
            ids = ids[:count]
        if shard is not None:
            ids = ids[shard[0]::shard[1]]
        self.selected = ids
        self.identity = deepcopy(data.identity)
        self.identity.update(selected_ids=ids, full_validation=split == "final" and ids == data.ids("final")
                             and data.identity.get("full_validation", False), shard=shard, requested_count=count)

    def ids(self, split):
        return self.selected if split == self.split else self.data.ids(split)

    def batches(self, split):
        if split != self.split:
            yield from self.data.batches(split)
            return
        selected = set(self.selected)
        for row in self.data.rows[split]:
            if row["sample_id"] in selected:
                yield self.data.backend.data_adapter.load_image(row).to(self.data.backend.device), row["sample_id"]


def evaluate(backend, cfg, data, model, plan, split, count=None, shard=None, *, native=True, sites=()):
    from ptq_agent.execution import evaluation as numerical
    view = DataView(data, split, count, shard)
    original = numerical.OutputErrors
    class ErrorsWithSums(original):
        def result(self):
            return {**super().result(), "raw_error_sums": deepcopy(self.sums)}
    # 只增加归约所需证据字段，不改旧Evaluator任何计算、顺序或检查。
    numerical.OutputErrors = ErrorsWithSums
    try:
        result = numerical.Evaluator(backend, cfg, view).evaluate(model, plan, split, native=native, sites=sites)
    finally:
        numerical.OutputErrors = original
    result["metrics"]["delta_top1_pp"] = (result["metrics"]["top1"] - result["metrics"]["baseline_top1"]
                                             if "top1" in result["metrics"] else None)
    result["used_for_search"] = True  # 用户允许完整测评反馈Lead，明确非隐藏测试集。
    return result


def aggregate(results):
    """独立逐图分片合并；样本集合严格验证，不平均分片百分比。"""
    from ptq_agent.execution.metrics import OutputErrors
    rows = [x["data"] for x in results]
    first = rows[0]
    for row in rows:
        if any(row[key] != first[key] for key in ("code_files", "proposal_id", "plan_key", "model_dir")):
            raise ValueError("分片代码/方案/模型身份不一致")
    evaluations = [row["evaluation"] for row in rows]
    split = evaluations[0]["split"]
    ids = [sid for evaluation in evaluations for sid in evaluation["sample_ids"]]
    expected = first["data_protocol"]["split_ids"][split]
    if len(ids) != len(set(ids)):
        raise ValueError("分片包含重复样本")
    selection = evaluations[0]["protocol"].get("requested_count")
    expected = expected if selection is None else expected[:selection]
    if set(ids) != set(expected):
        raise ValueError("分片未覆盖指定全部样本")
    errors = OutputErrors()
    for evaluation in evaluations:
        for key, sums in evaluation["metrics"]["raw_error_sums"].items():
            dest = errors.sums.setdefault(key, {field: 0 for field in sums})
            for field, value in sums.items():
                dest[field] = max(dest[field], value) if field == "max_abs_error" else dest[field] + value
    metrics = errors.result()
    counts = None
    if evaluations[0]["classification_counts"] is not None:
        counts = {k: sum(e["classification_counts"][k] for e in evaluations)
                  for k in evaluations[0]["classification_counts"]}
        metrics.update({k: 100 * v / len(ids) for k, v in counts.items()})
        metrics["delta_top1_pp"] = metrics["top1"] - metrics["baseline_top1"]
        metrics["top1_drop_pp"] = -metrics["delta_top1_pp"]
    full = split == "final" and first["data_protocol"].get("full_validation", False) and selection is None
    return {"status": "completed", "data": {k: v for k, v in first.items() if k != "evaluation"} | {
        "evaluation": {"status": "completed", "split": split, "sample_count": len(ids), "sample_ids": expected,
                       "classification_counts": counts, "metrics": metrics,
                       "full_dataset_evaluation": full, "used_for_search": True,
                       "health": {k: all(e["health"][k] for e in evaluations) for k in evaluations[0]["health"]},
                       "coverage": {k: sum(e["coverage"][k] for e in evaluations) for k in evaluations[0]["coverage"]},
                       "protocol": first["data_protocol"], "scope": "complete ImageNet validation 50000" if full else "subset",
                       "shard_timings": [e["timing"] for e in evaluations], "native_low_precision_kernel": False,
                       "latency_interpretation": "per-worker reference QDQ overhead, no native speedup claim"}},
        "cpu_seconds": sum(r["cpu_seconds"] for r in results),
        "worker_wall_seconds_sum": sum(r["elapsed_seconds"] for r in results),
        "peak_rss_kib_by_worker": [r["peak_rss_kib"] for r in results],
        "aggregation": "sum integer correct counts and FP64 error accumulators; same single-image forwards"}


def storage_report(backend, model_dir):
    from ptq_agent.execution.capabilities import estimate
    manifest = read(Path(model_dir) / "manifest.json")
    return {"theoretical": estimate(backend, manifest["plan"]),
            "file_bytes": sum(p.stat().st_size for p in Path(model_dir).rglob("*") if p.is_file()),
            "weights_sha256": file_hash(Path(model_dir) / "weights.pt"),
            "representation": manifest["representation"],
            "native_int4_latency_ms": None, "native_int4_speedup": None,
            "native_status": "unsupported: no deployment kernel; QDQ timings are reference overhead only"}
