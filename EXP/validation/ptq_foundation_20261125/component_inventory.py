"""Build the operation-to-source index from real exported plans, preserving the original primitive names."""

from collections import defaultdict
import json, re
from pathlib import Path
from audit_foundation import HERE, KNOW, ROOT, STATUS, read, save

status = read(STATUS)
prefix = HERE.relative_to(ROOT).as_posix()
observed = defaultdict(lambda: {"sources": set(), "requests": set(), "variants": set()})


def include(group, stage, entry, source, request, version):
    row = observed[(group, stage, entry)]
    row["sources"].add(source)
    row["requests"].add(request)
    row["variants"].add(version)


for row in status["methods"]:
    key = row["literature_id"]
    for call in row["source_requests"]:
        path, _, selector = call["receipt"].partition("#")
        receipt = read(ROOT / path)
        record = receipt[selector] if selector else receipt
        if path.endswith("source_request.json"):
            folder = (ROOT / path).parent / "model"
        else:
            folder = HERE / record.get("package", record.get("model_dir", ""))
        if not (folder / "manifest.json").is_file():
            raise ValueError((key, folder))
        manifest = read(folder / "manifest.json")
        plan = manifest["plan"]

        def add(group, stage, entry):
            include(group, stage, entry, key, call["receipt"], call["version"])

        for site in plan["layers"]:
            for operand in ("weight", "activation"):
                name = site.get(operand + "_method")
                if not name:
                    continue
                add(operand, name, "ptq_agent.execution.methods.get_target_method")
                calibration = site.get(operand + "_calibration", {}).get("strategy")
                if calibration:
                    add("calibration", calibration, "QuantizationExecutor.fit_target")
        for item in plan.get("preprocess", []):
            add("preprocess", item["method"], "QuantizationExecutor.preprocess")
        for item in plan.get("postfit", []):
            add("postfit", item["kind"], "QuantizationExecutor.prepare")
        for item in plan.get("structure", []):
            add("structure", item["kind"], "ptq_agent.execution.structure.configure")
            candidates = item.get("candidates")
            name = item.get("settings", {}).get("function")
            name = name or (
                candidates[0].get("function")
                if isinstance(candidates, list)
                and candidates
                and isinstance(candidates[0], dict)
                else None
            )
            if name:
                add("operator", name, "ptq_agent.execution.structure.configure")
        for item in plan.get("operators", []):
            add("operator", item["method"], "ptq_agent.execution.operators")
        req = record.get("request", record.get("source_request", record))
        if isinstance(req, dict) and req.get("allocation"):
            config = req["allocation"]
            add(
                "allocation",
                config.get(
                    "allocation_rule",
                    "configuration_grid"
                    if config.get("configuration_groups")
                    else "budget_search",
                ),
                "ptq_agent.execution.precision.select",
            )
        if key in ("clamp-vit", "psaq-vit", "dfq-vit", "sardfq"):
            add(
                "calibration_generator",
                {
                    "clamp-vit": "clamp_cyclic"
                    if "cyclic_images" in path
                    else "clamp_local_adapter",
                    "psaq-vit": "psaq_kde",
                    "dfq-vit": "e2h_crop",
                    "sardfq": "sardfq_gmm",
                }[key],
                "execution.calibration_generator",
            )

bindings = {
    "outlier.sparse_bypass": "encodings.apply",
    "rep.shift_log2": "encodings.shift_decode",
    "loss.adfq_module": "block_objectives.objective",
    "weight.mergeable_low_rank_correction": "block_objectives.merged_fit",
    "rep.learned_focus_interval": "region_grids.fit",
    "search.rank_allocation": "rank_search.search",
    "schedule.easy_to_hard_reconstruction": "block_objectives.merged_fit",
    "rep.tangent_companding": "encodings.apply",
    "aggregate.robust_channel_scale": "edge_transforms",
    "rep.bit_window_two_scale": "encodings.bit_window",
    "group.static_outlier_channels": "region_grids.fit",
    "search.shift_aligned_scales": "region_grids.fit",
    "pipeline.ptf_integer_ln": "operators.integer_layernorm",
    "pipeline.lis": "operators.SoftmaxSite",
    "observer.learned_dual_clip": "edge_transforms.bilateral_clip",
    "transform.repq_shared_grid": "QuantizationExecutor.preprocess",
    "schedule.activation_then_weight": "QuantizationExecutor.fit_model",
    "observer.row_percentile_select": "fixed_grids.first_batch_rows",
    "opt.scale_only_reconstruction": "reconstruction.fit_stage",
    "layout.fixed_token_scale": "fixed_grids",
    "transform.fixed_noise_shift": "QuantizationExecutor.preprocess",
    "transform.fold_shift_with_decoded_weight": "PipelineLinear.effective_bias",
    "group.instance_range_assignment": "encodings.fit_groups / encodings.apply",
    "search.group_count_budget": "source_allocation.group_schedule",
    "reduce.group_scaled_sum": "encodings.apply (分组反量化后的浮点矩阵乘，软件参考)",
    "init.distribution_scale_shift": "edge_transforms.domain_affine",
    "opt.learned_clip_and_weight_correction": "domain_fit.fit",
    "loss.domain_feature_alignment": "domain_fit.fit",
    "structure.register_candidate_bank": "structure.token_bank",
    "structure.cached_kv_prefix": "structure.configure",
    "structure.outlier_token_policy": "structure.configure",
}
aliases = {}
for r in status["methods"]:
    path = ROOT / "QAViT/docs" / r["read_asset"]["asset"]
    for label in re.findall(r"`([a-z][a-z0-9_]+\.[a-z][a-z0-9_.]+)`", path.read_text()):
        if label.endswith((".sh", ".py", ".json")):
            continue
        item = aliases.setdefault(
            label,
            {
                "uses": [],
                "implementation": bindings.get(label),
                "scope": "软件入口"
                if label in bindings
                else "文献研究问题或待独立测量的诊断",
                "source_asset": r["read_asset"],
            },
        )
        if r["literature_id"] not in item["uses"]:
            item["uses"].append(r["literature_id"])
out = {
    "schema": "qavit.ptq.component_evidence/1",
    "dependency_map": "08_METHOD_COMPONENT_MAP.md",
    "source_versions": "implementation_status.json",
    "observed": [
        {
            "kind": g,
            "name": n,
            "public_entry": ep,
            **{k: sorted(v) for k, v in vals.items()},
        }
        for (g, n, ep), vals in sorted(observed.items())
    ],
    "literature_identifiers": aliases,
}
save(KNOW / "component_status.json", out)
print(len(out["observed"]), len(aliases))
