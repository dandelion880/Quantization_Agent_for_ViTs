"""实验输入的数据回执：只读已有配置/清单及校验身份，不加载模型或选择科学参数。"""
from copy import deepcopy
from pathlib import Path

from core.cards import fingerprint
from core.journal import file_sha256, read_json
from runtime.paths import ROOT


def data_receipt(brief, execution=None, *, evidence=()):
    execution = execution or {}
    adapter = execution.get("backend", brief["model"].get("adapter", {}))
    result = {"source": "ResearchBrief.model.adapter + backend config + explicit backend_overrides",
              "available_data": [], "selected_data": {k: deepcopy(execution.get(k)) for k in (
                  "calibration", "evaluation", "evaluations")},
              "selection_source": "experiment execution" if execution else "尚未提交本实验数据选择",
              "unknowns": [], "checks": {}, "prior_validation": None,
              "meaning": "available_data 是现有数据池，不是本实验样本数。selected_data 仅转述已声明视图；"
                         "整个视图为 null 表示未声明；视图内显式 count=null 沿原接口表示完整或剩余数据范围。"
                         "可用容量为 null 表示未知，不从池大小补科学参数。此回执不执行实验，不代表图像/模型已通过运行时校验。"}
    if not isinstance(adapter, dict):
        result["unknowns"].append("backend/ResearchBrief.model.adapter 不是配置对象，数据来源与可用范围未知。")
        return result
    if execution.get("service") == "profile":
        result["selected_data"]["profile_arguments"] = {
            k: execution.get("arguments", {}).get(k) for k in ("count", "offset", "seed")}
    try:
        from ptq_agent.configuration import load_config
        from ptq_agent.schemas import PTQError
        config_path = adapter.get("config_asset", {}).get("path") or adapter.get("config_path")
        base = ROOT
        if config_path:
            path = Path(config_path)
            path = (path if path.is_absolute() else ROOT / path).resolve()
            cfg = load_config(path)
            base = path.parent
            config_sha = file_sha256(path)
            bound_sha = adapter.get("config_asset", {}).get("sha256")
            result["config_source"] = {"path": str(path), "sha256": config_sha, "frozen_sha256": bound_sha}
            result["checks"]["frozen_config_matches"] = config_sha == bound_sha if bound_sha else None
            if bound_sha and config_sha != bound_sha:
                result["unknowns"].append("backend.config_asset.sha256 与当前配置不符；冻结配置已变化，不能确认可用范围。")
        else:
            cfg = deepcopy(adapter.get("config", {}))
        backend = deepcopy(cfg.get("backend", {}))
        backend.update(deepcopy(adapter.get("backend_overrides", {})))
        counts = {s: backend.get(s + "_samples") for s in ("calibration", "search", "final")}
        result["configured_pool_counts"] = counts
        result["ordering_seeds"] = deepcopy(backend.get("data_seeds"))
        if adapter.get("kind") in ("fixture", "factory"):
            # 自定义 factory 可能完全不使用继承配置中的真实数据路径，不能据此宣称可用。
            result["unknowns"].append("自定义数据提供器未执行：dataset、split 实际容量及身份未知；配置池数量仅为声明。")
        elif not backend.get("data_path"):
            result["unknowns"].append("backend.data_path 未声明，数据集与可用范围未知。")
        else:
            path = Path(backend["data_path"])
            path = (path if path.is_absolute() else base / path).resolve()
            manifest = read_json(path)
            actual = file_sha256(path)
            expected = backend.get("data_sha256")
            result["manifest_source"] = {"path": str(path), "sha256": actual, "expected_sha256": expected}
            known = manifest.get("schema") in ("imagefolder_manifest_v1", "imagefolder_manifest_v2")
            matched = bool(expected) and actual == expected
            result["checks"].update(manifest_sha256_matches_config=matched,
                                    schema_matches_config=known and manifest.get("schema") == backend.get("data_protocol"),
                                    image_contents_rechecked=False)
            if not matched:
                result["unknowns"].append("backend.data_sha256 缺失或与清单不符；不能确认可用范围。")
            if not known or manifest.get("schema") != backend.get("data_protocol"):
                result["unknowns"].append("清单 schema 未被当前数据协议确认；dataset 与可用范围未知。")
            result["identity_summary"] = {"schema": manifest.get("schema"),
                "preprocessing_sha256": fingerprint(manifest["preprocessing"]) if "preprocessing" in manifest else None,
                "class_mapping_sha256": fingerprint(manifest["class_to_idx"]) if "class_to_idx" in manifest else None}
            for split, configured in counts.items():
                rows = manifest.get("splits", {}).get(split)
                total = len(rows) if isinstance(rows, list) else None
                consistent = type(configured) is int and configured > 0 and total == configured
                available = total if (consistent and matched and result["checks"]["schema_matches_config"]
                                      and result["checks"].get("frozen_config_matches") is not False) else None
                sources = sorted({r["dataset_split"] for r in rows if r.get("dataset_split")}) if isinstance(rows, list) else []
                result["available_data"].append({"dataset": "ImageNet-1k" if known else None, "split": split,
                    "source_splits": sources, "configured_samples": configured, "manifest_samples": total,
                    "available_samples": available, "index_range": [0, available] if available is not None else None,
                    "range_convention": "zero-based, end exclusive; runtime ordering_seeds apply",
                    "source": str(path), "counts_match": consistent})
                if not consistent:
                    result["unknowns"].append(f"backend.{split}_samples={configured!r} 与清单 splits.{split} 数量 {total!r} 不一致；可用范围未知。")
            # 只复用本次运行与当前清单、数量和顺序一致的已提交校验身份；不导入旧科研 Memory。
            for card in reversed(evidence):
                payload = card["payload"]
                provenance = payload.get("provenance", {})
                identity = provenance.get("data_identity", {})
                if (matched and not result["unknowns"] and payload.get("status") == "completed" and payload.get("measurement_source") == "real"
                        and provenance.get("model_identity", {}).get("data", {}).get("manifest_sha256") == actual
                        and identity.get("counts") == counts and identity.get("ordering_seeds") == backend.get("data_seeds")):
                    result["prior_validation"] = {"evidence_ref": card["id"], "experiment_ref": payload["experiment_ref"],
                        "data_identity_sha256": fingerprint(identity), "full_validation": identity.get("full_validation"),
                        "scope": "已有执行的数据身份校验；本次展示没有重读图像或运行模型"}
                    break
    except (OSError, ValueError, KeyError, TypeError, PTQError) as exc:
        result["unknowns"].append("资产配置/清单读取或校验失败，可用数据未知：" + str(exc))
    evaluations = execution.get("evaluations", brief.get("evaluation_protocol", {}).get("evaluations", []))
    if not isinstance(evaluations, list) or any(not isinstance(row, dict) for row in evaluations):
        result["unknowns"].append("evaluations 不是结构化评测列表；不据此推断额外数据来源。")
        evaluations = []
    if any(row.get("kind") == "ade_frozen" for row in evaluations or []):
        # 使用评测器现有 SETTINGS，不建立第二份 ADE 配置或把适配器预期数量当成实际容量。
        try:
            from tools.segmentation import SETTINGS
            settings = read_json(SETTINGS)
            root = Path(settings["data_root"])
            images = sorted((root / "images/validation").glob("*.jpg"))
            paired = bool(images) and all((root / "annotations/validation" / (p.stem + ".png")).is_file() for p in images)
            result["available_data"].append({"dataset": "ADE20K", "split": "ade_validation",
                "available_samples": len(images) if paired else None, "index_range": [0, len(images)] if paired else None,
                "source": {"path": str(SETTINGS), "sha256": file_sha256(SETTINGS), "data_root": str(root)},
                "identity_summary": {"ordered_paths_sha256": fingerprint([str(p) for p in images]),
                                     "pipeline_sha256": fingerprint(settings["pipeline"])},
                "checks": "仅列举图像路径及对应标注是否存在；未读取内容、核验 head 或执行数值验证。"})
            if not paired:
                result["unknowns"].append("ADE20K 图像/标注目录缺失或不成对，可用范围未知。")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["unknowns"].append("ADE 配置读取失败，可用范围未知：" + str(exc))
    return result
