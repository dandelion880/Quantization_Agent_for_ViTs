"""Verify the portable source/evidence joins and the local exported models used for this integration."""

from pathlib import Path
import gzip, hashlib, json, re, sys, subprocess
from audit_foundation import HERE, ROOT, KNOW, STATUS, read, save, intake, catalog, CAT


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    logs = HERE / "verification_logs"
    logs.mkdir(exist_ok=True)
    for arg in sys.argv[1:]:
        label, raw = arg.split("=", 1)
        p = Path(raw)
        assert p.is_file() and re.fullmatch("[a-z0-9_]+", label)
        (logs / (label + ".log.gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
    raw_suite = gzip.decompress((logs / "regression.log.gz").read_bytes()).decode()
    assert raw_suite.rstrip().endswith("\nOK")
    count, duration = re.search(r"Ran (\d+) tests in ([\d.]+)s", raw_suite).groups()
    ingest, ledger = intake()
    ix, status = catalog(ledger)
    assert ix == read(CAT) and status == read(STATUS)
    references = {
        source["id"]
        for source in ledger
        if source["coverage_category"] == "excluded_qat"
    }
    assert not references & {r.get("literature_id") for r in ix["methods"]}
    num = read(HERE / "numerical/summary.json")
    cold = read(HERE / "numerical/process_cold.json")
    assert num.keys() == cold.keys()
    checked = {}
    for k, r in num.items():
        assert r["status"] == "passed" and r["cold_max_abs"] == 0
        assert (
            cold[k]["state_sha256"] == r["state_sha256"]
            and cold[k]["heldout_output_sha256"] == r["output_sha256"]
        )
        p = HERE / "numerical" / k / "model"
        m = read(p / "manifest.json")
        assert digest(p / "weights.pt") == m["weights_sha256"] == r["weights_sha256"]
        checked[k] = m["weights_sha256"]
    source_models = set()
    for row in status["methods"]:
        for c in row["source_requests"]:
            path, _, key = c["receipt"].partition("#")
            src = ROOT / path
            d = read(src)
            if key:
                d = d[key]
            p = (
                src.parent / "model"
                if src.name == "source_request.json"
                else HERE / d.get("package", d.get("model_dir", ""))
            )
            if p not in source_models:
                assert (
                    digest(p / "weights.pt")
                    == read(p / "manifest.json")["weights_sha256"]
                )
                source_models.add(p)
    service = HERE / "service"
    worker = {}
    for mode in ("prepare", "reuse", "child"):
        job = service / mode
        r = read(job / "receipt.json")
        e = r["observations"]["evaluations"]["classification"]
        g = read(job / "gradient_audit.json")
        assert g == r["provenance"]["gradient_audit"]
        entry = {
            "request": "service/" + mode + "/request.json",
            "request_sha256": digest(job / "request.json"),
            "model_ref": r["model_ref"],
            "sample_ids": e["sample_ids"],
            "stable_metrics": {
                k: e["metrics"][k] for k in ("mse", "cosine_error", "top1", "top5")
            },
            "gradient_audit": g,
            "frozen_evaluation_sha256": digest(job / "frozen_evaluation.json"),
        }
        if mode != "reuse":
            manifest = read(job / "model/manifest.json")
            assert digest(job / "model/weights.pt") == manifest["weights_sha256"]
            entry.update(
                weights_sha256=manifest["weights_sha256"],
                construction=manifest["construction"],
                ancestry=manifest.get("preparation_ancestry"),
                refinement=manifest.get("refinement"),
                retained_plan=manifest.get("recipe_binding"),
            )
        worker[mode] = entry
    p, r, c = worker["prepare"], worker["reuse"], worker["child"]
    assert (
        p["model_ref"] == r["model_ref"]
        and p["stable_metrics"] == r["stable_metrics"]
        and p["sample_ids"] == r["sample_ids"]
    )
    assert p["gradient_audit"]["optimizer_steps"] == 2 and all(
        v == 0 for m in (r, c) for v in m["gradient_audit"].values()
    )
    assert c["model_ref"] != p["model_ref"] and c["construction"] == p["construction"]
    comparison = read(service / "child/comparison.json")
    scores = read(service / "child/scores.json")
    assert comparison["parent"] == p["model_ref"] and len(scores["candidates"]) == 2
    save(
        HERE / "service_evidence.json",
        {
            "fixture": "预训练分类模型与真实 ImageNet 原图；局部覆盖扩展，L2 有界验证",
            "stages": worker,
            "frozen_comparison": {
                "selected": comparison["selected"],
                "candidates": {
                    row["id"]: row["scores"] for row in scores["candidates"]
                },
            },
        },
    )
    audits = []
    for f in sorted((HERE / "role_calls").glob("*/*/llm_audit.json")):
        a = read(f)
        audits.append(
            {
                "role": a["role"],
                "provider": a["provider"],
                "request_sha256": a["request_sha256"],
                "exit_code": a["exit_code"],
                "response_present": bool(a["response"]),
                "audit_sha256": digest(f),
                "path": str(f.relative_to(HERE)),
            }
        )
    memory = read(HERE / "memory_restore.json")
    assert (
        memory["recorder_ref"].endswith("@1")
        and memory["evidence_ref"].endswith("@1")
        and memory["content_sha256"]
    )
    save(
        HERE / "role_evidence.json",
        {
            "calls": audits,
            "static_read": read(HERE / "generation.json")[0],
            "evidence": memory["evidence_ref"],
            "recorder": memory["recorder_ref"],
            "restored_digest": memory["content_sha256"],
        },
    )
    primary = read(HERE / "primary_protocol.json")
    alloc = read(HERE / "allocation_recipes.json")
    quality = read(HERE / "selection_quality.json")
    assert (
        not set(quality["calibration_ids"]) & set(quality["heldout_ids"])
        and quality["result"]["bootstrap"]["resamples"] == 64
    )
    optional_dense = read(HERE / "per_original_dense.json")
    assert len(optional_dense["originals"]) == 1
    verify = {
        "scope": "实现及有限输入的集成验证",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "suite": {
            "tests": int(count),
            "seconds": float(duration),
            "result": "OK",
            "log": "verification_logs/regression.log.gz",
        },
        "source": {
            "material_members": ingest["payload_members_verified"],
            "source_records": len(ledger),
            "source_revisions_measured": len(status["methods"]),
            "catalog_ids": len(ix["methods"]),
            "named_component_entries": len(
                read(KNOW / "component_status.json")["observed"]
            ),
        },
        "execution": {
            "numerical_cases": len(num),
            "separate_process_matches": len(cold),
            "distinct_source_bound_packages_verified": len(source_models),
            "author_graphs": sorted(read(HERE / "native_recipes.json")),
            "allocation_versions": sorted(alloc),
            "strict_static_covered_boundaries": len(primary["covered_boundaries"]),
            "strict_static_optimizer_steps": primary["construction"]["max_steps"],
            "service_steps": ["prepare", "reuse", "child"],
        },
        "independent_checks": {
            "wide_bit_pack_and_signed_requant": "edge_oracles.json",
            "real_deployed_nonlinear_reductions": "nonlinear_oracles.json",
            "paired_ridge_and_allocation": "component_checks.json",
            "dense_original": "per_original_dense.json",
        },
        "frozen_selector": {
            "proxy": quality["proxy_field"],
            "independent_task": quality["task_field"],
            "calibration_images": len(quality["calibration_ids"]),
            "heldout_images": len(quality["heldout_ids"]),
            "results": "selection_quality.json",
        },
        "research": {
            "role_receipt": "role_evidence.json",
            "evidence_ref": memory["evidence_ref"],
            "restored": memory["content_sha256"],
        },
        "logs": {p.name: digest(p) for p in sorted(logs.iterdir())},
    }
    save(HERE / "verification.json", verify)
    print(
        verify["suite"], verify["execution"]["separate_process_matches"], "cold models"
    )


if __name__ == "__main__":
    main()
