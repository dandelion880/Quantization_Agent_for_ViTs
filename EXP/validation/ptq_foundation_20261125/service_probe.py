"""Two independent worker processes, one bounded pixel fit and one reuse of the identical exported package."""

import sys
from pathlib import Path
from runtime.paths import ROOT
from core.journal import read_json, atomic_json
from core.ptq_protocol import reusable_manifest
from core.cards import fingerprint
from tools.experiment_worker import run_experiment

OUT = Path(__file__).resolve().parent
mode = sys.argv[1]
job = OUT / "service" / mode
job.mkdir(exist_ok=True, parents=True)
base = read_json(OUT / "focus_measurement.json")
gen = read_json(OUT / "task_metrics.json")["generated"]["generator"]
is_fit = mode == "prepare"
contract = {
    "protocol": "ptq_extension/1",
    "extension_dimensions": ["coverage_exceptions"],
}
spec = {
    "service": "quantize_evaluate" if is_fit else "evaluate_artifact",
    "backend": {
        "config_path": "QAViT/configs/deitb_backend.json",
        "backend_overrides": {"device": "cuda:0"},
    },
    "resolved_choices": {"training": is_fit},
    "quantization_contract": contract,
    "research_quantization_contract": contract,
    "requires_code": False,
    "seed": 11,
    "evaluations": [{"kind": "classification", "split": "search", "count": 1}],
}
if is_fit:
    spec.update(
        calibration_generator=gen,
        calibration={"split": "calibration", "count": gen["images"]},
        recipe={
            "source_id": "psaq-vit",
            "version": "paper+author reference",
            "profile": "bounded pixel optimization and single-site coverage",
            "base": base["plan"],
            "settings": {},
            "evidence": ["methods/recipes/psaq-vit.md"],
        },
        construction={
            "regime": "finite_calibration_ptq",
            "split": "calibration",
            "max_steps": sum(s["steps"] for s in gen["phases"]),
            "objective": "teacher CE + TV residual - patch KDE entropy",
            "update_targets": ["synthetic image pixels"],
            "recipe_version": "psaq_kde bounded/1",
        },
    )
elif mode == "reuse":
    spec["model_dir"] = str(OUT / "service" / "prepare" / "model")
    assert reusable_manifest(
        read_json(Path(spec["model_dir"]) / "manifest.json"), "ptq_extension/1"
    )
elif mode == "child":
    from tools.experiment_worker import backend_and_data, data_view
    from tools.frozen import load_refinement_parent
    from core.journal import file_sha256
    from ptq_agent.execution.frozen_trials import score_frozen_candidates
    import gc
    import torch

    parent_dir = OUT / "service" / "prepare"
    original = read_json(parent_dir / "receipt.json")
    folder = parent_dir / "model"
    ref = original["model_ref"]
    receipt = parent_dir / "frozen_evaluation.json"
    binding = {
        "model_ref": ref,
        "model_dir": str(folder),
        "evidence_ref": str(receipt),
        "assets": [
            {"path": str(p), "sha256": file_sha256(p)}
            for p in folder.rglob("*")
            if p.is_file()
        ],
        "verified_evaluation": {"path": str(receipt), "sha256": file_sha256(receipt)},
    }
    backend, cfg, data = backend_and_data(spec)
    model, manifest = load_refinement_parent(
        binding, backend, cfg, data, protocol=contract["protocol"]
    )
    assert not reusable_manifest(manifest, "static_w4a4/1")
    site = base["site"]
    rows = [
        {"id": str(factor), "scales": {site: {"activation": factor}}}
        for factor in (0.97, 1.03)
    ]
    obs = [
        {
            "id": "class",
            "view": "classification",
            "output": "classification_logits",
            "metrics": ["mse"],
        }
    ]
    scores = score_frozen_candidates(
        backend,
        model,
        data_view(data, {"split": "calibration", "count": 2}),
        rows,
        observations_spec=obs,
        source_ref=ref,
        input_shapes=manifest["input_shapes"],
    )
    atomic_json(job / "scores.json", scores)
    selected = min(
        scores["candidates"],
        key=lambda r: (
            r["scores"]["class.mse"] if "scores" in r else r["metrics"]["class.mse"]
        ),
    )
    spec.update(
        service="quantize_evaluate",
        parent_ref=ref,
        frozen_sources=[binding],
        accept_trial={
            "candidate_id": selected["id"],
            "evidence_ref": str(job / "scores.json"),
        },
        accepted_receipt=selected["receipt"],
        scored_calibration_ids=scores["sample_ids"],
    )
    del backend, data, model
    gc.collect()
    torch.cuda.empty_cache()
else:
    raise ValueError(mode)
payload = {
    "level": "L2",
    "resources": {"cpu_threads": 2},
    "execution": spec,
    "validation": {},
}
request = {"experiment": {"payload": payload}, "gpus": []}
atomic_json(job / "request.json", request)
report = run_experiment(request, job)
atomic_json(job / "receipt.json", report)
if mode == "reuse":
    p = read_json(OUT / "service" / "prepare" / "receipt.json")
    assert p["model_ref"] == report["model_ref"]
    a, b = [
        item["observations"]["evaluations"]["classification"] for item in (p, report)
    ]
    assert a["sample_ids"] == b["sample_ids"] and all(
        a["metrics"][k] == b["metrics"][k]
        for k in ("mse", "cosine_error", "top1", "top5")
    )
    atomic_json(
        job / "comparison.json",
        {
            "same_model": True,
            "sample_ids": a["sample_ids"],
            "stable_metrics": {
                k: a["metrics"][k] for k in ("mse", "cosine_error", "top1", "top5")
            },
            "gradient_audit": report["provenance"]["gradient_audit"],
        },
    )


if mode == "child":
    child = read_json(job / "model" / "manifest.json")
    assert (
        child["construction"]
        == read_json(parent_dir / "model" / "manifest.json")["construction"]
        and child["preparation_training"]
    )
    atomic_json(
        job / "comparison.json",
        {
            "parent": ref,
            "child": report["model_ref"],
            "selected": selected["id"],
            "refinement": child["refinement"],
            "frozen_evaluation": read_json(job / "frozen_evaluation.json"),
            "construction": child["construction"],
            "gradient_audit": report["provenance"]["gradient_audit"],
        },
    )
