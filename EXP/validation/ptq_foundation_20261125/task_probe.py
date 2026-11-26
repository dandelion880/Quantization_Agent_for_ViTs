"""Prepare and score two complete frozen states on real task supervision and generated calibration inputs."""

from pathlib import Path
from copy import deepcopy
from runtime.paths import ROOT
import torch
from core.journal import atomic_json, file_sha256, read_json
from ptq_agent.execution.frozen_trials import score_frozen_candidates
from ptq_agent.execution.task_sensitivity import prepare_window
from ptq_agent.execution.research_metrics import score
from ptq_agent.execution.calibration_sources import synthesize, CalibrationView
from ptq_agent.execution.model import tensor_digest, forward
from ptq_agent.execution.pipeline import QuantizationExecutor
from tools.experiment_worker import backend_and_data, data_view
from tools.frozen import FrozenData
from tools.segmentation import DenseTask
from ptq_agent.execution.calibration_search import state_identity
from ptq_agent.execution.sites import site_catalog

OUT = Path(__file__).resolve().parent
backend, cfg, data = backend_and_data(
    {
        "backend": {
            "config_path": "QAViT/configs/deitb_backend.json",
            "backend_overrides": {"device": "cuda:0"},
        }
    }
)


def load(path):
    model, manifest = QuantizationExecutor(backend, cfg, FrozenData(data)).load(path)
    return model, {"model_ref": manifest["content_sha256"]}


import sys

tap = backend.spec["block_paths"][0]
options = {"target": "labels", "gamma": 0.75, "mask_rule": "text_top_error"}
if "--dense-only" not in sys.argv:
    import tempfile

    run_dir = Path(tempfile.mkdtemp(dir=OUT, prefix="task_packages_"))
    base = read_json(OUT / "focus_measurement.json")
    fit_view = data_view(data, {"split": "calibration", "count": len(base["fit"])})
    executor = QuantizationExecutor(backend, cfg, fit_view)
    model, weights, activations, audit = executor.prepare(base["plan"], {})
    real_package = run_dir / "real"
    executor.export(model, base["plan"], weights, activations, real_package, audit)
    del model, weights, activations, executor
    model, src = load(real_package)
    tap = (
        next(iter(backend.spec["block_paths"]))
        if backend.spec.get("block_paths")
        else read_json(OUT / "focus_measurement.json")["site"].split(".attn")[0]
    )
    view = data_view(data, {"split": "calibration", "count": 2})
    options = {"target": "labels", "gamma": 0.75, "mask_rule": "text_top_error"}
    obs = {
        "id": "quant",
        "view": "classification",
        "block": tap,
        "output": "block",
        "metrics": ["mse", "apq_masked_gradient_error", "aph_official_abs"],
        "estimators": {
            "apq_masked_gradient_error": options,
            "aph_official_abs": {"target": "labels", "delta": 1e-3},
        },
    }
    base = read_json(OUT / "focus_measurement.json")
    shape = next(view.batches("calibration"))[0].shape
    params = {
        "variant": "psaq_kde",
        "attention_modules": [base["site"].rsplit(".", 1)[0]],
        "images": 2,
        "classes": backend.model.head.out_features,
        "input_chw": list(shape[1:]),
        "seed": 8,
        "lr": 0.002,
        "betas": [0.5, 0.9],
        "adam_epsilon": 1e-8,
        "channel_limits": [[-2.1, -2.0, -1.8], [2.2, 2.4, 2.6]],
        "class_weight": 1.0,
        "tv_weight": 0.05,
        "feature_weight": 1.0,
        "tv_target": [200.0, 240.0],
        "phases": [{"steps": 2, "jitter": 1, "warmup": 0}],
        "bandwidth": 0.01,
        "kde_samples": 10,
        "kde_epsilon": 1e-4,
        "flip": True,
    }
    directory = run_dir / "synthetic_data"
    receipt = synthesize(backend, params, directory)
    generated = CalibrationView(data, backend, directory)
    executor = QuantizationExecutor(backend, cfg, generated)
    m, w, a, audit = executor.prepare(base["plan"], {})
    synth_pkg = run_dir / "synthetic_model"
    if not synth_pkg.exists():
        executor.export(m, base["plan"], w, a, synth_pkg, audit)
    cold, sref = load(synth_pkg)
    x, sid = next(view.batches("calibration"))
    assert torch.equal(forward(backend, cold, x), forward(backend, m, x))
    del m, cold, executor
    paths = {"real_calibration": real_package, "synthetic_calibration": synth_pkg}
    frozen = score_frozen_candidates(
        backend,
        model,
        view,
        [{"id": k, "source_ref": k} for k in paths],
        observations_spec=[obs],
        objectives={"apq": {"quant.apq_masked_gradient_error": 1.0}},
        load_candidate=lambda s: load(paths[s]),
        source_ref=src["model_ref"],
    )
    atomic_json(OUT / "task_metrics.json", {"generated": receipt, "frozen": frozen})
    del model
dense = DenseTask(backend)
from tools.vision_models import segmentation_assets

_, settings, _, _ = segmentation_assets(backend.config)
root = Path(settings["data_root"])
im = next(iter(sorted((root / "images/training").glob("*.jpg"))))
lbl = root / "annotations/training" / (im.stem + ".png")
row = {
    "path": str(im),
    "file_sha256": file_sha256(im),
    "seg_map_path": str(lbl),
    "annotation_sha256": file_sha256(lbl),
}
window, meta, geometry = next(dense.windows(row))
labels = meta.pop("_qavit_labels")
s = {"id": "dense", "view": "dense", "block": tap, "output": "block"}
binding = {
    "reference": "fp_model",
    "input_state": tensor_digest(window),
    "suffix_state": backend.original_state,
    "head": dense.identity,
    "task": "dense",
    "view": geometry,
    "sample_ids": [row["file_sha256"]],
    "observation": tap,
    "axes": "B,T,C",
    "reduction": "valid pixels",
    "annotation": row["annotation_sha256"],
}
opts = {**options, "ignore_index": dense.segmentor.decode_head.ignore_index}
state, record = prepare_window(
    backend,
    backend.model,
    window,
    meta,
    s,
    "apq_masked_gradient_error",
    opts,
    binding,
    dense=dense,
    label=labels,
)
atomic_json(
    OUT / "dense_sensitivity.json",
    {
        "binding": binding,
        "preparation": record,
        "valid_pixels": int((labels != opts["ignore_index"]).sum()),
        "row": row,
    },
)
