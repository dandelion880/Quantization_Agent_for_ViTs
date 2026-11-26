"""Measure genuinely distinct frozen candidates on real labeled images and preserve the paired selection inputs."""

from pathlib import Path
from runtime.paths import ROOT
from core.journal import read_json, atomic_json
from tools.experiment_worker import backend_and_data, data_view
from tools.frozen import FrozenData
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.frozen_trials import score_frozen_candidates
from tools.evaluation.selection import compare

out = Path(__file__).resolve().parent
job = out / "service" / "prepare"
spec = read_json(job / "request.json")["experiment"]["payload"]["execution"]
b, cfg, data = backend_and_data(spec)
m, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(job / "model")
ref = read_json(job / "receipt.json")["model_ref"]
site = read_json(out / "focus_measurement.json")["site"]
rows = [
    {"id": "parent", "scales": {}},
    *({"id": str(f), "scales": {site: {"activation": f}}} for f in (0.7, 1.3)),
]
fields = {"calibration": "mse", "search": "task_nll"}
receipts = {}
for split, field in fields.items():
    obs = [
        {
            "id": "classification",
            "view": "classification",
            "output": "classification_logits",
            "metrics": [field],
        }
    ]
    receipts[split] = score_frozen_candidates(
        b,
        m,
        data_view(data, {"split": split, "count": 3}),
        rows,
        observations_spec=obs,
        split=split,
        source_ref=ref,
        input_shapes=manifest["input_shapes"],
    )
    atomic_json(out / ("selection_" + split + ".json"), receipts[split])
quality = compare(
    receipts["calibration"],
    receipts["search"],
    proxy="classification.mse",
    task="classification.task_nll",
    direction="min",
    baseline="parent",
    k=2,
    resamples=64,
    seed=1701,
)
atomic_json(out / "selection_quality.json", quality)
# Fold complete windows with the real task pipeline before counting per-original class intersections.
from tools.segmentation import evaluate
import tempfile

folder = Path(tempfile.mkdtemp(prefix="paired_ade_", dir=out))
dense = evaluate(m, b, folder, 1, per_image_confusion=True)
counts = dense["per_original_confusion"]
assert len(counts) == 1
import torch

for mode, s in next(iter(counts.values())).items():
    flat = torch.zeros(s["classes"] ** 2, dtype=torch.long)
    flat[s["indices"]] = torch.tensor(s["counts"])
    assert torch.equal(
        flat.reshape(s["classes"], -1),
        torch.tensor(dense["metrics"][mode]["confusion_matrix"]),
    )
atomic_json(
    out / "per_original_dense.json",
    {
        "invocation": {"count": 1, "per_image_confusion": True},
        "fp_identity": dense["fp_identity"],
        "task_cost_folder": str(folder.relative_to(out)),
        "originals": counts,
        "model_ref": ref,
        "miou_percent": {n: r["mIoU"] for n, r in dense["metrics"].items()},
    },
)
