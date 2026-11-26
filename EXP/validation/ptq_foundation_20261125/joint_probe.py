"""A real classification image and an annotated ADE training window fit one static grid."""

from pathlib import Path
import torch
from runtime.paths import ROOT
from core.journal import atomic_json, read_json
from tools.experiment_worker import backend_and_data
from ptq_agent.execution.task_views import TaskView
from ptq_agent.execution.pipeline import QuantizationExecutor, quantizers
from ptq_agent.execution.model import forward, tensor_digest
from tools.frozen import FrozenData

OUT = Path(__file__).resolve().parent
backend, cfg, data = backend_and_data(
    {
        "backend": {
            "config_path": "QAViT/configs/deitb_backend.json",
            "backend_overrides": {"device": "cuda:0"},
        }
    }
)
row = read_json(OUT / "dense_sensitivity.json")["row"]
spec = {
    "mode": "joint_calibration_same_backbone",
    "classification": {"split": "calibration", "count": 1},
    "dense": [{**row, "windows": [0]}],
}
view = TaskView(backend, data, spec)
plan = read_json(OUT / "focus_measurement.json")["plan"]
exe = QuantizationExecutor(view.backend, cfg, view)
model, w, a, audit = exe.prepare(plan, {})
package = OUT / "joint_model"
exe.export(model, plan, w, a, package, audit)
cold, manifest = QuantizationExecutor(backend, cfg, FrozenData(data)).load(package)
outputs = {}
for x, sid in view.batches("calibration"):
    y = forward(view.backend, model, x)
    z = forward(view.backend, cold, x)
    assert torch.equal(y, z)
    outputs[sid] = {
        "input": list(x.shape),
        "output": list(y.shape),
        "output_sha256": tensor_digest(y),
    }
atomic_json(
    OUT / "joint_views.json",
    {
        "request": spec,
        "frozen_package": manifest["weights_sha256"],
        "calibration_ids": manifest["calibration_ids"],
        "views": view.descriptor,
        "outputs": outputs,
    },
)
