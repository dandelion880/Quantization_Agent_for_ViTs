from pathlib import Path
import tempfile
import torch
from runtime.paths import ROOT
from core.journal import atomic_json
from tests.test_calibration_search import fixture
from tools.frozen import FrozenData
from ptq_agent.execution.calibration_sources import CalibrationView
from ptq_agent.execution.cyclic_calibration import run
from ptq_agent.execution.assembled_recipes import assemble
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.model import forward, tensor_digest

out = Path(__file__).resolve().parent
b, cfg, data, base = fixture()
cfg["constraints"]["allowed_formats"] += ["INT4", "INT8"]
names = [u + ".attn" for u in ["blocks.0", "blocks.1"]]
targets = torch.zeros((3, 1000))
targets[range(3), [160, 357, 825]] = 1
settings = {
    "variant": "clamp_cyclic",
    "images": 3,
    "seed": 52,
    "input_chw": [3, 32, 32],
    "groups": [
        {
            "output": names[0].rsplit(".", 1)[0],
            "sites": [names[0] + ".qkv"],
            "bits": [4],
            "initial_scale": 0.023,
        }
    ],
    "attention_layout": {n: {"prefix": 1, "grid": [2, 2]} for n in names},
    "target_vectors": targets.tolist(),
    "target_space": "probability",
    "fixed_activation_bits": 4,
    "activation_rule": "fixed",
    "initial_activation_scale": 0.08,
    "scale_perturbation": [-0.001, 0.001],
    "min_scale": 1e-5,
    "initial_steps": 2,
    "refresh_steps": 1,
    "refresh_every": 1,
    "passes": 1,
    "cycles": 2,
    "population": 2,
    "diversity_parents": 2,
    "lr": 0.01,
    "window": 3,
    "positives": 2,
    "temperature": 0.1,
}
req = {
    "source_id": "clamp-vit",
    "base": base,
    "version": "ECCV paper, explicit perturbation bounds",
    "profile": "bounded fixed-width cyclic",
    "settings": {},
    "evidence": ["methods/recipes/clamp-vit.md"],
}
folder = Path(tempfile.mkdtemp(prefix="clamp_cyclic_", dir=out))
rec = run(b, cfg, data, req, settings, folder / "pool")
view = CalibrationView(data, b, folder / "pool")
p, binding = assemble(b, req, view)
e = QuantizationExecutor(b, cfg, view)
m, w, a, audit = e.prepare(p, {})
e.export(m, p, w, a, folder / "model", audit)
cold, manifest = QuantizationExecutor(b, cfg, FrozenData(view)).load(folder / "model")
x, sid = next(data.batches("search"))
y = forward(b, m, x)
assert torch.equal(y, forward(b, cold, x))
pixels = [r for r in rec["trajectory"] if r["stage"] == "pixels"]
assert pixels[1]["before"] == pixels[0]["after"] and all(
    r["before"] != r["after"] for r in pixels
)
atomic_json(
    out / "cyclic_images.json",
    {
        "request": req,
        "generator": settings,
        "binding": binding,
        "model_dir": str((folder / "model").relative_to(out)),
        "generated_ids": view.names
        if hasattr(view, "names")
        else view.ids("calibration"),
        "trajectory": rec["trajectory"],
        "heldout_id": sid,
        "heldout_sha256": tensor_digest(y),
        "weights_sha256": manifest["weights_sha256"],
    },
)
