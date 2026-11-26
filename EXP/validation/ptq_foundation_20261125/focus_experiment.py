"""Execute Generation's fixed-prefix four-arm experiment on a bound pretrained classifier."""

from pathlib import Path
import argparse
import torch
from runtime.paths import ROOT
from core.journal import atomic_json, read_json, file_sha256
from ptq_agent.execution.model import forward, tensor_digest
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.recipes import spec
from ptq_agent.execution.sites import site_catalog
from tools.experiment_worker import backend_and_data, data_view
from tools.frozen import FrozenData

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["prepare", "cold"], required=True)
args = parser.parse_args()
backend_spec = {"kind": "server", "config_path": "QAViT/configs/deitb_backend.json"}
backend, cfg, data = backend_and_data({"backend": backend_spec})
calibration = data_view(data, {"split": "calibration", "count": 4})
search = data_view(data, {"split": "search", "count": 4})
assert set(calibration.ids("calibration")).isdisjoint(search.ids("search"))
site = next(
    row["name"]
    for row in site_catalog(backend).values()
    if row["kind"] == "linear" and row.get("semantic") == "post_layernorm"
)
plan = {
    "schema_version": 3,
    "unlisted_modules": "keep_fp",
    "unresolved": [],
    "preprocess": [],
    "execution": {
        "mode": "reference_qdq",
        "compute_dtype": "FP32",
        "activation_calibration": "static",
        "order": "w_then_a",
    },
    "layers": [
        {
            "module": site,
            "kind": "linear",
            "weight_method": "uniform_affine",
            "weight": spec(4, False, "per_channel", 0),
            "activation_method": "uniform_affine",
            "activation": spec(4, False),
        }
    ],
}
directory = OUT / "focus_model"
if args.phase == "prepare":
    assert read_json(OUT / "generation.json")[-1]["decision"]["content"]["next_step"]
    executor = QuantizationExecutor(backend, cfg, calibration)
    model, w, a, audit = executor.prepare(plan, {})
    executor.export(model, plan, w, a, directory, audit)
else:
    model, _ = QuantizationExecutor(backend, cfg, FrozenData(calibration)).load(
        directory
    )


def paired(view, split):
    pairs = []

    def record(module, args, y):
        fp = backend.model.get_submodule(site)(*args)
        pairs.append((y.detach().double().cpu(), (fp - y).double().cpu()))

    hook = model.get_submodule(site).register_forward_hook(record)
    logits = []
    try:
        for x, sid in view.batches(split):
            y = forward(backend, model, x)
            logits.append([sid, tensor_digest(y)])
    finally:
        hook.remove()
    assert len(pairs) == len(logits) and all(q.ndim == 3 for q, e in pairs)
    return torch.cat([q for q, e in pairs]), torch.cat([e for q, e in pairs]), logits


q, e, fit_ids = paired(calibration, "calibration")
h, r, eval_ids = paired(search, "search")
pos = e.mean(0, keepdim=True)
ch = e.mean((0, 1), keepdim=True)
mu = q.mean(0, keepdim=True)
v = q - mu
den = v.square().sum((0, 1), keepdim=True)
slope = torch.where(
    den > 0, (v * (e - pos)).sum((0, 1), keepdim=True) / den.clamp_min(1e-24), 0.0
)
states = {
    k: v.float()
    for k, v in {"position": pos, "channel": ch, "center": mu, "slope": slope}.items()
}
state_file = OUT / "focus_fit.pt"
if args.phase == "prepare":
    torch.save(states, state_file)
else:
    saved = torch.load(state_file, map_location="cpu", weights_only=True)
    assert all(torch.equal(saved[k], v) for k, v in states.items())
    states = saved


def arm_errors(y, err, control=False):
    z = (y - states["center"]) * states["slope"]
    estimates = {
        "zero": 0.0,
        "channel_bias": states["channel"],
        "position_bias": states["position"],
        "position_centered_slope": states["position"] + z,
    }
    if control:
        estimates["cross_image_slope"] = states["position"] + z.roll(1, 0)
    return {k: (err - x).square().mean((1, 2)).tolist() for k, x in estimates.items()}


report = {
    "evidence": "bounded local representation experiment",
    "protocol": "ptq_extension/1",
    "dimensions": ["additional_computation", "coverage_exceptions"],
    "reference": "FP original module on identical actual candidate input",
    "model": backend.identity(),
    "site": site,
    "plan": plan,
    "fit": fit_ids,
    "heldout": eval_ids,
    "training_mse_by_image": arm_errors(q, e),
    "heldout_mse_by_image": arm_errors(h, r, True),
    "state_bytes": {k: v.nbytes for k, v in states.items()},
    "shape": list(q.shape),
    "fitted_state_sha256": file_sha256(state_file),
    "weights_sha256": read_json(directory / "manifest.json")["weights_sha256"],
    "decision": file_sha256(OUT / "generation.json"),
}
if args.phase == "cold":
    assert report == read_json(OUT / "focus_measurement.json")
    atomic_json(
        OUT / "focus_cold.json",
        {
            "verified": True,
            "heldout": eval_ids,
            "fitted_state_sha256": file_sha256(state_file),
        },
    )
else:
    atomic_json(OUT / "focus_measurement.json", report)
print(site, report["heldout_mse_by_image"], flush=True)
