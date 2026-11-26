"""Validate an optimized full-body true W4A4 graph and a weight edit made from its deployed parent."""

from pathlib import Path
from copy import deepcopy
import torch
from runtime.paths import ROOT
from core.journal import read_json, atomic_json
from core.static_w4a4 import check_plan, native_audit, check_audit
from core.ptq_protocol import validate_construction
from tools.frozen import FrozenData
from tests.test_calibration_search import fixture
from ptq_agent.execution.pipeline import QuantizationExecutor, quantizers
from ptq_agent.execution.sites import site_catalog
from ptq_agent.execution.calibration_search import state_identity, apply_grid_changes
from ptq_agent.execution.model import forward, tensor_digest
from ptq_agent.execution import grids

out = Path(__file__).resolve().parent
folder = out / "numerical/recipe_qdrop/model"
b, cfg, data, _ = fixture()
model, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(folder)
plan = manifest["plan"]
cat = site_catalog(b)
required = check_plan(plan, cat, protocol="static_w4a4/2")
x, sid = next(data.batches("search"))
pre = forward(b, model, x)
covered = check_audit(native_audit(model, plan, cat, required), required)
log = manifest["calibration_report"]["postfit"]
assert log and all(
    s["optimizer_steps"] > 0 and s["rounding_sites"] and s["scale_sites"] for s in log
)
binding = {
    "regime": "finite_calibration_ptq",
    "split": "calibration",
    "update_targets": ["rounding", "activation_scale"],
    "max_steps": sum(i["optimizer_steps"] for i in log),
    "recipe_version": "QDrop, bounded/1",
    "objective": "FP block output reconstruction and hard rounding",
}
validate_construction({"construction": binding}, "static_w4a4/2", True)
qmap = quantizers(model)
site = next(n for n in log[0]["rounding_sites"] if qmap[n].grid_edit_from_packed_state)
q = qmap[site]
original = state_identity(model)
w = q.decoded_weight.clone()
rec = deepcopy(q.weight_record)
scale = rec["params"]["scale"] * 1.1
new = {**rec["params"], "scale": scale}
spec = q.layer["weight"]
oracle = grids.apply(w, spec, rec["method"], new)
original_fp = grids.apply(q.reference_weight, spec, rec["method"], new)
receipt = apply_grid_changes(model, {site: {"weight": 1.1}}, manifest["input_shapes"])
assert (
    torch.equal(q.decoded_weight, oracle)
    and not torch.equal(original_fp, oracle)
    and original != state_identity(model)
)
post = forward(b, model, x)
assert not torch.equal(post, pre)
atomic_json(
    out / "primary_protocol.json",
    {
        "protocol": "static_w4a4/2",
        "request": "numerical/recipe_qdrop/source_request.json",
        "package": str(folder.relative_to(out)),
        "weights_sha256": manifest["weights_sha256"],
        "construction": binding,
        "optimizer_receipts": log,
        "covered_boundaries": covered["boundaries"],
        "heldout_sample": sid,
        "heldout_output_sha256": tensor_digest(pre),
        "weight_edit": {
            "site": site,
            "receipt": receipt,
            "from_packed_reference_max_abs": float(
                (oracle - q.decoded_weight).abs().max()
            ),
            "against_refitting_original_fp_max_abs": float(
                (oracle - original_fp).abs().max()
            ),
            "heldout_output_sha256": tensor_digest(post),
        },
    },
)
