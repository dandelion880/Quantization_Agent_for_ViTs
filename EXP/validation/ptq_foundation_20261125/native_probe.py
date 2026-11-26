"""Exercise complete legacy paper-code graphs under a reduced numerical integration budget."""

from copy import deepcopy
from pathlib import Path
import tempfile
import torch
from runtime.paths import ROOT
from tests.test_calibration_search import fixture
from ptq_agent.execution.assembled_recipes import assemble
from ptq_agent.execution.recipes import SOURCES
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.calibration_search import state_identity
from ptq_agent.execution.model import forward, tensor_digest
from core.journal import atomic_json
from tools.frozen import FrozenData

out = Path(__file__).resolve().parent
b, cfg, data, _ = fixture()
# This model owns and executes its 1000-way head, with seeded fixture weights.
b.spec["native_classifier"] = True
cfg["loop"]["quantize_classifier"] = True
cfg["constraints"]["allowed_formats"] = sorted(
    set(cfg["constraints"]["allowed_formats"]) | {"INT4", "UINT4", "INT8", "UINT8"}
)
for name in (
    "allowed_activation_methods",
    "allowed_preprocess",
    "allowed_weight_methods",
):
    cfg["loop"].pop(name, None)
for k in ("excluded_modules", "protected_modules"):
    cfg["constraints"][k] = [s for s in cfg["constraints"][k] if s != "head"]
reports = {}
for source, short in (
    ("repq_vit", "repq"),
    ("fq_vit", "fq_vit"),
    ("ptq4vit", "ptq4vit"),
):
    req = {
        "base": {"author_recipe": short, "bits": 8 if short == "fq_vit" else 4},
        "source_id": source,
        "version": SOURCES[short],
        "settings": {},
        "profile": "source graph, reduced calibration",
        "evidence": ["methods/recipes/" + source + ".md"],
    }
    plan, binding = assemble(b, req, data)
    for layer in plan["layers"]:
        for target in ("activation", "weight"):
            opts = layer.get(target + "_calibration", {})
            if opts.get("strategy") == "coordinate":
                opts["rounds"] = 1
                if "factor_grid" in opts:
                    opts["factor_grid"].update(start=0.85, stop=1.1, count=3)
    exe = QuantizationExecutor(b, cfg, data)
    model, w, a, audit = exe.prepare(plan, {})
    path = Path(tempfile.mkdtemp(prefix="native_" + short + "_", dir=out)) / "model"
    exe.export(model, plan, w, a, path, audit)
    cold, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(path)
    x, sid = next(data.batches("search"))
    expected = forward(b, model, x)
    actual = forward(b, cold, x)
    assert torch.equal(expected, actual) and state_identity(model) == state_identity(
        cold
    )
    reports[source] = {
        "source_request": req,
        "binding": binding,
        "bounded_plan": plan,
        "heldout_sample": sid,
        "heldout_output": tensor_digest(actual),
        "state": state_identity(cold),
        "package": str(path.relative_to(out)),
        "operators": plan.get("operators", []),
        "preprocess": plan["preprocess"],
        "original_budget": False,
    }
    atomic_json(out / "native_recipes.json", reports)
