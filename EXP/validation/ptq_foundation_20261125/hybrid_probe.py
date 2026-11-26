"""Whole-bridge scoring on a real MobileViT block with the original folding and local/global fusion."""

from pathlib import Path
from copy import deepcopy
import hashlib, sys, tempfile
import torch
from runtime.paths import ROOT
from core.journal import read_json, atomic_json
from tests.test_calibration_search import fixture
from tools.frozen import FrozenData
from ptq_agent.backends.server import state_sha256
from ptq_agent.execution.assembled_recipes import assemble
from ptq_agent.execution.precision import select
from ptq_agent.execution.model import forward, tensor_digest
from ptq_agent.execution.pipeline import QuantizationExecutor

out = Path(__file__).resolve().parent
# Load the hybrid-capable installed timm in this separate process before the classifier fixture imports timm.
sys.path.insert(0, str(ROOT / "QAViT/.runtime_pkgs/segmentation"))
from timm.models import mobilevit

b, cfg, data, base = fixture()
from torch import nn


class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=1)
        self.bridge = mobilevit.MobileVitBlock(
            8, 8, transformer_dim=16, transformer_depth=1, num_heads=2, patch_size=3
        )
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        return self.head(self.bridge(self.stem(x)).mean((2, 3)))


torch.manual_seed(147)
b.model = Hybrid().eval().requires_grad_(False)
b.original_state = state_sha256(b.model)
b.spec = {"task_forward": "forward"}
b._modules = [
    {
        "name": k,
        "type": type(m).__name__,
        "protected": False,
        "weight_shape": list(m.weight.shape),
        "block": None,
    }
    for k, m in b.model.named_modules()
    if isinstance(m, (nn.Linear, nn.Conv2d))
]
for k in ("allowed_activation_methods", "allowed_preprocess", "allowed_weight_methods"):
    cfg["loop"].pop(k, None)
cfg["constraints"]["allowed_formats"] = list(
    set(cfg["constraints"]["allowed_formats"]) | {"INT4", "UINT4"}
)
site = "bridge.conv_1x1"
projection = deepcopy(next(row for row in base["layers"] if row["kind"] == "conv2d"))
projection["module"] = site
base["layers"] = [projection]
req = deepcopy(read_json(out / "allocation_recipes.json")["q-hyvit"]["request"])
req.update(
    base=base,
    units=["bridge"],
    profile="upstream timm MobileViT hybrid block; bounded W4A4; one projection",
)
req["settings"]["fit"]["operand_order"] = [[site, "activation"], [site, "weight"]]
group = req["allocation"]["configuration_groups"][0]
group["sites"] = {site: ["scheme"]}
req["allocation"]["bridge"]["module"] = "bridge"
plan, binding = assemble(b, req, data)
chosen, search = select(b, cfg, data, plan, req["allocation"])
exe = QuantizationExecutor(b, cfg, data)
model, ws, acts, audit = exe.prepare(chosen, {})
dest = Path(tempfile.mkdtemp(prefix="mobilevit_", dir=out)) / "model"
exe.export(model, chosen, ws, acts, dest, audit)
cold, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(dest)
x, sid = next(data.batches("search"))
observed = {}


def observe(name):
    def save(m, a, y):
        observed[name] = [list(a[0].shape), list(y.shape)]

    return save


hs = [
    model.get_submodule(n).register_forward_hook(observe(n))
    for n in (
        "bridge.conv_1x1",
        "bridge.transformer",
        "bridge.conv_proj",
        "bridge.conv_fusion",
        "bridge",
    )
]
try:
    y = forward(b, model, x)
finally:
    for h in hs:
        h.remove()
z = forward(b, cold, x)
assert (
    torch.equal(z, y)
    and len(observed) == 5
    and observed["bridge.transformer"][0] != observed["bridge.conv_1x1"][1]
)
src = Path(mobilevit.__file__)
atomic_json(
    out / "hybrid_bridge.json",
    {
        "request": req,
        "binding": binding,
        "implementation": str(src),
        "implementation_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "model_dir": str(dest.relative_to(out)),
        "decision": search,
        "observed_shapes": observed,
        "scale_search": manifest["calibration_report"]["postfit"],
        "heldout_id": sid,
        "heldout_sha256": tensor_digest(y),
        "reload_max_abs": float((y - z).abs().max()),
    },
)
