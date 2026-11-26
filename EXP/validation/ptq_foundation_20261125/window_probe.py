"""Round trip against the upstream detector's shifted Swin implementation on a bounded real block."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import torch
from torch import nn
from torch.nn import functional as F
from runtime.paths import ROOT

from core.journal import atomic_json
from ptq_agent.backends.server import state_sha256
from ptq_agent.execution.assembled_recipes import assemble
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.model import forward, tensor_digest
from ptq_agent.execution.window_routes import branch, partition
from tests.test_calibration_search import fixture
from tools.frozen import FrozenData

out = Path(__file__).resolve().parent
b, cfg, data, base = fixture()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

src = (
    ROOT.parent
    / "ptq_training_free/third_party/repq_reference/detection/mmdet/models/backbones/swin_transformer.py"
)
# Use the unmodified module definitions without importing the optional mmdet training framework.
tree = ast.parse(src.read_text())
names = {
    "Mlp",
    "MatMul",
    "WindowAttention",
    "window_partition",
    "window_reverse",
    "SwinTransformerBlock",
}
tree.body = [n for n in tree.body if getattr(n, "name", "") in names]
env = dict(
    torch=torch,
    nn=nn,
    F=F,
    DropPath=DropPath,
    to_2tuple=to_2tuple,
    trunc_normal_=trunc_normal_,
    __name__="asset_swin",
)
exec(compile(tree, str(src), "exec"), env)
Block = env["SwinTransformerBlock"]


class WindowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 8, stride=8)
        self.blocks = nn.ModuleList(
            Block(8, 2, window_size=2, shift_size=i, mlp_ratio=2) for i in (0, 1)
        )

    def forward(self, x):
        g = self.patch(x)
        h, w = g.shape[-2:]
        tokens = g.flatten(2).transpose(1, 2)
        labels = tokens.new_zeros(1, h + (-h) % 2, w + (-w) % 2, 1)
        for iy, ys in enumerate((slice(None, -2), slice(-2, -1), slice(-1, None))):
            for ix, xs in enumerate((slice(None, -2), slice(-2, -1), slice(-1, None))):
                labels[:, ys, xs, :] = 3 * iy + ix
        lab = partition(labels, (2, 2)).squeeze(-1)
        mask = (lab[:, None] != lab[:, :, None]).to(g) * -100
        for b in self.blocks:
            b.H, b.W = h, w
            tokens = b(tokens, mask)
        return tokens.mean(1)


torch.manual_seed(945)
b.model = WindowNet().eval().requires_grad_(False)
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
base["layers"] = [r for r in base["layers"] if r["kind"] == "linear"]
for r in base["layers"]:
    r["activation"].update(bits=8, format="UINT8")
cfg["constraints"]["allowed_formats"] = list(
    set(cfg["constraints"].get("allowed_formats", [])) | {"UINT8", "UINT4"}
)
stages = [
    {
        "kind": "window_branches",
        "module": f"blocks.{i}",
        "settings": {
            "stage": "encoder0",
            "first": i == 0,
            "last": i == 1,
            "low_fraction": 0.5,
            "low_bits": 4,
        },
    }
    for i in range(2)
]
request = {
    "source_id": "mixa-q",
    "version": "ICCV 2025 COCO PTQ",
    "profile": "bounded shared W4 / A8,4",
    "base": base,
    "evidence": ["methods/recipes/mixa-q.md"],
    "units": ["blocks.0", "blocks.1"],
    "settings": {"stages": stages},
}
plan, binding = assemble(b, request, data)
from ptq_agent.execution.precision import select

settings = {
    "configuration_groups": [
        {"choices": [0.0, 0.5, 1.0], "window_blocks": [s["module"] for s in stages]}
    ],
    "max_configurations": 3,
    "limits": {},
    "objectives": ["output_mse"],
    "priority": ["output_mse"],
}
winner, search = select(b, cfg, data, plan, settings)
assert search["shared_window_grids"] and len(search["candidates"]) == 3
executor = QuantizationExecutor(b, cfg, data)
m, ws, acts, audit = executor.prepare(plan, {})
folder = Path(tempfile.mkdtemp(prefix="swin_", dir=out)) / "model"
executor.export(m, plan, ws, acts, folder, audit)
cold, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(folder)
x, sid = next(data.batches("search"))
x = torch.cat((x, x.flip((-1,))), 0)


def run(model):
    seen = []
    handle = model.blocks[1].attn.qkv.register_forward_pre_hook(
        lambda mod, args: seen.append([branch.get(), len(args[0])])
    )
    try:
        y = forward(b, model, x)
    finally:
        handle.remove()
    assert seen == [[None, 4], [4, 4]], seen
    return y, seen


y, seen = run(m)
z, seen2 = run(cold)
assert torch.equal(y, z) and seen == seen2
# Turn off the low precision reduction to compare against the original masked graph and a separate all-high group.
for block in cold.blocks:
    block.ptq_window["low_fraction"] = 0.0
high = forward(b, cold, x)
for block in cold.blocks:
    block.ptq_routing_active = False
original = forward(b, cold, x)
assert torch.allclose(high, original, rtol=1e-5, atol=1e-6) and not torch.equal(y, high)
atomic_json(
    out / "window_graph.json",
    {
        "request": request,
        "binding": binding,
        "ratio_search": search,
        "ratio_settings": settings,
        "reference_source": str(src),
        "reference_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "model_dir": str(folder.relative_to(out)),
        "heldout_id": sid,
        "orientation_views": ["original", "flipped"],
        "shifted_calls": [{"low_bits_override": n, "windows": v} for n, v in seen],
        "all_high_original_max_abs": float((original - high).abs().max()),
        "mixed_all_high_max_abs": float((high - y).abs().max()),
        "heldout_sha256": tensor_digest(z),
        "reload_max_abs": float((z - y).abs().max()),
    },
)
