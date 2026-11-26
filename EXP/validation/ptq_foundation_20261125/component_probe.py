"""Independent numerical references, graph-specific round trips and data/precision stages."""

from pathlib import Path
from copy import deepcopy
import itertools, sys, torch
from torch import nn
from runtime.paths import ROOT
from core.journal import atomic_json, read_json
from tests.test_calibration_search import fixture
from ptq_agent.execution.paired_weights import correct_weight
from ptq_agent.execution.precision import budget_search, select
from ptq_agent.execution.research_metrics import prepare
from ptq_agent.execution.efficient_topology import denominator, lite_forward
from ptq_agent.execution.region_grids import three_fields
from ptq_agent.execution.model import tensor_digest, forward
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.calibration_sources import CalibrationView, synthesize
from ptq_agent.execution.assembled_recipes import assemble
from ptq_agent.execution.calibration_search import state_identity
from ptq_agent.backends.server import state_sha256
from tools.frozen import FrozenData

out = Path(__file__).resolve().parent
b, cfg, data, base = fixture()
for k in ("allowed_preprocess", "allowed_weight_methods", "allowed_activation_methods"):
    cfg["loop"].pop(k, None)
book = {}


def persist():
    atomic_json(out / "component_checks.json", book)


def package(backend, view, plan, dest):
    import tempfile

    exe = QuantizationExecutor(backend, cfg, view)
    m, w, a, audit = exe.prepare(plan, {})
    folder = Path(tempfile.mkdtemp(prefix=dest + "_", dir=out)) / "model"
    exe.export(m, plan, w, a, folder, audit)
    cold, manifest = QuantizationExecutor(backend, cfg, FrozenData(view)).load(folder)
    x, sid = next(data.batches("search"))
    warm = forward(backend, m, x)
    y = forward(backend, cold, x)
    assert torch.equal(warm, y) and state_identity(m) == state_identity(cold)
    return {
        "model_dir": str(folder.relative_to(out)),
        "heldout_id": sid,
        "sha256": tensor_digest(y),
        "state": state_identity(cold),
        "weights": manifest["weights_sha256"],
        "postfit": manifest["calibration_report"].get("postfit", []),
    }


torch.manual_seed(421)
w = torch.randn(4, 5, dtype=torch.float64)
x = torch.randn(13, 5, dtype=w.dtype)
q = x + torch.randn_like(x) * 0.3
actual, H = correct_weight(w, x, q, alpha=1.0, damping=0.13, normalization="sum")
ridge = torch.eye(5, dtype=w.dtype) * 0.13**0.5
expected = torch.linalg.lstsq(
    torch.cat((q, ridge)), torch.cat((x @ w.T, ridge @ w.T))
).solution.T
assert torch.allclose(expected, actual)
book["qep_augmented_least_squares"] = {
    "dtype": "float64",
    "maximum_difference": (expected - actual).abs().max().item(),
}
S = torch.tensor([[2.0, 3.0], [3.0, 5.0]], dtype=w.dtype)
z = torch.randn(4, 2, dtype=w.dtype)
binding = dict(
    reference="FP",
    input_state="four rows",
    suffix_state=S.tolist(),
    head="quadratic",
    task="arithmetic",
    view="known Hessian",
    sample_ids=list(range(4)),
    observation="z",
    axes="BF",
    reduction="sum",
)
prep = prepare(
    "aph_signed_directional",
    z,
    lambda v: (0.5 * (v @ S) * v).sum(),
    binding=binding,
    options={"delta": 0.001},
)
assert torch.allclose(prep["weight"], S.sum(-1))
book["aph_offdiagonal"] = {
    "weighted_direction": prep["weight"].tolist(),
    "diagonal": S.diag().tolist(),
}
sites = {
    "a": [
        {"index": i, "loss": l, "cost": {"b": c, "cycles": k}}
        for i, (l, c, k) in enumerate([(1.0, 2, 3), (2.0, 1, 2), (0.1, 4, 3)])
    ],
    "b": [
        {"index": i, "loss": l, "cost": {"b": c, "cycles": k}}
        for i, (l, c, k) in enumerate([(4.0, 1, 1), (0.2, 2, 3), (1.0, 3, 1)])
    ],
}
budget = {"b": 5, "cycles": 5}
opt = budget_search(sites, budget)
brute = min(
    sum(r["loss"] for r in rows)
    for rows in itertools.product(*sites.values())
    if all(sum(r["cost"][k] for r in rows) <= v for k, v in budget.items())
)
assert opt["predicted_objective"] == brute
book["integer_allocation_exhaustive"] = {**opt, "brute": brute}
settings = {
    "mid_shift": torch.tensor(1),
    "high_shift": torch.tensor(3),
    "scale": torch.tensor(0.25),
}
inputs = torch.arange(-256, 257) * 0.25
fields, y = three_fields(inputs, 4, settings)
oracle = torch.tensor(
    [
        ((code & 7) * 8 if code & 8 else -(code & 3) if code & 4 else (code & 3) * 2)
        * 0.25
        for code in fields.tolist()
    ]
)
assert torch.equal(y, oracle) and fields.min() >= 0 and fields.max() < 16
book["three_region_payload"] = {
    "codes": fields.unique().tolist(),
    "decoded": y.unique().tolist(),
}
persist()
for kind in ("psaq_kde", "sardfq_gmm", "e2h_crop", "clamp_local_adapter"):
    import tempfile

    holder = Path(tempfile.mkdtemp(prefix="source_", dir=out)) / "pool"
    c = deepcopy(read_json(out / "task_metrics.json")["generated"]["generator"])
    c.update(
        variant=kind,
        images=2,
        input_chw=[3, 32, 32],
        classes=1000,
        attention_modules=["blocks.0.attn"],
        crop_min_area=0.5,
        patch_grid=[2, 2],
        window=3,
        temperature=0.5,
        positives=2,
    )
    if kind == "sardfq_gmm":
        c.update(
            prior_sites={"blocks.0.attn": {"grid": [2, 2], "depth_fraction": 0.5}},
            gaussians=[1, 2],
            grid_extents=[4.0, 6.0],
            center_bounds=[-5.0, 5.0],
            covariance_scale=10.0,
            eigen_bounds=[0.05, 5.0],
            normalization="paper_mass",
            max_crops=2,
            semantic_logit_range=[5.0, 10.0],
        )
    rec = synthesize(b, c, holder)
    v = CalibrationView(data, b, holder)
    source, settings = {
        "psaq_kde": ("psaq-vit", {}),
        "sardfq_gmm": (
            "sardfq",
            {
                "fit": {
                    "seed": 7,
                    "steps": 2,
                    "lr": 0.0001,
                    "batch_images": 2,
                    "cache_mib": 16,
                }
            },
        ),
        "e2h_crop": (
            "dfq-vit",
            {"compensation_axes": {"blocks.0.mlp.fc2": [0]}, "reference": "fp_model"},
        ),
        "clamp_local_adapter": (
            "clamp-vit",
            {
                "search": {
                    "variant": "cpt_paper",
                    "population": 3,
                    "tournament": 2,
                    "cycles": 2,
                    "radius": 0.001,
                    "temperature": 2.0,
                    "normalize": True,
                    "seed": 15,
                }
            },
        ),
    }[kind]
    req = {
        "base": base,
        "source_id": source,
        "version": "paper, explicit reduced adapter",
        "profile": "integration",
        "evidence": ["methods/recipes/" + source + ".md"],
        "units": ["blocks.0"],
        "settings": settings,
    }
    plan, binding0 = assemble(b, req, v)
    book[kind] = {
        "source": source,
        "pool": str(holder.relative_to(out)),
        "source_request": req,
        "generated_ids": v.ids("calibration"),
        "phase_losses": rec["loss"],
        "recipe_binding": binding0,
        **package(b, v, plan, source),
    }
    persist()
site = "blocks.0.attn.qkv"
n = b.model.get_submodule(site).weight.numel()
costs = {
    "alternatives": {
        site: [
            {"weight": "UINT4", "activation": "UINT4"},
            {"weight": "UINT8", "activation": "UINT8"},
        ]
    },
    "budgets": {"weight_bits": 4 * n, "bitops": 4 * 4 * n * 5},
    "loss": "output_mse",
    "mac_counts": {site: n * 5},
}
plan, choice = select(b, cfg, data, base, costs)
assert choice["assignment"][site] == 0
book["precision_search"] = {
    "request": costs,
    "assignment": choice,
    **package(b, data, plan, "allocation"),
}
persist()


class Lite(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Conv2d(3, 9, 1)
        self.proj = nn.Conv2d(3, 3, 1)
        self.aggreg = nn.ModuleList()
        self.kernel_func = nn.ReLU()
        self.dim = 3

    def forward(self, x):
        q, k, v = self.qkv(x).flatten(2).transpose(1, 2).chunk(3, -1)
        q = q.relu()
        k = k.relu()
        scores = q @ k.transpose(-1, -2)
        y = (scores @ v) / scores.sum(-1, keepdim=True).clamp_min(1e-5)
        return self.proj(y.transpose(1, 2).reshape_as(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(3, 3, 3, padding=1, groups=3)
        self.pw = nn.Conv2d(3, 3, 1)
        self.att = Lite()

    def forward(self, x):
        return self.att(self.pw(self.dw(x))).mean((2, 3))


b.model = Net().eval().requires_grad_(False)
b.original_state = state_sha256(b.model)
b.spec = {
    "family": "bounded MBConv / normalized linear attention",
    "task_forward": "forward",
}
b._modules = [
    {
        "name": k,
        "type": "Conv2d",
        "protected": False,
        "weight_shape": list(m.weight.shape),
        "block": None,
    }
    for k, m in b.model.named_modules()
    if isinstance(m, nn.Conv2d)
]
p = deepcopy(base)
row = p["layers"][0]
p["layers"] = [{**deepcopy(row), "module": n, "kind": "conv2d"} for n in ("dw", "pw")]
req = {
    "source_id": "trio-vit",
    "version": "5628eb21; bounded exponent grid adapts capacity",
    "profile": "architecture bounded verification",
    "base": p,
    "units": [],
    "evidence": ["methods/recipes/trio-vit.md"],
    "settings": {
        "mbconv_edges": [
            {
                "method": "trio_dw_scale",
                "module": "dw",
                "norm": None,
                "settings": {"percentile": 0.99, "ratio_bounds": [0.1, 64.0]},
            },
            {
                "method": "trio_pw_shift",
                "module": "pw",
                "norm": None,
                "settings": {"center": "extrema_midpoint"},
            },
        ],
        "denominators": [
            {
                "kind": "lite_denominator",
                "module": "att",
                "settings": {
                    "exponent_bounds": [-8, 7],
                    "bits": 4,
                    "version": "bounded_nearest",
                },
            }
        ],
    },
}
p, binding0 = assemble(b, req, data)
values = torch.cat((torch.zeros(1), torch.exp2(torch.arange(-25, 20).float())))
paper = denominator(
    values, {"exponent_bounds": [-8, 7], "bits": 4, "version": "bounded_nearest"}
)
auth = denominator(values, {"exponent_bounds": [-12, 11], "version": "author_5628eb2"})
assert paper.min() > 0 and paper.unique().numel() == 16 and auth.unique().numel() == 24
book["trio_topology"] = {
    "request": req,
    "paper_adapter_levels": 16,
    "author_exponents": 24,
    **package(b, data, p, "trio"),
}
persist()
