"""Replay configuration choices, numerical evidence and cold state for precision-allocation recipes."""

from copy import deepcopy
from pathlib import Path
import sys, tempfile
import torch
from runtime.paths import ROOT
from core.journal import atomic_json, read_json
from tests.test_calibration_search import fixture
from tools.frozen import FrozenData
from ptq_agent.execution.assembled_recipes import assemble
from ptq_agent.execution.precision import select
from ptq_agent.execution.model import forward, tensor_digest
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.calibration_search import state_identity
from ptq_agent.execution.layer_importance import positive_linear
from ptq_agent.execution.plugin_contracts import site_id

out = Path(__file__).resolve().parent
b, cfg, data, base = fixture()
for k in ("allowed_weight_methods", "allowed_activation_methods", "allowed_preprocess"):
    cfg["loop"].pop(k, None)
cfg["constraints"]["allowed_formats"] += [
    "UINT2",
    "UINT3",
    "UINT5",
    "UINT6",
    "UINT8",
    "INT2",
    "INT3",
    "INT5",
    "INT8",
]
sites = ["blocks.0.attn.qkv", "blocks.1.attn.qkv"]
blocks = [s.rsplit(".attn", 1)[0] for s in sites]
N = b.model.get_submodule(sites[0]).weight.numel()
for x, sid in data.batches("calibration"):
    data.labels[sid] = int(forward(b, b.model, x).argmax(-1))
config = {
    "alternatives": {
        s: [
            {"weight": "UINT" + str(i), "activation": "UINT" + str(i)}
            for i in (2, 3, 4)
        ]
        for s in sites
    },
    "budgets": {"weight_bits": 6 * N},
    "loss": "output_mse",
}
common = {
    "source_id": "",
    "version": "paper; explicit reduced settings",
    "profile": "bounded allocation and packed software QDQ",
    "base": base,
    "units": blocks,
    "settings": {},
    "evidence": [],
}
requests = {}


def add(source, allocation, settings, base_plan=base):
    req = {
        **common,
        "source_id": source,
        "base": base_plan,
        "settings": settings,
        "allocation": allocation,
        "evidence": [
            "methods/"
            + ("adalog.md" if source == "adalog" else "recipes/" + source + ".md")
        ],
    }
    requests[source] = req


repq = {
    "repq": {
        "round_shared_zero": True,
        "learned_clip": {"initial_logit": 2.0, "steps": 1, "lr": 0.01},
    }
}
rel = {
    "blocks": [{"block": u, "attention": u + ".attn"} for u in blocks],
    "target": "teacher_class",
    "jacobian_chunk": 16,
    "site_blocks": dict(zip(sites, blocks)),
}
add("lrp-qvit", {**config, "relevance": rel}, repq)
add(
    "mix-qvit",
    {
        **config,
        "allocation_rule": "normalized_relative_loss",
        "relevance": rel,
        "mac_counts": {s: 5 * N for s in sites},
        "budgets": {"weight_bits": 6 * N, "bitops": 2 * N * 5 * 3 * 3},
    },
    repq,
)
add(
    "lampq",
    {
        **config,
        "loss": "top1_error",
        "allocation_rule": "lampq_initial",
        "gamma": 2.0,
        "fisher": {"statistic": "trace", "target": "teacher_class", "seed": 19},
        "type_calibration": {
            "types": {s: "qkv" for s in sites},
            "probes": {"qkv": {sites[0]: {"weight": "UINT2", "activation": "UINT2"}}},
        },
        "feedback": {
            "max_updates": 2,
            "formula": "neighbor_difference",
            "budget_rule": "require_feasible",
        },
    },
    {},
)
groups = [
    {
        "choices": [4, 8],
        "sites": {
            site_id(r): [v for v in ("weight", "activation") if r[v]]
            for r in base["layers"]
            if r["module"].startswith(u + ".")
        },
    }
    for u in blocks
]
pom = {
    "configuration_groups": groups,
    "encoder_units": [{"attention": u + ".attn", "output": u} for u in blocks],
    "max_configurations": 4,
    "limits": {},
    "objectives": ["E1_qkv", "E2_attention", "E3_encoder", "latency"],
    "priority": ["E3_encoder", "latency"],
    "error_norm": 1,
    "latency": {"kind": "software_qdq", "warmup": 1, "repeats": 3},
}
add("pomq-vit", pom, {})
patch = read_json(out / "numerical/recipe_patchwise-mpq-vit/source_request.json")
add(
    "patchwise-mpq-vit",
    {
        **config,
        "fisher": {"target": "model_distribution", "seed": 9, "statistic": "gsm"},
    },
    {k: v for k, v in patch["settings"].items() if k != "fixed_precision"},
)
for source in ("amp-vit", "mptq-vit"):
    start = read_json(out / ("numerical/recipe_" + source + "/source_request.json"))
    add(
        source,
        {
            "allocation_rule": "greedy_operand_sqnr",
            "sites": {
                "weight": [sites[0], sites[1]],
                "activation": ["blocks.0.mlp.fc2", "blocks.1.mlp.fc2"],
            },
            "minimum": {"weight": 3, "activation": 3},
            "average_bits": {"weight": 3.5, "activation": 3.5},
            "operand_order": ["weight", "activation"],
        },
        start["settings"],
    )
rankreq = read_json(out / "numerical/recipe_ptq-vit-2021/source_request.json")
rankreq["allocation"] = {
    "configuration_groups": groups,
    "max_configurations": 4,
    "limits": {},
    "objectives": ["nuclear_error", "quantized_weight_bits"],
    "priority": ["quantized_weight_bits", "nuclear_error"],
    "nuclear_attention": [u + ".attn" for u in blocks],
    "nuclear_taps": [p for u in blocks for p in (u + ".attn._ptq_softmax", u + ".mlp")],
}
requests["ptq-vit-2021"] = rankreq
bridge = read_json(out / "numerical/recipe_q-hyvit/source_request.json")
bridge["allocation"] = {
    "configuration_groups": [
        {
            "sites": {sites[0]: ["scheme"]},
            "choices": [
                {
                    "weight": {
                        "method": "uniform_minmax",
                        "spec": {
                            "granularity": g,
                            "axis": axis,
                            "format": fmt,
                            "bits": 4,
                            "symmetric": fmt == "INT4",
                            "group_size": None,
                        },
                    }
                }
                for g, axis, fmt in (
                    ("per_tensor", None, "INT4"),
                    ("per_channel", 0, "UINT4"),
                )
            ],
        }
    ],
    "bridge": {"module": "blocks.1", "target": "teacher_class", "cache_mib": 16},
    "max_configurations": 2,
    "limits": {},
    "objectives": ["squared_gradient_error"],
    "priority": ["squared_gradient_error"],
}
requests["q-hyvit"] = bridge
igq = read_json(out / "numerical/recipe_igq_vit/source_request.json")
groupsites = ["blocks.0.attn.proj", "blocks.1.attn.proj"]
costs = {}
choices = {}
for n in groupsites:
    tokens = 1 + int(b.model.patch_embed.num_patches)
    c = b.model.get_submodule(n).in_features
    d = b.model.get_submodule(n).out_features
    choices[n] = [2, 3]
    costs[n] = {
        str(g): {
            "minmax": 2 * tokens * c * 32,
            "assignment": 4 * g * c * 32,
            "scaled_sum": 2 * g * tokens * d * 32,
            "matmul": tokens * c * d * 4 * 4,
        }
        for g in choices[n]
    }
igq["allocation"] = {
    "allocation_rule": "group_count_schedule",
    "groups": choices,
    "alternating_steps": 4,
    "reallocate_every": 2,
    "budget": sum(sum(costs[n][str(v)].values()) for n, v in zip(groupsites, (2, 3))),
    "fixed_bops": 0,
    "costs": costs,
    "cost_scope": {
        "minmax": "two 32-bit comparisons per observation",
        "assignment": "four 32-bit range operations per channel and group",
        "scaled_sum": "multiply and add per group/output, 32-bit",
        "matmul": "W4 x A4 MACs in selected sites; others excluded",
    },
}
requests["igq_vit"] = igq
pot = read_json(out / "numerical/recipe_p2-vit/source_request.json")
pot["allocation"] = {
    "allocation_rule": "hessian_genetic",
    "sites": sites,
    "widths": [4, 8],
    "weight_bits": 12 * N,
    "fitness": "teacher_top1",
    "hessian": {"target": "teacher_class", "seed": 231, "vectors": 2},
    "genetic": {
        "seed": 73,
        "population": 3,
        "generations": 2,
        "mutations": 4,
        "crossovers": 4,
        "probability": 0.5,
    },
}
requests["p2-vit"] = pot
ids = sys.argv[1:] or list(requests)
path = out / "allocation_recipes.json"
book = read_json(path) if path.exists() else {}
for source in ids:
    req = requests[source]
    plan, binding = assemble(b, req, data)
    chosen, log = select(b, cfg, data, plan, req["allocation"])
    exe = QuantizationExecutor(b, cfg, data)
    m, w, a, au = exe.prepare(chosen, {})
    dest = Path(tempfile.mkdtemp(prefix="allocated_" + source + "_", dir=out)) / "model"
    exe.export(m, chosen, w, a, dest, au)
    cold, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(dest)
    x, sid = next(data.batches("search"))
    old = forward(b, m, x)
    y = forward(b, cold, x)
    assert torch.equal(old, y) and state_identity(cold) == state_identity(m)
    book[source] = {
        "request": req,
        "binding": binding,
        "decision": log,
        "model_dir": str(dest.relative_to(out)),
        "heldout_id": sid,
        "heldout_sha256": tensor_digest(y),
        "state_sha256": state_identity(m),
        "code_payload_sha256": manifest["weights_sha256"],
    }
    atomic_json(path, book)
    print(source, "success", flush=True)
