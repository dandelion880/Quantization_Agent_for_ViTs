"""Independent byte and signed requantization oracles for deployed payloads."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import hashlib
import torch
from runtime.paths import ROOT
from core.journal import atomic_json
from ptq_agent.execution import codecs, grids, integer_affine
from ptq_agent.execution.layer_importance import positive_linear
from ptq_agent.execution.paired_weights import correct_weight, erq, decode

out = Path(__file__).resolve().parent
rows = {}
for bits in range(2, 11):
    top = (1 << bits) - 1
    values = [0, 1, top, top - 1, top // 2, top // 2 + 1, 1]
    source = torch.tensor(values, dtype=torch.int16)
    packed = codecs.pack(source, bits)
    oracle = sum(v << (i * bits) for i, v in enumerate(values)).to_bytes(
        (len(values) * bits + 7) // 8, "little"
    )
    assert bytes(packed.tolist()) == oracle and torch.equal(
        codecs.unpack(packed, bits, source.shape, "cpu").long(), source.long()
    )
    spec = {
        "format": f"UINT{bits}",
        "bits": bits,
        "granularity": "per_tensor",
        "axis": None,
        "symmetric": False,
        "scale": "minmax",
        "group_size": None,
    }
    w = source.float().reshape(1, -1)
    pack = grids.encode(
        w,
        spec,
        {"scale": torch.ones(1, 1), "zero": torch.zeros(1, 1)},
        method="uniform_affine",
    )
    assert (
        torch.equal(grids.decode(pack, "cpu"), w)
        and bytes(pack["data"].tolist()) == oracle
    )
    rows[str(bits)] = {"bytes": oracle.hex(), "values": values}

# Three true output channels exercise negative multiplier, signed ties, constant zero slope and saturation.
spec = {"format": "UINT4", "bits": 4, "granularity": "per_tensor", "symmetric": False}
q = SimpleNamespace(
    layer={
        "kind": "linear",
        "weight": spec,
        "activation": spec,
        "activation_method": "uniform_affine",
    },
    weight_record={
        "method": "uniform_affine",
        "params": {"zero": torch.tensor(5.0), "scale": torch.tensor(1.0)},
    },
    calls=0,
)
q.params = lambda: {"scale": torch.tensor(1.0), "zero": torch.tensor(6.0)}
q.effective_bias = lambda: torch.zeros(3)
q.decoded_weight = torch.tensor([[1.0], [-1.0], [1.0]])
q.reference_weight = q.decoded_weight
q.calibration_input = lambda x: (x.round() + 6).clamp(0, 15) - 6
cfg = {
    "output_bits": 3,
    "output_scale": 1.0,
    "output_zero": 3,
    "fractional_bits": 3,
    "accumulator_bits": 16,
}
alpha = torch.tensor([-0.5, 1.5, 0.0])
beta = torch.tensor([0.0, 0.0, 2.0])
p = integer_affine.compile_state(q, alpha, beta, cfg)
q.ptq_output_correction = SimpleNamespace(**p)
inputs = list(range(-6, 10))
y = integer_affine.integer_forward(
    q, torch.tensor(inputs).float().reshape(-1, 1)
).long()
expected = [
    [
        min(7, max(0, round(v) + 3)) - 3
        for v in (Fraction(-x, 2), Fraction(-3 * x, 2), 2)
    ]
    for x in inputs
]
assert y.tolist() == expected and p["multiplier"].tolist() == [-4, 12, 0]

x = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 4.0, 2.0]], dtype=torch.float64)
w = torch.tensor([[2.0, -1.0, -3.0], [-2.0, 1.0, 2.0]], dtype=torch.float64)
R = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64)
measured, lost = positive_linear(x, w, R, zero_policy="retain_unassigned")
contrib = (x[:, None, :] * w[None, :, :]).clamp_min(0.0)
den = contrib.sum(-1, keepdim=True)
ref = (R[:, :, None] * torch.where(den > 0, contrib / den, 0.0)).sum(1)
assert (
    torch.allclose(ref, measured)
    and abs(float(R.sum() - measured.sum() - lost)) < 1e-12
)
# Bias remains a floating last coordinate after both solves. Confirm against an independent columnwise ridge solve.
import numpy as np

a = np.array([[0.0], [1.0], [-2.0], [0.25]])
b = np.array([[0.3], [0.85], [-1.7], [0.6]])
X = np.column_stack((b, np.ones(len(a))))
target = a * 1.7 + 0.45
orig = np.array([[1.7], [0.45]])
beta = orig + np.linalg.solve(X.T @ X + 0.2 * np.eye(2), X.T @ (target - X @ orig))
spec = {"bits": 4}
old = torch.tensor([0.45], dtype=torch.double)
rec, info = erq(
    torch.tensor([[1.7]], dtype=torch.double),
    torch.tensor(a),
    torch.tensor(b),
    spec,
    {
        "erq": {
            "lambda_a": 0.2,
            "lambda_w": 0.3,
            "moments": "sum",
            "flip_steps": 1,
            "flip_k": 1,
            "augment_bias": True,
        }
    },
    old,
)
hard = float(decode(rec, "cpu")[0, 0])
expected_bias = beta[1, 0] - (hard - beta[0, 0]) * (X[:, 0] @ X[:, 1]) / (
    (X[:, 1] @ X[:, 1]) + 0.3
)
observed = float(old + rec["params"]["bias_delta"])
assert abs(expected_bias - observed) < 1e-6
atomic_json(
    out / "edge_oracles.json",
    {
        "integer_payload_little_endian": rows,
        "channel_affine": {
            "independent": "exact rational multiplication, Python ties-to-even, integer saturation",
            "codes": (y + 3).tolist(),
            "multipliers": p["multiplier"].tolist(),
        },
        "positive_relevance": {
            "max_abs": float((ref - measured).abs().max()),
            "unassigned": float(lost),
        },
        "erq_unrounded_bias": {
            "numpy_augmented_ridge": expected_bias,
            "decoded_bias": observed,
            "absolute_error": abs(expected_bias - observed),
        },
    },
)
