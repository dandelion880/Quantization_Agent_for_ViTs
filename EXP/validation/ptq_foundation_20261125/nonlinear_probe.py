"""Compare frozen operator outputs with independent NumPy reductions and expanded GELU arithmetic."""

from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch
from runtime.paths import ROOT
from core.journal import atomic_json
from tests.test_calibration_search import fixture
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.model import forward, tensor_digest
from tools.frozen import FrozenData

out = Path(__file__).resolve().parent
b, cfg, data, base = fixture()
x, sid = next(data.batches("search"))
rows = {}
for recipe in ("h-vit", "practicaledge"):
    target = out / ("numerical/recipe_" + recipe) / "model"
    m, manifest = QuantizationExecutor(b, cfg, FrozenData(data)).load(target)
    all_calls = []

    def hook(n):
        def collect(module, args, result):
            all_calls.append(
                (
                    n,
                    module,
                    args[0].detach().cpu().numpy(),
                    result.detach().cpu().numpy(),
                )
            )

        return collect

    with ExitStack() as scope:
        for n, mod in m.named_modules():
            if hasattr(mod, "ptq_nonlinear"):
                scope.callback(mod.register_forward_hook(hook(n)).remove)
        forward(b, m, x)
    for n, mod, z, actual in all_calls:
        opt = mod.ptq_nonlinear_options
        p = {k: v.cpu().numpy() for k, v in mod.ptq_nonlinear.named_buffers()}
        op = opt["function"]
        ib, ob = opt["input_bits"], opt["output_bits"]
        if op == "hvit_layernorm":
            s = np.exp2(p["exponent"].astype(np.float64))
            y = np.clip(np.rint(z / s), -(2 ** (ib - 1)), 2 ** (ib - 1) - 1) * s
            norm = (y - y.mean(axis=-1, keepdims=True)) / np.sqrt(
                y.var(axis=-1, keepdims=True) + mod.eps
            )
            norm = norm * mod.weight.detach().numpy() + mod.bias.detach().numpy()
            so = np.exp2(p["token_exponent"].astype(np.float64))[None, :, None]
            expected = (
                np.clip(np.rint(norm / so), -(2 ** (ob - 1)), 2 ** (ob - 1) - 1) * so
            )
        else:
            q = (
                np.clip(np.rint(z / p["scale"] + p["zero"]), 0, 2**ib - 1).astype(
                    np.int64
                )
                - p["zero"]
            )
            if op == "edge_layernorm":
                mu = np.rint(q.mean(axis=-1, keepdims=True))
                den = np.rint(
                    np.sqrt(
                        np.maximum(
                            1,
                            np.rint(np.mean(q * q, axis=-1, keepdims=True))
                            - mu * mu
                            + 1,
                        )
                    )
                )
                gain = 2 ** opt["left_shift"]
                ratio = np.rint((q - mu) * gain / den)
                code = (
                    np.rint(
                        ratio * mod.weight.detach().numpy() / (gain * p["out_scale"])
                    )
                    + np.rint(mod.bias.detach().numpy() / p["out_scale"])
                    + p["out_zero"]
                )
            else:
                val = np.minimum(abs(q), -p["b_int"])
                polynomial = (
                    val * val + 2 * p["b_int"] * val + p["b_int"] ** 2 + p["c_int"]
                )
                code = (
                    np.rint(
                        q
                        * (np.sign(q) * polynomial + round(1 / float(p["erf_scale"])))
                        * float(p["scale"])
                        * float(p["erf_scale"])
                        / (2 * float(p["out_scale"]))
                    )
                    + p["out_zero"]
                )
            expected = (np.clip(code, 0, 2**ob - 1) - p["out_zero"]) * p["out_scale"]
        diff = float(np.max(abs(expected.astype(actual.dtype) - actual)))
        assert diff <= 2e-6, (op, n, diff)
        rows[recipe + "/" + n] = {
            "function": op,
            "request": str(target.parent.relative_to(out)) + "/request.json",
            "heldout_id": sid,
            "shape": list(z.shape),
            "max_abs": diff,
        }
assert {v["function"] for v in rows.values()} == {
    "hvit_layernorm",
    "edge_layernorm",
    "edge_gelu",
}
atomic_json(out / "nonlinear_oracles.json", rows)
