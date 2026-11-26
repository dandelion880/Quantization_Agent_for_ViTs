from pathlib import Path
from copy import deepcopy
import sys, torch, traceback, json
from runtime.paths import ROOT
from tests.test_calibration_search import fixture
from ptq_agent.execution.model import tensor_digest
from tools.frozen import FrozenData
from ptq_agent.execution.pipeline import QuantizationExecutor, quantizers
from ptq_agent.execution.model import forward
from ptq_agent.execution.calibration_search import state_identity
from ptq_agent.execution.frozen_trials import score_frozen_candidates
from core.journal import atomic_json

root = Path(__file__).resolve().parent
out = root / "numerical"
out.mkdir(exist_ok=True)
b, cfg, data, base = fixture()
cfg["loop"].pop("allowed_weight_methods", None)
site = "blocks.0.attn.qkv"
b1 = "blocks.1"
cases = {}
for kind in (
    "fixed_token",
    "fixed_group",
    "qep",
    "erq",
    "duq",
    "qwt",
    "bias_compensation",
    "channel_affine",
    "qdrop",
    "scale_only",
    "evol",
):
    p = deepcopy(base)
    qs = {l["module"]: l for l in p["layers"] if l["kind"] == "linear"}
    l = qs[site]
    if kind.startswith("fixed"):
        a = l["activation"]
        a.update(
            granularity="per_token" if kind.endswith("token") else "per_group",
            axis=2,
            group_size=None if kind.endswith("token") else 4,
        )
        l["activation_method"] = "rtn"
    elif kind in ("qep", "erq", "duq"):
        p["execution"].update(order="a_then_w", within_stage="sequential")
        if kind == "qep":
            l.update(
                weight_method="rtn",
                options={"qep": {"alpha": 0.5, "relative_damping": 1.0}},
            )
        else:
            l.update(
                weight_method="erq",
                options={
                    "erq": {
                        "lambda_a": 1.0,
                        "lambda_w": 0.01,
                        "moments": "mean",
                        "flip_steps": 3,
                        "flip_k": 1,
                        **({"tail_rule": "author_code"} if kind == "duq" else {}),
                    }
                },
            )
    elif kind in ("qwt", "bias_compensation", "channel_affine"):
        p["postfit"] = (
            [
                {
                    "kind": kind,
                    "module": f"blocks.{i}",
                    "reference": "local_fp_candidate",
                }
                for i in (0, 1)
            ]
            if kind == "qwt"
            else [
                {
                    "kind": kind,
                    "module": site,
                    "reference": "local_fp_candidate",
                    "reduce_axes": [0] if kind == "bias_compensation" else [0, 1],
                }
            ]
        )
    elif kind == "evol":
        p["postfit"] = [
            {
                "kind": "scale_search",
                "module": b1,
                "reference": "fp_model",
                "settings": {
                    "variant": "evol_paper",
                    "population": 3,
                    "tournament": 2,
                    "cycles": 2,
                    "radius": 0.0001,
                    "temperature": 3.0,
                    "normalize": True,
                    "seed": 5,
                },
            }
        ]
    else:
        p["postfit"] = [
            {
                "kind": "block_reconstruction",
                "module": b1,
                "reference": "fp_model",
                "settings": {
                    "update": "scale" if kind == "scale_only" else "rounding_and_scale",
                    "seed": 2,
                    "steps": 4,
                    "batch_images": 2,
                    "drop_probability": 0 if kind == "scale_only" else 0.5,
                    "scale_lr": 0.001,
                    "round_lr": 0.001,
                },
            }
        ]
    cases[kind] = p
for name, cfg0 in {
    "two_channel_scales": {
        "channel_axis": 2,
        "kmeans_steps": 5,
        "shifts": [0, 1, 2],
        "base_factors": [0.8, 1.0, 1.2],
    },
    "patch_precision": {
        "prefix": 1,
        "importance": "incoming_adapter",
        "unbiased": False,
        "percentile": 1.0,
    },
    "three_region": {
        "high_shift_floor": 1,
        "upper_percentile": 1.0,
        "base_factors": [0.5, 1.0, 1.5],
    },
    "focus_interval": {"initial_bounds": [0.01, 0.95]},
    "tanq": {"pairs": [[2.0, 0.5], [2.2, 0.45]], "inverse_round": "analytic"},
    "sulq": {"eta": [0.001, 0.01, 0.02]},
    "shift_log": {
        "shift_mode": "fixed_calibration",
        "steps_per_octave": 1,
        "reserve_underflow": True,
        "epsilon": 1e-6,
        "percentiles": [0.999, 1.0],
    },
    "bit_window": {"domain": "probability", "shift_bits": 4},
    "pot_error": {"clip_factors": [0.75, 1.0], "neighbors": 1},
    "igq_ranges": {"mode": "channel", "groups": 3, "steps": 3},
    "adfq_outliers": {"threshold": 1.5, "percentiles": [0.999, 1.0]},
    "hc_rows": {"curvature": 0.5, "iterations": 4, "seed": 8},
    "noisy_shift": {
        "seed": 17,
        "distribution": "gaussian",
        "objective": "linear_output",
        "amplitudes": [0.001, 0.05],
        "means": [0.0],
    },
    "median_reparam": {"aggregate": "median", "round_shared_zero": True},
    "centered_smoothing": {"alpha": 0.5, "pot_scale": True},
}.items():
    p = deepcopy(base)
    site0 = next(row for row in p["layers"] if row["module"] == site)
    if name in ("tanq", "sulq", "bit_window", "patch_precision", "focus_interval"):
        site0 = next(row for row in p["layers"] if row["kind"] == "post_softmax")
    if name in ("shift_log", "three_region"):
        site0 = next(row for row in p["layers"] if row["module"].endswith("mlp.fc2"))
    if name == "hc_rows":
        site0.update(
            weight_method=name,
            options={"row_codebook": cfg0},
            activation=None,
            activation_method=None,
        )
        site0.pop("activation_calibration")
    elif name in ("noisy_shift", "median_reparam", "centered_smoothing"):
        p["preprocess"] = [
            {
                "module": site,
                "method": "repq_ln" if name == "median_reparam" else name,
                "norm": None if name == "noisy_shift" else "blocks.0.norm1",
                "settings": cfg0,
            }
        ]
    else:
        site0.update(activation_method=name, options={"encoding": cfg0})
        if name == "adfq_outliers":
            site0["activation"].update(granularity="per_token", axis=2)
        if name in ("two_channel_scales", "pot_error"):
            site0["activation"].update(format="INT4", symmetric=True)
    if name == "hc_rows":
        p["layers"] = [site0]
    cases[name] = p
for key in ("allowed_activation_methods", "allowed_preprocess"):
    cfg["loop"].pop(key, None)
for kind in (
    "gelu_to_relu",
    "cached_kv",
    "first_batch_rows",
    "fpcs",
    "pd_code",
    "pd_paper",
    "mgrq",
    "low_rank",
    "full_correction",
):
    p = deepcopy(base)
    if kind == "gelu_to_relu":
        p["structure"] = [{"kind": kind, "module": "blocks.0.mlp.act"}]
    elif kind == "cached_kv":
        p["structure"] = [
            {"kind": kind, "module": "blocks.0.attn", "copies": 2, "candidates": 4}
        ]
    elif kind == "first_batch_rows":
        row = next(r for r in p["layers"] if r["module"] == site)
        row["activation"].update(granularity="per_token", axis=2)
        row.update(
            activation_method="rtn",
            activation_calibration={
                "strategy": kind,
                "first_batch_images": 2,
                "percentiles": [0.99, 0.999],
            },
        )
    elif kind == "fpcs":
        row = next(r for r in p["layers"] if r["kind"] == "post_softmax")
        row.update(
            activation_method="adalog",
            options={"reserve_zero": True},
            activation_calibration={
                "strategy": "fpcs",
                "metric": "mse",
                "base_values": [16, 32, 48],
                "factors": [0.7, 1.0, 1.2],
                "beam": 2,
                "subdivisions": 3,
                "refinements": 2,
            },
        )
    elif kind in ("pd_code", "pd_paper", "mgrq"):
        objective = {
            "kind": kind,
            **(
                {"alpha": 0.2, "beta": 0.3, "inner_modules": ["mlp.fc2", "attn.qkv"]}
                if kind == "mgrq"
                else {"lambda_r": 0.5, "temperature": 2.0}
            ),
        }
        p["postfit"] = [
            {
                "module": b1,
                "kind": "block_reconstruction",
                "reference": "fp_model",
                "settings": {
                    "update": "scale",
                    "drop_probability": 0.0,
                    "steps": 3,
                    "batch_images": 2,
                    "seed": 2,
                    "scale_lr": 0.0001,
                    "objective": objective,
                },
            }
        ]
    else:
        ranks = {
            r["module"]: 2
            for r in p["layers"]
            if r["kind"] == "linear" and r["module"].startswith(b1 + ".")
        }
        p["postfit"] = [
            {
                "module": b1,
                "kind": "merged_weight_fit",
                "reference": "fp_model",
                "settings": {
                    "parameterization": kind,
                    "batch_images": 2,
                    "prefix": "quantized",
                    "ranks": ranks,
                    "init_std": 0.02,
                    "easy_to_hard": True,
                    "steps": 3,
                    "seed": 13,
                    "lr": 0.002,
                },
            }
        ]
    cases[kind] = p
from ptq_agent.execution.assembled_recipes import assemble
from source_cases import representation_cases, scope

selected_names = (
    set(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--only" in sys.argv
    else None
)
for source, settings in {
    **representation_cases(b, base, data, cases),
    "aiqvit": {
        "focus_interval": {"initial_bounds": [0.005, 0.8]},
        "fit": {
            "parameterization": "low_rank",
            "prefix": "quantized",
            "batch_images": 2,
            "init_std": 0.03,
            "lr": 0.001,
            "steps": 3,
            "seed": 19,
            "learn_intervals": True,
            "loss": "frobenius",
            "easy_to_hard": True,
            "rank_search": {
                "weight_ids": data.ids("calibration")[:2],
                "architecture_ids": data.ids("calibration")[2:],
                "candidates": {
                    r["module"]: [1, 2] for r in base["layers"] if r["kind"] == "linear"
                },
                "init_std": 0.03,
                "drop_path": 0.1,
                "steps": 3,
                "weights_lr": 0.001,
                "architecture_lr": 0.003,
            },
        },
    },
    "apq-vit": {
        "fit": {
            "metric": {
                "kind": "apq_masked_gradient_error",
                "target": "teacher_class",
                "mask_rule": "text_top_error",
                "gamma": 0.75,
            },
            "rounds": 2,
            "factors": [0.8, 1.0, 1.2],
            "operand_order": [[site, "activation"], [site, "weight"]],
        }
    },
    "ptq-vit-2021": {
        "fit": {
            "attention_ranking": {
                "module": "blocks.0.attn",
                "key_axis": -1,
                "ties": "paper_zero_sign",
                "pair_reduction": "sum",
                "margin": 0.1,
                "gamma": 0.01,
            },
            "mean_bias_sites": [site],
            "metric": {"kind": "pearson_error", "axes": [1, 2]},
            "rounds": 2,
            "factors": [0.85, 1.0, 1.15],
            "operand_order": [[site, "activation"], [site, "weight"]],
        }
    },
    "q-hyvit": {
        "fit": {
            "metric": {"kind": "squared_gradient_error", "target": "teacher_class"},
            "rounds": 2,
            "factors": [0.8, 1.0, 1.2],
            "operand_order": [[site, "activation"]],
            "upstream_bridge": True,
        }
    },
    "bit-shrinking": {
        "fit": {
            "start_bits": 6,
            "lr": 0.00001,
            "steps_per_width": 2,
            "sharpness_images": 2,
            "plateau": {"patience": 1},
            "batch_images": 2,
            "seed": 12,
            "max_stages": 16,
            "ceiling_fraction": 0.45,
            "tolerance_fraction": 0.075,
            "bisection_steps": 12,
        }
    },
    "biascomp": {"reference": "local_fp_candidate"},
    "qwt": {},
    "qwt-v2": {},
    "erq-icml2024": {
        "repq": {"round_shared_zero": True},
        "erq": {
            "lambda_a": 1.0,
            "lambda_w": 0.1,
            "moments": "sum",
            "flip_steps": 3,
            "flip_k": 1,
        },
    },
    "erq-tpami2025-duq": {
        "repq": {"round_shared_zero": True},
        "erq": {
            "lambda_a": 1.0,
            "lambda_w": 0.1,
            "moments": "sum",
            "flip_steps": 3,
            "flip_k": 1,
            "tail_rule": "author_code",
            "augment_bias": True,
            "partition_reference": "post_aqer",
        },
    },
    "qep": {"qep": {"alpha": 0.5, "relative_damping": 0.5}},
    "repquant": {
        "repq": {
            "round_shared_zero": True,
            "learned_clip": {"initial_logit": 2.0, "steps": 4, "lr": 0.01},
        },
        "gptq": {"blocksize": 8, "damping": 0.1, "actorder": False},
    },
    "fima-q": {
        "fit": {
            "steps": 3,
            "seed": 5,
            "batch_images": 2,
            "scale_lr": 0.0003,
            "drop_probability": 0.0,
            "update": "scale",
            "objective": {
                "kind": "fima_official_abs_normalized",
                "temperature": 2.0,
                "solver": "pinv",
                "rank": 2,
                "refresh_interval": 1,
            },
        }
    },
    "aphq-vit": {
        "mlp_replacements": [
            {
                "kind": "gelu_to_relu",
                "module": "blocks.0.mlp.act",
                "fit": {
                    "steps": 2,
                    "seed": 5,
                    "batch_images": 2,
                    "lr": 0.0004,
                    "percentile": 0.98,
                    "cache_mib": 32,
                    "mlp": "blocks.0.mlp",
                    "weight_modules": ["blocks.0.mlp.fc1", "blocks.0.mlp.fc2"],
                    "objective": {
                        "kind": "aph_official_abs",
                        "temperature": 2.0,
                        "delta": 0.0001,
                    },
                },
            }
        ],
        "fit": {
            "steps": 2,
            "seed": 5,
            "batch_images": 2,
            "scale_lr": 0.0002,
            "round_lr": 0.001,
            "drop_probability": 0.5,
            "update": "rounding_and_scale",
            "objective": {
                "kind": "aph_official_abs",
                "temperature": 2.0,
                "delta": 0.0001,
            },
        },
    },
    "iptq-vit": {
        "stages": [
            {
                "kind": "integer_function",
                "module": "blocks.0.mlp.act",
                "operator": "gelu",
                "candidates": [
                    {
                        "function": "quartic_gelu",
                        "input_bits": 8,
                        "right_shift": 8,
                        "offsets": [-2.4, -2.7, -3.0],
                        "operation_count": 36,
                    },
                    {
                        "function": "quartic_gelu",
                        "input_bits": 8,
                        "right_shift": 12,
                        "coefficients": [-0.019913, -2.698088],
                        "operation_count": 36,
                    },
                ],
            },
            {
                "kind": "integer_function",
                "module": "blocks.1.attn",
                "operator": "softmax",
                "candidates": [
                    {
                        "function": "efficient_bit_softmax",
                        "input_bits": 8,
                        "fraction_shift": 6,
                        "reciprocal_bits": 40,
                        "output_bits": 6,
                        "operation_count": 55,
                    }
                ],
            },
        ]
    },
    "practicaledge": {
        "stages": [
            {
                "kind": "integer_function",
                "module": "blocks.0.attn",
                "operator": "softmax",
                "candidates": [
                    {
                        "function": "edge_softmax",
                        "input_bits": 8,
                        "bound_bits": bits,
                        "exp_coefficients": [0.36, 0.97, 1.0],
                        "exp_shift": 3,
                        "requant_factor": 256,
                        "operation_count": 55,
                    }
                    for bits in (12, 14)
                ],
            }
        ]
    },
    "pfcr-vit": {
        "activation_phase": {
            "post_ln": "per_tensor",
            "steps": 2,
            "lr": 0.001,
            "batch_images": 2,
            "cache_mib": 32,
            "seed": 1,
        },
        "quantized_phase": {
            "steps": 2,
            "lr": 0.001,
            "batch_images": 2,
            "cache_mib": 32,
            "seed": 1,
        },
    },
    "is-vit": {
        "repq": {"round_shared_zero": True},
        "encoding": {"eta": [0.001, 0.01]},
        "activation_phase": {
            "post_ln": "per_channel",
            "steps": 2,
            "lr": 0.0004,
            "batch_images": 2,
            "cache_mib": 32,
            "seed": 15,
        },
        "quantized_phase": {
            "steps": 2,
            "lr": 0.0004,
            "batch_images": 2,
            "cache_mib": 32,
            "seed": 15,
        },
    },
    "qdrop": {
        "fit": {
            "steps": 3,
            "seed": 6,
            "batch_images": 2,
            "scale_lr": 0.001,
            "round_lr": 0.001,
            "drop_probability": 0.5,
            "update": "rounding_and_scale",
        }
    },
    "pd-quant": {
        "fit": {
            "steps": 2,
            "seed": 6,
            "batch_images": 2,
            "scale_lr": 0.001,
            "round_lr": 0.001,
            "drop_probability": 0.5,
            "update": "rounding_and_scale",
            "objective": {"kind": "pd_code", "temperature": 2.0, "lambda_r": 0.1},
        }
    },
    "mgrq": {
        "fit": {
            "steps": 2,
            "seed": 5,
            "batch_images": 2,
            "scale_lr": 0.0002,
            "drop_probability": 0.0,
            "update": "scale",
            "objective": {
                "kind": "mgrq",
                "alpha": 0.15,
                "beta": 0.2,
                "inner_modules": ["mlp.fc2", "attn.qkv"],
            },
        }
    },
    "cpt-v": {
        "passes": 1,
        "sweep_direction": "forward",
        "search": {
            "variant": "cpt_paper",
            "population": 3,
            "tournament": 2,
            "cycles": 2,
            "radius": 0.0001,
            "temperature": 2.0,
            "normalize": True,
            "seed": 5,
        },
    },
    "evol-q": {
        "passes": 1,
        "sweep_direction": "reverse",
        "search": {
            "targets": ["weight"],
            "multiplier": 6.0,
            "variant": "evol_qavit_fixed_final",
            "population": 3,
            "tournament": 2,
            "cycles": 2,
            "radius": 0.0001,
            "temperature": 3.0,
            "normalize": True,
            "seed": 5,
        },
    },
}.items():
    req = {
        "source_id": source,
        "version": "paper; explicit reduced settings in this request",
        "profile": "qavit_bounded_integration",
        "base": deepcopy(base),
        "settings": settings,
        "evidence": ["docs/methods/index.json#" + source],
    }
    if selected_names is not None and "recipe_" + source not in selected_names:
        continue
    scope(source, req)
    if source in ("apq-vit", "ptq-vit-2021", "bit-shrinking"):
        req["units"] = ["blocks.0"]
    if source == "q-hyvit":
        req["units"] = ["blocks.1"]
    if source == "qep":
        for row in req["base"]["layers"]:
            if row["kind"] == "linear":
                row.update(weight_method="rtn")
    if source == "practicaledge":
        req["settings"]["stages"].extend(
            [
                {
                    "kind": "calibrated_operator",
                    "module": "blocks.0.norm1",
                    "settings": {
                        "function": "edge_layernorm",
                        "input_bits": 16,
                        "output_bits": 8,
                        "left_shift": 15,
                    },
                },
                {
                    "kind": "calibrated_operator",
                    "module": "blocks.0.mlp.act",
                    "settings": {
                        "function": "edge_gelu",
                        "input_bits": 8,
                        "output_bits": 8,
                        "erf_coefficients": [-0.2888, -1.769, 1.0],
                    },
                },
            ]
        )
    if "--cold" in sys.argv:
        from core.journal import read_json

        p = None  # restoration uses only the frozen manifest and generated tensor state
    else:
        p, binding = assemble(b, req, data)
        dest = out / ("recipe_" + source)
        dest.mkdir(exist_ok=True)
        atomic_json(dest / "source_request.json", req)
        atomic_json(dest / "binding.json", binding)
    cases["recipe_" + source] = p
p = deepcopy(base)
p["structure"] = [
    {
        "kind": "outlier_tokens",
        "module": "blocks.0",
        "end_module": "blocks.1",
        "prefix_tokens": 1,
        "grid_hw": [
            a // p
            for a, p in zip(
                b.model.patch_embed.img_size, b.model.patch_embed.patch_size
            )
        ],
        "delete_count": 1,
    }
]
cases["token_route"] = p
from core.journal import read_json

for alias, source in (("daquant_frozen", "daquant"), ("qwt_v2_integer_fold", "qwt-v2")):
    if selected_names is not None and alias not in selected_names:
        continue
    req = read_json(out / ("recipe_" + source) / "source_request.json")
    if source == "daquant":
        req["settings"]["activation_clip"]["range_mode"] = "frozen_channels"
    else:
        req["settings"]["integer_fold"] = {
            "output_bits": 8,
            "output_scale": 0.005,
            "output_zero": 127,
            "fractional_bits": 28,
            "accumulator_bits": 32,
        }
    if "--cold" in sys.argv:
        cases[alias] = None
    else:
        cases[alias], binding = assemble(b, req, data)
        dest = out / alias
        dest.mkdir(exist_ok=True)
        atomic_json(dest / "source_request.json", req)
        atomic_json(dest / "binding.json", binding)
choose = (
    set(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--only" in sys.argv
    else set(cases)
)
require_names = set(cases)
if choose - set(cases):
    raise ValueError(choose - set(cases))
cases = {k: v for k, v in cases.items() if k in choose}
summ = (
    read_json(out / "summary.json")
    if "--only" in sys.argv and (out / "summary.json").exists()
    else {}
)
if "--cold" in sys.argv:
    from core.journal import read_json

    expected = read_json(out / "summary.json")
    restored = {}
    for kind, p in cases.items():
        assert expected[kind]["status"] == "passed", kind
        cold = QuantizationExecutor(b, cfg, FrozenData(data))
        m, manifest = cold.load(out / kind / "model")
        x, sid = next(data.batches("search"))
        sha = tensor_digest(forward(b, m, x))
        assert (
            sha == expected[kind]["output_sha256"]
            and state_identity(m) == expected[kind]["state_sha256"]
        )
        restored[kind] = {
            "heldout_output_sha256": sha,
            "state_sha256": state_identity(m),
        }
    atomic_json(out / "process_cold.json", restored)
    sys.exit()
for kind, p in cases.items():
    target = out / kind
    target.mkdir(exist_ok=True)
    atomic_json(target / "request.json", p)
    try:
        e = QuantizationExecutor(b, cfg, data)
        m, w, a, audit = e.prepare(p, {})
        package = target / "model"
        import tempfile, shutil

        work = Path(tempfile.mkdtemp(dir=target))
        try:
            e.export(m, p, w, a, work / "model", audit)
            if package.exists():
                shutil.rmtree(package)
            (work / "model").rename(package)
        finally:
            shutil.rmtree(work)
        cold = QuantizationExecutor(b, cfg, FrozenData(data))
        m2, manifest = cold.load(package)
        x, sid = next(data.batches("search"))
        f, y, z = [forward(b, n, x).double() for n in (b.model, m, m2)]
        assert torch.equal(y, z) and state_identity(m) == state_identity(m2)
        summ[kind] = {
            "status": "passed",
            "state_sha256": state_identity(m2),
            "weights_sha256": manifest["weights_sha256"],
            "output_sha256": tensor_digest(z.float()),
            "heldout_id": sid,
            "mse": float((f - z).square().mean()),
            "cold_max_abs": float((y - z).abs().max()),
            "postfit": manifest["calibration_report"].get("postfit", []),
        }
    except Exception:
        summ[kind] = {"status": "failed", "error": traceback.format_exc()}
    atomic_json(out / "summary.json", summ)
    print(kind, summ[kind]["status"], summ[kind].get("error", ""), flush=True)


assert all(summ[k]["status"] == "passed" for k in cases), [
    k for k in cases if summ[k]["status"] != "passed"
]
