"""Bounded representation and graph requests. Budget and target geometry belong to this fixture."""

from copy import deepcopy
from ptq_agent.execution.plugin_contracts import site_id
from ptq_agent.execution.sites import site_catalog


def representation_cases(backend, base, data, shared):
    catalog = site_catalog(backend)
    sample_ids = data.ids("calibration")

    def codec(name, **kwargs):
        source = shared[name]
        row = next(r for r in source["layers"] if r.get("activation_method") == name)
        return {
            k: deepcopy(row[k])
            for k in (
                "activation",
                "activation_method",
                "options",
                "activation_calibration",
            )
        } | kwargs

    def prep(name):
        return shared[name]["preprocess"][0]["settings"]

    post = "post_layernorm"
    probability = "post_softmax"
    gelu = "post_gelu"
    signed = codec("pot_error")
    method_rows = [
        r
        for r in base["layers"]
        if r["kind"]
        in ("linear", "attention_query", "attention_key", "attention_value")
    ]
    axes = {
        site_id(r): (
            2 if r["kind"] in ("linear", "attention_key", "attention_value") else 3
        )
        for r in method_rows
    }
    grouped = codec("igq_ranges")
    row_group = deepcopy(grouped)
    row_group["options"]["encoding"]["mode"] = "attention_row"
    grid = [
        i // p
        for i, p in zip(
            backend.model.patch_embed.img_size, backend.model.patch_embed.patch_size
        )
    ]
    cases = {
        "daquant": {
            "fit": {
                "parameterization": "full_correction",
                "prefix": "quantized",
                "steps": 3,
                "lr": 0.002,
                "seed": 17,
                "batch_images": 1,
                "objective": {
                    "kind": "paired_domain_features",
                    "target_pairs": {
                        sample_ids[0]: sample_ids[-1],
                        sample_ids[1]: sample_ids[-1],
                    },
                    "alignment_weight": 1.5,
                    "feature_axis": 2,
                    "kl_reduction": "batchmean",
                },
            }
        },
        "adalog": {
            "on_semantic": {
                probability: {
                    k: deepcopy(v)
                    for k, v in next(
                        r
                        for r in shared["fpcs"]["layers"]
                        if r["activation_method"] == "adalog"
                    ).items()
                    if k in ("options", "activation_calibration", "activation_method")
                }
            }
        },
        "adfq_vit": {
            "on_semantic": {post: codec("adfq_outliers"), gelu: codec("shift_log")}
        },
        "dopq_vit": {
            "round_shared_zero": True,
            "on_semantic": {probability: codec("tanq")},
        },
        "noisyquant": {"noise": prep("noisy_shift")},
        "igq_vit": {
            "on_semantic": {
                s: (row_group if s == probability else grouped)
                for s in (post, "linear_input", gelu, probability)
            }
        },
        "orq_vit": {
            "feature_axes": axes,
            "initial_observer": {
                "strategy": "first_batch_rows",
                "first_batch_images": 2,
                "percentiles": [0.999, 0.9999, 0.99999],
            },
            "fit": {
                "update": "scale",
                "drop_probability": 0.0,
                "steps": 3,
                "batch_images": 2,
                "seed": 6,
                "scale_lr": 0.0001,
            },
        },
        "tsptq_vit": {
            "on_semantic": {
                "pre_layernorm": codec("two_channel_scales"),
                probability: codec("bit_window"),
                gelu: codec(
                    "bit_window",
                    options={"encoding": {"domain": "gelu", "shift_bits": 3}},
                ),
            }
        },
        "regcache": {
            "stages": [
                {
                    "kind": "cached_kv",
                    "module": "blocks.0.attn",
                    "copies": 2,
                    "candidates": 3,
                },
                {
                    "kind": "outlier_tokens",
                    "module": "blocks.1",
                    "end_module": "blocks.1",
                    "prefix_tokens": 1,
                    "grid_hw": grid,
                    "delete_count": 1,
                },
            ]
        },
        "outlier-aware-slicing": {
            "slices": [
                {
                    "start": "blocks.0",
                    "end": "blocks.1",
                    "sites": [
                        site_id(r)
                        for r in base["layers"]
                        if r["module"].startswith("blocks.")
                    ],
                }
            ],
            "fit": {
                "steps": 4,
                "batch_images": 2,
                "seed": 13,
                "round_lr": 0.001,
                "scale_lr": 0.00004,
                "round_reg": 0.1,
                "drop_probability": 0.5,
                "beta_range": [20.0, 2.0],
                "beta_schedule": "exponential",
                "warmup": 0.2,
            },
        },
        "hc-ptq": {"row_codebook": {"curvature": 0.5, "iterations": 3, "seed": 7}},
        "patchwise-mpq-vit": {
            "fixed_precision": True,
            "patch_precision": codec("patch_precision")["options"]["encoding"],
        },
    }
    cases["daquant"] = {
        "domain_sites": [r["module"] for r in base["layers"] if r["kind"] == "linear"],
        "distribution": {
            "weight_statistic": "author_first_coordinate",
            "min_scale": 1e-5,
            "use_shift": True,
        },
        "activation_clip": {
            "range_mode": "instance_tokens",
            "initial_logit": 4.0,
            "min_scale": 1e-5,
            "max_scale": 1e4,
            "max_zero": 1e4,
        },
        "domain_fit": {
            "target_pairs": {
                sample_ids[0]: sample_ids[-1],
                sample_ids[1]: sample_ids[-1],
            },
            "alignment_weight": 1.5,
            "kl_reduction": "batchmean",
            "lr": {"DGA": 0.005, "LWC": 0.01, "LAC": 0.0001, "WRC": 0.00001},
            "weight_decay": 0.0,
            "phases": [
                {"kind": "source", "steps": 2},
                {"kind": "alignment", "steps": 2},
            ],
            "batch_images": 1,
            "seed": 17,
        },
    }
    for source in ("amp-vit", "mptq-vit", "p2-vit", "h-vit"):
        cases[source] = {
            "smoothing": {"alpha": 0.5, "pot_scale": source in ("p2-vit", "h-vit")},
            "on_semantic": (
                {gelu: codec("three_region")}
                if source in ("amp-vit", "mptq-vit")
                else {post: signed, gelu: signed}
            ),
        }
    for source in ("amp-vit", "mptq-vit"):
        cases[source]["on_semantic"][gelu]["options"]["encoding"]["gradient_loss"] = (
            "teacher_class_ce"
        )
    for source in ("h-vit", "p2-vit"):
        cases[source]["weight_pot"] = "linear_output"
    cases["h-vit"]["weight_pot"] = "weight"
    cases["h-vit"]["on_semantic"] = deepcopy(cases["h-vit"]["on_semantic"])
    for assignment in cases["h-vit"]["on_semantic"].values():
        assignment["options"]["encoding"]["neighbors"] = 0
    cases["h-vit"].pop("smoothing")
    cases["h-vit"]["nonlinear"] = [
        {
            "kind": "calibrated_operator",
            "module": m,
            "settings": {
                "function": "hvit_layernorm",
                "input_bits": 8,
                "output_bits": 8,
                "output_layout": "frozen_positions",
                "clip_factors": [0.9, 1.0],
            },
        }
        for m in (
            "blocks.0.norm1",
            "blocks.0.norm2",
            "blocks.1.norm1",
            "blocks.1.norm2",
        )
    ]
    return cases


def scope(source, req):
    base = req["base"]
    if source == "hc-ptq":
        base["layers"] = [r for r in base["layers"] if r["kind"] == "linear"]
        for row in base["layers"]:
            row["activation"] = row["activation_method"] = None
            row.pop("activation_calibration", None)
    if source == "tsptq_vit":
        root = next(r for r in base["layers"] if r["kind"] == "post_softmax")
        for block in ("blocks.0", "blocks.1"):
            for norm in ("norm1", "norm2"):
                r = deepcopy(root)
                r.update(module=block + "." + norm, kind="pre_layernorm")
                base["layers"].append(r)
