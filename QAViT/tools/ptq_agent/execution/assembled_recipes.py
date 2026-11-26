"""Expand a literature revision and a declared starting configuration into an ordinary editable v3 plan."""

from contextlib import ExitStack
from copy import deepcopy
import torch
from ..schemas import require
from .baselines import build_baseline
from .model import forward
from .plugin_contracts import site_id
from .sites import site_catalog, install_attention
from .recipes import build_recipe


def execution_order(backend, data, paths):
    """Bind order and output axes by an evaluation-mode forward, including the two real matmuls."""
    paths = set(paths)
    require(paths, "empty recipe target set")
    teacher = deepcopy(backend.model).eval()
    products = {
        p.rsplit(".", 1)[0]
        for p in paths
        if p.endswith(("._ptq_qk", "._ptq_probability"))
    }
    install_attention(teacher, products)
    observed = {}

    def capture(path):
        def hook(_m, _args, y):
            require(
                path not in observed and torch.is_tensor(y),
                "repeated/structured call needs a more specific recipe binding: "
                + path,
            )
            observed[path] = list(y.shape)

        return hook

    with ExitStack() as hooks:
        for p in paths:
            hooks.callback(
                teacher.get_submodule(p).register_forward_hook(capture(p)).remove
            )
        x, _ = next(iter(data.batches("calibration")))
        forward(backend, teacher, x)
    require(
        set(observed) == paths,
        "some requested modules were absent from the bound task forward",
    )
    return observed


def assemble(backend, request, data):
    source = request["source_id"]
    require(
        request.get("version") and request.get("profile") and request.get("evidence"),
        "recipe needs source version, declared variant and evidence",
    )
    base = request["base"]
    if isinstance(base, dict) and base.get("schema_version") == 3:
        plan = deepcopy(base)
    elif "author_recipe" in base:
        plan = build_recipe(
            backend, base["author_recipe"], base["bits"], request["evidence"]
        )
    else:
        plan = build_baseline(
            backend, base["baseline"], quantize_head=base["quantize_head"]
        )
    initial = deepcopy(plan)
    settings = deepcopy(request["settings"])
    catalog = site_catalog(backend)
    available = {site_id(l): l for l in plan["layers"]}
    selected = request.get("site_ids", list(available))
    require(
        len(set(selected)) == len(selected) and set(selected) <= set(available),
        "unbound recipe site",
    )
    layers = [available[k] for k in selected]
    attention = dict.fromkeys(
        r["module"] for r in layers if r["kind"] == "post_softmax"
    )
    blocks = list(dict.fromkeys(n.rsplit(".", 1)[0] for n in attention))
    units = request.get("units", blocks)
    require(len(units) == len(set(units)), "repeated recipe unit")
    for unit in units:
        backend.model.get_submodule(unit)

    def post(kind, modules, reference, config):
        for name in execution_order(backend, data, modules):
            plan.setdefault("postfit", []).append(
                {
                    "kind": kind,
                    "module": name,
                    "reference": reference,
                    "settings": deepcopy(config.get(name, config)),
                }
            )

    def repq(row, extra):
        sid = site_id(row)
        entry = next(
            (p for p in plan["preprocess"] if p["module"] == row["module"]), None
        )
        if entry is None:
            entry = {
                "method": "repq_ln",
                "module": row["module"],
                "norm": catalog[sid]["edge"].removesuffix(":output"),
                "observation": "float",
                "calibration": {"strategy": "minmax"},
            }
            plan["preprocess"].append(entry)
        require(entry["method"] == "repq_ln", "incompatible upstream edge transform")
        entry.setdefault("settings", {}).update(deepcopy(extra))

    if source in ("biascomp", "qwt", "qwt-v2"):
        if source == "qwt":
            post("qwt", units, "local_fp_candidate", {})
            for stage in plan["postfit"]:
                if stage["kind"] == "qwt":
                    stage.pop("settings", None)
        else:
            paths = [
                r["module"]
                for r in layers
                if r["kind"]
                in (("linear",) if source == "biascomp" else ("linear", "conv2d"))
            ]
            if source == "biascomp":
                paths += [
                    n + "." + child
                    for n in attention
                    for child in ("_ptq_qk", "_ptq_probability")
                ]
            for name, shape in execution_order(backend, data, paths).items():
                conv = name in catalog and catalog[name]["kind"] == "conv2d"
                axes = (
                    [0]
                    if source == "biascomp"
                    else [
                        i
                        for i in range(len(shape))
                        if i != (1 if conv else len(shape) - 1)
                    ]
                )
                plan.setdefault("postfit", []).append(
                    {
                        "kind": "bias_compensation"
                        if source == "biascomp"
                        else "channel_affine",
                        "module": name,
                        "reference": settings.get("reference", "local_fp_candidate"),
                        "reduce_axes": axes,
                    }
                )
                if source == "qwt-v2" and settings.get("integer_fold"):
                    plan["postfit"][-1]["settings"] = {
                        "integer_fold": deepcopy(settings["integer_fold"])
                    }
    elif source in ("erq-icml2024", "erq-tpami2025-duq", "qep", "repquant"):
        plan["execution"].update(order="a_then_w", within_stage="sequential")
        for row in layers:
            kind = row["kind"]
            if kind != "linear":
                continue
            if (
                catalog[site_id(row)]["semantic"] == "post_layernorm"
                and source != "qep"
            ):
                repq(row, settings.get("repq", {}))
            if source == "repquant":
                row.update(
                    weight_method="gptq", weight_calibration={"strategy": "minmax"}
                )
                row["weight"].update(
                    format="INT" + str(row["weight"]["bits"]),
                    symmetric=True,
                    granularity="per_channel",
                    group_size=None,
                    axis=0,
                )
                row.setdefault("options", {}).update(deepcopy(settings["gptq"]))
            elif source == "qep":
                row.setdefault("options", {})["qep"] = deepcopy(settings["qep"])
            else:
                erq = deepcopy(settings["erq"])
                require(
                    ("tail_rule" in erq) == (source == "erq-tpami2025-duq"),
                    "DUQ is a separate source revision",
                )
                row.update(
                    weight_method="erq", weight_calibration={"strategy": "minmax"}
                )
                row["weight"].update(
                    format="UINT" + str(row["weight"]["bits"]),
                    symmetric=False,
                    granularity="per_channel",
                    axis=0,
                    group_size=None,
                )
                row.setdefault("options", {})["erq"] = erq
    elif source in ("cpt-v", "evol-q"):
        order = list(
            execution_order(backend, data, request.get("units", list(attention)))
        )
        if settings["sweep_direction"] == "reverse":
            order.reverse()
        require(
            settings["sweep_direction"] in ("forward", "reverse")
            and settings["passes"] > 0,
            "invalid search schedule",
        )
        plan.setdefault("postfit", []).extend(
            {
                "kind": "scale_search",
                "module": n,
                "reference": "fp_model",
                "settings": deepcopy(settings["search"]),
            }
            for _ in range(settings["passes"])
            for n in order
        )
    elif source in ("apq-vit", "q-hyvit", "ptq-vit-2021", "bit-shrinking"):
        if source == "apq-vit":
            for row in layers:
                if row["kind"] == "post_softmax":
                    row.update(
                        activation_method="uniform_minmax",
                        activation_calibration={"strategy": "minmax"},
                        options={},
                    )
        post(
            "bit_shrinking" if source == "bit-shrinking" else "block_coordinate_search",
            units,
            "fp_model",
            settings["fit"],
        )
    elif source == "outlier-aware-slicing" and "slices" in settings:
        common = settings["fit"]
        for boundary in settings["slices"]:
            require(
                set(boundary["sites"]) <= set(selected) and boundary["sites"],
                "slice members must be real selected tensor sites",
            )
            plan.setdefault("postfit", []).append(
                {
                    "kind": "block_reconstruction",
                    "module": boundary["start"],
                    "reference": "fp_model",
                    "settings": {
                        **common,
                        "update": "rounding_and_scale",
                        "slice": boundary,
                    },
                }
            )
    elif (
        source in ("pfcr-vit", "outlier-aware-slicing")
        or source == "is-vit"
        and "activation_phase" in settings
    ):
        from .graph_fit import tree

        names = [
            r["module"] for r in layers if r.get("weight") and r["kind"] == "linear"
        ]
        if source == "pfcr-vit":
            initial = settings["activation_phase"]
            final = settings["quantized_phase"]
            plan["activation_first"] = {
                **initial,
                "groups": tree(
                    units, names, initial["steps"], initial["lr"], activation_only=True
                ),
            }
            groups = tree(units, names, final["steps"], final["lr"])
        elif source == "is-vit":
            initial = settings["activation_phase"]
            final = settings["quantized_phase"]

            def whole(fit):
                return [
                    {
                        "observe": u,
                        "weight_modules": [n for n in names if n.startswith(u + ".")],
                        "steps": fit["steps"],
                        "lr": fit["lr"],
                    }
                    for u in units
                ]

            plan["activation_first"] = {**initial, "groups": whole(initial)}
            groups = whole(final)
            for row in layers:
                if catalog[site_id(row)]["semantic"] == "post_layernorm":
                    repq(row, settings["repq"])
                if row["kind"] == "post_softmax":
                    row.update(
                        activation_method="sulq",
                        options={"encoding": settings["encoding"]},
                        activation_calibration={"strategy": "minmax"},
                    )
        else:
            final = settings["quantized_phase"]
            groups = final["groups"]
        plan.setdefault("postfit", []).append(
            {
                "kind": "graph_reconstruction",
                "module": units[0],
                "reference": "fp_model",
                "settings": {**final, "groups": groups},
            }
        )
    elif source in ("qdrop", "pd-quant", "mgrq", "orq_vit", "fima-q", "aphq-vit"):
        if source == "aphq-vit":
            plan.setdefault("structure", []).extend(
                deepcopy(settings["mlp_replacements"])
            )
        if source == "orq_vit":
            for row in layers:
                if row["kind"] not in (
                    "linear",
                    "attention_query",
                    "attention_key",
                    "attention_value",
                ):
                    continue
                # Explicit row/channel layout of each real GEMM operand, including transposed K and V.
                axis = settings["feature_axes"][site_id(row)]
                row["activation"].update(
                    granularity="per_token", axis=axis, group_size=None
                )
                row.update(
                    activation_method="rtn",
                    activation_calibration=deepcopy(settings["initial_observer"]),
                )
        post("block_reconstruction", units, "fp_model", settings["fit"])
    elif source == "daquant" and "domain_fit" in settings:
        targets = set(settings["domain_sites"])
        require(
            targets and targets <= set(selected),
            "DGA/LWC/LAC/WRC require bound Linear sites",
        )
        for sid in sorted(targets):
            row = available[sid]
            require(
                row["kind"] == "linear", "DA channel adaptation requires Linear axes"
            )
            row["weight"].update(
                format="UINT" + str(row["weight"]["bits"]),
                symmetric=False,
                granularity="per_channel",
                axis=0,
                group_size=None,
            )
            row["activation"].update(
                format="UINT" + str(row["activation"]["bits"]),
                symmetric=False,
                granularity="per_channel",
                axis=2,
                group_size=None,
            )
            row.update(
                weight_method="uniform_affine",
                weight_calibration={"strategy": "minmax"},
                activation_method="domain_clip",
                activation_calibration={"strategy": "minmax"},
                options={"encoding": deepcopy(settings["activation_clip"])},
            )
            norm = (
                catalog[sid]["edge"].removesuffix(":output")
                if catalog[sid]["semantic"] == "post_layernorm"
                else None
            )
            plan["preprocess"].append(
                {
                    "method": "domain_affine",
                    "module": sid,
                    "norm": norm,
                    "settings": deepcopy(settings["distribution"]),
                }
            )
        post("domain_affine_fit", units, "fp_model", settings["domain_fit"])
    elif source in ("aiqvit", "daquant"):
        if settings.get("focus_interval"):
            for row in layers:
                if row["kind"] == "post_softmax":
                    row.update(
                        activation_method="focus_interval",
                        options={"encoding": deepcopy(settings["focus_interval"])},
                        activation_calibration={"strategy": "minmax"},
                    )
        post("merged_weight_fit", units, "fp_model", settings["fit"])
    elif source in (
        "noisyquant",
        "amp-vit",
        "mptq-vit",
        "dopq_vit",
        "adfq_vit",
        "igq_vit",
        "tsptq_vit",
        "p2-vit",
        "h-vit",
        "hc-ptq",
        "is-vit",
        "adalog",
    ):
        for row in layers:
            info = catalog[site_id(row)]
            if source == "hc-ptq" and row["kind"] == "linear":
                row.update(
                    weight_method="hc_rows",
                    options={"row_codebook": deepcopy(settings["row_codebook"])},
                )
            if source == "noisyquant" and row["kind"] == "linear":
                plan["preprocess"].append(
                    {
                        "method": "noisy_shift",
                        "module": row["module"],
                        "norm": None,
                        "settings": deepcopy(settings["noise"]),
                    }
                )
            if source == "dopq_vit" and info["semantic"] == "post_layernorm":
                repq(
                    row,
                    {
                        "aggregate": "median",
                        "round_shared_zero": settings["round_shared_zero"],
                    },
                )
            if (
                source in ("amp-vit", "mptq-vit", "p2-vit")
                and info["semantic"] == "post_layernorm"
            ):
                plan["preprocess"].append(
                    {
                        "method": "centered_smoothing",
                        "module": row["module"],
                        "norm": info["edge"].removesuffix(":output"),
                        "settings": deepcopy(settings["smoothing"]),
                    }
                )
            if (
                source in ("p2-vit", "h-vit")
                and row["weight"]
                and settings.get("weight_pot")
            ):
                spec = row["weight"]
                spec.update(format="INT" + str(spec["bits"]), symmetric=True)
                if source == "h-vit":
                    spec.update(granularity="per_tensor", axis=None, group_size=None)
                row.update(
                    weight_method="uniform_minmax",
                    weight_calibration={
                        "strategy": "weight_pot",
                        "neighbors": 0 if source == "h-vit" else 1,
                        "error_domain": settings["weight_pot"]
                        if row["kind"] == "linear"
                        else "weight",
                    },
                )
            update = settings.get("on_semantic", {}).get(info["semantic"])
            if update:
                for name in ("activation_method", "activation_calibration", "options"):
                    if name in update:
                        row[name] = deepcopy(update[name])
                if "activation" in update:
                    row["activation"].update(deepcopy(update["activation"]))
        if source == "h-vit" and settings.get("nonlinear"):
            plan.setdefault("structure", []).extend(deepcopy(settings["nonlinear"]))
        if settings.get("fit"):
            post("block_reconstruction", units, "fp_model", settings["fit"])
    elif (
        source == "clamp-vit"
        and data.identity.get("generator_variant") == "clamp_cyclic"
    ):
        final = data.identity["cyclic_result"]
        require(
            final["pool_sha256"] and data.identity.get("synthetic_source"),
            "missing cyclic selection provenance",
        )
        plan = deepcopy(final["plan"])
    elif source in ("dfq-vit", "sardfq", "psaq-vit", "clamp-vit"):
        generated = data.identity.get("synthetic_source")
        require(
            generated
            and data.identity.get("generator_variant")
            == {
                "psaq-vit": "psaq_kde",
                "sardfq": "sardfq_gmm",
                "dfq-vit": "e2h_crop",
                "clamp-vit": "clamp_local_adapter",
            }[source],
            "source recipe needs its corresponding generated calibration pool",
        )
        if source == "dfq-vit":
            for n, axes in settings["compensation_axes"].items():
                backend.model.get_submodule(n)
            for n in execution_order(backend, data, settings["compensation_axes"]):
                plan.setdefault("postfit", []).append(
                    {
                        "kind": "bias_compensation",
                        "module": n,
                        "reference": settings["reference"],
                        "reduce_axes": settings["compensation_axes"][n],
                    }
                )
        elif source == "sardfq":
            fit = settings["fit"]
            names = [r["module"] for r in layers if r.get("weight")]
            groups = [
                {
                    "observe": n,
                    "weight_modules": [s for s in names if s.startswith(n + ".")],
                    "lr": fit["lr"],
                    "steps": fit["steps"],
                }
                for n in units
            ]
            plan.setdefault("postfit", []).append(
                {
                    "kind": "graph_reconstruction",
                    "module": units[0],
                    "reference": "fp_model",
                    "settings": {**fit, "groups": groups},
                }
            )
        elif source == "clamp-vit":
            post("scale_search", units, "fp_model", settings["search"])
    elif source in ("lrp-qvit", "mix-qvit", "lampq", "pomq-vit", "patchwise-mpq-vit"):
        # Allocation is executed by the worker before final preparation. Its measured table is kept with the frozen state.
        require(
            request.get("allocation") is not None or settings.get("fixed_precision"),
            "declare the source allocation or its fixed-grid ablation",
        )
        if source in ("lrp-qvit", "mix-qvit"):
            for row in layers:
                if catalog[site_id(row)]["semantic"] == "post_layernorm":
                    repq(row, settings["repq"])
        if source == "patchwise-mpq-vit":
            for row in layers:
                if row["kind"] == "post_softmax":
                    row.update(
                        activation_method="patch_precision",
                        options={"encoding": settings["patch_precision"]},
                        activation_calibration={"strategy": "minmax"},
                    )
    elif source == "mixa-q":
        require(
            all(s["kind"] == "window_branches" for s in settings["stages"]),
            "window branch configuration required",
        )
        plan.setdefault("structure", []).extend(deepcopy(settings["stages"]))
    elif source == "trio-vit":
        plan["preprocess"].extend(deepcopy(settings["mbconv_edges"]))
        plan.setdefault("structure", []).extend(deepcopy(settings["denominators"]))
        if settings.get("fit"):
            post("block_reconstruction", units, "fp_model", settings["fit"])
    elif source in ("regcache", "iptq-vit", "practicaledge"):
        plan.setdefault("structure", []).extend(deepcopy(settings["stages"]))
    elif source in ("ptq4vit", "repq_vit", "fq_vit"):
        require(
            base.get("author_recipe")
            == {"ptq4vit": "ptq4vit", "repq_vit": "repq", "fq_vit": "fq_vit"}[source],
            "source base must match",
        )
    else:
        require(
            False,
            "no bound source schedule for this ID; submit an explicit composed v3 plan",
        )
    require(
        plan != initial
        or source in ("ptq4vit", "repq_vit", "fq_vit", "psaq-vit")
        or bool(request.get("allocation")),
        "source binding changed no stage or tensor site in the supplied base",
    )
    return plan, {
        "source_id": source,
        "version": request["version"],
        "profile": request["profile"],
        "selected_sites": list(selected),
        "units": units,
        "evidence": list(request["evidence"]),
        "settings": settings,
    }
