"""Construction and deployment are separate identities. Historical briefs keep their own semantics."""

from core.cards import ContractError

LEGACY = "static_w4a4/1"
PRIMARY = "static_w4a4/2"
EXTENDED = "ptq_extension/1"
DIMENSIONS = {
    "coverage_exceptions",
    "dynamic_quantization",
    "mixed_precision",
    "additional_computation",
    "weight_only",
}


def protocol_of(execution):
    return (
        execution.get("research_quantization_contract")
        or execution.get("quantization_contract")
        or {}
    ).get("protocol")


def current(protocol):
    return protocol in (PRIMARY, EXTENDED)


def validate_profile(contract):
    protocol = contract.get("protocol")
    if protocol not in (LEGACY, PRIMARY, EXTENDED):
        raise ContractError("unknown quantization protocol")
    dimensions = contract.get("extension_dimensions", [])
    if (
        not isinstance(dimensions, list)
        or any(x not in DIMENSIONS for x in dimensions)
        or len(set(dimensions)) != len(dimensions)
        or bool(dimensions) != (protocol == EXTENDED)
    ):
        raise ContractError("extended PTQ requires its actual deployment dimensions")


def validate_construction(spec, protocol, training):
    construction = spec.get("construction")
    if not current(protocol):
        if construction is not None:
            raise ContractError(
                "construction declaration requires a versioned PTQ profile"
            )
        return
    if not training and construction is None:
        return
    if (
        not isinstance(construction, dict)
        or construction.get("regime") != "finite_calibration_ptq"
    ):
        raise ContractError("PTQ optimization requires a finite calibration recipe")
    scope = construction.get("update_targets")
    steps = construction.get("max_steps")
    if (
        training is not True
        or not isinstance(scope, list)
        or not scope
        or any(not isinstance(x, str) or not x.strip() for x in scope)
        or type(steps) is not int
        or steps < 1
        or not construction.get("objective")
        or not construction.get("recipe_version")
        or construction.get("split") != "calibration"
    ):
        raise ContractError(
            "declare recipe version, update targets, calibration split, objective and optimizer-step budget"
        )
    if spec.get("calibration", {}).get("split", "calibration") != "calibration":
        raise ContractError("PTQ fitting requires the calibration split")


def reusable(training, construction, protocol):
    if training is False:
        return True
    if training is not True or not current(protocol):
        return False
    try:
        validate_construction({"construction": construction}, protocol, training)
    except ContractError:
        return False
    return True


def reusable_execution(spec):
    return reusable(
        spec.get("resolved_choices", {}).get("training"),
        spec.get("construction"),
        protocol_of(spec),
    )


def reusable_manifest(manifest, protocol):
    return reusable(
        manifest.get("preparation_training"), manifest.get("construction"), protocol
    )
