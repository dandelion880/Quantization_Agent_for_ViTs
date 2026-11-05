"""Actual CPU numerical paths; builder stubs isolate dispatch, never fake measurements."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.execution_contract import check_plan, validate_plan_schema
from core.journal import read_json, atomic_json
from tests.test_infrastructure import TempCase
from tests.test_execution_contracts import plan_fixture
from tests.support import validation_config, experiment_payload
from tools.experiments import preflight_experiment, expand_plan, execution_fingerprint
from tools.experiment_worker import backend_and_data, evaluate_services
from ptq_agent.execution.contracts import normalize_plan, PLUGIN_PLAN_INPUT_SCHEMA
from ptq_agent.execution.capabilities import validate_plan, detect
from ptq_agent.execution.model import RTNExecutor
from ptq_agent.schemas import PTQError, validate


def numeric_plan(version=3, *, granularity="per_channel", dynamic=False):
    plan = plan_fixture()
    for row in plan["layers"]:
        for target in ("weight", "activation"):
            row[target + "_method"] = "rtn"
            row.pop(target + "_calibration", None)
        row["activation"].update(format="INT8", bits=8, symmetric=True,
                                 granularity=granularity, axis=None if granularity == "per_tensor" else 2,
                                 group_size=4 if granularity == "per_group" else None)
    plan["execution"]["activation_calibration"] = "dynamic" if dynamic else "static"
    if version == 2:
        plan["schema_version"] = 2
        plan.pop("preprocess")
        for field in ("order", "within_stage"):
            plan["execution"].pop(field)
        for row in plan["layers"]:
            for field in ("kind", "weight_method", "activation_method"):
                row.pop(field)
            row["method"] = "rtn"
    return plan


def abbreviated(plan):
    value = deepcopy(plan)
    value.pop("unlisted_modules")
    if value["schema_version"] == 3:
        value["execution"].pop("mode")
    for row in value["layers"]:
        if value["schema_version"] == 2:
            row.pop("method")
        for target in ("weight", "activation"):
            spec = row[target]
            if spec is None:
                row.pop(target + "_method", None)
                continue
            spec.pop("bits")
            if spec["granularity"] == "per_tensor":
                spec.pop("axis")
            if spec["granularity"] != "per_group":
                spec.pop("group_size")
    return value


class PlanExecutionConsistency(TempCase):
    def prepare(self, plan):
        brief = validation_config()["research_brief"]
        brief["hard_constraints"] = {"training": False}
        payload = experiment_payload("L2")
        payload["execution"] = {"service": "quantize_evaluate", "plan": plan,
            "resolved_choices": {"training": False}, "seed": 42,
            "evaluations": [{"kind": "classification", "split": "search", "count": 1}]}
        from ptq_agent.execution.pipeline import plan_uses_calibration
        if plan_uses_calibration(plan):
            payload["execution"]["calibration"] = {"split": "calibration", "count": 2}
        payload = preflight_experiment(payload, brief)
        spec = payload["execution"]
        backend, cfg, data = backend_and_data(spec)
        return payload, backend, cfg, data

    def test_shared_normalization_preserves_input_and_does_not_guess_science(self):
        for version in (2, 3):
            plan = numeric_plan(version, granularity="per_tensor")
            compact = abbreviated(plan)
            before = deepcopy(compact)
            self.assertEqual(expand_plan(compact), plan)
            self.assertEqual(validate_plan_schema(compact), plan)
            self.assertEqual(check_plan(compact, {})["plan"], plan)
            self.assertEqual(compact, before)
            self.assertEqual(normalize_plan(normalize_plan(compact)), plan)
            payload, *_ = self.prepare(plan)
            reduced, *_ = self.prepare(compact)
            self.assertEqual(execution_fingerprint({"payload": payload}),
                             execution_fingerprint({"payload": reduced}))
        empty = numeric_plan()
        empty["layers"][0].update(activation=None)
        empty["layers"][0].pop("activation_method")
        self.assertIsNone(validate_plan_schema(empty)["layers"][0]["activation_method"])
        for field in ("format", "symmetric", "granularity"):
            bad = abbreviated(numeric_plan())
            bad["layers"][0]["activation"].pop(field)
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_plan_schema(bad)
        missing_method = abbreviated(numeric_plan())
        missing_method["layers"][0].pop("activation_method")
        with self.assertRaises(ContractError):
            validate_plan_schema(missing_method)

    def test_explicit_conflicts_and_custom_representation_boundary(self):
        for field, value in (("bits", 4), ("axis", 1), ("group_size", 4)):
            bad = numeric_plan(granularity="per_tensor")
            bad["layers"][0]["activation"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "activation." + field):
                validate_plan_schema(bad)
        custom = {"schema_version": 3, "layers": [{"module": "site", "kind": "custom",
            "weight": {"format": "CUSTOM7", "bits": 7}, "activation": None,
            "weight_method": "new_algorithm"}]}
        self.assertEqual(check_plan(custom, {}, native=False)["plan"], custom)

    def test_format_intrinsic_fields_are_derived_but_signed_integer_choice_is_not(self):
        for version in (2, 3):
            plan = numeric_plan(version)
            plan["layers"][0]["activation"] = {"format": "FP16"}
            actual = validate_plan_schema(plan)["layers"][0]["activation"]
            self.assertEqual(actual, {"format": "FP16", "bits": 16, "granularity": "per_tensor",
                "scale": "none", "symmetric": True, "axis": None, "group_size": None})
            plan["layers"][0]["activation"] = {"format": "UINT4", "granularity": "per_tensor"}
            actual = validate_plan_schema(plan)["layers"][0]["activation"]
            self.assertEqual(actual["scale"], "minmax")
            self.assertFalse(actual["symmetric"])
            plan["layers"][0]["activation"] = {"format": "INT4", "granularity": "per_tensor"}
            with self.assertRaises(ContractError):
                validate_plan_schema(plan)
            plan["layers"][0]["activation"] = {"format": "FP16", "scale": "minmax"}
            with self.assertRaisesRegex(ContractError, "activation.scale"):
                validate_plan_schema(plan)

    def test_tool_preflight_defers_only_unknown_shapes_to_prepare(self):
        from tools.profiling import profile_host
        from ptq_agent.schemas import ToolCall
        plan = numeric_plan(granularity="per_channel")
        _, backend, cfg, _ = self.prepare(plan)
        host = profile_host(backend, cfg, self.root / "profile")
        response = host.registry.invoke("profiling", ToolCall("shape", "validate_experiment",
            {"plan": abbreviated(plan)}))
        self.assertEqual(response.status, "success", response.error)
        self.assertEqual(set(response.data["shape_checks_deferred_to_prepare"]),
                         {row["module"] for row in plan["layers"]})
        bad = deepcopy(plan)
        bad["layers"][0]["activation"]["group_size"] = 4
        response = host.registry.invoke("profiling", ToolCall("bad", "validate_experiment", {"plan": bad}))
        self.assertNotEqual(response.status, "success")

    def test_profile_inputs_reject_real_missing_choices_before_cost_estimation(self):
        from tools.profiling import profile_host
        from ptq_agent.schemas import ToolCall
        plan = numeric_plan(granularity="per_tensor")
        _, backend, cfg, _ = self.prepare(plan)
        host = profile_host(backend, cfg, self.root / "profile_missing")
        compact = abbreviated(plan)
        compact["layers"][0].pop("activation_method")
        response = host.registry.invoke("profiling", ToolCall("missing-method", "profile_ptq_candidates",
            {"plans": [compact], "count": 1, "seed": 42, "offset": 0}))
        self.assertNotEqual(response.status, "success")
        self.assertIn("activation_method", str(response.error))
        variant = {"name": "missing-signedness", "weight": None,
                   "activation": {"format": "INT4", "granularity": "per_tensor"}}
        response = host.registry.invoke("profiling", ToolCall("missing-signedness", "profile_sensitivity",
            {"modules": [plan["layers"][0]["module"]], "variants": [variant], "count": 1,
             "seed": 42, "offset": 0}))
        self.assertNotEqual(response.status, "success")
        self.assertIn("symmetric", str(response.error))

    def test_grouped_and_delta_inputs_share_normalization(self):
        from ptq_agent.execution.planning import expand_groups, apply_plan_patch
        from ptq_agent.execution.sites import site_catalog
        plan = numeric_plan(granularity="per_tensor")
        _, backend, cfg, _ = self.prepare(plan)
        compact = abbreviated(plan)
        validate(compact, PLUGIN_PLAN_INPUT_SCHEMA)
        fields = {k: v for k, v in compact.items() if k != "layers"}
        groups = [{"selector": {"pool": "available_sites", "kinds": ["linear"],
                   "site_ids": [row["module"]]},
                   "settings": {k: v for k, v in row.items() if k not in ("module", "kind")}}
                  for row in compact["layers"]]
        self.assertEqual(expand_groups(fields, groups, site_catalog(backend, cfg), cfg), plan)
        updated = apply_plan_patch(plan, [{"site_ids": [plan["layers"][0]["module"]],
                                          "changes": {"activation": None}}])
        self.assertIsNone(updated["layers"][0]["activation_method"])

    def test_real_direct_and_coding_plans_export_reload_and_independent_evaluation(self):
        from ptq_agent.execution.pipeline import QuantizationExecutor
        for version in (2, 3):
            for origin in ("direct", "coding", "coding_explicit_builtin"):
                with self.subTest(version=version, origin=origin):
                    plan = numeric_plan(version)
                    payload, backend, cfg, data = self.prepare(abbreviated(plan))
                    if origin != "direct":
                        payload["execution"].update(requires_code=True)
                        payload["execution"].pop("plan")
                    plugin = SimpleNamespace(build_plan=lambda *args: abbreviated(plan))
                    if origin == "coding_explicit_builtin":
                        plugin.make_executor = QuantizationExecutor
                    directory = self.root / f"v{version}_{origin}"
                    directory.mkdir()
                    with patch("tools.experiment_worker.module_from_approval", return_value=plugin):
                        result = evaluate_services({"payload": payload}, backend, cfg, data, directory, None)
                    model_dir = Path(result["model_dir"])
                    manifest = read_json(model_dir / "manifest.json")
                    self.assertEqual(manifest["schema_version"], 1 if version == 2 else 2)
                    self.assertEqual(manifest["plan"], plan)
                    self.assertTrue(result["observations"]["reload"]["host_verified"])
                    self.assertTrue(manifest["input_shapes"])
                    existing = deepcopy(payload)
                    existing["execution"].update(service="evaluate_artifact", model_dir=str(model_dir), requires_code=False)
                    restored_dir = self.root / (directory.name + "_existing")
                    restored_dir.mkdir()
                    with patch.object(RTNExecutor, "prepare", side_effect=AssertionError("must only load")):
                        restored = evaluate_services({"payload": existing}, backend, cfg, data, restored_dir, None)
                    self.assertEqual(restored["model_ref"], result["model_ref"])
                    self.assertEqual(restored["observations"]["evaluations"]["classification"]["metrics"]["mse"],
                                     result["observations"]["evaluations"]["classification"]["metrics"]["mse"])

    def test_shape_capture_in_same_prepare_and_actual_shape_conflicts(self):
        for version in (2, 3):
            plan = numeric_plan(version, granularity="per_group", dynamic=True)
            _, backend, cfg, data = self.prepare(plan)
            executor = RTNExecutor(backend, cfg, data)
            model, weights, activations, audit = executor.prepare(abbreviated(plan), {})
            self.assertEqual(activations, {})
            self.assertTrue(executor.input_shapes)
            validate_plan(backend, plan, cfg, detect(backend), executor.input_shapes)
            bad = deepcopy(plan)
            bad["layers"][0]["activation"]["axis"] = 1
            with self.assertRaises(PTQError):
                RTNExecutor(backend, cfg, data).prepare(bad, {})

    def test_v2_native_float_mode_keeps_actual_representation_and_zero_calibration(self):
        plan = numeric_plan(2)
        plan["execution"]["mode"] = "native"
        for row in plan["layers"]:
            row.update(weight={"format": "FP32"}, activation={"format": "FP32"})
        payload, backend, cfg, data = self.prepare(normalize_plan(plan))
        cfg["loop"]["execution_modes"] = ["native"]  # Explicit fixture permission, not a production default.
        directory = self.root / "native_float"
        directory.mkdir()
        result = evaluate_services({"payload": payload}, backend, cfg, data, directory, None)
        self.assertTrue(result["observations"]["representation"].startswith("native;"))
        self.assertNotIn("calibration", result["execution_facts"]["data"])
        self.assertEqual(read_json(Path(result["model_dir"]) / "manifest.json")["calibration_ids"], [])

    def test_generic_loader_reports_custom_format_boundary(self):
        _, backend, cfg, data = self.prepare(numeric_plan())
        directory = self.root / "custom_package"
        directory.mkdir()
        atomic_json(directory / "manifest.json", {"schema_version": "custom", "plan": {"schema_version": 3}})
        with self.assertRaisesRegex(PTQError, "对应批准.*make_executor.load"):
            RTNExecutor(backend, cfg, data).load(directory)
