"""CPU 合同正反例与真实 Data/Backend 小夹具；不调用 LLM/GPU。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card, fingerprint, validate_payload
from core.evidence_scope import canonical_scope, validate_claim_scope, bind_claim_scope, matches
from core.execution_contract import check_plan, validate_plan_schema
from core.journal import atomic_json
from core.memory import MemoryRetriever
from tests.support import validation_config, experiment_payload, review
from tests.test_infrastructure import TempCase
from tests import test_experiment_preflight as preflight_tests
from tools.experiments import preflight_experiment
from tools.experiment_worker import backend_and_data, evaluate_services
from tools.result_contract import check_results
from tools.runtime_interface import interface_fixture, execute_custom


def plan_fixture():
    weight = {"format": "INT8", "bits": 8, "granularity": "per_channel", "axis": 0,
              "group_size": None, "symmetric": True, "scale": "minmax"}
    activation = {"format": "UINT6", "bits": 6, "granularity": "per_tensor", "axis": None,
                  "group_size": None, "symmetric": False, "scale": "minmax"}
    return {"schema_version": 3, "unlisted_modules": "keep_fp", "preprocess": [], "unresolved": [],
        "execution": {"mode": "reference_qdq", "compute_dtype": "FP32", "activation_calibration": "static",
                      "order": "joint", "within_stage": "parallel"},
        "layers": [{"module": site, "kind": "linear", "weight": deepcopy(weight), "activation": deepcopy(activation),
                    "weight_method": "uniform_full", "activation_method": "uniform_affine",
                    "weight_calibration": {"strategy": "minmax"},
                    "activation_calibration": {"strategy": "percentile", "percentiles": [.91]} if i == 1 else {"strategy": "minmax"},
                    "reason": "CPU contract fixture", "evidence": ["fixture"]}
                   for i, site in enumerate(("blocks.0.mlp.fc1", "blocks.0.mlp.fc2"))]}


def full_fixture():
    plan = plan_fixture()
    brief = validation_config()["research_brief"]
    brief["hard_constraints"].update(weight_bits=8, activation_bits=6)
    brief["evaluation_protocol"] = {"evaluations": [{"kind": "classification", "split": "search", "count": 4}],
                                    "reload_before_evaluation": True}
    brief["quantization_contract"] = {
        "required_sites": [r["module"] for r in plan["layers"]],
        "specs": {t: deepcopy(plan["layers"][0][t]) for t in ("weight", "activation")},
        "methods": {"weight": "uniform_full", "activation": "uniform_affine"},
        "execution": deepcopy(plan["execution"]), "preprocess": [], "operators": [],
        "calibrations": {"weight": {"strategy": "minmax"}, "activation": {"strategy": "minmax"}},
        "site_overrides": {"blocks.0.mlp.fc2": {"activation_calibration": {"strategy": "percentile", "percentiles": [.91]}}},
        "parameters": {"p": .91},
        "parameter_bindings": [{"parameter": "p", "site": "blocks.0.mlp.fc2", "target": "activation",
                                "field": "percentiles", "as_list": True}],
        "fp_exceptions": {"modules": ["head"], "parameters": ["blocks.0.norm1.weight"]}}
    draft = experiment_payload("L3")
    draft["execution"] = {"service": "quantize_evaluate", "requires_code": True, "interface": "executor_v1", "seed": 9,
        "calibration": {"split": "calibration", "count": 2}, "parameters": {"p": .91},
        "evaluations": deepcopy(brief["evaluation_protocol"]["evaluations"]),
        "resolved_choices": {"training": False, "weight_bits": 8, "activation_bits": 6,
                             "weight_spec": deepcopy(plan["layers"][0]["weight"]),
                             "activation_granularity": "per_tensor", "compute_dtype": "FP32"}}
    return brief, draft, plan


class PlanContractTests(unittest.TestCase):
    def setUp(self):
        self.brief, self.draft, self.plan = full_fixture()

    def frozen(self):
        return preflight_experiment(self.draft, self.brief, executable=True)["execution"]

    def check(self, plan):
        spec = self.frozen()
        return check_plan(plan, spec["resolved_choices"], contract=spec["quantization_contract"])

    def test_normal_plan_and_aliases_are_idempotent_without_scientific_defaults(self):
        spec = self.frozen()
        self.assertEqual(self.check(self.plan)["weight_sites"], 2)
        self.assertEqual(spec["quantization_contract"]["parameters"], {"p": .91})
        self.assertEqual(spec["quantization_contract"]["specs"]["activation"]["bits"], 6)
        payload = preflight_experiment(self.draft, self.brief, executable=True)
        self.assertEqual(preflight_experiment(payload, self.brief, executable=True), payload)

    def test_deleted_site_null_target_and_duplicate_site_fail(self):
        for mutation in (lambda p: p["layers"].pop(), lambda p: p["layers"][0].update(weight=None),
                         lambda p: p["layers"].append(deepcopy(p["layers"][0]))):
            plan = deepcopy(self.plan); mutation(plan)
            with self.subTest(plan=plan), self.assertRaises(ContractError): self.check(plan)

    def test_granularity_axis_bits_parameter_order_and_fp_exception_fail(self):
        changes = [lambda p: p["layers"][0]["weight"].update(granularity="per_tensor", axis=None),
                   lambda p: p["layers"][0]["weight"].update(axis=1),
                   lambda p: p["layers"][0]["weight"].update(bits=4, format="INT4"),
                   lambda p: p["layers"][1]["activation_calibration"].update(percentiles=[.99]),
                   lambda p: p["execution"].update(order="w_then_a"),
                   lambda p: p["layers"].append({**deepcopy(p["layers"][0]), "module": "head"})]
        for mutation in changes:
            plan = deepcopy(self.plan); mutation(plan)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError): self.check(plan)

    def test_conflicting_alias_unknown_required_constraint_and_natural_language_fp_fail(self):
        for where, key, value in [("choices", "weight_granularity", "per_tensor"),
                                  ("brief", "unknown_must_execute", True),
                                  ("contract", "unknown_site_rule", []),
                                  ("contract", "fp_exceptions", "keep head FP")]:
            brief, draft, _ = full_fixture()
            target = draft["execution"]["resolved_choices"] if where == "choices" else brief["hard_constraints"] if where == "brief" else brief["quantization_contract"]
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError): preflight_experiment(draft, brief, executable=True)

    def test_custom_parameters_need_no_binding_but_mirror_conflicts_fail(self):
        self.brief["quantization_contract"]["parameter_bindings"] = []
        self.assertEqual(self.frozen()["parameters"], {"p": .91})
        self.brief, self.draft, _ = full_fixture()
        self.draft["execution"]["resolved_choices"]["p"] = .99
        with self.assertRaisesRegex(ContractError, "冲突"): self.frozen()

    def test_fixed_parameters_populate_execution_mirror(self):
        self.draft['execution'].pop('parameters')
        self.assertEqual(self.frozen()['parameters'], {'p': .91})

    def test_plan_schema_runs_before_capture_or_device_capability(self):
        self.plan["execution"]["compute_dtype"] = "float32"
        with patch("ptq_agent.execution.capabilities.detect", side_effect=AssertionError("device probe too early")), \
                patch("ptq_agent.execution.calibration.capture", side_effect=AssertionError("capture too early")):
            with self.assertRaisesRegex(ContractError, "Schema"): self.check(self.plan)

    def test_allowed_bits_are_resolved_from_task_choice(self):
        self.brief["hard_constraints"]["weight_bits"] = [6, 8]
        self.assertEqual(self.frozen()["quantization_contract"]["specs"]["weight"]["bits"], 8)
        self.draft["execution"]["resolved_choices"]["weight_bits"] = 4
        with self.assertRaises(ContractError): self.frozen()

    def test_explicit_site_override_and_brief_required_sites_are_enforced(self):
        self.brief["required_sites"] = self.brief["quantization_contract"].pop("required_sites")
        self.brief["quantization_contract"]["site_overrides"]["blocks.0.mlp.fc1"] = {
            "weight": {"bits": 6, "format": "INT6"}, "weight_method": "rtn"}
        self.plan["layers"][0]["weight"].update(bits=6, format="INT6")
        self.plan["layers"][0]["weight_method"] = "rtn"
        self.assertEqual(self.check(self.plan)["weight_sites"], 2)
        self.plan["layers"].pop()
        with self.assertRaisesRegex(ContractError, "遗漏"): self.check(self.plan)
        self.brief["required_unknown"] = True
        with self.assertRaisesRegex(ContractError, "未识别"): self.frozen()


class ReloadAndInterfaceTests(TempCase):
    def test_real_cpu_plan_export_reload_evaluates_loaded_instance(self):
        from ptq_agent.execution.pipeline import QuantizationExecutor
        from tools.evaluating import evaluate
        brief, draft, plan = full_fixture()
        payload = preflight_experiment(draft, brief, executable=True)
        backend, cfg, data = backend_and_data(payload["execution"])
        loaded = []
        original_load = QuantizationExecutor.load
        def load(executor, directory):
            value = original_load(executor, directory); loaded.append(value[0]); return value
        def measured(backend, cfg, data, model, *args):
            self.assertIs(model, loaded[-1])
            return evaluate(backend, cfg, data, model, *args)
        directory = self.root / "cpu_reload"; directory.mkdir()
        with patch("tools.experiment_worker.module_from_approval", return_value=SimpleNamespace(build_plan=lambda *a: deepcopy(plan))), \
                patch.object(QuantizationExecutor, "load", load), patch("tools.evaluating.evaluate", side_effect=measured):
            result = evaluate_services({"payload": payload}, backend, cfg, data, directory, None)
        self.assertEqual(result["observations"]["evaluations"]["classification"]["sample_count"], 4)
        self.assertEqual(result["observations"]["reload"]["evaluated_instance"], "reloaded")
        self.assertTrue(result["observations"]["reload"]["host_verified"])
        # custom L3 同样必须走宿主 loader 与实测；以下夹具仅替代审批/文件保护，
        # 模型包、Data、Backend、forward、loader 和 evaluator 均为实际 CPU 实现。
        from tools.experiment_worker import run_experiment
        custom = deepcopy(payload)
        custom["execution"].update(service="custom_experiment", interface="experiment_v1")
        def execute(payload, context):
            for x, sid in context["data"].batches("calibration"):
                context["backend"].forward_model(context["reference_model"], x)
            return {"observations": {"plugin_marker": "not an evaluation"}, "model_dir": str(directory / "model")}
        with patch("tools.experiment_worker.module_from_approval", return_value=SimpleNamespace(execute=execute)), \
                patch("tools.experiment_worker.validate_approval"), \
                patch("tools.experiment_worker.backend_and_data", return_value=(backend, cfg, data)), \
                patch("runtime.guards.protect_files", return_value={"source": "test_double"}), \
                patch("runtime.guards.install_gradient_guard"), patch.object(QuantizationExecutor, "load", load), \
                patch("tools.evaluating.evaluate", side_effect=measured):
            result = run_experiment({"experiment": {"payload": custom}, "approval": {}}, directory)
        self.assertEqual(result["observations"]["evaluations"]["classification"]["sample_count"], 4)
        self.assertEqual(result["provenance"]["execution_facts"]["measurement_kind"], "full_model")

    def interface(self):
        payload = experiment_payload()
        payload["execution"].update(service="custom_experiment", requires_code=True, interface="experiment_v1")
        payload["execution"].pop("evaluation")
        case = {"seed": 3, "shape": [1, 5, 8], "sites": ["blocks.0.mlp.fc2"],
                "counts": {"calibration": 2, "search": 4, "final": 2}}
        return payload, interface_fixture(payload["execution"], case)

    def test_runtime_fixture_uses_production_data_api_and_actual_forward(self):
        payload, (backend, cfg, data) = self.interface()
        from ptq_agent.execution.evaluation import VerifiedManifestData
        self.assertIsInstance(data, VerifiedManifestData)
        def execute(payload, context):
            out = []
            for x, sid in context["data"].batches("calibration"):
                out.append(context["backend"].forward_model(context["reference_model"], x).sum().item())
            return {"observations": {"sums": out}}
        result = execute_custom(SimpleNamespace(execute=execute), payload, backend, cfg, data, self.root)
        self.assertEqual(result["execution_facts"]["forward_calls"], 2)
        self.assertEqual(result["execution_facts"]["sites"], ["blocks.0.mlp.fc2"])
        self.assertEqual(result["execution_facts"]["data"]["calibration"]["sample_ids"], data.ids("calibration"))

    def test_custom_empty_artifact_mapping_survives_worker_closeout(self):
        from tools.experiment_worker import run_experiment
        payload, (backend, cfg, data) = self.interface()
        payload["execution"]["backend"] = {"kind": "fixture"}
        backend.identity = lambda: {"source": "CPU interface fixture"}
        def execute(payload, context):
            for x, sid in context["data"].batches("calibration"):
                context["backend"].forward_model(context["reference_model"], x)
            return {"observations": {"sum": 1}, "artifacts": {}}
        with patch("tools.experiment_worker.module_from_approval", return_value=SimpleNamespace(execute=execute)), \
                patch("tools.experiment_worker.validate_approval"), \
                patch("tools.experiment_worker.backend_and_data", return_value=(backend, cfg, data)), \
                patch("runtime.guards.protect_files", return_value={"source": "test_double"}), \
                patch("runtime.guards.install_gradient_guard"):
            result = run_experiment({"experiment": {"payload": payload}, "approval": {}}, self.root)
        self.assertEqual(len(result["artifacts"]), 1)
        self.assertTrue(result["artifacts"][0]["path"].endswith("gradient_audit.json"))

    def test_custom_artifact_list_is_preserved_and_invalid_outputs_route_to_coding(self):
        from runtime.task_worker import failure_origin
        from tools.experiments import asset
        payload, (backend, cfg, data) = self.interface()
        payload["execution"].pop("calibration")
        output = self.root / "measurement.json"
        atomic_json(output, {"value": 1})
        artifacts = [dict(asset(output), description="raw measurement")]
        plugin = SimpleNamespace(execute=lambda *args: {"observations": {"sum": 1}, "artifacts": artifacts})
        result = execute_custom(plugin, payload, backend, cfg, data, self.root)
        self.assertIs(result["artifacts"], artifacts)
        self.assertEqual(result["artifacts"][0]["description"], "raw measurement")
        request = {"approval": {"candidate": {"path": str(self.root / "approved_source.py")}}}
        invalid = ({"file": str(output)}, None, "path", (), [str(output)], [{}],
                   [{"path": str(output), "sha256": "bad", "bytes": 1}],
                   [dict(asset(output), path="")], [dict(asset(output), bytes=True)],
                   [dict(asset(output), bytes=-1)])
        for artifacts in invalid:
            with self.subTest(artifacts=artifacts):
                try:
                    execute_custom(plugin, payload, backend, cfg, data, self.root)
                    self.fail("Invalid artifacts must fail at the custom return boundary")
                except ContractError as exc:
                    self.assertIn("artifacts", str(exc))
                    self.assertEqual(failure_origin(exc, request), "implementation")

    def test_custom_evaluation_container_and_rows_fail_as_implementation_contracts(self):
        from runtime.task_worker import failure_origin
        payload, (backend, cfg, data) = self.interface()
        payload["execution"].pop("calibration")
        request = {"approval": {"candidate": {"path": str(self.root / "approved_source.py")}}}
        invalid = ([{"split": "search"}], None, "evaluations", {"classification": []},
                   {"classification": None}, {"classification": 1})
        for evaluations in invalid:
            with self.subTest(evaluations=evaluations):
                plugin = SimpleNamespace(execute=lambda *args: {"observations": {"evaluations": evaluations}})
                try:
                    execute_custom(plugin, payload, backend, cfg, data, self.root)
                    self.fail("Invalid evaluation return values must fail at the custom boundary")
                except ContractError as exc:
                    self.assertIn("observations.evaluations", str(exc))
                    self.assertEqual(failure_origin(exc, request), "implementation")

    def test_custom_evaluation_split_and_sample_ids_require_actual_reads(self):
        from runtime.task_worker import failure_origin
        payload, (backend, cfg, data) = self.interface()
        payload["execution"].pop("calibration")
        request = {"approval": {"candidate": {"path": str(self.root / "approved_source.py")}}}
        invalid = ({}, {"split": []}, {"split": {}}, {"split": "unread"},
                   {"split": "search", "sample_ids": ["invented"]},
                   {"split": "search", "sample_ids": data.ids("search"), "full_dataset": True})
        for row in invalid:
            with self.subTest(row=row):
                def execute(payload, context):
                    list(context["data"].batches("search"))
                    return {"observations": {"evaluations": {"classification": row}}}
                try:
                    execute_custom(SimpleNamespace(execute=execute), payload, backend, cfg, data, self.root)
                    self.fail("Unverified evaluation ranges must fail at the custom boundary")
                except ContractError as exc:
                    self.assertIn("自定义评测", str(exc))
                    self.assertEqual(failure_origin(exc, request), "implementation")

    def test_custom_evaluation_object_preserves_valid_result_and_optional_absence(self):
        payload, (backend, cfg, data) = self.interface()
        payload["execution"].pop("calibration")
        evaluations = {"classification": {"split": "search", "sample_ids": data.ids("search"),
                                         "sample_count": 4, "metrics": {"top1": 25.0}, "full_dataset": False}}
        def execute(payload, context):
            list(context["data"].batches("search"))
            return {"observations": {"evaluations": evaluations}}
        result = execute_custom(SimpleNamespace(execute=execute), payload, backend, cfg, data, self.root)
        self.assertIs(result["observations"]["evaluations"], evaluations)
        for observations in ({"statistic": 1}, {"statistic": 1, "evaluations": {}}):
            plugin = SimpleNamespace(execute=lambda *args: {"observations": observations})
            result = execute_custom(plugin, payload, backend, cfg, data, self.root)
            self.assertEqual(result["observations"], observations)

    def test_math_branch_cannot_hide_bad_split_batch_or_forward_signature(self):
        for bug in ("split", "batch", "forward"):
            payload, (backend, cfg, data) = self.interface()
            def execute(payload, context):
                if "activations" in context:
                    return {"observations": {"sum": sum(context["activations"])}}
                for row in context["data"].batches(payload["execution"]["calibration"] if bug == "split" else "calibration"):
                    if bug == "batch": context["backend"].forward_model(context["reference_model"], row)
                    if bug == "forward": context["backend"].forward_model(row[0])
                return {"observations": {"sum": 1}}
            self.assertEqual(execute(payload, {"activations": [1, 2]})["observations"]["sum"], 3)
            with self.subTest(bug=bug), self.assertRaises((ContractError, TypeError)):
                execute_custom(SimpleNamespace(execute=execute), payload, backend, cfg, data, self.root)

    def test_custom_preflight_requires_no_model_or_math_fixture(self):
        payload, _ = self.interface()
        payload.pop("validation")
        checked = preflight_experiment(payload, validation_config()["research_brief"], executable=True)
        self.assertEqual(checked["validation"], {})
        self.assertNotIn("interface_cases", checked["validation"])

    def test_fp_exception_parameter_is_checked_on_actual_model(self):
        from core.execution_contract import check_fp_state
        import torch
        brief, draft, _ = full_fixture()
        spec = preflight_experiment(draft, brief, executable=True)["execution"]
        backend, _, _ = backend_and_data(spec)
        model = deepcopy(backend.model)
        check_fp_state(model, backend, spec["quantization_contract"])
        with torch.no_grad(): model.blocks[0].norm1.weight[0] += 1
        with self.assertRaisesRegex(ContractError, "FP 例外参数"): check_fp_state(model, backend, spec["quantization_contract"])


class EvidenceBoundaryTests(TempCase):
    def measured(self):
        spec = preflight_experiment(experiment_payload(), validation_config()["research_brief"])
        data = {"measurement_source": "synthetic_computation", "observations": {"mse": .1},
                "provenance": {"model_identity": {"fixture": True}, "execution_facts": {
                    "measurement_kind": "local_activation_probe", "quantization": {"blocks.0.mlp.fc2": {
                        "weight": None, "activation": {"bits": 4, "granularity": "per_tensor", "axis": None}}},
                    "data": {"calibration": {"split": "calibration", "sample_count": 2,
                                              "sample_ids": ["a", "b"], "full_dataset": False}}}}}
        return spec, data

    def test_scope_binds_code_and_measurement_identity_but_not_artifact_location(self):
        payload, data = self.measured()
        payload['execution'].update(requires_code=True, measurement_implementation={'files': {'worker.py': 'v1'}})
        artifact = {'source_sha256': 'source-one', 'interface': 'quantizer_v1', 'id': 'job-one', 'path': '/one'}
        data['provenance']['code_artifact'] = artifact
        scope = canonical_scope({'payload': payload}, data)
        repeated = deepcopy(data)
        repeated['provenance']['code_artifact'].update(id='job-two', path='/two')
        self.assertEqual(canonical_scope({'payload': payload}, repeated), scope)
        self.assertEqual(bind_claim_scope({}, [scope, canonical_scope({'payload': payload}, repeated)]), scope)
        repeated['provenance']['code_artifact']['source_sha256'] = 'source-two'
        other = canonical_scope({'payload': payload}, repeated)
        with self.assertRaisesRegex(ContractError, '范围不同'):
            bind_claim_scope({}, [scope, other])
        payload['execution']['measurement_implementation']['files']['worker.py'] = 'v2'
        with self.assertRaisesRegex(ContractError, '范围不同'):
            bind_claim_scope({}, [scope, canonical_scope({'payload': payload}, data)])
        partial = deepcopy(scope)
        partial['implementation']['code'].pop('source_sha256')
        with self.assertRaisesRegex(ContractError, 'implementation'):
            validate_claim_scope(partial, scope)

    def test_scope_keeps_different_consumed_parameters_and_method_options_separate(self):
        payload, data = self.measured()
        facts = data['provenance']['execution_facts']
        facts['parameters'] = {'bits': 4, 'signed': True, 'group_size': None, 'alpha': .5}
        facts['quantization']['blocks.0.mlp.fc2']['method_options'] = {'clip_ratio': .99}
        first = canonical_scope({'payload': payload}, data)
        for change in (lambda f: f['parameters'].update(alpha=.7),
                       lambda f: f['parameters'].update(signed=False),
                       lambda f: f['quantization']['blocks.0.mlp.fc2']['method_options'].update(clip_ratio=.9)):
            altered = deepcopy(data)
            change(altered['provenance']['execution_facts'])
            with self.assertRaisesRegex(ContractError, '范围不同'):
                bind_claim_scope({}, [first, canonical_scope({'payload': payload}, altered)])

    def test_probe_records_full_consumed_spec_and_broker_binds_implementation(self):
        from tools.experiment_worker import run_experiment
        from ptq_agent.execution.recipes import spec as quant_spec
        wf = self.workflow(1)
        experiment = wf.broker.freeze(experiment_payload(), requester='generation')
        wf.journal.commit('fixture/spec', cards=[experiment])
        # Worker protection is process-scoped: never install Landlock in unittest's process.
        import time
        from core.journal import read_json
        tid = wf.queue.enqueue('probe', 'experiment', inputs={'experiment_ref': experiment['id']})
        try:
            wf.dispatch()
            deadline = time.monotonic() + 40
            while wf.queue.get(tid)['state'] == 'running' and time.monotonic() < deadline:
                time.sleep(.1)
                wf.poll()
            task = wf.queue.get(tid)
            self.assertEqual(task['state'], 'completed', task.get('error'))
            directory = Path(task['request_path']).parent
            data = read_json(directory / 'result.json')['data']
        finally:
            wf.runtime.stop()
        facts = data['provenance']['execution_facts']
        params = experiment['payload']['execution']['parameters']
        self.assertEqual(facts['parameters'], params)
        self.assertEqual(facts['quantization']['blocks.0.mlp.fc2']['activation'],
                         quant_spec(params['bits'], params['signed'], params['granularity'], params['axis'], params['group_size']))
        self.assertEqual(facts['quantization']['blocks.0.mlp.fc2']['method_options'], {})
        receipt = {'experiment_ref': experiment['id'], 'experiment_sha256': experiment['content_sha256'],
                   'execution_id': 'cpu-numeric-probe', 'status': 'completed', 'resources': {}, 'data': data}
        receipt_path = directory / 'receipt.json'
        atomic_json(receipt_path, receipt)
        evidence = wf.broker.evidence(experiment, receipt, receipt_path)
        self.assertIn('measurement_sha256', evidence['payload']['scope']['implementation'])
        wf.broker.validate_evidence_for(experiment, evidence)
        # 封装边界夹具：验证代码元数据取自宿主参数，而非 worker 原始 provenance。
        artifact = {'source_sha256': 'fixture-source', 'interface': 'quantizer_v1'}
        bound = wf.broker.evidence(experiment, receipt, receipt_path, code_artifact=artifact)
        self.assertEqual(bound['payload']['scope']['implementation']['code'], artifact)
        self.assertNotIn('code_artifact', data['provenance'])

    def test_scope_comes_from_verified_local_facts_and_never_target_w4(self):
        spec, data = self.measured()
        spec["scope"].update(model="unmeasured", site=["all"], weight_bits=4)
        scope = canonical_scope({"payload": spec}, data)
        self.assertEqual(scope["model"], "tiny_vit")
        self.assertNotIn("weight_bits", scope)
        self.assertEqual(scope["site"], ["blocks.0.mlp.fc2"])
        validate_claim_scope(scope, scope)
        for key, wrong in (("model", ["tiny_vit", "other"]), ("site", ["blocks.0.mlp.fc2", "other"]),
                           ("activation_bits", [4, 8]), ("weight_bits", 4), ("data", {"calibration": {"full_dataset": True}})):
            claim = deepcopy(scope); claim[key] = wrong
            with self.subTest(key=key), self.assertRaises(ContractError): validate_claim_scope(claim, scope)
        del scope["data"]
        with self.assertRaisesRegex(ContractError, "遗漏"): validate_claim_scope(scope, canonical_scope({"payload": spec}, data))

    def test_exact_queries_share_scope_and_do_not_use_suffix_fallback(self):
        spec, data = self.measured(); scope = canonical_scope({"payload": spec}, data)
        self.assertTrue(matches(scope, {"model": "tiny_vit", "activation_bits": 4, "site": "blocks.0.mlp.fc2"}))
        self.assertFalse(matches(scope, {"weight_bits": 4}))
        self.assertFalse(matches({"research_target": {"model": "other"}}, {"model": "other"}))

    def test_every_l3_service_rejects_missing_results_and_reloaded_claims(self):
        for service in ("quantize_evaluate", "evaluate_artifact", "custom_experiment"):
            brief, draft, _ = full_fixture()
            draft["execution"]["service"] = service
            result = {"observations": {"statistic": 1}, "provenance": {"execution_facts": {"measurement_kind": "contract_fixture_only"}}}
            with self.subTest(service=service), self.assertRaisesRegex(ContractError, "评测结果"):
                check_results(draft, brief, result)
            result["observations"]["evaluations"] = {"classification": {"sample_count": 4,
                "sample_ids": ["a", "b", "c", "d"], "split": "search", "full_dataset": False, "metrics": {"top1": 50.}}}
            result["provenance"]["execution_facts"]["data"] = deepcopy(result["observations"]["evaluations"])
            with self.assertRaisesRegex(ContractError, "重载"): check_results(draft, brief, result)
            result["observations"]["reload"] = {"host_verified": True, "evaluated_instance": "reloaded"}
            check_results(draft, brief, result)
            result["observations"]["evaluations"]["classification"]["sample_ids"] = ["a"] * 4
            with self.assertRaises(ContractError): check_results(draft, brief, result)

    def test_custom_required_result_schema_and_unknown_scope_fail_closed(self):
        draft = experiment_payload("L3"); draft["execution"]["service"] = "custom_experiment"
        brief = {"evaluation_protocol": {"required_results": [{"path": "stats.value", "schema": {"type": "number"}}]}}
        check_results(draft, brief, {"observations": {"stats": {"value": 3.}},
                                    "provenance": {"execution_facts": {"measurement_kind": "contract_fixture_only"}}})
        for obs in ({"stats": {}}, {"stats": {"value": "3"}}):
            with self.assertRaises(ContractError): check_results(draft, brief, {"observations": obs})

    def test_broker_rejects_custom_completed_receipt_before_evidence_publication(self):
        from core.workflow import Workflow
        brief, draft, _ = full_fixture()
        config = validation_config(); config.update(purpose="infrastructure_test", enable_l3=True)
        config["budget"]["min_free_bytes"] = 0; config["research_brief"] = brief
        wf = Workflow(self.root / "broker", config)
        draft["execution"].update(service="custom_experiment", interface="experiment_v1")
        draft["validation"]["interface_cases"] = [{"seed": 3, "shape": [1, 5, 8], "sites": ["blocks.0.mlp.fc2"],
            "counts": {"calibration": 2, "search": 4, "final": 2}}]
        spec = wf.broker.freeze(draft, requester="judger")
        wf.journal.commit("fixture/spec", cards=[spec])
        receipt = {"status": "completed", "experiment_ref": spec["id"], "experiment_sha256": spec["content_sha256"],
            "execution_id": "fixture-only", "resources": {}, "data": {"observations": {"looks_done": True},
                "provenance": {}, "measurement_source": "synthetic_computation"}}
        path = self.root / "receipt.json"; atomic_json(path, receipt)
        with self.assertRaisesRegex(ContractError, "评测结果"): wf.broker.evidence(spec, receipt, path)
        self.assertEqual(wf.journal.find_cards("EvidenceCard"), [])

    def test_memory_and_retrieval_use_same_canonical_measurement(self):
        from core.memory import ResearchMemory
        from tests.support import meta
        spec, data = self.measured()
        journal = self.journal(); memory = ResearchMemory(journal)
        experiment = make_card("ExperimentCard", journal.run_id, spec)
        scope = canonical_scope(experiment, data)
        evidence = make_card("EvidenceCard", journal.run_id, {**data, "scope": scope,
            "experiment_ref": experiment["id"], "execution_fingerprint": "fixture-only", "execution_id": "fixture-one",
            "status": "completed", "resources": {}, "artifacts": []}, refs=[experiment["id"]])
        journal.commit("fixture/evidence", cards=[experiment, evidence])
        payload = meta(1); payload["evidence_refs"] = [evidence["id"]]
        payload["knowledge_updates"] = [{"claim_id": "k", "text": "test fixture", "status": "observed_once",
            "scope": deepcopy(scope), "evidence_refs": [evidence["id"]], "rationale": "contract fixture only"}]
        card = make_card("MetareviewCard", journal.run_id, payload, author="recorder", refs=[evidence["id"]])
        journal.commit("fixture/knowledge", cards=[card], events=[memory.apply_metareview(card)])
        retrieved = MemoryRetriever(journal).retrieve("judger", query={"site": "blocks.0.mlp.fc2", "activation_bits": 4})
        self.assertIn(evidence["id"], {c["ref"] for c in retrieved["entries"]})
        self.assertEqual(len(MemoryRetriever(journal).retrieve("judger", query={"layer": "L2", "site": "blocks.0.mlp.fc2", "activation_bits": 4})["entries"]), 1)
        self.assertFalse(MemoryRetriever(journal).retrieve("judger", query={"layer": "L2", "weight_bits": 4})["entries"])
        for change in (lambda s: s.update(site=["blocks.0.mlp.fc2", "other"]),
                       lambda s: s["quantization"]["blocks.0.mlp.fc2"]["activation"].pop("granularity")):
            wrong = deepcopy(payload); change(wrong["knowledge_updates"][0]["scope"])
            with self.assertRaises(ContractError): memory.validate_metareview(make_card("MetareviewCard", journal.run_id, wrong, author="recorder"))

    def test_protocol_checks_actual_data_identity_and_calibration_count(self):
        brief, draft, _ = full_fixture()
        brief["evaluation_protocol"].update(calibration_samples=2, batch_size=1)
        data = {"observations": {"evaluations": {"classification": {"sample_count": 4, "sample_ids": ["a", "b", "c", "d"],
            "metrics": {"top1": 50.}, "split": "search", "full_dataset": False}},
            "reload": {"host_verified": True, "evaluated_instance": "reloaded"}}, "provenance": {
                "data_identity": {"split_ids": {"search": ["a", "b", "c", "d"]}, "batch_size": 1},
                "execution_facts": {"measurement_kind": "fixture-only", "data": {"calibration": {"sample_count": 2}}}}}
        data["provenance"]["execution_facts"]["data"].update(deepcopy(data["observations"]["evaluations"]))
        check_results(draft, brief, data)
        data["provenance"]["execution_facts"]["data"]["calibration"]["sample_count"] = 1
        with self.assertRaisesRegex(ContractError, "calibration_samples"): check_results(draft, brief, data)
        data["provenance"]["execution_facts"]["data"]["calibration"]["sample_count"] = 2
        data["observations"]["evaluations"]["classification"]["sample_ids"].reverse()
        with self.assertRaisesRegex(ContractError, "sample_ids"): check_results(draft, brief, data)

    def test_custom_full_dataset_flag_cannot_expand_host_measured_subset(self):
        draft = experiment_payload("L3")
        draft["execution"].update(service="custom_experiment", evaluations=[{"kind": "custom_metric", "split": "search", "count": 2}])
        row = {"sample_ids": ["a", "b"], "sample_count": 2, "split": "search", "full_dataset": False, "metrics": {"score": .1}}
        data = {"observations": {"evaluations": {"custom_metric": deepcopy(row)}},
                "provenance": {"execution_facts": {"measurement_kind": "fixture", "data": {"search": deepcopy(row)}}}}
        check_results(draft, {"evaluation_protocol": {}}, data)
        data["observations"]["evaluations"]["custom_metric"]["full_dataset"] = True
        with self.assertRaisesRegex(ContractError, "full_dataset"):
            check_results(draft, {"evaluation_protocol": {"full_dataset": True}}, data)


class InputFeedbackTests(TempCase):
    def test_wrong_query_types_and_whitespace_reasons_fail_as_contract_errors(self):
        memory = MemoryRetriever(self.journal())
        for query in ([], "x", {"text": {}}, {"evidence_ids": "id"}, {"evidence_ids": [""]}):
            with self.subTest(query=query), self.assertRaises(ContractError): memory.retrieve("judger", query=query)
        for query in ({}, {"text": " "}):
            self.assertEqual(memory.retrieve("judger", query=query)["entries"], [])
        for reasons in ([""], ["   "], [None], [{}], "reason", []):
            with self.subTest(reasons=reasons), self.assertRaises(ContractError):
                validate_payload("ReviewDecisionCard", review("screening", eligible=False,
                    hard_rejection_reasons=reasons, similarity_analysis="checked", evidence_gaps=[]))

    def test_invalid_query_batch_returns_to_original_task_then_succeeds(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role("fixture/query", "generate", refs=[wf.brief["id"]], context={})
        wf.queue.ready(); task = wf.queue.claim(tid)
        deliver = lambda task, response: preflight_tests.DraftRepairRegression.deliver(self, wf, task, response)
        calls = [{"id": "q", "name": "retrieve_memory", "arguments": {"query": {"text": []}}}]
        deliver(task, {"tool_calls": calls})
        self.assertEqual(len(wf.tasks()), 1)
        pending = wf.queue.get(tid)
        self.assertEqual((pending["state"], pending["repair_count"]), ("retry", 1))
        self.assertEqual(pending["repair_feedback"]["rejected_response"]["tool_calls"], calls)
        wf.queue.ready(); task = wf.queue.claim(tid)
        calls[0]["arguments"]["query"] = {"text": "fixture"}
        deliver(task, {"tool_calls": calls})
        self.assertEqual(wf.queue.get(tid)["state"], "waiting")
        with patch.object(wf.runtime, "launch", side_effect=AssertionError("no LLM")):
            wf.dispatch()
        self.assertEqual(wf.queue.get(wf.queue.get(tid)["waiting_for"][0])["state"], "completed")

    def test_screen_repair_is_bounded_and_unknown_program_exceptions_propagate(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role("fixture/screen", "screen", refs=[wf.brief["id"]], context={})
        wf.queue.ready(); task = wf.queue.claim(tid)
        response = {"decision": {"action": "submit", "content": review("screening", eligible=False,
            hard_rejection_reasons=[" "], similarity_analysis="checked", evidence_gaps=[])}}
        for attempt in range(3):
            preflight_tests.DraftRepairRegression.deliver(self, wf, task, response)
            self.assertEqual(wf.queue.get(tid)["state"], "retry" if attempt < 2 else "failed")
            if attempt < 2:
                wf.queue.ready(); task = wf.queue.claim(tid)
        self.assertEqual(wf.queue.get(tid)["repair_count"], 2)
        for exc in (RuntimeError("unexpected"), TypeError("program bug"), KeyError("program bug")):
            with patch.object(wf.runtime, "poll", return_value=[(task, {}, self.root / "receipt")]), \
                    patch.object(wf, "accept", side_effect=exc), self.assertRaises(type(exc)):
                wf.poll()


if __name__ == "__main__":
    unittest.main()
