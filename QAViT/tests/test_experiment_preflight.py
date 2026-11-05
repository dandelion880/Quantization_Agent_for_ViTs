"""草案到冻结的 CPU 合同回归；脚本回执不代表真实 LLM 或科研测量。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import atomic_json
from core.workflow import Workflow
from tests.support import experiment_payload, validation_config
from tests.test_infrastructure import TempCase
from tools.experiments import preflight_experiment, resolve_execution


class InterfaceFixtureRegression(unittest.TestCase):
    def configured_fixture_inputs(self):
        execution = {'backend': {'config_path': str(Path(__file__).resolve().parents[1] / 'configs/deitb_backend.json'),
                                 'backend_overrides': {'calibration_samples': 32}}}
        case = {'seed': 3, 'shape': [1, 5, 8], 'sites': ['blocks.0.mlp.fc1', 'blocks.0.mlp.fc2'],
                'counts': {'calibration': 2, 'search': 1, 'final': 1}}
        return execution, case

    def test_frozen_backend_configuration_supports_real_prepare_without_mutation(self):
        from tools.experiment_worker import backend_configuration
        from tools.runtime_interface import interface_fixture
        from ptq_agent.execution.pipeline import QuantizationExecutor
        from tests.test_execution_contracts import plan_fixture
        execution, case = self.configured_fixture_inputs()
        original = deepcopy(execution)
        effective = backend_configuration(execution)
        backend, cfg, data = interface_fixture(execution, case)
        self.assertEqual(cfg['constraints'], effective['constraints'])
        self.assertEqual({k: v for k, v in cfg['loop'].items() if k != 'evaluation'},
                         {k: v for k, v in effective['loop'].items() if k != 'evaluation'})
        self.assertEqual(cfg['backend']['device'], 'cpu')
        self.assertEqual(len(data.ids('calibration')), 2)
        executor = QuantizationExecutor(backend, cfg, data)
        plan = plan_fixture()
        model, weights, activations, _ = executor.prepare(plan, {})
        self.assertEqual(set(weights), set(case['sites']))
        self.assertEqual(set(activations), set(case['sites']))
        executor.audit(model, plan)
        backend.reset()
        self.assertEqual(execution, original)

    def test_frozen_backend_mode_scope_method_and_protection_restrictions_remain(self):
        from tools.experiment_worker import backend_configuration
        from tools.runtime_interface import interface_fixture
        from ptq_agent.execution.pipeline import QuantizationExecutor
        from ptq_agent.schemas import PTQError
        from tests.test_execution_contracts import plan_fixture
        execution, case = self.configured_fixture_inputs()
        restrictions = [('loop', 'execution_modes', ['native']),
                        ('loop', 'scope_modules', ['blocks.0.mlp.fc1']),
                        ('loop', 'allowed_activation_methods', ['rtn']),
                        ('loop', 'quantize_targets', ['weight']),
                        ('constraints', 'protected_modules', ['blocks.0.mlp.fc1'])]
        for section, key, value in restrictions:
            with self.subTest(key=key):
                cfg = backend_configuration(execution)
                cfg[section][key] = value
                backend, passed, data = interface_fixture({'backend': {'config': cfg}}, case)
                self.assertEqual(passed[section][key], value)
                with self.assertRaises(PTQError):
                    QuantizationExecutor(backend, passed, data).prepare(plan_fixture(), {})

    def test_native_calibration_preserves_fixture_and_reset_rejects_mutation(self):
        import torch
        from tools.runtime_interface import interface_fixture
        from ptq_agent.execution.pipeline import QuantizationExecutor, build_model
        from ptq_agent.schemas import PTQError
        from tests.test_execution_contracts import plan_fixture
        case = {'seed': 3, 'shape': [1, 5, 8],
                'sites': ['blocks.0.mlp.fc1', 'blocks.0.mlp.fc2'],
                'counts': {'calibration': 2, 'search': 1, 'final': 1}}
        backend, cfg, data = interface_fixture({}, case)
        plan = plan_fixture()
        model = build_model(backend, plan)
        executor = QuantizationExecutor(backend, cfg, data)
        weights, activations, report = executor.fit_model(model, plan)
        self.assertEqual(set(weights), set(case['sites']))
        self.assertEqual(set(activations), set(case['sites']))
        backend.reset()
        for mutation in ('training', 'gradient', 'weight', 'buffer'):
            with self.subTest(mutation=mutation):
                original = deepcopy(backend.model)
                if mutation == 'training':
                    backend.model.train()
                elif mutation == 'gradient':
                    next(backend.model.parameters()).requires_grad_(True)
                elif mutation == 'weight':
                    with torch.no_grad():
                        next(backend.model.parameters()).add_(1)
                else:
                    backend.model.register_buffer('unexpected', torch.ones(1))
                with self.assertRaises(PTQError):
                    backend.reset()
                backend.model = original
                backend.reset()

    def test_native_catalog_matches_constructed_fixture_without_inventing_targets(self):
        from tools.runtime_interface import interface_fixture
        from ptq_agent.execution.sites import site_catalog
        for depth in (1,12):
            with self.subTest(depth=depth):
                sites=['patch_embed.proj']+[f'blocks.{i}.{part}' for i in range(depth)
                    for part in ('attn.qkv','attn.proj','mlp.fc1','mlp.fc2')]
                case={'seed':42,'shape':[1,3,32,32],'sites':sites,
                      'counts':{'calibration':1,'search':1,'final':1}}
                backend,cfg,data=interface_fixture({},case)
                catalog=site_catalog(backend,cfg)
                required=set(sites)|{f'blocks.{i}.attn::{kind}' for i in range(depth)
                    for kind in ('attention_query','attention_key','attention_value','post_softmax')}
                self.assertEqual(len(required),1+8*depth)
                self.assertTrue(required<=catalog.keys())
                self.assertFalse(any(k.startswith(f'blocks.{depth}.') for k in catalog))
                self.assertEqual(catalog['patch_embed.proj']['activation_channel_axis'],1)
                self.assertEqual(catalog['blocks.0.attn::attention_key']['activation_channel_axis'],-2)
                self.assertTrue(catalog['head']['protected'])
                inventory=backend.modules();inventory[0]['weight_shape'].append(-1)
                self.assertNotEqual(inventory,backend.modules())
                self.assertFalse(data.identity.get('full_validation'))

    def test_vit_fixture_preserves_module_types_shapes_and_seeded_data(self):
        import torch
        from tools.runtime_interface import interface_fixture
        sites = ['patch_embed.proj'] + [f'blocks.{i}.{part}' for i in range(12)
                 for part in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')]
        case = {'seed': 42, 'shape': [1, 3, 224, 224], 'sites': sites,
                'counts': {'calibration': 2, 'search': 1, 'final': 1}}
        backend, _, data = interface_fixture({}, case)
        self.assertIsInstance(backend.model.get_submodule(sites[0]), torch.nn.Conv2d)
        observed = {}; handles = []
        for name in sites:
            module = backend.model.get_submodule(name)
            if name != sites[0]: self.assertIsInstance(module, torch.nn.Linear)
            handles.append(module.register_forward_pre_hook(lambda m, args, n=name: observed.setdefault(n, tuple(args[0].shape)) and None))
        image, sid = next(data.batches('calibration'))
        backend.forward_model(backend.model, image)
        for handle in handles: handle.remove()
        self.assertEqual(set(observed), set(sites))
        self.assertEqual(observed['patch_embed.proj'], (1, 3, 224, 224))
        self.assertTrue(all(len(observed[n]) == 3 for n in sites[1:]))
        repeat, _, other_data = interface_fixture({}, case)
        other_image, other_sid = next(other_data.batches('calibration'))
        self.assertTrue(torch.equal(image, other_image)); self.assertEqual(sid, other_sid)
        self.assertTrue(all(torch.equal(v, repeat.model.state_dict()[k]) for k, v in backend.model.state_dict().items()))
        self.assertFalse(data.identity.get('full_validation'))

    def test_non_image_interface_fixture_keeps_existing_linear_contract(self):
        import torch
        from tools.runtime_interface import interface_fixture
        case = {'seed': 7, 'shape': [1, 4, 8], 'sites': ['block.linear'],
                'counts': {'calibration': 2, 'search': 1, 'final': 1}}
        backend, _, data = interface_fixture({}, case)
        self.assertIsInstance(backend.model.get_submodule('block.linear'), torch.nn.Linear)
        image, _ = next(data.batches('calibration'))
        self.assertEqual(tuple(backend.forward_model(backend.model, image).shape), (1, 4, 8))


class CalibrationPreflight(unittest.TestCase):
    def setUp(self):
        self.brief = validation_config()["research_brief"]
        self.draft = experiment_payload()

    def check(self, draft=None):
        return preflight_experiment(self.draft if draft is None else draft, self.brief, executable=True)

    def test_legacy_count_completes_only_known_view_and_is_idempotent(self):
        del self.draft["execution"]["calibration"]
        original = deepcopy(self.draft)
        value = self.check()
        self.assertEqual(value["execution"]["calibration"], {"split": "calibration", "count": 2})
        self.assertEqual(self.check(value), value)
        self.assertEqual(self.draft, original)
        self.assertEqual(value["execution"]["resolved_choices"], original["execution"]["resolved_choices"])
        self.assertEqual({k: v for k, v in value.items() if k not in ("execution", "stage")},
                         {k: v for k, v in original.items() if k != "execution"})

    def test_canonical_view_generates_mirror_and_preserves_extensions(self):
        execution = self.draft["execution"]
        del execution["resolved_choices"]["calibration_samples"]
        execution["calibration"] = {"count": 1, "offset": 1, "identity_note": "explicit subset"}
        value = self.check()["execution"]
        self.assertEqual(value["resolved_choices"]["calibration_samples"], 1)
        self.assertEqual(value["calibration"], {"split": "calibration", "count": 1,
                                               "offset": 1, "identity_note": "explicit subset"})

    def test_missing_count_inherits_explicit_protocol_not_backend_pool(self):
        del self.draft["execution"]["calibration"]
        del self.draft["execution"]["resolved_choices"]["calibration_samples"]
        self.brief["evaluation_protocol"]["calibration_samples"] = 8
        self.assertEqual(self.check()["execution"]["calibration"]["count"], 8)
        del self.brief["evaluation_protocol"]["calibration_samples"]
        with self.assertRaisesRegex(ContractError, "execution.calibration.count 缺失"):
            self.check()

    def test_explicit_count_conflict_names_both_fields_without_mutation(self):
        self.draft["execution"]["calibration"]["count"] = 1
        original = deepcopy(self.draft)
        with self.assertRaisesRegex(ContractError,
                r"execution.calibration.count=1.*execution.resolved_choices.calibration_samples=2.*显式冲突"):
            self.check()
        self.assertEqual(self.draft, original)

    def test_invalid_types_and_counts_are_not_coerced(self):
        for field in ("count", "calibration_samples"):
            for value in (True, 0, -1, "2", 2.0):
                with self.subTest(field=field, value=value):
                    draft = deepcopy(self.draft)
                    target = draft["execution"]["calibration" if field == "count" else "resolved_choices"]
                    target[field] = value
                    with self.assertRaisesRegex(ContractError, "execution.*" + field):
                        self.check(draft)
        self.draft["execution"]["calibration"] = None
        with self.assertRaisesRegex(ContractError, "execution.calibration"):
            self.check()

    def test_fixed_split_conflict_is_not_silently_ignored_by_worker(self):
        self.draft["execution"]["calibration"]["split"] = "search"
        with self.assertRaisesRegex(ContractError, "execution.calibration.split='search'.*固定读取"):
            self.check()

    def test_custom_split_must_be_explicit_and_can_differ(self):
        execution = self.draft["execution"]
        execution.update(service="custom_experiment", requires_code=True, interface="experiment_v1")
        self.draft["validation"]["interface_cases"] = [{"seed": 1, "shape": [1, 4, 8], "sites": [execution["site"]],
            "counts": {"calibration": 2, "search": 4, "final": 2}}]
        del execution["calibration"]
        with self.assertRaisesRegex(ContractError, "execution.calibration.split 缺失.*无法唯一确定"):
            self.check()
        execution["calibration"] = {"split": "search"}
        self.assertEqual(self.check()["execution"]["calibration"], {"split": "search", "count": 2})

    def test_explicit_full_view_is_preserved_but_not_inferred(self):
        self.draft["execution"]["calibration"]["count"] = None
        del self.draft["execution"]["resolved_choices"]["calibration_samples"]
        value = self.check()["execution"]
        self.assertIsNone(value["calibration"]["count"])
        self.assertIsNone(value["resolved_choices"]["calibration_samples"])
        self.draft["execution"]["resolved_choices"]["calibration_samples"] = 2
        with self.assertRaisesRegex(ContractError, "count=None.*calibration_samples=2"):
            self.check()

    def test_missing_scientific_choices_require_repair(self):
        self.brief['hard_constraints'].pop('training')  # No unique user-supplied value to inherit.
        self.brief['hard_constraints'].pop('activation_bits')
        for path in (("parameters", "bits"),):
            with self.subTest(path=path):
                draft = deepcopy(self.draft)
                target = draft["execution"]
                for part in path[:-1]:
                    target = target[part]
                del target[path[-1]]
                draft["execution"]["resolved_choices"].pop("activation_bits", None)
                with self.assertRaisesRegex(ContractError, "execution." + ".".join(path)):
                    self.check(draft)

    def test_existing_constraints_and_service_levels_remain_enforced(self):
        self.brief["hard_constraints"]["calibration_samples"] = 1
        self.assertEqual(self.check()['execution']['calibration']['count'], 2)
        # Explicit exploratory sampling does not have to copy final-candidate calibration.
        del self.brief["hard_constraints"]["calibration_samples"]
        self.draft["level"] = "L3"
        with self.assertRaisesRegex(ContractError, "level=L3.*tensor_probe"):
            self.check()

    def test_services_without_calibration_do_not_acquire_scientific_defaults(self):
        for service, level in (("profile", "L1"), ("evaluate_artifact", "L3")):
            with self.subTest(service=service):
                draft = deepcopy(self.draft)
                draft["level"] = level
                draft["execution"]["service"] = service
                if service == 'profile':
                    draft['execution'].update(tool='query_structure', arguments={})
                draft["execution"]["parameters"] = {}
                if level == "L3":
                    draft["execution"]["evaluations"] = [{"kind": "classification", "split": "search", "count": 4}]
                del draft["execution"]["calibration"]
                del draft["execution"]["resolved_choices"]["calibration_samples"]
                value = self.check(draft)["execution"]
                self.assertNotIn("calibration", value)
                self.assertNotIn("calibration_samples", value["resolved_choices"])


class DraftRepairRegression(TempCase):
    def plan(self):
        config = validation_config()
        config.update(purpose="infrastructure_test", enable_l3=True)
        config["budget"]["min_free_bytes"] = 0
        wf = Workflow(self.root / "research", config)
        tid = wf.enqueue_role("r001/plan_l3", "plan_l3", refs=[wf.brief["id"]], context={"round": 1})
        wf.queue.ready()
        task = wf.queue.claim(tid)
        draft = experiment_payload("L3")
        draft["execution"].update(service="quantize_evaluate", requires_code=True, interface="executor_v1",
            parameters={}, evaluations=[{"kind": "classification", "split": "search", "count": 4}])
        return wf, task, draft

    def deliver(self, wf, task, response):
        path = wf.run_dir / "jobs" / (task["task_id"] + f"-a{task['attempt']:04d}") / "result.json"
        receipt = {"status": "completed", "data": {"response": {"source": "test_double", **response}}}
        atomic_json(path, receipt)
        with patch.object(wf.runtime, "poll", return_value=[(task, receipt, path)]), \
                patch.object(wf.runtime, "settle"):
            self.assertEqual(wf.poll(), 1)
        return receipt["data"]["response"]

    def submit_response(self, draft):
        return {"decision": {"action": "submit", "content": draft}, "tool_calls": []}

    def test_preflight_has_no_freeze_assets_budget_or_execution_side_effects(self):
        wf, task, draft = self.plan()
        del draft["execution"]["calibration"]
        before = wf.journal.sequence
        with patch.object(wf.broker, "freeze", side_effect=AssertionError("preflight froze")), \
                patch("tools.experiments.asset", side_effect=AssertionError("preflight read assets")), \
                patch("tools.experiments.runtime_identity", side_effect=AssertionError("preflight bound runtime")), \
                patch.object(wf.budget, "check", side_effect=AssertionError("preflight consumed budget")), \
                patch.object(wf.runtime, "launch", side_effect=AssertionError("preflight launched")):
            normalized = wf.broker.preflight(draft)
            self.assertEqual(wf.journal.sequence, before)
            self.deliver(wf, task, self.submit_response(draft))
        finished = wf.queue.get(task["task_id"])
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(finished["result"]["draft"], normalized)
        self.assertEqual(wf.journal.find_cards("ExperimentCard"), [])

    def test_original_plan_repairs_before_completion_then_freezes_once(self):
        wf, task, draft = self.plan()
        draft["execution"]["calibration"]["count"] = 1
        rejected = self.deliver(wf, task, self.submit_response(draft))
        retry = wf.queue.get(task["task_id"])
        self.assertEqual((retry["state"], retry["repair_count"]), ("retry", 1))
        self.assertFalse(any(t["kind"] == "experiment_request" for t in wf.tasks().values()))
        # 持久化重建后仍回到同一任务，原草案完整进入无状态调用上下文。
        wf = Workflow(wf.run_dir)
        self.assertEqual(wf.context(wf.queue.get(task["task_id"]))["repair_feedback"]["rejected_response"], rejected)
        wf.queue.ready()
        task = wf.queue.claim(task["task_id"])
        repaired = deepcopy(rejected["decision"]["content"])
        repaired["execution"]["calibration"]["count"] = 2
        with patch.object(wf.broker, "freeze", wraps=wf.broker.freeze) as freeze, \
                patch.object(wf.runtime, "launch", side_effect=AssertionError("unexpected worker")):
            self.deliver(wf, task, self.submit_response(repaired))
            freeze.assert_not_called()
            finished = wf.queue.get(task["task_id"])
            self.assertEqual(finished["state"], "completed")
            tid = wf.queue.enqueue("r001/L3", "experiment_request", role="supervisor", inputs={
                "draft": finished["result"]["draft"], "requester": "judger", "refs": task["inputs"]["refs"]})
            wf.queue.ready()
            trial = wf.queue.claim(tid)
            wf.progress_request(trial)
            freeze.assert_not_called()  # Only immutable request is saved before Coding.
        self.assertEqual(len(wf.journal.find_cards("ExperimentCard")), 1)
        self.assertEqual(wf.queue.get(tid)["state"], "waiting")
        self.assertEqual(wf.queue.get(task["task_id"])["attempt"], 2)

    def test_repair_limit_and_clarification_stay_in_original_task(self):
        wf, task, draft = self.plan()
        draft["execution"]["calibration"]["count"] = 1
        for attempt in range(3):
            self.deliver(wf, task, self.submit_response(draft))
            state = wf.queue.get(task["task_id"])
            self.assertEqual(state["state"], "retry" if attempt < 2 else "failed")
            if attempt < 2:
                wf.queue.ready()
                task = wf.queue.claim(task["task_id"])
        self.assertEqual(state["repair_count"], 2)
        self.assertEqual(wf.journal.find_cards("ExperimentCard"), [])
        self.assertEqual(len(wf.tasks()), 1)

    def test_ambiguous_draft_can_clarify_without_completion(self):
        wf, task, draft = self.plan()
        draft["execution"].update(service="custom_experiment", interface="experiment_v1")
        draft["execution"]["calibration"]["count"] = 1
        self.deliver(wf, task, self.submit_response(draft))
        self.assertIn("calibration", wf.queue.get(task["task_id"])["repair_feedback"]["error"])
        wf.queue.ready()
        task = wf.queue.claim(task["task_id"])
        self.deliver(wf, task, {"decision": {"action": "clarify", "content": {
            "question": "校准使用哪个 split？", "reason": "自定义接口未固定数据视图"}}, "tool_calls": []})
        self.assertEqual(wf.queue.get(task["task_id"])["state"], "needs_clarification")
        self.assertEqual(wf.journal.find_cards("ExperimentCard"), [])

    def test_tool_batch_preflights_before_enqueue_and_returns_original_batch(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role("fixture/generate", "generate", refs=[wf.brief["id"]], context={})
        wf.queue.ready()
        task = wf.queue.claim(tid)
        valid, bad = experiment_payload(), experiment_payload()
        del valid["execution"]["calibration"]
        bad["execution"]["calibration"]["count"] = 1
        calls = [{"id": name, "name": "request_experiment", "arguments": {"experiment": draft}}
                 for name, draft in (("first", valid), ("second", bad))]
        original = deepcopy(calls)
        with patch.object(wf.broker, "freeze", side_effect=AssertionError("tool submission froze")):
            rejected = self.deliver(wf, task, {"tool_calls": calls})
            self.assertEqual(len(wf.tasks()), 1)
            self.assertEqual(wf.queue.get(tid)["repair_feedback"]["rejected_response"], rejected)
            self.assertEqual(calls, original)
            wf.queue.ready()
            task = wf.queue.claim(tid)
            fixed = deepcopy(calls)
            fixed[1]["arguments"]["experiment"]["execution"]["calibration"]["count"] = 2
            self.deliver(wf, task, {"tool_calls": fixed})
        parent = wf.queue.get(tid)
        self.assertEqual(parent["state"], "waiting")
        children = [wf.queue.get(t) for t in parent["waiting_for"]]
        self.assertEqual(len(children), 2)
        for child in children:
            self.assertEqual(child["inputs"]["parent_task"], tid)
            self.assertEqual(child["inputs"]["draft"]["execution"]["calibration"], {"split": "calibration", "count": 2})
        self.assertEqual(parent["conversation"][-1]["content"]["tool_calls"], fixed)

    def test_custom_request_structure_error_returns_to_researcher_before_immutable_request(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role('fixture/custom-request', 'generate', refs=[wf.brief['id']], context={})
        wf.queue.ready(); task = wf.queue.claim(tid)
        draft = {'question': 'Local weight error', 'measurements': ['weight error'],
            'execution': {'service': 'custom_experiment', 'quantization_contract': {
                'weight': {'bits': 3}, 'methods': {'weight': 'new_source_method'}}}}
        calls = [{'id': 'local', 'name': 'request_experiment', 'arguments': {'experiment': draft}}]
        self.deliver(wf, task, {'tool_calls': calls})
        retry = wf.queue.get(tid)
        self.assertEqual(retry['state'], 'retry')
        self.assertIn("['weight']", retry['repair_feedback']['error'])
        self.assertEqual(retry['repair_feedback']['rejected_response']['tool_calls'], calls)
        self.assertEqual(wf.journal.find_cards('ExperimentCard'), [])
        self.assertEqual(len(wf.tasks()), 1)
        repaired = deepcopy(calls)
        contract = repaired[0]['arguments']['experiment']['execution']['quantization_contract']
        contract['specs'] = {'weight': contract.pop('weight')}
        wf.queue.ready(); task = wf.queue.claim(tid)
        self.deliver(wf, task, {'tool_calls': repaired})
        parent = wf.queue.get(tid)
        self.assertEqual(parent['state'], 'waiting')
        child = wf.queue.get(parent['waiting_for'][0])
        self.assertEqual(child['inputs']['parent_task'], tid)
        self.assertEqual(child['inputs']['original_request'], repaired[0]['arguments']['experiment'])
        self.assertEqual(wf.journal.find_cards('ExperimentCard'), [])

    def test_freeze_and_resolver_reject_same_conflict_before_asset_binding(self):
        wf, _, draft = self.plan()
        draft["execution"]["calibration"]["count"] = 1
        errors = []
        with patch("tools.experiments.asset", side_effect=AssertionError("invalid draft read assets")):
            for action in (lambda: wf.broker.preflight(draft),
                           lambda: wf.broker.freeze(draft, requester="judger"),
                           lambda: resolve_execution(draft["execution"], wf.brief["payload"])):
                with self.assertRaises(ContractError) as error:
                    action()
                errors.append(str(error.exception))
        self.assertTrue(all("calibration" in error and "冲突" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
