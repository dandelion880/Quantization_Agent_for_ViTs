"""Actual CPU computation; fixture data and scripted roles are not research evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.experiment_requests import ExperimentRequests
from core.journal import read_json
from runtime.guards import install_gradient_guard, GRADIENT_AUDIT
from tests.test_infrastructure import TempCase
from tests import test_single_implementation as single
from tests.test_execution_contracts import plan_fixture
from tools.experiment_worker import backend_and_data, evaluate_services
from tools.experiments import preflight_experiment
from tests.support import validation_config


class ReadonlyGradientTests(unittest.TestCase):
    def test_autograd_and_backward_are_readonly_diagnostics_but_optimizer_updates_fail(self):
        import torch
        x = torch.tensor([2., 3.], requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=.1)
        guard = install_gradient_guard(True)
        try:
            self.assertTrue(torch.equal(torch.autograd.grad(x.square().sum(), x)[0], 2 * x))
            x.square().sum().backward()
            self.assertTrue(torch.equal(x.grad, 2 * x))
            before = x.detach().clone()
            with self.assertRaisesRegex(ValueError, '优化器更新'):
                optimizer.step()
            self.assertTrue(torch.equal(x, before))
            self.assertEqual(GRADIENT_AUDIT, {'optimizer_steps': 0, 'autograd_grad_calls': 1, 'backward_calls': 1})
        finally:
            guard.remove()

    def test_implementation_origin_controls_local_repair(self):
        for code in ('NameError', 'UnboundLocalError', 'IndexError', 'ZeroDivisionError', 'NotImplementedError'):
            with self.subTest(code=code):
                error = {'code': code, 'message': 'implementation failed', 'origin': 'implementation'}
                self.assertTrue(ExperimentRequests.revisable_failure({'error': error}))
                error['origin'] = 'host'
                self.assertFalse(ExperimentRequests.revisable_failure({'error': error}))


class CustomWorkerTests(TempCase):
    begin = single.SingleImplementation.begin
    progress = single.SingleImplementation.progress
    approve = single.SingleImplementation.approve

    def custom_start(self):
        wf = self.workflow(1)
        author = wf.enqueue_role('r001/generate/H1', 'generate', refs=[wf.brief['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        researcher, _ = self.begin(wf, wf.queue.get(author))
        draft = {'question': 'Inspect local FP weight gradient without reading images',
                 'measurements': ['gradient_squared_norm'],
                 'execution': {'service': 'custom_experiment', 'requires_code': True}}
        wf.receive_tools(researcher, {'tool_calls': [{'id': 'experiment', 'name': 'request_experiment',
            'arguments': {'experiment': draft}}]}, 'SCRIPTED')
        wf.queue.ready()
        request = wf.queue.claim(wf.queue.get(author)['waiting_for'][0])
        wf.progress_request(request)
        return wf, researcher, wf.queue.get(request['task_id'])

    def custom_coding(self, wf, request, source):
        suffix = '/revision' + str(request['code_revision']) if request.get('code_revision') else ''
        task, _ = self.begin(wf, wf.tasks()[request['key'] + '/code' + suffix])
        wf.submit(task, {'source': source, 'execution_choices': {
            'parameters': {'alpha': .73}, 'method': 'weight_gradient_diagnostic_v1',
            'resolved_choices': {'training': False}}}, 'SCRIPTED')
        return self.progress(wf, request)

    def execute(self, wf, request):
        suffix = '/revision' + str(request['code_revision']) if request.get('code_revision') else ''
        tid = wf.tasks()[request['key'] + '/execute' + suffix]['task_id']
        wf.dispatch()
        deadline = time.monotonic() + 40
        while wf.queue.get(tid)['state'] == 'running' and time.monotonic() < deadline:
            time.sleep(.1)
            wf.poll()
        result = wf.queue.get(tid)
        self.assertNotEqual(result['state'], 'running')
        return result

    def test_no_image_custom_statistic_recovers_from_name_error_in_same_request(self):
        source = '''import torch
def execute(payload, context):
    weight = context['reference_model'].get_submodule('blocks.0.mlp.fc2').weight.detach().clone().requires_grad_(True)
    loss = context['parameters']['alpha'] * weight.square().sum()
    gradient = torch.autograd.grad(loss, weight)[0]
    return {'observations': {'gradient_squared_norm': float(gradient.square().sum())}}
'''
        wf, researcher, request = self.custom_start()
        try:
            request = self.approve(wf, self.custom_coding(wf, request, source.replace('float(gradient.square().sum())', 'missing_statistic')))
            failed = self.execute(wf, request)
            self.assertEqual(failed['state'], 'failed', failed.get('error'))
            self.assertEqual(failed['error']['code'], 'NameError')
            self.assertEqual(failed['error']['origin'], 'implementation')
            original = request['request_ref']
            request = self.progress(wf, request)
            self.assertEqual(request['code_revision'], 1)
            request = self.progress(wf, request)
            request = self.approve(wf, self.custom_coding(wf, request, source))
            self.assertEqual(request['request_ref'], original)
            completed = self.execute(wf, request)
            self.assertEqual(completed['state'], 'completed', completed.get('error'))
            evidence = wf.journal.card(completed['outputs'][0])
            value = evidence['payload']
            self.assertEqual(value['measurement_source'], 'synthetic_computation')
            self.assertGreater(value['observations']['gradient_squared_norm'], 0)
            facts = value['provenance']['execution_facts']
            self.assertEqual(facts['data'], {})
            self.assertEqual(facts['forward_calls'], 0)
            self.assertEqual(facts['parameters'], {'alpha': .73})
            self.assertTrue(value['provenance']['model_unchanged'])
            self.assertEqual(value['provenance']['gradient_audit']['autograd_grad_calls'], 1)
            request = self.progress(wf, request)
            self.assertEqual(request['result']['evidence_ref'], evidence['id'])
            resumed, _ = self.begin(wf, wf.queue.get(researcher['task_id']))
            self.assertIn(evidence['id'], wf.required_refs(resumed))
        finally:
            wf.runtime.stop()


class CustomExecutorTests(TempCase):
    def test_native_weight_only_and_fp_control_do_not_read_calibration_images(self):
        for weight_only in (False, True):
            plan = plan_fixture()
            plan['layers'] = plan['layers'][:1] if weight_only else []
            if weight_only:
                plan['layers'][0].update(activation=None, activation_method=None)
                plan['layers'][0].pop('activation_calibration')
            draft = {'level': 'L2', 'question': 'Weight-only or FP control', 'measurements': ['output_error'],
                'execution': {'service': 'quantize_evaluate', 'requires_code': False, 'plan': plan,
                    'evaluations': [{'kind': 'classification', 'split': 'search', 'count': 1}],
                    'resolved_choices': {'training': False}}}
            payload = preflight_experiment(draft, validation_config()['research_brief'], executable=True)
            self.assertNotIn('calibration', payload['execution'])
            backend, cfg, data = backend_and_data(payload['execution'])
            original = data.batches
            def no_calibration(split):
                self.assertNotEqual(split, 'calibration')
                yield from original(split)
            directory = self.root / str(weight_only); directory.mkdir()
            with patch.object(data, 'batches', side_effect=no_calibration):
                result = evaluate_services({'payload': payload}, backend, cfg, data, directory, None)
            self.assertNotIn('calibration', result['execution_facts']['data'])
            self.assertEqual(read_json(directory / 'model/manifest.json')['calibration_ids'], [])
            self.assertTrue(result['observations']['reload']['host_verified'])

    def test_custom_method_observer_and_local_intervention_survive_actual_reload_and_global_evaluation(self):
        import torch
        plan = plan_fixture()
        plan['layers'] = plan['layers'][:1]
        row = plan['layers'][0]
        row.update(weight_method='new_weight_rule', activation=None,
                   weight_calibration={'strategy': 'new_weight_observer', 'alpha': .73})
        row['weight'].update(bits=3, format='INT3')
        draft = {'level': 'L2', 'question': 'Local weight intervention, observe whole-model output',
            'measurements': ['output_error'], 'execution': {'service': 'quantize_evaluate', 'requires_code': True,
                'parameters': {'alpha': .73}, 'evaluations': [{'kind': 'classification', 'split': 'search', 'count': 1}],
                'resolved_choices': {'training': False}}}
        payload = preflight_experiment(draft, validation_config()['research_brief'], executable=True)
        self.assertNotIn('calibration', payload['execution'])
        backend, cfg, data = backend_and_data(payload['execution'])
        class CustomExecutor:
            def __init__(self, backend, cfg, data):
                self.backend = backend
            def prepare(self, plan, shapes):
                candidate = deepcopy(self.backend.model)
                with torch.no_grad():
                    weight = candidate.get_submodule(row['module']).weight
                    scale = weight.abs().max() / 3
                    weight.copy_((weight / scale).round().clamp(-3, 3) * scale)
                return candidate, {}, {}, {}
            def export(self, model, plan, weights, activations, directory, audit):
                directory.mkdir()
                torch.save(model.state_dict(), directory / 'custom_state.pt')
            def load(self, directory):
                loaded = deepcopy(self.backend.model)
                loaded.load_state_dict(torch.load(directory / 'custom_state.pt', weights_only=True))
                return loaded, {}
        plugin = SimpleNamespace(build_plan=lambda *args: deepcopy(plan), make_executor=CustomExecutor)
        directory = self.root / 'custom'; directory.mkdir()
        with patch('tools.experiment_worker.module_from_approval', return_value=plugin):
            result = evaluate_services({'payload': payload}, backend, cfg, data, directory, None)
        self.assertTrue(result['observations']['reload']['host_verified'])
        self.assertEqual(result['observations']['reload']['loader'], 'approved_custom_executor')
        self.assertNotIn('calibration', result['execution_facts']['data'])
        self.assertEqual(result['execution_facts']['data']['classification']['sample_count'], 1)
        evaluated = read_json(directory / 'classification.json')
        self.assertGreater(evaluated['coverage'][row['module']], 0)
        self.assertGreater(evaluated['metrics']['relative_mse'], 0)
        self.assertTrue(evaluated['health']['original_state_unchanged'])
        opaque = {'algorithm': 'custom_tensor_transform', 'layers': {'private_algorithm_field': 1}, 'alpha': .73}
        plugin.build_plan = lambda *args: deepcopy(opaque)
        directory = self.root / 'opaque'; directory.mkdir()
        with patch('tools.experiment_worker.module_from_approval', return_value=plugin):
            result = evaluate_services({'payload': payload}, backend, cfg, data, directory, None)
        self.assertTrue(result['observations']['reload']['host_verified'])
        self.assertEqual(result['execution_facts']['quantization'], {})
        self.assertEqual(read_json(directory / 'classification.json')['coverage'], {})
        # An explicit data-use promise cannot be satisfied by no reads.
        payload['execution']['calibration'] = {'split': 'calibration', 'count': 2}
        directory = self.root / 'unused_calibration'; directory.mkdir()
        with patch('tools.experiment_worker.module_from_approval', return_value=plugin):
            with self.assertRaisesRegex(ContractError, '实际校准样本'):
                evaluate_services({'payload': payload}, backend, cfg, data, directory, None)
        # Returned-value failures happen after the implementation frame exits;
        # the actual failing host contract must still route to this implementation.
        from runtime.task_worker import failure_origin
        request = {'approval': {'candidate': {'path': str(self.root / 'approved_source.py')}}}
        original_prepare = CustomExecutor.prepare
        prepared = []
        def prepare(executor, *args):
            result = original_prepare(executor, *args)
            prepared.append(result[0])
            return result
        payload['execution'].pop('calibration')
        directory = self.root / 'invalid_reload'; directory.mkdir()
        with patch('tools.experiment_worker.module_from_approval', return_value=plugin), \
                patch.object(CustomExecutor, 'prepare', prepare), \
                patch.object(CustomExecutor, 'load', side_effect=lambda *args: (prepared[-1], {})):
            try:
                evaluate_services({'payload': payload}, backend, cfg, data, directory, None)
                self.fail('Returning the same model is not a reload')
            except ContractError as exc:
                self.assertIn('实际重建实例', str(exc))
                self.assertEqual(failure_origin(exc, request), 'implementation')
        with patch('tools.experiment_worker.module_from_approval', side_effect=OSError('host filesystem unavailable')):
            try:
                evaluate_services({'payload': payload}, backend, cfg, data, directory, None)
                self.fail('Host failure must propagate')
            except OSError as exc:
                self.assertEqual(failure_origin(exc, request), 'host')


if __name__ == '__main__':
    unittest.main()
