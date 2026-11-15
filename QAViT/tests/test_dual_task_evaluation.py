"""Dual-task contracts and real child lifecycle; GPU smoke is separately recorded."""
from pathlib import Path
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import atomic_json, file_sha256, read_json
from runtime.gpu_resources import placement
from runtime.task_queue import process_alive
from tools.result_contract import dual_task_fallback
from tools.experiment_worker import evaluation_summary
from tools.dual_task_evaluation import check_identity, evaluate_parallel
from runtime.failures import RemoteWorkerError, failure_details
from tests.test_adaptive_resources import classification_payload
from tests.test_infrastructure import TempCase


def payload():
    p = classification_payload()
    p['resources'].pop('gpu_count')
    p['execution']['evaluations'].append({'kind': 'ade_frozen', 'count': 1})
    return p


class DualTaskTests(TempCase):
    def test_capability_dynamic_default_and_explicit_serial(self):
        p = payload()
        self.assertIsNone(dual_task_fallback(p))
        choice = placement({'payload': p}, {'allowed_gpus': [0, 1, 2, 3]})
        self.assertEqual(choice['gpu_count'], 2)
        self.assertTrue(choice['parallelism']['dual_task_supported'])
        self.assertFalse(choice['parallelism']['classification_shards_supported'])
        p['resources']['gpu_count'] = 1
        self.assertEqual(placement({'payload': p}, {'allowed_gpus': [0, 1]})['gpu_count'], 1)
        p['resources']['gpu_count'] = 8
        self.assertEqual(placement({'payload': p}, {'allowed_gpus': [0, 1, 2]})['gpu_count'], 2)
        p['execution']['requires_code'] = True
        self.assertIsNone(dual_task_fallback(p))  # existing make_executor.load contract
        p['execution']['resolved_choices']['training'] = True
        self.assertIsNotNone(dual_task_fallback(p))

    def test_custom_resource_request_requires_hashed_literal_capability(self):
        p = payload()
        p['execution']['requires_code'] = True
        p['resources']['gpu_count'] = 8
        self.assertEqual(placement({'payload': p}, {'allowed_gpus': [0, 1]})['gpu_count'], 1)
        path = self.root / 'executor.py'
        for text, expected in [('parallel_frozen_load = True', 2), ('parallel_frozen_load = False', 1),
                               ('parallel_frozen_load = bool(1)', 1), ('x = True', 1),
                               ('parallel_frozen_load = True\nparallel_frozen_load = False', 1)]:
            path.write_text(text + "\nraise AssertionError('must not import during placement')\n")
            approval = {'status': 'approved', 'candidate': {'path': str(path), 'source_sha256': file_sha256(path)}}
            result = placement({'payload': p}, {'allowed_gpus': [0, 1]}, approval=approval)
            self.assertEqual(result['gpu_count'], expected)
        path.write_text('parallel_frozen_load = True')
        with self.assertRaises(ContractError):
            placement({'payload': p}, {'allowed_gpus': [0, 1]}, approval=approval)

    def test_pair_preservation_and_missing_baseline_not_fabricated(self):
        raw = {'metrics': {'fp': {'mIoU': 40., 'mAcc': 60., 'aAcc': 80.},
                           'quantized': {'mIoU': 4., 'mAcc': 7., 'aAcc': 34.}},
               'images': 2, 'sample_ids': ['a', 'b'], 'full_dataset_evaluation': False}
        _, row = evaluation_summary({'kind': 'ade_frozen'}, raw)
        self.assertEqual(row['metrics']['delta_mIoU_pp'], -36.)
        self.assertEqual(row['metric_units']['baseline_mIoU'], 'percent')
        self.assertEqual(row['metric_units']['delta_mIoU_pp'], 'percentage_points')
        del raw['metrics']['fp']
        _, row = evaluation_summary({'kind': 'ade_frozen'}, raw)
        self.assertNotIn('baseline_mIoU', row['metrics'])
        raw = {'metrics': {'top1': 20., 'baseline_top1': 80., 'top5': 50., 'baseline_top5': 95.},
               'sample_count': 1, 'sample_ids': ['a'], 'split': 'final', 'full_dataset_evaluation': False}
        _, row = evaluation_summary({'kind': 'classification'}, raw)
        self.assertEqual(row['metrics']['delta_top5_pp'], -45.)

    def test_all_frozen_identities_and_request_are_required(self):
        request = {k: k for k in ('model_digest', 'baseline_digest', 'plan_sha256', 'model_assets')}
        path = self.root / 'request.json'
        atomic_json(path, request)
        good = {**request, 'request_sha256': file_sha256(path), 'status': 'completed'}
        check_identity(good, request, path)
        for key in good:
            with self.subTest(key=key), self.assertRaises(ContractError):
                check_identity({**good, key: 'different'}, request, path)

    def test_custom_loader_reconstructs_real_tensors_without_prepare(self):
        from copy import deepcopy
        from core.cards import fingerprint
        from tools.runtime_interface import interface_fixture
        from tools.experiment_worker import plan_constraints
        from tools.dual_task_evaluation import run, package_files
        from tools.safeguards import write_artifact, approval_receipt
        from ptq_agent.execution.baselines import build_baseline
        from ptq_agent.execution.model import RTNExecutor
        from ptq_agent.execution.evaluation import model_digest
        spec = {'backend': {'config_path': 'QAViT/configs/deitb_backend.json'}, 'seed': 42,
                'resolved_choices': {'training': False}, 'interface': 'executor_v1'}
        fixture = {'shape': [1, 4], 'sites': ['fc'], 'seed': 3,
                   'counts': {'calibration': 2, 'search': 1, 'final': 1}}
        backend, cfg, data = interface_fixture(spec, fixture)
        backend.identity = lambda: {'fixture': 'seed3', 'state': backend.original_state}
        plan = build_baseline(backend, 'repq')
        executor = RTNExecutor(backend, cfg, data)
        model, weights, acts, audit = executor.prepare(plan, {})
        folder = self.root / 'custom_model'
        executor.export(model, plan, weights, acts, folder, audit)
        experiment = {'id': 'custom-test@1', 'payload': {'execution': spec, 'resources': {'cpu_threads': 1}}}
        experiment['content_sha256'] = fingerprint(experiment)
        source = """from ptq_agent.execution.model import RTNExecutor
parallel_frozen_load = True
class FrozenLoader:
    def __init__(self, backend, cfg, data):
        self.inner = RTNExecutor(backend, cfg, data)
    def load(self, directory):
        return self.inner.load(directory)
    def prepare(self, *args):
        raise AssertionError('independent load must never prepare')
def make_executor(backend, cfg, data):
    return FrozenLoader(backend, cfg, data)
def build_plan(*args):
    raise AssertionError('independent load must not build a new plan')
"""
        candidate = write_artifact(self.root, experiment, {'source': source}, 'coding')
        reviewed = {'experiment_ref': experiment['id'], 'experiment_sha256': experiment['content_sha256'],
                    'request_ref': experiment['id'], 'request_sha256': experiment['content_sha256'],
                    'source_sha256': candidate['source_sha256']}
        review = {'id': 'scripted-engineering-review', 'content_sha256': 'fixture',
                  'payload': {'approved': True, 'phase': 'safeguard', 'reviewed_binding': reviewed}}
        approval = approval_receipt(experiment, candidate, review)
        request = {'experiment': experiment, 'approval': approval, 'native': False, 'model_dir': str(folder),
                   'model_assets': package_files(folder), 'model_digest': model_digest(model),
                   'baseline_digest': model_digest(backend.model), 'plan': plan, 'plan_sha256': fingerprint(plan),
                   'requested': {'kind': 'classification', 'split': 'search', 'count': 1}}
        path = self.root / 'custom_request.json'
        atomic_json(path, request)
        # Only the tiny backend/data provider is substituted. Approved source
        # import, strict package load, tensors and evaluator execute for real.
        with patch('tools.experiment_worker.backend_and_data', return_value=(backend, cfg, data)):
            receipt = run(request, path)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['model_assets'], package_files(folder))
        self.assertEqual(receipt['evaluation']['sample_count'], 1)
        attempted_fit = source.replace('self.inner = RTNExecutor(backend, cfg, data)',
                                       'self.inner = RTNExecutor(backend, cfg, data); self.data = data').replace(
            'return self.inner.load(directory)', "list(self.data.batches('calibration')); return self.inner.load(directory)")
        candidate = write_artifact(self.root, experiment, {'source': attempted_fit}, 'coding')
        review['payload']['reviewed_binding']['source_sha256'] = candidate['source_sha256']
        request['approval'] = approval_receipt(experiment, candidate, review)
        atomic_json(path, request)
        with patch('tools.experiment_worker.backend_and_data', return_value=(backend, cfg, data)), self.assertRaises(ContractError):
            run(request, path)

    def exercise_child_failure(self, *, deadline=False, parent_failure=False, reported_error=None, actual_request=False):
        model_dir = self.root / 'model'
        model_dir.mkdir()
        (model_dir / 'weights.pt').write_bytes(b'immutable')
        effective_deadline = time.time() - 1 if deadline else time.time() + 60
        atomic_json(self.root / 'request.json', {'deadline_unix': time.time() - 1 if actual_request else effective_deadline})
        request_path = self.root / ('request-resumed.json' if actual_request else 'request.json')
        if actual_request:
            atomic_json(request_path, {'deadline_unix': effective_deadline})
        popen = subprocess.Popen
        children = []
        def launch(command, **kwargs):
            # Actual OS process/identity/group cleanup; no numerical claims.
            script = """import json, hashlib, time, sys
from pathlib import Path
p = Path(sys.argv[1]); r = json.loads(p.read_text())
ready = {k:r[k] for k in ('model_digest','baseline_digest','plan_sha256','model_assets')}
ready.update(status='completed',request_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
tmp = p.parent/'ready.tmp'; tmp.write_text(json.dumps(ready)); tmp.rename(p.parent/'ready.json')
while not (p.parent/'start.json').exists(): time.sleep(.02)
"""
            if reported_error:
                script += "\nready.update(status='failed',error=" + repr(reported_error) + ")\n"
                script += "tmp=p.parent/'result.tmp'; tmp.write_text(json.dumps(ready)); tmp.rename(p.parent/'result.json')\nraise SystemExit(1)"
            else:
                script += 'time.sleep(30)' if deadline or parent_failure else 'raise SystemExit(7)'
            child = popen([sys.executable, '-B', '-c', script, command[-1]], **kwargs)
            children.append(child)
            return child
        raw = {'metrics': {'top1': 20., 'baseline_top1': 80.}, 'sample_count': 1,
               'sample_ids': ['a'], 'split': 'search', 'full_dataset_evaluation': False}
        with patch('tools.dual_task_evaluation.subprocess.Popen', side_effect=launch), \
             patch.dict(os.environ, {'QAVIT_TASK_REQUEST_PATH': str(request_path)}), \
             patch('ptq_agent.execution.evaluation.model_digest', return_value='model'), \
             patch('tools.experiment_worker.evaluate_task', side_effect=RuntimeError('parent failed') if parent_failure else None,
                   return_value=raw):
            with self.assertRaises((ContractError, TimeoutError, RuntimeError)) as caught:
                evaluate_parallel({'id': 'x', 'payload': payload()}, SimpleNamespace(model=None), {}, None,
                                  None, {}, {}, True, model_dir, self.root, [0, 1], None,
                                  deadline_unix=read_json(request_path)['deadline_unix'])
        error = failure_details(caught.exception, {})
        if reported_error:
            self.assertEqual(error['code'], reported_error['code'])
            self.assertEqual(error['origin'], reported_error['origin'])
            self.assertEqual(error['subprocess']['worker'], 'ade_frozen')
            self.assertTrue(Path(error['subprocess']['worker_log']).is_file())
            self.assertEqual(read_json(self.root / 'dual_task_failure.json')['error'], error)
            self.assertNotIn('dual_tasks/segmentation/result.json',
                             read_json(self.root / 'dual_task_failure.json')['partial_results'])
        elif not deadline and not parent_failure:
            self.assertEqual(error['code'], 'ChildProcessError')
            self.assertEqual(error['origin'], 'host')
            self.assertEqual(error['returncode'], 7)
        self.assertEqual(len(children), 1)
        self.assertEqual(read_json(self.root / 'dual_tasks/segmentation/request.json')['deadline_unix'], effective_deadline)
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(process_alive(read_json(self.root / 'processes/dual-segmentation.json')))
        self.assertEqual(read_json(self.root / 'dual_task_failure.json')['status'], 'failed')
        if not parent_failure and not deadline:
            self.assertTrue((self.root / 'classification.json').is_file())

    def test_child_failure_preserves_completed_sibling(self):
        self.exercise_child_failure()

    def test_timeout_cleans_actual_child(self):
        self.exercise_child_failure(deadline=True)

    def test_parent_failure_cleans_actual_child(self):
        self.exercise_child_failure(parent_failure=True)

    def test_reported_implementation_error_survives_actual_child_boundary(self):
        self.exercise_child_failure(reported_error={'code': 'TypeError', 'origin': 'implementation',
                                                    'message': 'custom loader argument mismatch'})

    def test_reported_timeout_stays_host_timeout(self):
        self.exercise_child_failure(reported_error={'code': 'TimeoutError', 'origin': 'host',
                                                    'message': 'dual-task deadline exceeded'})

    def test_resumed_actual_request_supplies_child_deadline(self):
        self.exercise_child_failure(actual_request=True)

    def test_invalid_identity_cannot_relay_claimed_implementation_failure(self):
        request = {k: k for k in ('model_digest', 'baseline_digest', 'plan_sha256', 'model_assets')}
        path = self.root / 'request.json'
        atomic_json(path, request)
        failed = {**request, 'model_digest': 'different', 'request_sha256': file_sha256(path),
                  'status': 'failed', 'error': {'code': 'TypeError', 'message': 'bad plugin', 'origin': 'implementation'}}
        try:
            check_identity(failed, request, path)
            self.fail('identity mismatch must reject the claimed remote provenance')
        except ContractError as exc:
            self.assertNotIsInstance(exc, RemoteWorkerError)
            self.assertEqual(failure_details(exc, {})['origin'], 'host')

    def test_child_main_captures_actual_plugin_frames_and_original_type(self):
        from tools import dual_task_evaluation
        source = self.root / 'approved.py'
        source.write_text("def load():\n    raise TypeError('actual plugin failure')\n")
        namespace = {}
        exec(compile(source.read_text(), str(source), 'exec'), namespace)
        request = {k: k for k in ('model_digest', 'baseline_digest', 'plan_sha256', 'model_assets')}
        request['approval'] = {'candidate': {'path': str(source)}}
        path = self.root / 'request.json'
        atomic_json(path, request)
        with patch.object(sys, 'argv', ['dual-task', str(path)]), \
             patch.object(dual_task_evaluation, 'run', side_effect=lambda *args: namespace['load']()), \
             patch('tools.dual_task_evaluation.traceback.print_exc'), self.assertRaises(SystemExit):
            dual_task_evaluation.main()
        receipt = read_json(self.root / 'result.json')
        with self.assertRaises(RemoteWorkerError) as caught:
            check_identity(receipt, request, path)
        self.assertEqual(failure_details(caught.exception, request)['code'], 'TypeError')
        self.assertEqual(failure_details(caught.exception, request)['origin'], 'implementation')
