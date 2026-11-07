"""Offline device simulation and real subprocess lifecycle; no science claims."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import read_json
from core.workflow import Workflow
from runtime.gpu_resources import (resolve_runtime_resources, available_inventory,
    gpu_bindings, placement, acquire, ResourceBusy)
from runtime.task_runtime import TaskRuntime
from tests.support import validation_config
from tests.test_infrastructure import TempCase
from tools.result_contract import classification_shard_fallback


def devices(count):
    return [{'index': n, 'uuid': 'GPU-test-' + str(n), 'device_node': '/dev/nvidia' + str(n + 10)}
            for n in range(count)]


def classification_payload():
    return {'level': 'L3', 'resources': {'gpu_count': 6}, 'execution': {
        'service': 'evaluate_artifact', 'resolved_choices': {'training': False},
        'backend': {'kind': 'server', 'config': {'backend': {'device': 'cuda:6'}}},
        'evaluations': [{'kind': 'classification', 'split': 'final', 'count': 17, 'metrics': ['top1', 'relative_mse']}]}}


class AdaptiveDevices(TempCase):
    def test_pool_sizes_and_explicit_limits(self):
        for count in (0, 1, 4, 6):
            rows = devices(count)
            with self.subTest(count=count), patch('runtime.gpu_resources.available_inventory', return_value=rows):
                value = {'allowed_gpus': 'auto', 'experiment_concurrency': 'auto'}
                resolve_runtime_resources(value)
                self.assertEqual(value['allowed_gpus'], list(range(count)))
                self.assertEqual(value['experiment_concurrency'], max(1, count))
                if count > 1:
                    value = {'allowed_gpus': [rows[-1]['uuid'], 0], 'experiment_concurrency': 1}
                    resolve_runtime_resources(value)
                    self.assertEqual(value['allowed_gpus'], [0, count - 1])
                    self.assertEqual(value['experiment_concurrency'], 1)
        with patch('runtime.gpu_resources.available_inventory', side_effect=AssertionError('CPU needs no probe')):
            value = {'allowed_gpus': []}
            resolve_runtime_resources(value)
            self.assertEqual(value, {'allowed_gpus': [], 'experiment_concurrency': 1})

    def test_visibility_allocation_and_authorization_intersect(self):
        rows = devices(6)
        with patch.dict(os.environ, {'SLURM_JOB_GPUS': '1-4', 'SLURM_STEP_GPUS': '2,4'}, clear=True), \
             patch('runtime.gpu_resources.inventory', return_value=rows), \
             patch('runtime.gpu_resources.visible_uuids', return_value=[rows[n]['uuid'] for n in (0, 1, 2, 3)]):
            value = {'allowed_gpus': [0, 2, 4, 5]}
            resolve_runtime_resources(value)
            self.assertEqual(value['allowed_gpus'], [2])
            self.assertEqual(value['experiment_concurrency'], 1)
            self.assertEqual(value['gpu_selection']['requested'], [0, 2, 4, 5])
        with patch.dict(os.environ, {'NVIDIA_VISIBLE_DEVICES': rows[1]['uuid']}, clear=True), \
             patch('runtime.gpu_resources.inventory', return_value=rows), \
             patch('runtime.gpu_resources.visible_uuids', return_value=[r['uuid'] for r in rows]):
            self.assertEqual(available_inventory({}), [rows[1]])
        with patch.dict(os.environ, {'NVIDIA_VISIBLE_DEVICES': '1,3'}, clear=True), \
             patch('runtime.gpu_resources.inventory', return_value=rows), \
             patch('runtime.gpu_resources.visible_uuids', return_value=[r['uuid'] for r in rows]):
            self.assertEqual(available_inventory({}), [rows[1], rows[3]])
        with patch.dict(os.environ, {'NVIDIA_VISIBLE_DEVICES': '7,8'}, clear=True), \
             patch('runtime.gpu_resources.inventory', return_value=rows), \
             patch('runtime.gpu_resources.visible_uuids', return_value=[r['uuid'] for r in rows]):
            with self.assertRaisesRegex(ContractError, '数字无法对应'):
                available_inventory({})

    def test_recovery_binds_uuid_despite_index_and_minor_changes(self):
        old = devices(1)[0]
        current = {**old, 'index': 5, 'device_node': '/dev/nvidia7'}
        with patch('runtime.gpu_resources.available_inventory', return_value=[current]):
            self.assertEqual(gpu_bindings([0], {'gpu_bindings': [old]}), [{**current, 'index': 0}])
        with patch('runtime.gpu_resources.available_inventory', return_value=devices(6)[1:]):
            self.assertEqual(gpu_bindings([0], {'gpu_bindings': [old]}), [])

    def test_explicit_hardware_description_and_uncertain_detection(self):
        rows = devices(4)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': rows[2]['uuid']}, clear=True), \
             patch('runtime.gpu_resources.visible_uuids', side_effect=OSError('driver unavailable')):
            value = {'allowed_gpus': [rows[2]['uuid']], 'gpu_inventory': rows}
            resolve_runtime_resources(value)
            self.assertEqual(value['allowed_gpus'], [2])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}, clear=True), \
             patch('runtime.gpu_resources.visible_uuids', side_effect=OSError('driver unavailable')):
            with self.assertRaisesRegex(ContractError, '无法可靠'):
                resolve_runtime_resources({'allowed_gpus': 'auto', 'gpu_inventory': rows})

    def test_no_auto_rediscovery_or_budget_reset_when_reading_frozen_run(self):
        config = validation_config()
        config['runtime'].update(allowed_gpus='auto', experiment_concurrency='auto')
        with patch('runtime.gpu_resources.available_inventory', return_value=devices(4)):
            wf = Workflow(self.root / 'frozen', config)
        saved = deepcopy(wf.config)
        with patch('runtime.gpu_resources.available_inventory', side_effect=AssertionError('do not rewrite frozen authorization')):
            restored = Workflow(wf.run_dir)
        self.assertEqual(restored.config, saved)
        self.assertEqual(restored.config['runtime']['experiment_concurrency'], 4)
        self.assertEqual(restored.budget.snapshot()['started_unix'], wf.budget.snapshot()['started_unix'])

    def test_uuid_lease_excludes_other_container_ordinal(self):
        uuid = 'GPU-lease-alias-' + self.root.name
        with patch('runtime.gpu_resources.externally_busy', return_value=set()):
            _, fds = acquire([100], bindings=[{'index': 100, 'uuid': uuid}])
            try:
                with self.assertRaises(ResourceBusy):
                    acquire([200], bindings=[{'index': 200, 'uuid': uuid}])
            finally:
                for fd in fds:
                    os.close(fd)
            selected, fds = acquire([200], bindings=[{'index': 200, 'uuid': uuid}])
            self.assertEqual(selected, [200])
            for fd in fds:
                os.close(fd)

    def test_busy_poll_does_not_reinitialize_cuda_discovery(self):
        wf = self.workflow()
        wf.runtime.settings = {'allowed_gpus': [2]}
        tid = wf.queue.enqueue('queued', 'experiment')
        wf.queue.promote()
        body = {'kind': 'test_control', 'gpu_pool': [2], 'gpu_count': 1, 'pinned_gpu': None}
        row = devices(3)[2]
        with patch('runtime.gpu_resources.gpu_bindings', return_value=[row]) as discover, \
             patch('runtime.gpu_resources.externally_busy', side_effect=[{row['uuid']}, {row['uuid']}, set()]), \
             patch.object(wf.runtime, '_launch', return_value='launched'):
            for _ in range(2):
                with self.assertRaises(ResourceBusy):
                    wf.runtime.launch(wf.queue.get(tid), body, input_versions={})
            self.assertEqual(wf.queue.get(tid)['attempt'], 0)
            self.assertEqual(wf.budget.snapshot()['reservations'], {})
            self.assertEqual(wf.runtime.launch(wf.queue.get(tid), body, input_versions={}), 'launched')
        discover.assert_called_once()

    def test_ade_setup_accepts_uuid_and_cpu_without_resetting_thread_limit(self):
        import torch
        from tools.vision_models import setup_segmentation
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for visible in ('GPU-a-real-style-uuid', '', '6'):
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}):
                    setup_segmentation()
                    self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(threads)


class CapabilityFallback(TempCase):
    def test_required_metrics_keep_original_protocol_and_choose_serial(self):
        for metric in ('latency_ms', 'baseline_latency_ms', 'measured_speedup', 'cuda_peak_memory_mib', 'cuda_increment_mib'):
            payload = classification_payload()
            payload['execution']['evaluations'][0]['metrics'].append(metric)
            before = deepcopy(payload)
            choice = placement({'payload': payload}, {'allowed_gpus': list(range(6))})
            self.assertEqual(choice['gpu_count'], 1)
            self.assertIn(metric, choice['parallelism']['fallback'])
            self.assertEqual(payload, before)
        payload = classification_payload()
        choice = placement({'payload': payload}, {'allowed_gpus': list(range(4))})
        self.assertEqual(choice['gpu_count'], 6)  # request ceiling, acquisition uses available subset
        self.assertEqual(choice['gpu_pool'], [0, 1, 2, 3])
        self.assertTrue(choice['parallelism']['classification_shards_supported'])

    def test_frozen_l3_required_results_and_composite_schemas(self):
        for path in ('evaluations.classification.metrics.latency_ms', 'evaluations.classification.metrics',
                     'evaluations.classification', 'evaluations'):
            payload = classification_payload()
            payload['execution']['result_protocol'] = {'required_results': [{'path': path, 'schema': {'type': 'number'}}]}
            self.assertIsNotNone(classification_shard_fallback(payload, {}))
        payload['execution']['result_protocol']['required_results'][0]['path'] = 'evaluations.classification.metrics.top1'
        self.assertIsNone(classification_shard_fallback(payload, {}))

    def test_other_backends_training_code_and_mixed_evaluations_fallback(self):
        for kind in ('factory', 'fixture'):
            payload = classification_payload()
            payload['execution']['backend']['kind'] = kind
            self.assertIsNotNone(classification_shard_fallback(payload, {}))
        for field, value in [('requires_code', True), ('resolved_choices', {'training': True}),
                             ('service', 'custom_experiment'), ('evaluations', [{'kind': 'ade_frozen', 'count': 1}])]:
            payload = classification_payload()
            payload['execution'][field] = value
            self.assertIsNotNone(classification_shard_fallback(payload, {}))


class DurableResultCleanup(TempCase):
    def exercise_cleanup(self, delay, grace, *, descendant=False, restore=False):
        config = validation_config()
        config['budget'] = {'wall_seconds': 20, 'clock_mode': 'active_segments', 'exclude_numeric_wait': True}
        config['runtime']['worker_exit_grace_seconds'] = grace
        wf = Workflow(self.root / 'cleanup', config)
        wf.budget.begin_segment()
        tid = wf.queue.enqueue('numeric', 'experiment')
        wf.queue.promote()
        popen = subprocess.Popen
        script = '''
import atexit, runpy, sys, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parents[1]))
from core.journal import atomic_json
from runtime.task_queue import process_identity
if sys.argv[4] == 'yes':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    atomic_json(Path(sys.argv[2]).parent / 'processes/child.json', process_identity(child.pid))
atexit.register(time.sleep, float(sys.argv[3]))
sys.argv = sys.argv[1:3]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        def launch(command, **kwargs):
            return popen([sys.executable, '-B', '-c', script, command[-2], command[-1], str(delay),
                          'yes' if descendant else 'no'], **kwargs)
        original = wf.runtime
        try:
            with patch('runtime.task_runtime.subprocess.Popen', side_effect=launch):
                wf.runtime.launch(wf.queue.get(tid), {'kind': 'test_control', 'estimated_seconds': 2}, input_versions={})
            wf.budget.numeric_wait(True)
            path = Path(wf.queue.get(tid)['request_path']).parent / 'result.json'
            until = time.monotonic() + 10
            while not path.exists() and time.monotonic() < until:
                time.sleep(.02)
            self.assertTrue(path.exists())
            self.assertEqual(wf.runtime.poll(), [])
            self.assertEqual(wf.queue.get(tid)['state'], 'running')
            self.assertTrue(wf.runtime.live_processes())
            self.assertEqual(wf.budget.snapshot()['settlements'], {})
            if restore:
                wf = Workflow(wf.run_dir)
            result = []
            while not result and time.monotonic() < until:
                result = wf.runtime.poll()
                time.sleep(.02)
            self.assertTrue(result, 'cleanup must eventually finish')
            self.assertFalse(wf.runtime.live_processes())
            _, receipt, _ = result[0]
            self.assertEqual(receipt['status'], 'completed')
            wf.runtime.settle(receipt)
            wf.runtime.settle(receipt)
            wf.queue.finish(tid, result=receipt['data'])
            self.assertEqual(wf.runtime.poll(), [])
            self.assertEqual((wf.queue.get(tid)['attempt'], wf.queue.get(tid)['retry_count']), (1, 0))
            self.assertEqual(len(wf.budget.snapshot()['settlements']), 1)
            self.assertGreater(wf.budget.snapshot()['remaining_seconds'], 18)
            wf.budget.end_segment(processes_stopped=True, reason='test complete')
        finally:
            original.stop()
            wf.runtime.stop()

    def test_slow_teardown_past_worker_deadline_preserves_result_on_resume(self):
        self.exercise_cleanup(2.5, 10, restore=True)

    def test_durable_result_hung_teardown_and_registered_child_are_cleaned(self):
        self.exercise_cleanup(60, .05, descendant=True)
