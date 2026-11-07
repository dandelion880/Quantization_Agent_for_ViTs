"""Operational concurrency and budget checks; no simulated scientific success."""
from pathlib import Path
import os
import sys
import time
import subprocess
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.journal import Journal, read_json
from runtime.budget import BudgetLedger, RequestBudgetInsufficient
from runtime.gpu_resources import acquire, ResourceBusy, placement
from runtime.task_queue import GlobalTaskQueue
from runtime.task_runtime import TaskRuntime
from tests.test_infrastructure import TempCase
from tests.support import experiment_payload


class NumericBudget(TempCase):
    def test_budget_checkpoint_failure_cannot_skip_worker_cleanup(self):
        from runtime.budget import BudgetExhausted
        wf = self.workflow()
        with patch.object(wf.budget, 'check', side_effect=BudgetExhausted('test stop')), \
             patch.object(wf.budget, 'numeric_wait', side_effect=OSError('test storage failure')), \
             patch.object(wf.runtime, 'stop') as stop:
            result = wf.run()
        stop.assert_called_once()
        self.assertFalse(result['state_saved'])
        self.assertIn('close_numeric_wait', [r['step'] for r in result['finalization_errors']])

    def test_outer_monitor_tracks_numeric_wait_past_original_deadline(self):
        from core.workflow import Workflow
        from runtime.run_segments import run_managed
        from runtime.paths import QAVIT
        from tests.support import validation_config
        cfg = validation_config()
        cfg['budget'] = {'wall_seconds': 2, 'clock_mode': 'active_segments', 'exclude_numeric_wait': True}
        wf = Workflow(self.root / 'monitor', cfg)
        wf.queue.enqueue('numeric', 'experiment')
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
from core.workflow import Workflow
wf = Workflow(sys.argv[2])
wf.advance = lambda: all(t['state']=='completed' for t in wf.queue.snapshot().values())
wf.request_body = lambda task: {'kind':'test_control', 'sleep_seconds':3, 'estimated_seconds':10}
def accept(task, receipt, path):
    wf.runtime.settle(receipt)
    if receipt['status']=='completed': wf.queue.finish(task['task_id'], result=receipt['data'])
    else: wf.queue.fail(task['task_id'], receipt['error'])
wf.accept = accept
report = wf.run()
assert report['status']=='completed', report
'''
        # Real monitor, Supervisor and task processes; only numerical work is a delay.
        popen = subprocess.Popen
        def supervisor(command, **kwargs):
            return popen([sys.executable, '-B', '-c', script, str(QAVIT), str(wf.run_dir)], **kwargs)
        # dispatch does not need an ExperimentCard for this explicit test control.
        # Supply one to preserve the normal production dispatch and cache checks.
        card = wf.broker.preflight(experiment_payload())
        # Freeze only after run_managed opens the budget segment.
        def readiness(current, deadline):
            experiment = current.broker.freeze(card, requester='user')
            current.journal.commit('test/spec', cards=[experiment])
            task = next(iter(current.queue.snapshot().values()))
            current.queue.change(task['task_id'], 'pending', patch={'inputs':{'experiment_ref':experiment['id']}})
        with patch('runtime.run_segments.subprocess.Popen', side_effect=supervisor):
            result = run_managed(wf, started_unix=time.time(), readiness=readiness)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertGreater(result['budget']['excluded_numeric_wait_seconds'], 2)
        self.assertIsNone(result['budget']['active_segment'])
        self.assertFalse(wf.runtime.live_processes())

    def test_union_overlap_pause_resume_and_estimate(self):
        clock = [100.]
        j = self.journal()
        budget = BudgetLedger(j, {'wall_seconds': 50, 'clock_mode': 'active_segments',
                                  'exclude_numeric_wait': True}, clock=lambda: clock[0])
        budget.begin_segment()
        clock[0] = 110
        budget.numeric_wait(True)
        clock[0] = 150
        budget.numeric_wait(True)  # overlapping experiments do not open another interval
        self.assertEqual(budget.snapshot()['elapsed_seconds'], 10)
        budget.reserve('experiment', 'experiment', estimated_seconds=1000)
        budget.numeric_wait(False)  # Coding starts while numerical work continues
        clock[0] = 160
        self.assertEqual(budget.snapshot()['elapsed_seconds'], 20)
        budget.numeric_wait(True)
        clock[0] = 200
        budget.end_segment(processes_stopped=True, reason='pause')
        self.assertEqual(budget.snapshot()['elapsed_seconds'], 20)
        clock[0] = 1000
        restored = BudgetLedger(Journal(j.root, j.run_id), clock=lambda: clock[0])
        self.assertEqual(restored.snapshot()['remaining_seconds'], 30)
        restored.begin_segment()
        clock[0] = 1005
        self.assertEqual(restored.snapshot()['elapsed_seconds'], 25)
        self.assertIn('单任务超时', restored.input_receipt()['meaning'])
        self.assertNotIn('numeric_waits', restored.input_receipt())

    def test_unknown_crash_interval_is_charged(self):
        clock = [0.]
        budget = BudgetLedger(self.journal(), {'wall_seconds': 100, 'clock_mode': 'active_segments',
                              'exclude_numeric_wait': True}, clock=lambda: clock[0])
        budget.begin_segment()
        clock[0] = 10
        budget.numeric_wait(True)
        clock[0] = 20
        budget.numeric_wait(True)
        clock[0] = 60
        with patch('runtime.budget.process_alive', return_value=False):
            budget.end_segment(processes_stopped=True, reason='crash recovery', recovered=True)
        self.assertEqual(budget.snapshot()['elapsed_seconds'], 50)
        self.assertEqual(budget.snapshot()['excluded_numeric_wait_seconds'], 10)

    def test_numeric_worker_outlives_original_wall_boundary_and_settles_once(self):
        j = self.journal()
        budget = BudgetLedger(j, {'wall_seconds': 1, 'clock_mode': 'active_segments', 'exclude_numeric_wait': True})
        budget.begin_segment()
        queue = GlobalTaskQueue(j)
        tid = queue.enqueue('numeric', 'experiment')
        queue.promote()
        runtime = TaskRuntime(j.root, queue, budget, {})
        try:
            runtime.launch(queue.get(tid), {'kind': 'test_control', 'sleep_seconds': 1.3,
                           'estimated_seconds': 10}, input_versions={})
            budget.numeric_wait(True)
            for _ in range(200):
                result = runtime.poll()
                if result:
                    break
                time.sleep(.02)
            task, receipt, path = result[0]
            self.assertEqual(receipt['status'], 'completed', receipt)
            self.assertGreater(receipt['finished_unix'], budget.snapshot()['started_unix'] + 1)
            runtime.settle(receipt)
            queue.finish(tid)
            budget.numeric_wait(False)
            self.assertEqual(runtime.poll(), [])
            self.assertEqual(len(budget.snapshot()['settlements']), 1)
            self.assertGreater(budget.snapshot()['remaining_seconds'], 0)
        finally:
            runtime.stop()


class GpuLeases(TempCase):
    def test_exited_worker_lease_does_not_race_unaccepted_queue_ownership(self):
        wf = self.workflow()
        first = wf.queue.enqueue('finished_process', 'experiment', resource_keys=['gpu:0'])
        second = wf.queue.enqueue('next', 'experiment')
        wf.queue.promote();wf.queue.claim(first)
        body = {'kind':'test_control','gpu_pool':[0],'gpu_count':1,'pinned_gpu':None}
        with patch('runtime.gpu_resources.externally_busy', return_value=set()), \
             patch.object(wf.runtime, '_launch', side_effect=AssertionError('must queue until receipt accepted')):
            with self.assertRaises(ResourceBusy):
                wf.runtime.launch(wf.queue.get(second), body, input_versions={})
        self.assertEqual(wf.queue.get(second)['state'], 'ready')
        self.assertEqual(wf.queue.get(second)['attempt'], 0)
        self.assertEqual(wf.budget.snapshot()['reservations'], {})

    def test_container_device_minor_is_resolved_by_uuid_not_gpu_index(self):
        from types import SimpleNamespace
        from runtime.gpu_resources import gpu_bindings
        info = SimpleNamespace(read_text=lambda: 'GPU UUID: GPU-last\nDevice Minor: 6\n')
        with patch('runtime.gpu_resources.subprocess.run', return_value=SimpleNamespace(stdout='4, GPU-last\n')), \
             patch('runtime.gpu_resources.Path.glob', return_value=[info]):
            self.assertEqual(gpu_bindings([4]), [{'index':4,'uuid':'GPU-last','device_node':'/dev/nvidia6'}])

    def test_worker_inherits_lease_until_killed_and_no_stale_lock_after_exit(self):
        from runtime.codex_cli import parent_guard
        with patch('runtime.gpu_resources.externally_busy', return_value=set()):
            _, fds = acquire([102])
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                     pass_fds=tuple(fds), preexec_fn=parent_guard(os.getpid()))
            for fd in fds: os.close(fd)
            try:
                with self.assertRaises(ResourceBusy): acquire([102])
            finally:
                child.kill(); child.wait(timeout=5)
            _, fds = acquire([102])
            for fd in fds: os.close(fd)

    def test_probe_binding_reports_all_missing_parameters_in_one_repair(self):
        wf = self.workflow()
        draft = experiment_payload()
        draft['execution'].pop('parameters')
        draft['execution']['resolved_choices'] = {'training':False}
        wf.brief['payload']['hard_constraints'] = {'training':False}
        from tools.experiments import preflight_experiment
        from core.cards import ContractError
        with self.assertRaises(ContractError) as error:
            preflight_experiment(draft, wf.brief['payload'], executable=True)
        for field in ('bits', 'signed', 'granularity', 'axis', 'group_size'):
            self.assertIn(field, str(error.exception))
    def test_shared_lease_exclusion_partial_allocation_and_release(self):
        with patch('runtime.gpu_resources.externally_busy', return_value=set()):
            selected, fds = acquire([100, 101], 1)
            try:
                self.assertEqual(selected, [100])
                other, second = acquire([100, 101], 2)
                self.assertEqual(other, [101])
                try:
                    with self.assertRaises(ResourceBusy): acquire([100, 101])
                finally:
                    for fd in second: os.close(fd)
            finally:
                for fd in fds: os.close(fd)
            again, fds = acquire([100, 101], 2)
            for fd in fds: os.close(fd)
            self.assertEqual(again, [100, 101])

    def test_busy_gpu_does_not_claim_fail_or_reserve(self):
        wf = self.workflow()
        card = wf.broker.freeze(experiment_payload(), requester='user')
        wf.journal.commit('numeric', cards=[card])
        tid = wf.queue.enqueue('numeric', 'experiment', inputs={'experiment_ref': card['id']})
        body = {'kind': 'test_control', 'gpu_pool': [0], 'gpu_count': 1, 'pinned_gpu': None}
        with patch.object(wf, 'request_body', return_value=body), patch('runtime.gpu_resources.externally_busy', return_value={0}):
            self.assertEqual(wf.dispatch(), 0)
        task = wf.queue.get(tid)
        self.assertEqual((task['state'], task['attempt']), ('ready', 0))
        self.assertEqual(wf.budget.snapshot()['reservations'], {})

    def test_multi_gpu_falls_back_for_existing_services(self):
        wf = self.workflow()
        p = experiment_payload()
        p['resources']['gpu_count'] = 5
        card = wf.broker.freeze(p, requester='user')
        value = placement(card, {'allowed_gpus': [0,1,2,3,4]})
        self.assertEqual(value['gpu_count'], 1)
        self.assertFalse(value['parallelism']['classification_shards_supported'])
