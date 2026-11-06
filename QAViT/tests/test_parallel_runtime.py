"""有界并发回归：真实本地进程+可控延迟；不调用模型或声称科研测量。"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from runtime.configuration import config_check
from runtime.identity import code_identity
from runtime.paths import QAVIT
from tests.support import validation_config
from tests.test_infrastructure import TempCase


class ParallelRuntime(TempCase):
    def enqueue(self, wf, pair, side='A'):
        return wf.enqueue_role('fixture/' + '-'.join(pair) + '/' + side, 'position',
            refs=[wf.brief['id']], context={'round': 1, 'pair': pair, 'exchange': 1, 'side': side})

    def test_configuration_has_bounded_opt_in_and_keeps_old_serial_configs(self):
        config = validation_config()
        self.assertEqual(config['runtime']['llm_concurrency'], 4)
        del config['runtime']['llm_concurrency']
        self.assertEqual(config_check(config)['runtime']['llm_concurrency'], 4)
        for cap in (1, 2, 3, 4):
            config['runtime']['llm_concurrency'] = cap
            config_check(config)
        for cap in (0, 5, True, 3.0):
            config['runtime']['llm_concurrency'] = cap
            with self.assertRaises(ContractError): config_check(config)
        config['runtime']['llm_concurrency'] = 4
        config['runtime']['experiment_concurrency'] = 2
        with self.assertRaises(ContractError): config_check(config)

    def test_dispatch_starts_three_overlapping_workers_and_recovery_accepts_once(self):
        wf = self.workflow(1)
        ids = [self.enqueue(wf, pair) for pair in (['H1', 'H2'], ['H1', 'H3'], ['H2', 'H3'])]
        # Full persistent worker lifecycle, with a test_control body instead of a paid LLM.
        def controlled(task):
            return {'kind': 'test_control', 'sleep_seconds': 1.2,
                    'value': task['task_id'], 'estimated_seconds': 15}
        try:
            with patch.object(wf, 'request_body', side_effect=controlled):
                self.assertEqual(wf.dispatch(), 3)
                self.assertEqual(wf.dispatch(), 0)
            self.assertEqual([wf.queue.get(t)['state'] for t in ids], ['running'] * 3)
            restored = Workflow(wf.run_dir)
            self.assertEqual(restored.dispatch(), 0)
            deadline = time.monotonic() + 20
            rows = []
            while len(rows) < 3 and time.monotonic() < deadline:
                for task, receipt, path in restored.runtime.poll():
                    self.assertEqual(receipt['status'], 'completed', receipt)
                    self.assertEqual(receipt['data']['value'], task['task_id'])
                    restored.runtime.settle(receipt)
                    restored.queue.finish(task['task_id'], result=receipt['data'])
                    rows.append((task, receipt, path))
                time.sleep(.02)
            self.assertEqual(len(rows), 3)
            intervals = [(read_json(path.parent / 'started.json')['started_unix'], receipt['finished_unix'])
                         for task, receipt, path in rows]
            overlap = min(end for start, end in intervals) - max(start for start, end in intervals)
            self.assertGreater(overlap, .4, intervals)
            self.assertEqual(restored.runtime.poll(), [])
            self.assertTrue(all(restored.queue.get(t)['attempt'] == 1 for t in ids))
            ledger = restored.budget.snapshot()
            self.assertEqual(len(ledger['settlements']), 3)
            # Each attempt retains its duration; elapsed is one wall clock interval.
            worker_total = sum(r['resources']['wall_seconds'] for r in ledger['settlements'].values())
            self.assertGreater(worker_total, ledger['elapsed_seconds'])
            self.assertEqual(restored.dispatch(), 0)
        finally:
            wf.runtime.stop()

    def test_global_and_role_caps_pair_exclusion_and_legacy_ready_task_migration(self):
        wf = self.workflow(1)
        for pair in (['H1', 'H2'], ['H1', 'H3'], ['H2', 'H3']):
            for side in ('A', 'B'): self.enqueue(wf, pair, side)
        wf.enqueue_role('fixture/code1', 'code', context={})
        wf.enqueue_role('fixture/code2', 'code', context={})
        wf.enqueue_role('fixture/generate', 'generate', context={})
        # Simulate an old persisted resource key without rewriting task identity/history.
        old = next(iter(wf.tasks().values()))
        wf.queue.change(old['task_id'], old['state'], patch={'resource_keys': ['llm:subscription']})
        def launch(task, body, *, input_versions):
            return wf.queue.claim(task['task_id'], input_versions=input_versions)
        with patch.object(wf, 'request_body', return_value={}), patch.object(wf.runtime, 'launch', side_effect=launch):
            self.assertEqual(wf.dispatch(), 4)
            self.assertEqual(wf.dispatch(), 0)
        active = [t for t in wf.tasks().values() if t['state'] == 'running']
        self.assertEqual(sum(t['role'] == 'judger' for t in active), 3)
        self.assertEqual(sum(t['role'] == 'coding' for t in active), 1)
        self.assertEqual(len({tuple(t['inputs']['context']['pair']) for t in active if t['role'] == 'judger'}), 3)
        self.assertTrue(wf.queue.get(old['task_id'])['resource_keys'][0].startswith('llm:judger:'))

    def test_single_slot_fairness_survives_retry_and_resume(self):
        config = validation_config()
        config['runtime']['llm_concurrency'] = 1
        wf = Workflow(self.root / 'fairness', config)
        ids = [self.enqueue(wf, pair) for pair in (['H1', 'H2'], ['H1', 'H3'], ['H2', 'H3'])]
        def launch(task, body, *, input_versions):
            wf.queue.claim(task['task_id'], input_versions=input_versions)
        with patch.object(wf, 'request_body', return_value={}), patch.object(wf.runtime, 'launch', side_effect=launch):
            self.assertEqual(wf.dispatch(), 1)
        wf.queue.fail(ids[0], {'code': 'llm_transport_error'}, retryable=True)
        wf.queue.change(ids[0], 'retry', patch={'retry_after': 0})
        wf = Workflow(wf.run_dir)
        with patch.object(wf, 'request_body', return_value={}), patch.object(wf.runtime, 'launch', side_effect=launch):
            self.assertEqual(wf.dispatch(), 1)
            self.assertEqual(wf.queue.get(ids[1])['state'], 'running')
            wf.queue.finish(ids[1])
            self.assertEqual(wf.dispatch(), 1)
            self.assertEqual(wf.queue.get(ids[2])['state'], 'running')
            wf.queue.finish(ids[2])
            self.assertEqual(wf.dispatch(), 1)
        self.assertEqual(wf.queue.get(ids[0])['attempt'], 2)
        self.assertEqual(wf.queue.get(ids[0])['retry_count'], 1)

    def test_project_subscription_slots_overlap_and_bound_roles_across_processes(self):
        script = '''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from core.journal import atomic_json
from runtime.model_router import subscription_slot
with subscription_slot(sys.argv[2], time.time() + 15) as slot:
    start = time.time()
    time.sleep(.7)
    atomic_json(Path(sys.argv[3]), {'role':sys.argv[2], 'slot':slot, 'start':start, 'end':time.time()})
'''
        roles = ['judger'] * 4 + ['coding', 'generation']
        processes = [subprocess.Popen([sys.executable, '-B', '-c', script, str(QAVIT), role,
                     str(self.root / ('slot' + str(i) + '.json'))]) for i, role in enumerate(roles)]
        try:
            for process in processes: self.assertEqual(process.wait(timeout=20), 0)
        finally:
            for process in processes:
                if process.poll() is None: process.terminate(); process.wait(timeout=3)
        rows = [read_json(self.root / ('slot' + str(i) + '.json')) for i in range(len(roles))]
        maximum = {'judger': 0, 'other': 0, 'total': 0}
        for t in [r['start'] + .001 for r in rows]:
            active = [r for r in rows if r['start'] <= t < r['end']]
            maximum['judger'] = max(maximum['judger'], sum(r['role'] == 'judger' for r in active))
            maximum['other'] = max(maximum['other'], sum(r['role'] != 'judger' for r in active))
            maximum['total'] = max(maximum['total'], len(active))
        self.assertEqual(maximum, {'judger': 3, 'other': 1, 'total': 4})

    def test_numeric_workers_from_two_runs_share_one_global_slot(self):
        script = '''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from core.journal import atomic_json
from runtime import task_worker
path = sys.argv[2]
def controlled(request, directory):
    start = time.time()
    time.sleep(.7)
    atomic_json(directory / 'controlled_interval.json', {'start':start, 'end':time.time()})
    return {'source':'test_double'}
task_worker.work = controlled
sys.argv = ['task_worker.py', path]
task_worker.main()
'''
        paths = [self.root / run / 'jobs' / 'trial' / 'request.json' for run in ('run-a', 'run-b')]
        for i, path in enumerate(paths):
            atomic_json(path, {'kind': 'experiment', 'task_id': 'controlled-' + str(i),
                'execution_id': 'controlled-' + str(i), 'deadline_unix': time.time() + 15,
                'core_sha256': code_identity()['core_sha256'], 'resource_keys': []})
        processes = [subprocess.Popen([sys.executable, '-B', '-c', script, str(QAVIT), str(path)]) for path in paths]
        try:
            for process in processes: self.assertEqual(process.wait(timeout=20), 0)
        finally:
            for process in processes:
                if process.poll() is None: process.terminate(); process.wait(timeout=3)
        receipts = [read_json(path.parent / 'result.json') for path in paths]
        self.assertEqual([r['status'] for r in receipts], ['completed', 'completed'])
        intervals = sorted((read_json(path.parent / 'controlled_interval.json') for path in paths),
                           key=lambda row: row['start'])
        self.assertLessEqual(intervals[0]['end'], intervals[1]['start'])
        # Both workers exist concurrently; only entry to numerical work is serialized.
        starts = [read_json(path.parent / 'started.json')['started_unix'] for path in paths]
        self.assertLess(max(starts), min(r['finished_unix'] for r in receipts))

