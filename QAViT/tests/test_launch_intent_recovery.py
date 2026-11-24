"""Real workers consume exactly the recovered request; no numerical evidence."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import time
from unittest.mock import patch

from core.journal import read_json, file_sha256
from runtime.budget import BudgetLedger
from runtime.task_queue import GlobalTaskQueue
from runtime.task_runtime import TaskRuntime
from tests.test_infrastructure import TempCase


class LaunchIntentRecovery(TempCase):
    def exercise(self, *, changed=False, expired=False):
        journal = self.journal()
        queue = GlobalTaskQueue(journal)
        budget = BudgetLedger(journal, {'wall_seconds': 60})
        runtime = TaskRuntime(journal.root, queue, budget, {'cpu_threads': 1})
        self.addCleanup(runtime.stop)
        tid = queue.enqueue('orphan-intent', 'test_control')
        queue.promote()
        body = {'kind': 'test_control', 'value': 'original', 'cpu_threads': 1,
                'estimated_seconds': .05 if expired else 15}
        # Durable request + reservation, interruption immediately before queue claim.
        with patch.object(queue, 'claim', side_effect=RuntimeError('injected before claim')):
            with self.assertRaisesRegex(RuntimeError, 'injected before claim'):
                runtime.launch(queue.get(tid), deepcopy(body), input_versions={})
        directory = journal.root / 'jobs' / (tid + '-a0001')
        original = directory / 'request.json'
        original_bytes = original.read_bytes()
        if expired:
            time.sleep(.08)
            body['estimated_seconds'] = 15
        if changed:
            body.update(value='changed', cpu_threads=2)
        captured = {}
        popen = subprocess.Popen
        def launch(command, **kwargs):
            captured.update(command=command, env=kwargs['env'])
            return popen(command, **kwargs)
        with patch('runtime.task_runtime.subprocess.Popen', side_effect=launch), \
             patch.dict(os.environ, {'QAVIT_ASSIGNED_GPU_UUIDS': 'stale-parent-device'}):
            runtime.launch(queue.get(tid), deepcopy(body), input_versions={})
        request_path = Path(queue.get(tid)['request_path'])
        self.assertEqual(captured['command'][-1], str(request_path))
        self.assertEqual(captured['env']['QAVIT_TASK_REQUEST_PATH'], str(request_path.resolve()))
        self.assertEqual(captured['env']['OMP_NUM_THREADS'], str(body['cpu_threads']))
        self.assertNotIn('QAVIT_ASSIGNED_GPU_UUIDS', captured['env'])
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(request_path != original, changed or expired)
        for _ in range(500):
            results = runtime.poll()
            if results:
                break
            time.sleep(.01)
        self.assertEqual(len(results), 1)
        receipt = results[0][1]
        self.assertEqual(receipt['status'], 'completed', receipt)
        self.assertEqual(receipt['data']['value'], body['value'])
        self.assertEqual(receipt['request_sha256'], file_sha256(request_path))
        runtime.settle(receipt)
        queue.finish(tid)
        self.assertEqual(len(budget.snapshot()['reservations']), 1)
        self.assertEqual(len(budget.snapshot()['settlements']), 1)

    def test_unclaimed_current_intent_reuses_exact_request_without_deadline_extension(self):
        self.exercise()

    def test_changed_body_gets_new_immutable_request_and_executes_new_value(self):
        self.exercise(changed=True)

    def test_expired_unclaimed_intent_gets_new_deadline_and_keeps_old_request(self):
        self.exercise(expired=True)
