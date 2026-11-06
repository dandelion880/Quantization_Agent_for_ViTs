"""Three match service chains; scripted decisions/measurements, real host queue."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import make_card
from core.debate import INITIAL_PAIRS
from core.journal import atomic_json
from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.test_single_implementation import SOURCE
from tests.support import review


class ParallelRequests(TempCase):
    begin = delivery.DeliveryRegression.begin

    def test_busy_coding_queues_all_matches_and_returns_only_own_evidence_after_restore(self):
        wf = self.workflow(1)
        owners, requests, starts, returned = [], [], [], {}
        choices = {'site': 'blocks.0.mlp.fc2',
                   'parameters': {'bits': 4, 'signed': True, 'granularity': 'per_tensor'},
                   'calibration': {'split': 'calibration', 'count': 2},
                   'evaluation': {'split': 'search', 'count': 1},
                   'resolved_choices': {'training': False, 'activation_calibration': 'static'}}

        def launch(task, body, *, input_versions):
            # Persist precisely the per-call read snapshot as the real launcher does.
            path = wf.run_dir / 'jobs' / task['task_id'] / 'request.json'
            atomic_json(path, body)
            wf.queue.claim(task['task_id'], input_versions=input_versions, request_path=str(path))
            starts.append((task['task_id'], tick))
            if task['task_id'] in owners:
                context = body['spec']['context']
                result, = context['tool_results']
                own_request = requests[owners.index(task['task_id'])]
                self.assertEqual(result['call_id'], 'same-id-in-each-match')
                self.assertEqual(result['task'], own_request)
                returned[task['task_id']] = result['outputs']
                self.assertEqual(result['outputs'], wf.queue.get(own_request)['outputs'])
                wf.submit(wf.queue.get(task['task_id']), review('position', arguments=['fixture']), 'SCRIPTED')

        # Publish the first request and keep its Coding worker busy before the
        # other two scientists request work. Busy Coding must never reject them.
        tick = 0
        for pair in INITIAL_PAIRS:
            tid = wf.enqueue_role('fixture/' + '-'.join(pair), 'position', refs=[wf.brief['id']],
                context={'round': 1, 'pair': list(pair), 'exchange': 1, 'side': 'A', 'position_for': pair[0]})
            owners.append(tid)
            task, _ = self.begin(wf, wf.queue.get(tid))
            wf.receive_tools(task, {'tool_calls': [{'id': 'same-id-in-each-match', 'name': 'request_experiment',
                'arguments': {'experiment': {'question': 'fixture ' + '-'.join(pair),
                    'measurements': ['input_relative_mse'],
                    'execution': {'service': 'tensor_probe', 'requires_code': True}}}}]}, 'SCRIPTED')
            requests.append(wf.queue.get(tid)['waiting_for'][0])
            with patch.object(wf.runtime, 'launch', side_effect=launch):
                wf.dispatch(); wf.dispatch()
        self.assertTrue(all(wf.queue.get(t)['state'] == 'waiting' for t in requests))
        self.assertEqual(sum(t['state'] == 'running' and t['role'] == 'coding'
                             for t in wf.queue.snapshot().values()), 1)

        pipeline, restored, evidence_refs = False, False, []
        for tick in range(1, 50):
            with patch.object(wf.runtime, 'launch', side_effect=launch):
                wf.dispatch()
            active = [t for t in wf.queue.snapshot().values() if t['state'] == 'running']
            self.assertLessEqual(sum(t['role'] == 'coding' for t in active), 1)
            self.assertLessEqual(sum(t['kind'] == 'experiment' for t in active), 1)
            pipeline |= (any(t['kind'] == 'experiment' for t in active)
                         and any(t['role'] in ('coding', 'safeguard') for t in active))
            for task in active:
                if task['role'] == 'coding':
                    wf.submit(task, {'source': SOURCE, 'execution_choices': choices}, 'SCRIPTED')
                elif task['role'] == 'safeguard':
                    wf.submit(task, {'approved': True, 'rationale': 'fixture approval'}, 'SCRIPTED')
                elif task['kind'] == 'experiment':
                    started = next(at for tid, at in starts if tid == task['task_id'])
                    if tick - started < 3:
                        continue  # Leave numerical work active while other stages progress.
                    ref = task['inputs']['experiment_ref']
                    ev = make_card('EvidenceCard', wf.journal.run_id, {
                        'experiment_ref': ref, 'execution_fingerprint': task['task_id'], 'status': 'completed',
                        'observations': {'fixture_request': task['inputs']['parent_task']},
                        'provenance': {}, 'resources': {}, 'artifacts': [],
                        'measurement_source': 'test_double', 'execution_id': task['task_id']}, refs=[ref])
                    wf.queue.finish(task['task_id'], cards=[ev], outputs=[ev['id']])
                    evidence_refs.append(ev['id'])
            if not restored and any(t['kind'] == 'experiment' for t in active):
                wf = Workflow(wf.run_dir)
                restored = True
            if len(returned) == 3:
                break
        self.assertTrue(restored)
        self.assertTrue(pipeline)
        self.assertEqual(len(returned), 3)
        self.assertEqual(len(evidence_refs), 3)
        self.assertEqual({refs[0] for refs in returned.values()}, set(evidence_refs))
        self.assertEqual(len(starts), len({tid for tid, _ in starts}))
        self.assertEqual(sum(t['role'] == 'coding' for t in wf.queue.snapshot().values()), 3)
        self.assertEqual(sum(t['role'] == 'safeguard' for t in wf.queue.snapshot().values()), 3)
        self.assertFalse(any(t['inputs'].get('purpose') == 'reference' for t in wf.queue.snapshot().values()))
        before = len(wf.queue.snapshot())
        with patch.object(wf.runtime, 'launch', side_effect=AssertionError('completed work relaunched')):
            wf.dispatch(); wf.dispatch()
        self.assertEqual(len(wf.queue.snapshot()), before)
