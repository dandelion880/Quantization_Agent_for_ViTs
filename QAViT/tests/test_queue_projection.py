"""Queue selection must preserve snapshot isolation, transitions and replay."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cards import ContractError, canonical
from core.journal import Journal
from runtime.task_queue import GlobalTaskQueue
from tests.test_infrastructure import TempCase


class QueueProjection(TempCase):
    def test_filtered_snapshots_keep_order_missing_fields_and_independent_values(self):
        queue = GlobalTaskQueue(self.journal())
        a = queue.enqueue('a', 'test_control', inputs={'nested': [1]}, resource_keys=['gpu:0'])
        b = queue.enqueue('b', 'llm', inputs={'nested': [2]})
        queue.ready(); queue.claim(a)
        complete = queue.snapshot()
        selected = queue.snapshot(states={'running'}, kinds={'test_control'}, fields=('inputs', 'absent'))
        self.assertEqual(selected, {a: {'inputs': {'nested': [1]}}})
        selected[a]['inputs']['nested'].append(99)
        self.assertEqual(queue.get(a)['inputs']['nested'], [1])
        self.assertEqual(queue.snapshot(states=set()), {})
        self.assertEqual(queue.snapshot(kinds=set()), {})
        self.assertEqual(queue.snapshot(fields=()), {a: {}, b: {}})
        self.assertEqual(list(queue.snapshot()), [a, b])
        queue.finish(a, result={'complete': True})
        self.assertEqual(complete[a]['state'], 'running')
        self.assertEqual(queue.snapshot()[a]['state'], 'completed')

    def test_key_lookup_and_resource_scan_refresh_without_changing_history(self):
        journal = self.journal()
        queue = GlobalTaskQueue(journal)
        a = queue.enqueue('stable', 'test_control', resource_keys=['gpu:0'])
        queue.ready()
        before = canonical(journal.events)
        self.assertEqual(queue.enqueue('stable', 'test_control'), a)
        self.assertEqual(queue.get_by_key('stable'), queue.get(a))
        self.assertIsNone(queue.get_by_key('missing'))
        sentinel = object()
        self.assertIs(queue.get_by_key('missing', sentinel), sentinel)
        self.assertEqual(canonical(journal.events), before)
        reader = GlobalTaskQueue(Journal(journal.root, journal.run_id))
        self.assertEqual(reader.held_resources(), set())
        queue.claim(a)
        held = reader.held_resources()
        self.assertEqual(held, {'gpu:0'})
        held.add('gpu:9')
        self.assertEqual(reader.held_resources(), {'gpu:0'})
        looked_up = reader.get_by_key('stable')
        looked_up['resource_keys'].append('gpu:9')
        self.assertEqual(reader.get(a)['resource_keys'], ['gpu:0'])

    def test_batch_replay_and_ready_results_match_persisted_latest_state(self):
        journal = self.journal()
        queue = GlobalTaskQueue(journal)
        a = queue.enqueue('a', 'test_control', resource_keys=['shared'])
        b = queue.enqueue('b', 'test_control', depends_on=[a])
        c = queue.enqueue('c', 'test_control', resource_keys=['shared'])
        queue.ready(); queue.claim(a)
        with self.assertRaises(ContractError):
            queue.claim(c)
        queue.fail(a, {'code': 'retry'}, retryable=True)
        queue.change(a, 'retry', patch={'retry_after': 0})
        queue.ready(); queue.claim(a); queue.finish(a, result={'ok': True})
        ready = queue.ready()
        self.assertEqual([t['task_id'] for t in ready], [b, c])
        ready[0]['inputs']['foreign'] = True
        self.assertNotIn('foreign', queue.get(b)['inputs'])
        recovered = GlobalTaskQueue(Journal(journal.root, journal.run_id))
        self.assertEqual(recovered.snapshot(), queue.snapshot())
        self.assertEqual(recovered.get(a)['attempt'], 2)
        self.assertEqual(journal.card(queue.get(a)['task_card'])['payload']['state'], 'completed')
        queue.cancel_unfinished('fixture_stop')
        self.assertEqual([t['state'] for t in queue.snapshot().values()], ['completed', 'cancelled', 'cancelled'])
        self.assertEqual(recovered.snapshot(), queue.snapshot())

    def test_failed_promotion_does_not_modify_prior_results_or_events(self):
        journal = self.journal()
        queue = GlobalTaskQueue(journal)
        parent = queue.enqueue('parent', 'llm')
        child = queue.enqueue('child', 'test_control')
        queue.ready(); queue.claim(child); queue.finish(child, result={'ok': True})
        queue.claim(parent)
        queue.change(parent, 'waiting', patch={'waiting_for': [child],
            'tool_results': [{'task': child, 'result': {'large': 'x' * 25000}}]})
        before = deepcopy(queue.get(parent))
        events = canonical(journal.events)
        with patch.object(queue, 'change', side_effect=RuntimeError('before commit')):
            with self.assertRaises(RuntimeError):
                queue.promote()
        self.assertEqual(queue.get(parent), before)
        self.assertEqual(canonical(journal.events), events)
        queue.promote()
        self.assertEqual(queue.get(parent)['state'], 'ready')
        self.assertEqual(queue.get(parent)['tool_results'][0]['result']['complete_result_asset'],
                         str(journal.card_path(queue.get(child)['task_card'])))
