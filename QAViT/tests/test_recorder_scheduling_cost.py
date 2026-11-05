"""Recorder boundary scheduling: CPU fixtures, no model calls or scientific claims."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import make_card
from tests.support import hypothesis
from tests.test_infrastructure import TempCase


class RecorderSchedulingCost(TempCase):
    def terminal_engineering(self, wf, key='fixture/engineering', *, failed=False, status='completed'):
        identifier = wf.queue.enqueue(key, 'experiment_request')
        wf.queue.ready()
        wf.queue.claim(identifier)
        if failed:
            wf.queue.fail(identifier, {'code': 'fixture_failure', 'message': 'measurement unavailable'})
        else:
            wf.queue.finish(identifier, result={'status': status})
        return wf.queue.get(identifier)

    def science(self, wf):
        card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/science', cards=[card])
        return card

    def test_engineering_only_is_archived_without_recorder(self):
        wf = self.workflow(1)
        task = self.terminal_engineering(wf)
        batch = wf.memory.recording_batch()
        self.assertIn(task['task_card'], batch['source_refs'])
        self.assertFalse(batch['review_required'])
        self.assertIsNone(wf.enqueue_recorder(1, incomplete=True))
        self.assertFalse(wf.journal.find_cards('MetareviewCard'))
        self.assertTrue(wf.summary()['scientific_review_complete'])
        self.assertFalse(any(e['type'] == 'recorder_batch_completed' for e in wf.journal.events))

    def test_successful_late_status_does_not_start_closeout_review(self):
        wf = self.workflow(1)
        self.science(wf)
        identifier = wf.enqueue_recorder(1, incomplete=True)
        wf.queue.ready()
        wf.submit(wf.queue.claim(identifier), {'scientific_analysis': 'CPU fixture only.'}, 'SCRIPTED')
        wf.event('round_finished', 'fixture/finished', round=1)
        self.terminal_engineering(wf)
        task_count = len(wf.tasks())
        self.assertTrue(wf.advance())
        self.assertEqual(len(wf.tasks()), task_count)
        self.assertEqual(len(wf.journal.find_cards('MetareviewCard')), 1)
        self.assertTrue(wf.summary()['scientific_review_complete'])

    def test_new_failure_remains_scientific_material(self):
        for failed, status in ((True, 'completed'), (False, 'rejected')):
            with self.subTest(failed=failed, status=status):
                self.root = self.root / status
                wf = self.workflow(1)
                task = self.terminal_engineering(wf, failed=failed, status=status)
                batch = wf.memory.recording_batch()
                self.assertTrue(batch['review_required'])
                self.assertIn(task['task_card'], batch['scientific_failure_refs'])
                self.assertFalse(wf.summary()['scientific_review_complete'])
                self.assertIsNotNone(wf.enqueue_recorder(1, incomplete=True))

    def test_engineering_update_does_not_retry_deferred_science_but_explicit_retry_resumes(self):
        wf = self.workflow(1)
        self.science(wf)
        identifier = wf.enqueue_recorder(1, incomplete=True)
        wf.queue.ready()
        wf.queue.claim(identifier)
        wf.queue.fail(identifier, {'code': 'fixture_recorder_failure'})
        self.assertTrue(wf.defer_recorder(wf.queue.get(identifier)))
        wf.event('round_finished', 'fixture/finished', round=1)
        self.terminal_engineering(wf)
        self.assertTrue(wf.recording_deferred(wf.memory.recording_batch()))
        task_count = len(wf.tasks())
        self.assertTrue(wf.advance())
        self.assertEqual(len(wf.tasks()), task_count)
        self.assertFalse(wf.summary()['scientific_review_complete'])
        wf.retry_failed('Explicit CPU fixture retry')
        self.assertFalse(wf.recording_deferred(wf.memory.recording_batch()))
        self.assertEqual(wf.enqueue_recorder(1, key='fixture/closeout'), identifier)
        self.assertEqual(len(wf.tasks()), task_count)

    def test_budget_closeout_without_science_uses_no_recorder(self):
        wf = self.workflow(1)
        self.terminal_engineering(wf)
        wf.event('budget_closeout_started', 'budget/closeout', round=1)
        with patch.object(wf.runtime, 'launch', side_effect=AssertionError('No model should launch')):
            report = wf.run()
        self.assertEqual(report['status'], 'budget_stopped')
        self.assertFalse(any(t['inputs'].get('purpose') == 'record' for t in wf.tasks().values()))
        self.assertFalse(wf.journal.find_cards('MetareviewCard'))
        self.assertTrue(report['scientific_review_complete'])
