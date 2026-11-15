"""Recorded before/after retained B, without modifying research decisions or Memory."""
from unittest.mock import patch

from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests import test_fallback_solutions as fallback, test_gonly as gonly


class RecorderBContext(TempCase):
    promote = fallback.FallbackTests.promote
    workflow_case = gonly.GOnlyRegression.workflow_case
    drive = gonly.GOnlyRegression.drive

    def setUp(self):
        super().setUp()
        guard = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('No model/worker allowed'))
        guard.start(); self.addCleanup(guard.stop)

    def selected(self, *, prior=True):
        wf = self.workflow_case()
        if prior:
            self.promote(wf, 10, 80, 40)
        wf, select = self.drive(wf, stop=lambda t: t['inputs']['purpose'] == 'select')
        created = next(e['commit_sequence'] for e in wf.journal.events
                       if e['type'] == 'task_snapshot' and e['task']['task_id'] == select['task_id'])
        wf.submit(select, {'selected_slot': 'H2', 'rationale': 'scripted CPU selection'}, None)
        return wf, created

    def test_promotion_is_distinct_from_prior_B_and_recovery_keeps_batch_boundary(self):
        wf, created = self.selected()
        self.promote(wf, 1, 81, 41)
        tid = wf.enqueue_recorder(1)
        context = wf.context(wf.queue.get(tid))
        transition = context['B_transition']
        self.assertEqual(transition['before_l3']['candidate']['model_ref'], 'model10')
        self.assertEqual(transition['after_l3']['candidate']['model_ref'], 'model1')
        self.assertEqual(transition['before_l3']['snapshot_sequence'], created)
        self.assertEqual(transition['before_l3']['source']['kind'], 'selection_task_created')
        self.assertEqual(context['fallback_B']['candidate']['model_ref'], 'model1')
        # A later round can change current fallback but cannot rewrite this batch's B.
        self.promote(wf, 2, 82, 42)
        restored = Workflow(wf.run_dir)
        sequence, memory = restored.journal.sequence, restored.memory.snapshot()
        context = restored.context(restored.queue.get(tid))
        self.assertEqual(context['B_transition'], transition)
        self.assertEqual(context['fallback_B']['candidate']['model_ref'], 'model2')
        self.assertEqual((restored.journal.sequence, restored.memory.snapshot()), (sequence, memory))

    def test_first_valid_result_has_explicit_pending_prior_B(self):
        wf, _ = self.selected(prior=False)
        self.promote(wf, 1, 81, 41)
        context = wf.context(wf.queue.get(wf.enqueue_recorder(1)))['B_transition']
        self.assertEqual(context['before_l3']['status'], 'pending_no_verified_result')
        self.assertIsNone(context['before_l3']['candidate'])
        self.assertEqual(context['after_l3']['candidate']['model_ref'], 'model1')

    def test_failed_l3_closeout_retains_B_without_inventing_promotion(self):
        wf, _ = self.selected()
        self.promote(wf, 1, 99, 99, valid=False)
        tid = wf.enqueue_recorder(1, incomplete=True, key='closeout/record/test')
        transition = wf.context(wf.queue.get(tid))['B_transition']
        self.assertEqual(transition['before_l3']['candidate']['model_ref'], 'model10')
        self.assertEqual(transition['after_l3']['candidate']['model_ref'], 'model10')
        self.assertFalse(transition['solution_considerations'][0]['selected'])

    def test_missing_selection_uses_recorded_round_start_or_explicit_unknown(self):
        wf = self.workflow_case()
        tid = wf.enqueue_role('fixture/record', 'record', context={'round': 1})
        task = wf.queue.get(tid)
        self.assertEqual(wf.context(task)['B_transition']['before_l3']['status'], 'unavailable_no_pre_l3_anchor')
        wf.budget.start_round(1)
        # A new recording batch captures the round-start source without any L3.
        tid = wf.enqueue_role('closeout/record/no_l3', 'record', context={'round': 1})
        transition = wf.context(wf.queue.get(tid))['B_transition']
        self.assertEqual(transition['before_l3']['source']['kind'], 'round_started')
        self.assertEqual(transition['after_l3']['status'], 'pending_no_verified_result')
        self.assertEqual(transition['solution_considerations'], [])
