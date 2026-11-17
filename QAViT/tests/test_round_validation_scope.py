"""Report scope and resumed planning; fixtures are never scientific results."""
from copy import deepcopy
from unittest.mock import patch
from tests.test_minimal_fixes import L3Regression
from tests.test_infrastructure import TempCase
from tests.support import validation_config
from core.workflow import Workflow


class RoundValidationScope(TempCase):
    planned = L3Regression.planned
    evidence = L3Regression.evidence

    def test_new_planning_revision_is_current_before_request_exists(self):
        wf, trial = self.planned('planning')
        wf.queue.fail(trial['task_id'], {'code': 'fixture_failure'})
        wf.advance()
        plan = wf.tasks()['r001/plan_l3/revision1']
        for state in ('pending', 'ready', 'running', 'failed'):
            wf.queue.change(plan['task_id'], state)
            outcome = wf.l3_outcome(1)['candidates']['H1']
            self.assertEqual(outcome['status'], state)
            self.assertEqual(outcome['planning_state'], state)
            self.assertEqual(outcome['plan_revision'], 1)
            self.assertEqual(outcome['request_state'], 'failed')
        wf = Workflow(wf.run_dir)
        self.assertEqual(wf.l3_outcome(1)['candidates']['H1']['status'], 'failed')

    def test_aggregate_covers_all_requested_rounds_and_scope(self):
        config = validation_config()
        config.update(purpose='infrastructure_test', enable_l3=True)
        config['budget'].update(max_rounds=3, min_free_bytes=0)
        wf = Workflow(self.root / 'coverage', config)
        self.assertEqual(wf.summary()['expected_rounds'], [1, 2, 3])
        for number in (1, 2, 3):
            wf.budget.start_round(number)
            with patch.object(wf, 'l3_outcome', side_effect=lambda n: {
                    'formal_validation_success': n <= number, 'task_refs': ['fixture']}):
                for status in ('running', 'blocked', 'completed'):
                    report = wf.summary(status)
                    self.assertEqual(report['validated_rounds'], list(range(1, number + 1)))
                    self.assertEqual(report['formal_validation_success'], number == 3)
                wf.config['scope_target'] = {'round': 3, 'stage': 'generation'}
                self.assertFalse(wf.summary('scope_target_reached')['formal_validation_success'])
                del wf.config['scope_target']
        wf = Workflow(wf.run_dir)
        self.assertFalse(wf.summary('completed')['formal_validation_success'])

    def test_measured_round_does_not_cover_missing_later_round(self):
        wf, trial = self.planned('measured')
        _, evidence = self.evidence(wf, trial)
        wf.queue.finish(trial['task_id'], outputs=[evidence['id']], result={'status': 'completed'})
        self.assertTrue(wf.l3_outcome(1)['candidates']['H1']['formal_validation_success'])
        self.assertFalse(wf.l3_outcome(1)['formal_validation_success'])
        wf.config = deepcopy(wf.config)
        wf.config['budget']['max_rounds'] = 3
        report = wf.summary('completed')
        self.assertFalse(report['formal_validation_success'])
        self.assertEqual(report['validated_rounds'], [])  # H2/H3 explicitly deferred.
