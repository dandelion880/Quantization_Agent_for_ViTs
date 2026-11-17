"""Three-candidate queue/recovery tests; scripted science, no formal experiments."""
from copy import deepcopy
from itertools import permutations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.workflow import Workflow
from tests.support import validation_config, hypothesis, experiment_payload
from tests.test_infrastructure import TempCase
from tests import test_minimal_fixes as l3_fixtures


class MultiCandidateL3(TempCase):
    def prepare(self, name, *, missing=None):
        config = validation_config()
        config.update(enable_l3=True, purpose='infrastructure_test')
        config['budget'].update(max_rounds=1, min_free_bytes=0)
        wf = Workflow(self.root / name, config)
        wf.advance()
        for task in wf.queue.ready():
            task = wf.queue.claim(task['task_id'])
            slot = task['inputs']['context']['slot']
            if slot == missing:
                wf.queue.fail(task['task_id'], {'code': 'scripted_generation_failure'})
            else:
                wf.submit(task, hypothesis(1, slot), 'scripted')
        wf.advance()
        task = wf.queue.claim(wf.queue.ready()[0]['task_id'])
        wf.submit(task, {'rationale': 'Compare all three; no language ranking excludes a candidate.'}, 'scripted')
        wf.advance()
        for ready in wf.queue.ready():
            task = wf.queue.claim(ready['task_id'])
            draft = experiment_payload('L3')
            draft['execution'].update(service='evaluate_artifact', parameters={}, evaluations=[{
                'kind': 'classification', 'split': 'final', 'count': 2}])
            wf.submit(task, draft, 'scripted')
        wf.advance()
        requests = {t['inputs']['slot']: t for t in wf.queue.ready() if t['kind'] == 'experiment_request'}
        return wf, requests

    def measured(self, wf, trial):
        current = wf.queue.get(trial['task_id'])
        wf.queue.claim(current['task_id'])
        current = wf.queue.get(current['task_id'])
        _, evidence = l3_fixtures.L3Regression.evidence(self, wf, current)
        wf.queue.finish(current['task_id'], outputs=[evidence['id']], result={'status': 'completed'})
        return evidence

    def test_all_completion_orders_preserve_exact_binding_and_receive_all_results(self):
        for order in permutations(('H1', 'H2', 'H3')):
            wf, requests = self.prepare(''.join(order))
            self.assertEqual(set(requests), {'H1', 'H2', 'H3'})
            self.assertEqual(len({r['key'] for r in requests.values()}), 3)
            evidence = {}
            for slot in order:
                evidence[slot] = self.measured(wf, requests[slot])
                wf = Workflow(wf.run_dir)
                outcome = wf.l3_outcome(1)
                self.assertEqual(set(outcome['validated_slots']), set(evidence))
                self.assertEqual(outcome['formal_validation_success'], len(evidence) == 3)
                for observed in evidence:
                    self.assertEqual(outcome['candidates'][observed]['valid_evidence_refs'], [evidence[observed]['id']])
                    self.assertEqual(outcome['candidates'][observed]['hypothesis_ref'], wf.candidate_refs(1)[observed])
                if len(evidence) < 3:
                    self.assertFalse(wf.advance())
                    self.assertNotIn('r001/record', wf.tasks())
            wf.advance()
            self.assertEqual(set(wf.tasks()['r001/record']['inputs']['context']['l3_outcome']['valid_evidence_refs']),
                             {e['id'] for e in evidence.values()})
            considered = [e for e in wf.journal.events if e['type'] == 'solution_considered']
            self.assertEqual({e['evidence_ref'] for e in considered}, {e['id'] for e in evidence.values()})
            # These synthetic subset receipts must not enter the real solution Top-3.
            self.assertFalse(any(r['status'] == 'evaluated' for r in wf.solutions.snapshot()))
            sequence = wf.journal.sequence
            wf.consider_l3(1)
            self.assertEqual(wf.journal.sequence, sequence)

    def test_one_failure_does_not_block_siblings_and_resume_does_not_repeat_completed_work(self):
        wf, requests = self.prepare('partial-failure')
        wf.queue.claim(requests['H1']['task_id'])
        wf.queue.fail(requests['H1']['task_id'], {'code': 'scripted_calibration_failure'})
        e3 = self.measured(wf, requests['H3'])
        wf = Workflow(wf.run_dir)
        self.assertFalse(wf.advance())
        self.assertNotIn('r001/record', wf.tasks())
        e2 = self.measured(wf, requests['H2'])
        plan_root = wf.l3_request_keys(1, 'H1', wf.candidate_refs(1)['H1'])[0]
        wf.advance()
        revision = wf.tasks()[plan_root + '/revision1']
        wf.queue.ready()
        wf.submit(wf.queue.claim(revision['task_id']), {'defer_reason': 'Numerical failure retained; needs follow-up.'}, 'scripted')
        wf.advance()
        outcome = wf.l3_outcome(1)
        self.assertEqual(set(outcome['valid_evidence_refs']), {e2['id'], e3['id']})
        self.assertFalse(outcome['formal_validation_success'])
        self.assertEqual(outcome['status'], 'partial')
        self.assertTrue(wf.tasks()['r001/record']['inputs']['context']['incomplete_round'])
        self.assertEqual(len([t for t in wf.tasks().values() if t['kind'] == 'experiment_request']), 3)

    def test_missing_slot_is_not_all_complete_and_cross_candidate_cache_is_rejected(self):
        wf, requests = self.prepare('missing', missing='H2')
        e1 = self.measured(wf, requests['H1'])
        h3 = wf.queue.get(requests['H3']['task_id'])
        experiment = wf.broker.freeze(h3['inputs']['draft'], requester='generation')
        def cached(spec, *, visible, **kwargs):
            return e1 if visible(e1) else None
        with patch.object(wf.broker, 'cached', side_effect=cached):
            self.assertIsNone(wf.cached_experiment(experiment, h3))
        self.measured(wf, requests['H3'])
        self.assertEqual(wf.l3_outcome(1)['missing_slots'], ['H2'])
        self.assertFalse(wf.summary('completed')['formal_validation_success'])

    def test_all_real_result_shapes_enter_library_with_their_own_hypothesis(self):
        wf, requests = self.prepare('library')
        selection = wf.selection_card(1)
        for index, slot in enumerate(('H3', 'H1', 'H2')):
            exp = {'id': 'exp-' + slot, 'payload': {'level': 'L3', 'execution': {},
                'candidate_binding': {'selection_ref': selection['id'], 'hypothesis_ref': wf.candidate_refs(1)[slot]}}}
            evidence = {'id': 'ev-' + slot, 'payload': {'measurement_source': 'real', 'status': 'completed',
                'provenance': {'model_ref': 'model-' + slot}, 'observations': {'evaluations': {
                    'classification': {'full_dataset': True, 'metrics': {'top1': 80 + index}},
                    'segmentation': {'full_dataset': True, 'metrics': {'mIoU': 40 + index}}}}, 'artifacts': []}}
            # Isolate state/library handling from numerical receipts in this test.
            wf.solutions.consider_round(1, evidence, exp, selection, validator=lambda *a, **k: None)
            before = wf.journal.sequence
            wf.solutions.consider_round(1, evidence, exp, selection, validator=lambda *a, **k: None)
            self.assertEqual(before, wf.journal.sequence)
        rows = wf.solutions.snapshot()
        self.assertEqual({r['slot'] for r in rows}, {'H1', 'H2', 'H3'})
        for row in rows:
            self.assertEqual(row['hypothesis_ref'], wf.candidate_refs(1)[row['slot']])
        self.assertEqual(wf.solutions.fallback()['candidate']['model_ref'], 'model-H2')

    def test_completed_requests_without_evidence_are_incomplete(self):
        wf, requests = self.prepare('missing-receipts')
        for trial in requests.values():
            wf.queue.claim(trial['task_id'])
            wf.queue.finish(trial['task_id'], result={'status': 'completed'})
        outcome = wf.l3_outcome(1)
        self.assertEqual(outcome['status'], 'incomplete')
        self.assertFalse(outcome['formal_validation_success'])
        self.assertTrue(all(row['status'] == 'completed' for row in outcome['candidates'].values()))
        self.assertTrue(all('tasks' not in row for row in outcome['candidates'].values()))
        self.assertEqual(len(outcome['task_refs']), len(set(outcome['task_refs'])))

    def test_misbound_recovered_request_cannot_move_evidence_between_candidates(self):
        wf, requests = self.prepare('misbound-request')
        e1 = self.measured(wf, requests['H1'])
        e3 = self.measured(wf, requests['H3'])
        wrong = wf.queue.get(requests['H3']['task_id'])
        # Emulate a corrupted historical request-root association while keeping
        # the immutable numerical receipt bound to its actual H3 source.
        fake = deepcopy(wrong)
        fake['inputs']['request_root'] = requests['H1']['inputs']['request_root']
        real_tasks = wf.tasks()
        with patch.object(wf, 'tasks', return_value={**real_tasks, wrong['key']: fake}):
            outcome = wf.l3_outcome(1)
        self.assertEqual(outcome['candidates']['H1']['valid_evidence_refs'], [e1['id']])
        self.assertEqual(outcome['candidates']['H3']['valid_evidence_refs'], [e3['id']])
        self.assertTrue(outcome['candidates']['H1']['evidence_issues'])

    def test_finished_historical_single_candidate_keeps_original_scope(self):
        wf, requests = self.prepare('historical-scope')
        self.measured(wf, requests['H1'])
        selection = deepcopy(wf.selection_card(1))
        selection['payload'].update(selection_stage='gonly_preselection', selected_slot='H1',
                                    selected_hypothesis_ref=wf.candidate_refs(1)['H1'])
        wf.event('round_finished', 'fixture/old-finish', round=1)
        with patch.object(wf, 'selection_card', return_value=selection):
            outcome = wf.l3_outcome(1)
        self.assertEqual(outcome['expected_slots'], ['H1'])
        self.assertTrue(outcome['historical_single_candidate'])
        self.assertTrue(outcome['formal_validation_success'])

    def test_only_explicit_legacy_selection_allows_missing_binding_in_library(self):
        wf = self.workflow(1)
        selection = {'id': 'legacy-selection', 'payload': {
            'selection_stage': 'gonly_preselection', 'candidate_refs': {'H1': 'hypothesis@1'},
            'selected_slot': 'H1', 'selected_hypothesis_ref': 'hypothesis@1'}}
        experiment = {'id': 'legacy-experiment', 'payload': {'level': 'L3', 'execution': {}}}
        evidence = {'id': 'legacy-evidence', 'payload': {'measurement_source': 'real', 'status': 'completed',
            'provenance': {'model_ref': 'legacy-model'}, 'observations': {'evaluations': {
                'classification': {'full_dataset': True, 'metrics': {'top1': 80}},
                'segmentation': {'full_dataset': True, 'metrics': {'mIoU': 40}}}}, 'artifacts': []}}
        wf.solutions.consider_round(1, evidence, experiment, selection, validator=lambda *a, **k: None)
        self.assertEqual(wf.solutions.snapshot()[0]['hypothesis_ref'], 'hypothesis@1')
        for index, binding in enumerate(({'selection_ref': 'wrong', 'hypothesis_ref': 'hypothesis@1'}, {})):
            bad_selection = deepcopy(selection)
            if not binding:
                bad_selection['payload']['selection_stage'] = 'gonly_synthesis'
            bad_evidence = deepcopy(evidence)
            bad_evidence['id'] += '-' + str(index)
            bad_experiment = deepcopy(experiment)
            bad_experiment['payload']['candidate_binding'] = binding
            wf.solutions.consider_round(1, bad_evidence, bad_experiment, bad_selection, validator=lambda *a, **k: None)
            event = [e for e in wf.journal.events if e['type'] == 'solution_considered'][-1]
            self.assertFalse(event['selected'])
            self.assertIn('exact synthesized candidate', event['reason'])


if __name__ == '__main__':
    unittest.main()
