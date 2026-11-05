"""Scientific submission boundaries; scripted control flow, no measurement claims."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.roles import submission_schema
from core.cards import ContractError, make_card, validate_schema
from core.debate import validate_verdict
from tests.support import meta, hypothesis, experiment_payload
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery


class SubmissionGateCleanup(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    debate_task = delivery.DeliveryRegression.debate_task

    @staticmethod
    def minimal_hypothesis():
        return {'title': 'Local mechanism study', 'research_question': 'Which local change matters?',
                'mechanism': 'A proposed local mechanism; no observed performance claim.',
                'intervention': {'site': 'blocks.0.mlp.fc2'}}

    def test_minimal_hypothesis_with_previous_feedback_needs_no_appendices(self):
        wf = self.workflow(1)
        previous = make_card('MetareviewCard', wf.journal.run_id, meta(1), author='recorder')
        wf.journal.commit('prior_review', cards=[previous], events=[wf.memory.apply_metareview(previous)])
        task, _ = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id'], previous['id']],
            round=2, slot='H3', hypothesis_type='new_algorithm'))
        value = self.minimal_hypothesis()
        validate_schema(value, submission_schema('generate'), 'submission')
        wf.submit(task, value, 'SCRIPTED')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual((payload['round'], payload['slot']), (2, 'H3'))
        for name in ('predictions', 'sources', 'mechanism_relations', 'feedback_used'):
            self.assertEqual(payload[name], [])
        self.assertNotIn('implementation', payload)
        self.assertNotIn('validation_design', payload)

    def test_optional_rows_keep_valid_items_and_archive_specific_issues(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id']],
            round=1, slot='H3', hypothesis_type='new_algorithm'))
        value = self.minimal_hypothesis()
        value.update(predictions=[{'statement': 'A testable direction'}, {'measurement': 'missing claim'}],
            mechanism_relations=[{'relation': 'changes', 'target': 'local scaling', 'explanation': 'the mechanism'},
                                 {'relation': 'missing target and explanation'}],
            sources=[{'ref': 'C999999', 'meaning': 'Unknown optional source'}],
            feedback_used='mistyped appendix')
        wf.submit(task, value, 'SCRIPTED')
        card = wf.result_card(wf.queue.get(task['task_id']))
        payload = card['payload']
        self.assertEqual(payload['mechanism'], value['mechanism'])
        self.assertEqual(payload['predictions'], [{'id': 'p1', 'statement': 'A testable direction'}])
        self.assertEqual(len(payload['mechanism_relations']), 1)
        self.assertEqual(payload['sources'], [])
        self.assertEqual(payload['feedback_used'], [])
        issues = payload['submission_issues']
        self.assertEqual({row['field'] for row in issues},
                         {'predictions', 'mechanism_relations', 'sources', 'feedback_used'})
        self.assertTrue(all(row['issue'] and 'submission' in row for row in issues))
        self.assertNotIn('C999999', card['refs'])

    def test_position_and_screen_accept_only_their_scientific_core(self):
        wf = self.workflow(1)
        position, _ = self.begin(wf, self.debate_task(wf))
        wf.submit(position, {'rationale': 'No discriminating result is available yet.'}, 'SCRIPTED')
        self.assertEqual(wf.queue.get(position['task_id'])['state'], 'completed')
        screen, _ = self.begin(wf, self.task(wf, 'screen', refs=[wf.brief['id']]))
        wf.submit(screen, {'eligible': True, 'rationale': 'The core claim is testable.'}, 'SCRIPTED')
        payload = wf.result_card(wf.queue.get(screen['task_id']))['payload']
        self.assertEqual(payload['hard_rejection_reasons'], [])
        self.assertNotIn('similarity_analysis', payload)
        self.assertNotIn('arguments', payload)

    def test_explicit_decision_and_identity_conflicts_still_fail(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.task(wf, 'screen', round=1))
        for value in ({'eligible': False, 'rationale': 'No hard rejection reason supplied'},
                      {'eligible': True, 'hard_rejection_reasons': ['Explicit incompatible condition'],
                       'rationale': 'Contradictory eligibility'},
                      {'eligible': True, 'round': 2, 'rationale': 'Wrong assigned round'}):
            with self.subTest(value=value), self.assertRaises(ContractError):
                wf.submit(task, value, 'SCRIPTED')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'running')

    def test_verdict_uses_rationale_without_duplicate_decisive_reason(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.task(wf, 'arbitrate', round=1,
            pair=['H1', 'H2'], exchange=1))
        wf.submit(task, {'outcome': 'A', 'rationale': 'Under the matched conditions, A addresses the decisive error.'},
                  'SCRIPTED')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(payload['outcome'], 'A')
        self.assertNotIn('decisive_reason', payload)
        validate_verdict({'outcome': 'undetermined', 'rationale': 'No discriminating evidence is available.'}, 1)
        with self.assertRaises(ContractError):
            validate_verdict({'outcome': 'undetermined', 'continuation_assessment': {'experiment_available': True}}, 1)
        with self.assertRaises(ContractError):
            validate_verdict({'outcome': 'continue'}, 5)

    def test_selection_mirrors_rationale_and_derives_known_provisional_status(self):
        from tests.test_minimal_fixes import RankingRegression
        wf = self.workflow(1)
        task, matches = RankingRegression.rank_task(self, wf, ['A', 'B', 'A'])
        reason = 'The results form a cycle; H2 is a provisional choice for the stated scientific objective.'
        with patch.object(wf, 'matches', return_value=matches):
            wf.submit(task, {'selected_slot': 'H2', 'rationale': reason, 'selection_basis': ''}, 'SCRIPTED')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(payload['selection_basis'], reason)
        self.assertIs(payload['provisional'], True)
        self.assertEqual(payload['submission_issues'][0]['field'], 'selection_basis')

    def test_recorder_optional_issues_do_not_block_analysis_or_archive(self):
        wf = self.workflow(1)
        judgment = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'test-double', 'status': 'failed',
            'observations': {}, 'provenance': {}, 'resources': {}, 'artifacts': [],
            'measurement_source': 'test_double', 'execution_id': 'test-double'}, refs=[spec['id']])
        wf.journal.commit('scientific_sources', cards=[judgment, spec, evidence])
        task, _ = self.begin(wf, wf.queue.get(wf.enqueue_recorder(1, incomplete=True)))
        valid_audit = {'judgment_ref': judgment['id'], 'assessment': 'untested',
                       'evidence_refs': [evidence['id']], 'reasoning': 'A failed scripted receipt proves no mechanism.'}
        value = {'scientific_analysis': 'This is analysis of a failed control-flow fixture, not a measurement.',
                 'judgment_audit': [valid_audit, {'judgment_ref': judgment['id']}],
                 'predictive_metrics': [{'metric': 'proxy', 'evidence_refs': [judgment['id']]}],
                 'next_questions': ['Which real observation discriminates the mechanism?', 17],
                 'knowledge_updates': [{'text': 'Must not publish as measured', 'status': 'observed_once',
                    'evidence_refs': [evidence['id']], 'rationale': 'Invalid evidence for observed knowledge'}]}
        validate_schema(value, submission_schema('record'), 'submission')
        wf.submit(task, value, 'SCRIPTED')
        card = wf.result_card(wf.queue.get(task['task_id']))
        payload = card['payload']
        self.assertEqual(payload['scientific_analysis'], value['scientific_analysis'])
        self.assertEqual(payload['judgment_audit'], [valid_audit])
        self.assertEqual(payload['predictive_metrics'], [])
        self.assertEqual(payload['next_questions'], value['next_questions'][:1])
        self.assertEqual({row['field'] for row in payload['submission_issues']},
                         {'judgment_audit', 'predictive_metrics', 'next_questions'})
        self.assertEqual(payload['knowledge_updates'], [])
        self.assertEqual(len(payload['pending_knowledge_updates']), 1)
        self.assertEqual(wf.memory.snapshot()['L2'], {})
        self.assertEqual(wf.memory.recording_status()['scientific_review'], 'completed')
        self.assertEqual(wf.journal.card(evidence['id']), evidence)

    def test_minimal_recorder_and_malformed_optional_knowledge_container(self):
        wf = self.workflow(1)
        judgment = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('unmeasured_scientific_source', cards=[judgment])
        task, _ = self.begin(wf, wf.queue.get(wf.enqueue_recorder(1, incomplete=True)))
        value = {'scientific_analysis': 'No measurement is available; retain the unresolved question.',
                 'knowledge_updates': 'mistyped optional container'}
        wf.submit(task, value, 'SCRIPTED')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(payload['generation_guidance'], '')
        self.assertNotIn('next_questions', payload)
        self.assertEqual(payload['pending_knowledge_updates'][0]['submission'], value['knowledge_updates'])


if __name__ == '__main__':
    unittest.main()
