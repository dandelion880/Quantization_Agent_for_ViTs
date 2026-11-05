"""Explicit evidence handoffs across research and implementation; scripted CPU only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, make_card
from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery, test_branch_acl as acl
from tests.support import experiment_payload, hypothesis, review
from tools.safeguards import write_artifact


class EvidenceHandoffRegression(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    read = delivery.DeliveryRegression.read
    science = delivery.DeliveryRegression.science
    debate_task = delivery.DeliveryRegression.debate_task

    def setUp(self):
        super().setUp()
        guard = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('scripted CPU only'))
        guard.start()
        self.addCleanup(guard.stop)

    def measurement(self, wf, label, *, observations=None, prior=None):
        payload = experiment_payload()
        payload['question'] = 'FIXTURE ' + label
        if prior:
            payload['execution']['trial_evidence_refs'] = [prior['id']]
        spec = make_card('ExperimentCard', wf.journal.run_id, payload,
                         refs=[prior['id']] if prior else [])
        facts = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': label, 'status': 'completed',
            'observations': observations or {'error': 0.125, 'explanation': 'OBSERVATION_' + label},
            'scope': deepcopy(payload['scope']), 'warnings': ['fixture: local measurement only'],
            'limitations': ['fixture: no full-model accuracy conclusion'],
            'provenance': {}, 'resources': {}, 'artifacts': [],
            'measurement_source': 'test_double', 'execution_id': label}, refs=[spec['id']])
        wf.journal.commit('measurement/' + label, cards=[spec, facts])
        return spec, facts

    def subject(self, wf, evidence, *, slot='H1', sources=False):
        payload = hypothesis(1, slot)
        payload['evidence_refs'] = [] if sources else [evidence['id']]
        if sources:
            payload['sources'] = [{'ref': evidence['id'], 'reason': 'fixture direct support'}]
        card = make_card('HypothesisCard', wf.journal.run_id, payload, refs=[evidence['id']])
        wf.journal.commit('subject/' + card['id'], cards=[card])
        return card

    def assert_measurement(self, context, spec, evidence):
        cards = {c['id']: c for c in context['memory']['cards']}
        self.assertIn(spec['id'], cards)
        self.assertIn(evidence['id'], cards)
        payload = cards[evidence['id']]['payload']
        for key in ('status', 'scope', 'warnings', 'limitations', 'measurement_source', 'execution_id'):
            self.assertEqual(payload[key], evidence['payload'][key])
        self.assertEqual(cards[spec['id']]['payload']['execution'], spec['payload']['execution'])
        self.assertEqual(len(cards), len(context['memory']['cards']))
        return payload

    def test_current_subject_support_reaches_research_purposes_once(self):
        wf = self.workflow(1)
        spec, evidence = self.measurement(wf, 'CURRENT')
        _, unrelated = self.measurement(wf, 'UNRELATED')
        h = self.subject(wf, evidence, sources=True)
        decision = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('ranking', evidence_refs=[evidence['id']]), refs=[h['id'], evidence['id']])
        wf.journal.commit('decision', cards=[decision])
        for purpose in ('screen', 'position', 'arbitrate', 'evolve', 'plan_l3', 'generate'):
            with self.subTest(purpose=purpose):
                task = self.task(wf, purpose, refs=[h['id'], decision['id']], round=1,
                    hypothesis_ref=h['id'], parent_ref=h['id'], slot='H4' if purpose == 'evolve' else 'H1',
                    hypothesis_type='evolved' if purpose == 'evolve' else 'existing_method',
                    previous_review_pending=True, exchange=1, pair=['H1', 'H2'], side='A')
                _, context = self.begin(wf, task)
                payload = self.assert_measurement(context, spec, evidence)
                self.assertEqual(payload['observations'], evidence['payload']['observations'])
                self.assertNotIn(unrelated['id'], [c['id'] for c in context['memory']['cards']])
                self.assertEqual(canonical(context).count('OBSERVATION_CURRENT'), 1)
                wf.queue.finish(task['task_id'], result={'fixture': 'context inspected only'})

    def test_ranking_receives_candidate_and_verdict_observations_without_repeats(self):
        wf = self.workflow(1)
        spec, evidence = self.measurement(wf, 'RANK')
        verdict_spec, verdict_evidence = self.measurement(wf, 'VERDICT')
        hypotheses = [self.subject(wf, evidence, slot=slot) for slot in ('H1', 'H2', 'H3')]
        verdict = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('verdict', outcome='tie', evidence_refs=[verdict_evidence['id']]),
            refs=[verdict_evidence['id']])
        wf.journal.commit('verdicts', cards=[verdict], events=[{
            'type': 'match_closed', 'round': 1, 'pair': pair, 'outcome': 'tie',
            'exchanges': 1, 'verdict_ref': verdict['id']} for pair in ('H1-H2', 'H1-H3', 'H2-H3')])
        task = self.task(wf, 'rank', round=1, selection_stage='parent',
            candidate_refs={h['payload']['slot']: h['id'] for h in hypotheses}, eligible=['H1', 'H2', 'H3'])
        _, context = self.begin(wf, task)
        self.assert_measurement(context, spec, evidence)
        self.assert_measurement(context, verdict_spec, verdict_evidence)
        self.assertEqual(canonical(context).count('OBSERVATION_RANK'), 1)

    def test_coding_and_safeguard_receive_request_subject_and_trial_evidence(self):
        wf = self.workflow(1)
        spec, evidence = self.measurement(wf, 'IMPLEMENT')
        trial_spec, trial_evidence = self.measurement(wf, 'TRIAL')
        h = self.subject(wf, evidence)
        payload = experiment_payload()
        payload['execution'].update(requires_code=True, interface='quantizer_v1',
                                    trial_evidence_refs=[trial_evidence['id']])
        request = make_card('ExperimentCard', wf.journal.run_id, payload, refs=[wf.brief['id'], h['id']])
        frozen = make_card('ExperimentCard', wf.journal.run_id,
            {**payload, 'request_ref': request['id'], 'request_sha256': request['content_sha256']}, refs=[request['id']])
        wf.journal.commit('implementation-request', cards=[request, frozen])
        artifact = write_artifact(wf.run_dir, request, {'source':
            'def fit(inputs, context):\n    return {}\ndef apply(value, state, context):\n    return value\n'}, 'coding')
        for purpose in ('code', 'safeguard'):
            with self.subTest(purpose=purpose):
                task = self.task(wf, purpose, refs=[request['id'], frozen['id']],
                    request_ref=request['id'], experiment_ref=frozen['id'], candidate=artifact)
                _, context = self.begin(wf, task)
                self.assert_measurement(context, spec, evidence)
                self.assert_measurement(context, trial_spec, trial_evidence)
                self.assertEqual(context['original_request']['id'], request['id'])
                self.assertEqual(context['frozen_experiment']['id'], frozen['id'])
                self.assertEqual(canonical(context).count('OBSERVATION_IMPLEMENT'), 1)
                wf.queue.finish(task['task_id'], result={'fixture': 'context inspected only'})

    def test_large_support_is_bounded_and_exactly_readable_after_restore(self):
        wf = self.workflow(1)
        rows = [{'site': 'blocks.' + str(i) + '.attn', 'token_type': 'cls',
                 **{'metric_' + str(j): (-i if i % 2 else i) + j for j in range(24)}} for i in range(500)]
        spec, evidence = self.measurement(wf, 'LARGE', observations={'rows': rows,
            'limitations': 'fixture: opposite signs across sites; descriptive statistics only'})
        h = self.subject(wf, evidence)
        task, context = self.begin(wf, self.task(wf, refs=[h['id']], round=1, hypothesis_ref=h['id']))
        payload = self.assert_measurement(context, spec, evidence)
        self.assertLess(len(canonical(payload['observations'])), 12000)
        self.assertIn('display_coverage', payload['observations']['rows'])
        self.assertIn('opposite signs', canonical(payload['observations']))
        wf = Workflow(wf.run_dir)
        task = wf.queue.get(task['task_id'])
        exact = self.read(wf, task, 'read_card', ref=evidence['id'], field='/payload/observations/rows',
                          select={'site': 'blocks.499.attn', 'token_type': 'cls', 'metric': 'metric_23'})
        self.assertEqual(exact['entries'][0]['value'], rows[499]['metric_23'])
        self.assertEqual(wf.journal.card(evidence['id'])['payload']['observations']['rows'], rows)
        wf.submit(task, review('screening', eligible=True, hard_rejection_reasons=[]), 'SCRIPTED')

    def test_bound_feedback_brings_own_evidence_without_unrolling_prior_trials(self):
        wf = self.workflow(1)
        _, historical = self.measurement(wf, 'HISTORICAL')
        spec, evidence = self.measurement(wf, 'LATEST', prior=historical)
        h = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        feedback = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('interpretation', evidence_refs=[evidence['id']]), refs=[h['id'], evidence['id']])
        wf.journal.commit('feedback', cards=[h, feedback])
        _, context = self.begin(wf, self.task(wf, refs=[h['id']], round=1, hypothesis_ref=h['id']))
        self.assert_measurement(context, spec, evidence)
        self.assertNotIn(historical['id'], [c['id'] for c in context['memory']['cards']])

    def test_free_research_extensions_and_malformed_optional_sources_do_not_break_delivery(self):
        wf = self.workflow(1)
        spec, evidence = self.measurement(wf, 'VALID_SOURCE')
        _, unused = self.measurement(wf, 'NOT_DECLARED_SUPPORT')
        h = make_card('HypothesisCard', wf.journal.run_id, {
            **hypothesis(1, 'H1'), 'execution': 'A scientific explanation, not an executable contract',
            'evidence_refs': [evidence['id']]}, refs=[evidence['id']])
        decision = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('interpretation', execution={'trial_evidence_refs': [unused['id']]},
                   evidence_refs=[evidence['id']]), refs=[h['id'], evidence['id']])
        payload = experiment_payload()
        payload.update(sources={'ref': unused['id']}, original_request={
            'evidence_refs': [None, {}, 'missing-source'],
            'sources': [0, {'ref': []}, {'ref': evidence['id']}]})
        payload['execution'].update(trial_evidence_refs={'ref': unused['id']},
                                   trial_bindings='not a binding array',
                                   profile_evidence_bindings=[None, {'evidence_ref': {}}, {'evidence_ref': evidence['id']}])
        request = make_card('ExperimentCard', wf.journal.run_id, payload, refs=[h['id'], decision['id']])
        wf.journal.commit('optional-extension-shapes', cards=[h, decision, request])
        _, context = self.begin(wf, self.task(wf, 'code', refs=[request['id']], experiment_ref=request['id']))
        self.assert_measurement(context, spec, evidence)
        self.assertNotIn(unused['id'], [c['id'] for c in context['memory']['cards']])

    def test_handoff_cannot_grant_opposite_exchange_or_future_evidence_access(self):
        wf = self.workflow(1)
        own, opposite = self.debate_task(wf), self.debate_task(wf, side='B')
        secret = acl.private_result(self, wf, opposite)
        public = self.subject(wf, secret[1])
        # ACL derives only persisted ancestry, so create a real linked child.
        linked = wf.enqueue_role('linked-code', 'code', refs=[public['id']],
            context={'experiment_ref': secret[0]['id']}, parent_task=own['task_id'])
        with self.assertRaises(ContractError):
            wf.context(wf.queue.get(linked))
        running, _ = self.begin(wf, self.task(wf, 'generate', round=1, slot='H1', hypothesis_type='existing_method'))
        _, future = self.measurement(wf, 'FUTURE')
        with self.assertRaises(ContractError):
            self.read(wf, running, 'read_card', ref=future['id'])


if __name__ == '__main__':
    unittest.main()
