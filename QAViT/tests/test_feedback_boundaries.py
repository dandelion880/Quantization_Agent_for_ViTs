"""CPU request-chain regression: current feedback subjects versus historical provenance."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, make_card
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.support import hypothesis, review, meta


class FeedbackBoundariesRegression(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    read = delivery.DeliveryRegression.read
    index = delivery.DeliveryRegression.index
    science = delivery.DeliveryRegression.science

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU only'))
        worker.start(); self.addCleanup(worker.stop)

    def publish(self, wf, kind, body, refs=()):
        card = make_card(kind, wf.journal.run_id, body, refs=refs)
        wf.journal.commit('fixture/' + card['id'], cards=[card])
        return card

    def test_old_exchange_whole_body_stays_optional_but_old_limitation_survives(self):
        wf = self.workflow(1)
        chosen = self.publish(wf, 'HypothesisCard', hypothesis(1, 'H1'))
        old = self.publish(wf, 'ReviewDecisionCard', review('position', exchange=1,
            rationale='OLD_RATIONALE_CONDITION', uncertainties=['OLD_EXPLICIT_LIMITATION'],
            arguments=['OPTIONAL_OLD_ARGUMENT_BODY'], objections=['OLD_OBJECTION'],
            responses_to_prior=['OLD_RESPONSE']), [chosen['id']])
        task, context = self.begin(wf, self.task(wf, 'position', refs=[chosen['id'], old['id']],
            round=1, pair=['H1', 'H2'], exchange=4, side='A', position_for='H1'))
        self.assertNotIn(old['id'], context['memory']['read_manifest']['refs'])
        self.assertFalse(any(r['ref'] == old['id'] and r['field'] == '/payload'
                             for r in context['memory']['required_reads']))
        self.assertNotIn('OPTIONAL_OLD_ARGUMENT_BODY', canonical(context))
        for marker in ('OLD_RATIONALE_CONDITION', 'OLD_EXPLICIT_LIMITATION', 'OLD_OBJECTION', 'OLD_RESPONSE'):
            self.assertIn(marker, canonical(context))
        self.assertIn(old['id'], self.index(wf, task))
        task, context = self.tools(wf, task, [('read_card', {'ref': old['id']})])
        self.assertIn('OPTIONAL_OLD_ARGUMENT_BODY', canonical(context))

    def test_shared_evidence_does_not_bind_other_hypothesis_review(self):
        wf = self.workflow(1)
        _, ev, chosen, current = self.science(wf, 'shared')
        other = self.publish(wf, 'HypothesisCard', hypothesis(1, 'H2'), [ev['id']])
        wrong = self.publish(wf, 'ReviewDecisionCard', review('interpretation',
            rationale='OTHER_HYPOTHESIS_REVIEW_' + 'x' * 9000), [other['id'], ev['id']])
        task, context = self.begin(wf, self.task(wf, refs=[chosen['id'], ev['id']],
            round=1, hypothesis_ref=chosen['id']))
        self.assertFalse(any(r['ref'] == wrong['id'] for r in context['memory']['required_reads']))
        self.assertNotIn(wrong['id'], context['memory']['read_manifest']['refs'])
        self.assertIn(current['id'], context['memory']['read_manifest']['refs'])
        self.assertIn(wrong['id'], self.index(wf, task))
        self.assertIn('OTHER_HYPOTHESIS_REVIEW_', canonical(self.read(wf, task, 'read_card', ref=wrong['id'])))

    def test_referenced_other_hypothesis_is_provenance_not_current_subject(self):
        wf = self.workflow(1)
        old = self.publish(wf, 'HypothesisCard', hypothesis(1, 'H2'))
        chosen = self.publish(wf, 'HypothesisCard', hypothesis(1, 'H1'), [old['id']])
        unrelated = self.publish(wf, 'ReviewDecisionCard', review('interpretation',
            rationale='PROVENANCE_ONLY_REVIEW'), [old['id']])
        _, context = self.begin(wf, self.task(wf, refs=[chosen['id']], round=1, hypothesis_ref=chosen['id']))
        self.assertNotIn('PROVENANCE_ONLY_REVIEW', canonical(context))
        self.assertNotIn(unrelated['id'], context['memory']['read_manifest']['refs'])

    def test_ambiguous_evidence_only_review_is_retained_conservatively(self):
        wf = self.workflow(1)
        _, ev, chosen, _ = self.science(wf, 'ambiguous')
        ambiguous = self.publish(wf, 'ReviewDecisionCard', review('interpretation',
            rationale='AMBIGUOUS_MEASUREMENT_CONDITION'), [ev['id']])
        _, context = self.begin(wf, self.task(wf, refs=[chosen['id'], ev['id']], round=1, hypothesis_ref=chosen['id']))
        self.assertIn(ambiguous['id'], context['memory']['read_manifest']['refs'])
        self.assertIn('ambiguous_subject', canonical(context['memory'].get('feedback_bindings', [])))

    def test_old_long_limitation_remains_required_without_requiring_old_card(self):
        wf = self.workflow(1)
        chosen = self.publish(wf, 'HypothesisCard', hypothesis(1, 'H1'))
        old = self.publish(wf, 'ReviewDecisionCard', review('verdict', exchange=1,
            next_questions=['KNOWN_LONG_LIMITATION_' + 'x' * 18000]), [chosen['id']])
        _, context = self.begin(wf, self.task(wf, 'arbitrate', refs=[chosen['id'], old['id']],
            round=1, pair=['H1', 'H2'], exchange=4))
        required = [r['field'] for r in context['memory']['required_reads'] if r['ref'] == old['id']]
        self.assertIn('/payload/next_questions', required)
        self.assertNotIn('/payload', required)

    def test_explicit_assignments_apply_to_record_and_supervise(self):
        wf = self.workflow(1)
        card = self.publish(wf, 'MetareviewCard', {**meta(1), 'generation_guidance': 'EXPLICIT_ASSIGNED_BOUNDARY'})
        for purpose in ('record', 'supervise'):
            extra = {'recording_batch': wf.memory.recording_batch()} if purpose == 'record' else {}
            task = self.task(wf, purpose, feedback_refs=[card['id']], **extra)
            context = wf.request_body(task)['spec']['context']
            self.assertIn('EXPLICIT_ASSIGNED_BOUNDARY', canonical(context['memory'].get('bound_feedback', [])))

    def test_explicit_experiment_condition_is_bound_without_a_hypothesis_subject(self):
        wf = self.workflow(1)
        spec, _, _, interpretation = self.science(wf, 'explicit-experiment')
        task = self.task(wf, 'interpret', refs=[spec['id']], experiment_ref=spec['id'])
        _, context = self.begin(wf, task)
        self.assertIn(interpretation['id'], context['memory']['read_manifest']['refs'])

    def test_recent_exchange_and_explicit_old_assignment_remain_required(self):
        wf = self.workflow(1)
        chosen = self.publish(wf, 'HypothesisCard', hypothesis(1, 'H1'))
        old = self.publish(wf, 'ReviewDecisionCard', review('verdict', exchange=1,
            rationale='ASSIGNED_OLD_VERDICT', extended_analysis='ASSIGNED_LONG_' + 'x' * 18000), [chosen['id']])
        recent = self.publish(wf, 'ReviewDecisionCard', review('verdict', exchange=3,
            rationale='CURRENT_RECENT_DISPUTE'), [chosen['id']])
        task = self.task(wf, 'arbitrate', refs=[chosen['id'], old['id'], recent['id']],
            round=1, pair=['H1', 'H2'], exchange=4, feedback_refs=[old['id']])
        _, context = self.begin(wf, task)
        self.assertIn(recent['id'], context['memory']['read_manifest']['refs'])
        self.assertTrue(any(r['ref'] == old['id'] and r['field'] == '/payload'
                            for r in context['memory']['required_reads']))
