"""CPU-only feedback binding and L2 lexical retrieval regression."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, make_card
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.support import hypothesis, review, meta, experiment_payload


class FeedbackDeliveryRegression(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    read = delivery.DeliveryRegression.read
    tools = delivery.DeliveryRegression.tools

    def publish_meta(self, wf, body, refs=()):
        card = make_card('MetareviewCard', wf.journal.run_id, body, author='recorder', refs=refs)
        event = wf.memory.apply_metareview(card)
        wf.journal.commit('meta/' + card['id'], cards=[card], events=[event])
        return card

    def chosen(self, wf):
        card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('chosen', cards=[card])
        return card

    def test_l2_lexical_order_changes_and_pagination_is_stable_with_filters(self):
        wf = self.workflow(1)
        value = meta(1)
        value['knowledge_updates'] = [{'claim_id': word, 'text': word + ' mechanism', 'status': 'hypothesis',
                'scope': {'model': model}, 'evidence_refs': [], 'rationale': 'CPU hypothetical retrieval fixture'}
                for word, model in [('alpha', 'm'), ('beta', 'm'), ('beta_other', 'other')]]
        card = self.publish_meta(wf, value)
        def query(text, **kwargs):
            return wf.retriever.retrieve('generation', query={'layer':'L2','text':text,'model':'m'}, **kwargs)
        self.assertEqual(query('beta')['entries'][0]['claim_id'], 'beta')
        self.assertEqual(query('alpha')['entries'][0]['claim_id'], 'alpha')
        pages = [query('beta', limit=1, offset=i)['entries'][0]['claim_id'] for i in range(2)]
        self.assertEqual(pages, [x['claim_id'] for x in query('beta')['entries']])
        self.assertEqual(query('beta', visible=lambda c: False)['total'], 0)
        self.assertEqual(query('beta', after_version=wf.journal.card_versions[card['id']])['total'], 0)
        self.assertEqual(query('beta', sequence=1)['total'], 0)

    def test_unrelated_long_global_feedback_is_optional(self):
        wf = self.workflow(1); chosen = self.chosen(wf)
        value = meta(1); value['next_questions'] = ['UNRELATED_GLOBAL_' + 'x' * 19000]
        latest = self.publish_meta(wf, value)
        _, context = self.begin(wf, self.task(wf, refs=[chosen['id']], round=2, hypothesis_ref=chosen['id']))
        self.assertFalse(any(r['ref'] == latest['id'] for r in context['memory']['required_reads']))
        self.assertIn(latest['id'], canonical(context['memory']))
        self.assertNotIn('x' * 2000, canonical(context))

    def test_old_object_linked_limitation_is_delivered_beyond_global_latest(self):
        wf = self.workflow(1); chosen = self.chosen(wf)
        limitation = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('interpretation', rationale='KNOWN_OLDER_LIMITATION: applies only at one activation site'), refs=[chosen['id']])
        wf.journal.commit('old_limitation', cards=[limitation])
        unrelated_chain = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('interpretation', rationale='INDIRECT_REVIEW_NOT_AUTOMATIC'), refs=[limitation['id']])
        wf.journal.commit('indirect_review', cards=[unrelated_chain])
        for i in range(15):
            unrelated = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation', rationale='unrelated_' + str(i)))
            wf.journal.commit('unrelated/' + str(i), cards=[unrelated])
        _, context = self.begin(wf, self.task(wf, refs=[chosen['id']], round=2, hypothesis_ref=chosen['id']))
        self.assertIn('KNOWN_OLDER_LIMITATION', canonical(context))
        self.assertIn(limitation['id'], context['memory']['read_manifest']['refs'])
        self.assertNotIn('INDIRECT_REVIEW_NOT_AUTOMATIC', canonical(context))

    def test_linked_judgment_audit_is_required_but_global_long_field_is_not(self):
        wf = self.workflow(1); chosen = self.chosen(wf)
        value = meta(1, [chosen['id']]); value['next_questions'] = ['global_' + 'z' * 18000]
        value['judgment_audit'][0]['reasoning'] = 'BOUND_AUDIT_' + 'a' * 18000
        latest = self.publish_meta(wf, value, [chosen['id']])
        _, context = self.begin(wf, self.task(wf, refs=[chosen['id']], round=2, hypothesis_ref=chosen['id']))
        required = context['memory']['required_reads']
        self.assertTrue(any(r['ref'] == latest['id'] and r['field'] == '/payload/judgment_audit/0' for r in required))
        self.assertFalse(any(r['field'] == '/payload/next_questions' for r in required))

    def test_explicit_supervisor_feedback_result_is_delivered(self):
        wf = self.workflow(1); chosen = self.chosen(wf)
        note = self.task(wf, 'supervise'); wf.queue.ready(); wf.queue.claim(note['task_id'])
        wf.queue.finish(note['task_id'], result={'status_analysis': 'ASSIGNED_SUPERVISOR_BOUNDARY', 'unresolved_boundaries': ['keep the one-site scope']})
        note = wf.queue.get(note['task_id'])
        _, context = self.begin(wf, self.task(wf, refs=[chosen['id'], note['task_card']], round=2, hypothesis_ref=chosen['id']))
        self.assertIn('ASSIGNED_SUPERVISOR_BOUNDARY', canonical(context))

    def test_assigned_long_feedback_is_required_and_short_field_deduplicates(self):
        wf = self.workflow(1); chosen = self.chosen(wf)
        value = meta(1, [chosen['id']]); value['judgment_audit'][0]['reasoning'] = 'UNIQUE_SHORT_AUDIT_REASON'
        card = self.publish_meta(wf, value)
        task, context = self.begin(wf, self.task(wf, refs=[chosen['id']], round=2, hypothesis_ref=chosen['id']))
        task, context = self.tools(wf, task, [('read_card', {'ref': card['id'], 'field': '/payload/judgment_audit/0'})])
        self.assertEqual(canonical(context).count('UNIQUE_SHORT_AUDIT_REASON'), 1)
        value = meta(2); value['generation_guidance'] = 'EXPLICIT_LONG_FEEDBACK_' + 'b' * 18000
        assigned = self.publish_meta(wf, value)
        context = wf.context(self.task(wf, refs=[chosen['id']], round=3, hypothesis_ref=chosen['id'], feedback_refs=[assigned['id']]))
        self.assertTrue(any(r['ref'] == assigned['id'] and r['field'] == '/payload' for r in context['memory']['required_reads']))

    def test_conditional_claim_keeps_conditions_and_is_not_counterevidence(self):
        wf = self.workflow(1)
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        wf.journal.commit('spec', cards=[spec])
        scope = {'schema_version': 'qavit.evidence_scope/1', 'model': 'scalar_CPU_fixture',
                 'measurement_kind': 'scalar_fixture', 'data': {}}
        # Actual trivial scalar computation; no model or dataset result is claimed.
        ev = make_card('EvidenceCard', wf.journal.run_id, {'experiment_ref':spec['id'], 'status':'completed',
            'observations':{'sum':sum([1,2])}, 'execution_fingerprint':'scalar_fixture', 'execution_id':'scalar_fixture',
            'provenance':{}, 'resources':{}, 'artifacts':[], 'scope':scope, 'measurement_source':'synthetic_computation'}, refs=[spec['id']])
        wf.journal.commit('ev', cards=[ev])
        chosen = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1,'H1'), refs=[ev['id']])
        wf.journal.commit('chosen', cards=[chosen])
        value = meta(1); value['evidence_refs']=[ev['id']]
        value['knowledge_updates']=[{'claim_id':'conditional_scalar','text':'Conditional arithmetic fixture',
            'status':'conditional','scope':scope,'evidence_refs':[ev['id']], 'rationale':'ONLY_FOR_THE_DECLARED_SCALAR_INPUTS'}]
        self.publish_meta(wf,value,[ev['id']])
        _, context = self.begin(wf,self.task(wf,refs=[chosen['id']],round=2,hypothesis_ref=chosen['id']))
        self.assertIn('ONLY_FOR_THE_DECLARED_SCALAR_INPUTS', canonical(context))
        self.assertNotIn('conditional_scalar', canonical(context['memory'].get('counterevidence_reminders',{})))
