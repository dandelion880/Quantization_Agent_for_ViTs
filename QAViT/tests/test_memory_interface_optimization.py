"""Hosted tool/submission compatibility, without model calls or new measurement claims."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.roles import TOOLS, submission_schema
from core.cards import make_card, validate_schema
from tests.support import hypothesis
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery


class MemoryInterfaceOptimization(TempCase):
    begin = delivery.DeliveryRegression.begin
    read = delivery.DeliveryRegression.read
    task = delivery.DeliveryRegression.task

    def test_omitted_query_through_host_and_visible_schema(self):
        wf = self.workflow(1)
        h = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/h', cards=[h])
        task, _ = self.begin(wf, self.task(wf, 'generate', round=1, slot='H1', hypothesis_type='existing_method'))
        for args in ({}, {'query': {}}, {'query': {'text': ''}}):
            validate_schema(args, TOOLS['retrieve_memory']['parameters'], 'arguments')
            result = self.read(wf, task, 'retrieve_memory', **args)
            row = next(r for r in result['entries'] if r['ref'] == h['id'])
            self.assertEqual(row['body']['mechanism'], h['payload']['mechanism'])
        result = self.read(wf, task, 'retrieve_memory', query={'layer': 'L1'})
        self.assertTrue(result['current_research_state']['progress']['recent_hypotheses'])

    def test_old_read_declaration_does_not_resolve_untrusted_refs_or_reject_analysis(self):
        wf = self.workflow(1)
        h = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/h', cards=[h])
        task, _ = self.begin(wf, wf.queue.get(wf.enqueue_recorder(1, incomplete=True)))
        content = {'scientific_analysis': 'Fixture only; no experimental conclusion.',
                   'reviewed_refs': ['C999999', {'unexpected': 'old declaration'}]}
        wf.submit(task, content, 'SCRIPTED')
        card = wf.result_card(wf.queue.get(task['task_id']))
        self.assertEqual(card['payload']['scientific_analysis'], content['scientific_analysis'])
        self.assertNotIn('reviewed_refs', card['payload'])
        self.assertNotIn('next_questions', card['payload'])
        self.assertEqual(card['payload']['source_refs'], [h['id']])

    def test_optional_delta_fields_not_required_in_model_schema(self):
        schema = submission_schema('record')
        self.assertEqual(schema['required'], ['scientific_analysis'])
        for field in ('resolved_questions', 'abandoned_questions', 'next_questions'):
            self.assertIn(field, schema['properties'])
        self.assertNotIn('reviewed_refs', schema['properties'])

    def test_automatic_directory_is_small_but_explicit_browse_and_sources_remain_available(self):
        wf = self.workflow(1)
        cards = [make_card('HypothesisCard', wf.journal.run_id,
                          {**hypothesis(1, 'H1'), 'title': 'Fixture ' + str(i)}) for i in range(8)]
        wf.journal.commit('fixture/history', cards=cards)
        task, context = self.begin(wf, self.task(wf, 'generate', round=1, slot='H1', hypothesis_type='existing_method'))
        page = context['memory']['directory']
        self.assertEqual(len(page['entries']), 4)
        self.assertEqual(page['next_offset'], 4)
        found = self.read(wf, task, 'retrieve_memory', query={'kind': 'HypothesisCard'})
        self.assertEqual({r['ref'] for r in found['entries']}, {c['id'] for c in cards})
        self.assertNotIn('method_documents', found)
        self.assertTrue(all(r['body']['mechanism'] for r in found['entries']))
