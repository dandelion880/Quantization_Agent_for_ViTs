"""Bounded reusable memory and incremental research state, scripted CPU fixtures only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, make_card
from core.memory import MemoryRetriever, ResearchMemory, short_field
from tests.test_infrastructure import TempCase
from tests.support import experiment_payload, meta


class MemoryStateOptimization(TempCase):
    def evidence(self, journal, label, bits=4):
        spec = make_card('ExperimentCard', journal.run_id, {**experiment_payload(), 'question': 'Compare ' + label})
        evidence = make_card('EvidenceCard', journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': label, 'execution_id': label,
            'status': 'completed', 'observations': {'relative_mse': bits / 100, 'warning': 'synthetic feature result only'},
            'scope': {'schema_version': 'qavit.evidence_scope/1', 'model': 'cpu_fixture',
                      'measurement_kind': 'fixture', 'parameters': {'bits': bits}, 'data': {'count': 4}},
            'provenance': {}, 'resources': {}, 'artifacts': [], 'measurement_source': 'synthetic_computation'}, refs=[spec['id']])
        journal.commit('evidence/' + label, cards=[spec, evidence])
        return evidence

    def publish(self, journal, content):
        memory = ResearchMemory(journal)
        payload = memory.prepare_metareview(content, memory.recording_batch(), content.get('round', 1))
        card = make_card('MetareviewCard', journal.run_id, payload, author='recorder', refs=payload['evidence_refs'])
        journal.commit('review/' + str(journal.sequence), cards=[card], events=[memory.apply_metareview(card,
            evidence_refs=payload['evidence_refs'])])
        return card

    def test_default_retrieval_delivers_results_conditions_and_direct_documents(self):
        journal = self.journal(); evidence = self.evidence(journal, 'one')
        retrieved = MemoryRetriever(journal).retrieve('generation', query={})
        row = next(row for row in retrieved['entries'] if row['kind'] == 'EvidenceCard')
        self.assertEqual(row['body']['observations']['relative_mse'], .04)
        self.assertEqual(row['scope'], evidence['payload']['scope'])
        self.assertIn('warning', row['body']['observations'])
        document = retrieved['method_documents']['entries'][0]['read_asset']['asset']
        self.assertTrue(Path(document).is_file())
        self.assertEqual(MemoryRetriever(journal).retrieve('generation', query={'parameters.bits': 7})['entries'], [])
        self.assertTrue(MemoryRetriever(journal).retrieve('generation')['entries'])

    def test_journal_facts_and_questions_survive_selective_or_omitted_updates(self):
        journal = self.journal(); a = self.evidence(journal, 'a'); b = self.evidence(journal, 'b')
        content = meta(1); content['next_questions'] = ['why A?', 'why B?']; content['evidence_refs'] = [a['id']]
        self.publish(journal, content)
        state = ResearchMemory(journal).snapshot()['L1']
        self.assertEqual(set(state['recent_evidence']), {a['id'], b['id']})
        content = meta(1); content.pop('next_questions'); content['reviewed_refs'] = ['made-up declaration']
        card = self.publish(journal, content)
        self.assertNotIn('reviewed_refs', card['payload'])
        self.assertNotIn('next_questions', card['payload'])
        self.assertEqual(ResearchMemory(journal).snapshot()['L1']['open_questions'], ['why A?', 'why B?'])
        content = meta(1); content.update(next_questions=['why C?'], resolved_questions=['why A?'], abandoned_questions=['why B?'])
        self.publish(journal, content)
        self.assertEqual(ResearchMemory(journal).snapshot()['L1']['open_questions'], ['why C?'])
        context = ResearchMemory(journal).current_research_state()
        self.assertEqual(context['progress']['recent_evidence'], 2)
        hidden = ResearchMemory(journal).current_research_state(visible=lambda card: card['kind'] != 'MetareviewCard')
        self.assertEqual(hidden['open_questions'], [])

    def test_comparison_preserves_each_scope_and_distinguishes_replication(self):
        journal = self.journal(); a = self.evidence(journal, 'arm4', 4); b = self.evidence(journal, 'arm8', 8)
        claim = {'claim_id': 'comparison', 'text': 'Fixture arm difference', 'status': 'conditional',
                 'evidence_refs': [a['id'], b['id']], 'rationale': 'Different bits, small synthetic sample; not independent replication.'}
        content = meta(1); content['knowledge_updates'] = [claim]
        card = self.publish(journal, content)
        accepted = card['payload']['knowledge_updates'][0]
        self.assertEqual({arm['evidence_ref']: arm['scope'] for arm in accepted['scope']['comparison']},
                         {e['id']: e['payload']['scope'] for e in (a, b)})
        found = MemoryRetriever(journal).retrieve('generation', query={'layer': 'L2', 'parameters.bits': 4})['entries'][0]
        self.assertEqual([arm['matches_requested_conditions'] for arm in found['comparison']].count(True), 1)
        self.assertIn('parameters.bits', found['different_conditions'])
        self.assertIn('small synthetic', found['body']['rationale'])
        content['knowledge_updates'][0].update(status='replicated', replication_analysis='Two executions alone')
        pending = self.publish(journal, content)['payload']['pending_knowledge_updates'][0]
        self.assertIn('不等于独立重复', pending['issue'])

    def test_pending_correction_and_abandonment_keep_history_and_clear_todo(self):
        journal = self.journal(); evidence = self.evidence(journal, 'real-fixture')
        claim = {'claim_id': 'draft', 'text': 'Fixture observation', 'status': 'observed_once',
                 'evidence_refs': [], 'scope': {'model': 'cpu_fixture'}, 'rationale': 'No supporting source yet'}
        content = meta(1); content['knowledge_updates'] = [claim]
        draft = self.publish(journal, content)
        self.assertEqual(ResearchMemory(journal).recording_status()['pending_knowledge_count'], 1)
        content['knowledge_updates'][0]['evidence_refs'] = [evidence['id']]
        fixed = self.publish(journal, content)
        self.assertEqual(fixed['payload']['knowledge_updates'][0]['supersedes'], draft['id'])
        self.assertEqual(ResearchMemory(journal).recording_status()['pending_knowledge_count'], 0)
        self.assertTrue(journal.card(draft['id'])['payload']['pending_knowledge_updates'])
        claim = {**claim, 'claim_id': 'discard', 'text': 'Unsupported second idea', 'evidence_refs': []}
        content['knowledge_updates'] = [claim]; rejected = self.publish(journal, content)
        content['knowledge_updates'] = [{'claim_id': 'discard', 'status': 'abandoned', 'rationale': 'No longer relevant'}]
        abandoned = self.publish(journal, content)
        self.assertEqual(abandoned['payload']['knowledge_updates'][0]['status'], 'abandoned')
        self.assertEqual(ResearchMemory(journal).recording_status()['pending_knowledge_count'], 0)
        self.assertTrue(journal.card(rejected['id'])['payload']['pending_knowledge_updates'])
        self.assertNotIn('discard', [row['claim_id'] for row in MemoryRetriever(journal).retrieve('generation', query={'layer': 'L2'})['entries']])

    def test_malformed_optional_knowledge_does_not_lose_valid_analysis(self):
        journal = self.journal(); a = self.evidence(journal, 'a', 4); b = self.evidence(journal, 'b', 8)
        for scope in ('bad', {'comparison': [None]}, {'comparison': 'bad'}):
            content = meta(1); content['knowledge_updates'] = [{'text': 'Malformed scope fixture', 'status': 'conditional',
                'evidence_refs': [a['id'], b['id']], 'scope': scope, 'rationale': 'fixture'}]
            card = self.publish(journal, content)
            self.assertEqual(card['payload']['scientific_analysis'], content['scientific_analysis'])
            self.assertEqual(len(card['payload']['pending_knowledge_updates']), 1)

    def test_current_questions_are_paged_and_hidden_review_cannot_close_public_question(self):
        journal = self.journal()
        content = meta(1); content['next_questions'] = ['public question ' + str(i) for i in range(12)]
        first = self.publish(journal, content)
        content['next_questions'] = ['private question']; content['resolved_questions'] = ['public question 0']
        hidden = self.publish(journal, content)
        visible = lambda card: card['id'] != hidden['id']
        state = ResearchMemory(journal).current_research_state(visible=visible)
        self.assertEqual(state['open_question_count'], 12)
        self.assertIn('public question 0', state['open_questions'])
        self.assertNotIn('private question', canonical(state))
        self.assertEqual(state['latest_review']['ref'], first['id'])
        self.assertEqual(len(state['open_questions']), 8)
        second = MemoryRetriever(journal).retrieve('generation', visible=visible, **state['more_questions'])
        self.assertEqual(len(second['current_research_state']['open_questions']), 4)

    def test_pending_pages_and_repeated_failed_revision_keep_one_identity(self):
        journal = self.journal()
        content = meta(1); content['knowledge_updates'] = [{'claim_id': 'pending-' + str(i),
            'text': 'Incomplete fixture ' + str(i), 'status': 'observed_once', 'scope': {'model': 'fixture'},
            'rationale': 'No measurement yet'} for i in range(10)]
        first = self.publish(journal, content)
        status = ResearchMemory(journal).recording_status()
        self.assertEqual((status['pending_knowledge_count'], len(status['pending_knowledge_updates'])), (10, 8))
        second = MemoryRetriever(journal).retrieve('generation', **status['pending_knowledge_more'])
        self.assertEqual(len(second['pending_knowledge_updates']), 2)
        content['knowledge_updates'] = [{**content['knowledge_updates'][0], 'scope': 'still invalid'}]
        content['knowledge_updates'][0].pop('claim_id')
        fixed = self.publish(journal, content)
        self.assertEqual(fixed['payload']['pending_knowledge_updates'][0]['claim_id'], 'pending-0')
        self.assertEqual(ResearchMemory(journal).recording_status()['pending_knowledge_count'], 10)
        self.assertEqual(ResearchMemory(journal).recording_status(visible=lambda card: card['id'] not in {first['id'], fixed['id']})['pending_knowledge_count'], 0)

    def test_malformed_pending_can_be_abandoned_by_identity_without_rewriting_science(self):
        journal = self.journal()
        for malformed in ('unstructured old draft', None, {}, {'text': 42, 'scope': 'invalid'}):
            content = meta(1); content['knowledge_updates'] = malformed
            original = self.publish(journal, content)
            identity = original['payload']['pending_knowledge_updates'][0]['claim_id']
            content['knowledge_updates'] = [{'claim_id': identity, 'status': 'abandoned'}]
            result = self.publish(journal, content)['payload']['knowledge_updates'][0]
            self.assertEqual(result['supersedes'], original['id'])
            self.assertEqual(result['scope'], {'pending_draft': original['id']})
            self.assertEqual(result['text'], malformed if isinstance(malformed, str) else 'Abandoned pending draft')
            self.assertEqual(ResearchMemory(journal).recording_status()['pending_knowledge_count'], 0)
            self.assertEqual(journal.card(original['id'])['payload']['pending_knowledge_updates'][0]['submission'], malformed)

    def test_compact_state_avoids_envelopes_keeps_conditions_results_and_question_paging(self):
        journal = self.journal(); evidence = self.evidence(journal, 'compact')
        content = meta(1); content['next_questions'] = ['question ' + str(i) for i in range(8)]
        self.publish(journal, content)
        memory = ResearchMemory(journal)
        compact = memory.current_research_state(compact=True)
        full = memory.current_research_state()
        self.assertLess(len(canonical(compact)), len(canonical(full)))
        self.assertEqual(len(compact['open_questions']), 4)
        finding = compact['latest_findings'][0]
        self.assertEqual(finding['actual_conditions']['parameters'], {'bits': 4})
        self.assertEqual(finding['observations']['relative_mse'], .04)
        self.assertIn('warning', finding['observations'])
        self.assertNotIn('full_card_asset', finding)
        self.assertNotIn('observations', memory.current_research_state(compact=True,
            excluded_refs=[evidence['id']])['latest_findings'][0])
        self.assertEqual(finding['read_card'], {'ref': evidence['id']})

    def test_short_prose_never_splits_decimal_or_path_at_period(self):
        for content in ('value=0.123456 and more details', '/tmp/run.result.json and more details'):
            preview = short_field(content, 'ref', '/payload', 10)
            self.assertNotIn('excerpt', preview)


if __name__ == '__main__': unittest.main()
