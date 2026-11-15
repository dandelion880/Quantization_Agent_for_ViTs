"""Shared evidence views retain arm ownership, immutable provenance and question closure."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, fingerprint
from core.evidence import collection_summary, evidence_facts, project_observations, selected_leaves
from core.memory import ResearchMemory, question_id
from tests import test_memory_state_optimization as memory_tests
from tests.test_infrastructure import TempCase
from tests.support import meta
from tools.evaluation.judge import project_sources


class EvidenceDelivery(unittest.TestCase):
    def card(self):
        return {'id': 'Evidence@1', 'content_sha256': 'immutable', 'payload': {
            'status': 'completed', 'measurement_source': 'real', 'execution_id': 'measurement-1',
            'experiment_ref': 'Experiment@1', 'provenance': {'protocol_hash': 'protocol', 'model_ref': 'frozen-model'},
            'artifacts': [], 'observations': {'evaluations': {'segmentation': {
                'split': 'ade_validation', 'sample_count': 2, 'sample_ids': ['a', 'b'],
                'full_dataset': False, 'metrics': {'mIoU': 4., 'mAcc': 7., 'aAcc': 34.}}}}}}

    def test_historical_fp_is_hash_bound_scope_checked_and_never_written_back(self):
        with tempfile.TemporaryDirectory() as root:
            card = self.card()
            path = Path(root) / 'segmentation.json'
            raw = {'sample_ids': ['a', 'b'], 'metrics': {
                'fp': {'mIoU': 39., 'mAcc': 50., 'aAcc': 78.},
                'quantized': {'mIoU': 4., 'mAcc': 7., 'aAcc': 34.}}}
            path.write_text(json.dumps(raw))
            card['payload']['artifacts'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
            original = deepcopy(card)
            result = evidence_facts(card, root)
            pair = result['evaluations']['segmentation']['paired_metrics']['mIoU']
            self.assertEqual((pair['FP'], pair['Q'], pair['delta_Q_minus_FP']), (39., 4., -35.))
            self.assertEqual(pair['delta_unit'], 'percentage_points')
            self.assertEqual(card, original)
            path.write_text('{}')
            denied = evidence_facts(card, root)
            self.assertIsNone(denied['evaluations']['segmentation']['paired_metrics']['mIoU']['FP'])
            self.assertIn('hash mismatch', denied['source_issues'][0]['reason'])
            raw['sample_ids'] = ['a', 'different']
            path.write_text(json.dumps(raw))
            card['payload']['artifacts'][0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn('scope/metric mismatch', evidence_facts(card, root)['source_issues'][0]['reason'])

    def test_missing_fp_and_failure_are_explicit_not_fabricated(self):
        card = self.card(); card['payload'].update(status='failed', error='partial worker failure')
        result = evidence_facts(card)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error'], 'partial worker failure')
        self.assertEqual(result['identity']['model_ref'], 'frozen-model')
        self.assertEqual(result['evaluations']['segmentation']['paired_metrics']['mIoU']['pair_status'], 'missing_FP_or_Q')

    def test_arm_rows_never_share_descriptive_means(self):
        values = [{'arm': arm, 'variant': variant, 'loss': loss}
                  for arm, variant, loss in [('H2', 'pv', 2), ('H2', 'mass', -7), ('seed', 'original', 10)]]
        view = collection_summary(values, 'ref', '/payload/observations', max_characters=10000)
        groups = [v for k, v in view['numeric_summary'].items() if k.endswith('/loss')]
        self.assertEqual(sorted(v['mean'] for v in groups), [-7, 2, 10])
        self.assertTrue(all(v['count'] == 1 for v in groups))

    def test_arm_identity_survives_structured_labels_and_exact_leaf_selection(self):
        values = [{'arm_id': ident, 'variant_id': {'rule': variant}, 'loss': loss}
                  for ident, variant, loss in [('H2', 'pv', 2), ('H2', 'mass', -7)]]
        view = collection_summary(values, 'ref', '/payload/observations', max_characters=10000)
        groups = [v for k, v in view['numeric_summary'].items() if k.endswith('/loss')]
        self.assertEqual(sorted(v['mean'] for v in groups), [-7, 2])
        selected = list(selected_leaves(values, '/payload/observations', {'metric': 'loss', 'text': 'H2'}))
        self.assertEqual([row['labels']['arm_id'] for row in selected], ['H2', 'H2'])
        self.assertEqual(selected[0]['labels']['variant_id']['sha256'], fingerprint({'rule': 'pv'}))

    def test_explicit_arm_containers_keep_unlabelled_rows_and_numeric_style_keys_separate(self):
        for values in ([{'loss': 2}, {'loss': -7}],
                       {'block.0': {'loss': 2}, 'block.1': {'loss': -7}}):
            with self.subTest(container=type(values).__name__):
                original = deepcopy(values)
                view = collection_summary(values, 'ref', '/payload/observations/arms', max_characters=10000)
                groups = [v for k, v in view['numeric_summary'].items() if k.endswith('/loss')]
                self.assertEqual(sorted(v['mean'] for v in groups), [-7, 2])
                self.assertTrue(all(v['count'] == 1 for v in groups))
                self.assertEqual(values, original)

    def test_key_facts_keep_list_arm_failure_and_dictionary_arm_identity(self):
        card = self.card()
        rows = [{'arm_id': 'pv', 'status': 'failed', 'error': 'actual failure'},
                {'arm_id': 'seed', 'metrics': {'top1': 80}}]
        card['payload']['observations']['arms'] = rows
        facts = evidence_facts(card)
        self.assertEqual(facts['arm_results'], rows)
        self.assertEqual(facts['arm_source']['read_card']['field'], '/payload/observations/arms')
        card['payload']['observations']['arms'] = {'seed': rows[1]}
        self.assertEqual(evidence_facts(card)['arm_results']['seed']['arm_id'], 'seed')

    def test_list_identity_compaction_keeps_small_results_and_accurate_pointer(self):
        ids = ['image-' + str(i) for i in range(5000)]
        value = {'sample_ids': ids, 'metrics': {'baseline_top1': 90, 'top1': 80}, 'counterevidence': 'accuracy decreased'}
        view = project_observations(value, 'ref', '/payload/observations/arm')
        self.assertEqual(view['metrics'], value['metrics'])
        self.assertEqual(view['counterevidence'], value['counterevidence'])
        self.assertEqual(view['sample_ids']['sha256'], fingerprint(ids))
        self.assertEqual(view['sample_ids']['read_card']['field'], '/payload/observations/arm/sample_ids')
        self.assertLess(len(canonical(view)), 1000)

    def test_offline_exact_dedup_does_not_merge_arms_or_change_sources(self):
        value = {'arms': {'seed': {'top1': 89}, 'pv': {'top1': 87}},
                 'sample_ids': ['sample-' + str(i) for i in range(2000)], 'counterevidence': 'PV lost two points'}
        sources = [{'source_id': str(i), 'locator': '#/context', 'content': deepcopy(value)} for i in range(2)]
        original = deepcopy(sources)
        projected = project_sources(sources)
        self.assertEqual(sources, original)
        self.assertEqual(projected[0]['content']['arms'], value['arms'])
        self.assertEqual(projected[0]['content']['counterevidence'], value['counterevidence'])
        self.assertEqual(projected[1]['content']['same_content_as'], {'source_id': '0', 'locator': '#/context'})
        self.assertEqual(projected[0]['content']['sample_ids']['items'], 2000)


class FrozenTableDelivery(TempCase):
    def test_large_frozen_tables_and_selection_ids_have_exact_independent_sources(self):
        from tests import test_memory_delivery as fixtures
        from core.cards import make_card
        wf = self.workflow(1)
        cards = fixtures.DeliveryRegression.science(self, wf, 'frozen-tables', commit=False)
        payload = deepcopy(cards[1]['payload'])
        table = {f'blocks.{i}.attn': {'activation': {'bits': 4, 'format': 'UINT4', 'axis': -1},
                                   'weight': {'bits': 4, 'format': 'INT4', 'axis': 0}} for i in range(97)}
        ids = ['sample-image-' + str(i) for i in range(64)]
        payload['scope'] = {'model': 'fixture', 'weight_bits': [4], 'activation_bits': [4], 'quantization': table}
        payload['provenance'] = {'execution_facts': {'quantization': deepcopy(table)}, 'sample_selection': {'search': ids}}
        evidence = make_card('EvidenceCard', wf.journal.run_id, payload, refs=[cards[0]['id']])
        wf.journal.commit('frozen-view', cards=[cards[0], evidence])
        view = wf.present_card(evidence)['payload']
        for container, field in ((view['scope'], '/payload/scope'),
                                 (view['provenance']['execution_facts'], '/payload/provenance/execution_facts')):
            projected = container['quantization']
            self.assertEqual(projected['items'], 97)
            self.assertEqual(projected['sha256'], fingerprint(table))
            self.assertEqual(projected['read_card']['field'], field + '/quantization')
            self.assertEqual(wf.card_field(evidence, projected['read_card']['field']), table)
        selected = view['provenance']['sample_selection']['search']
        self.assertEqual(selected['items'], 64)
        self.assertEqual(wf.card_field(evidence, selected['read_card']['field']), ids)
        self.assertEqual(wf.journal.card(evidence['id'])['payload'], payload)
        self.assertLess(len(canonical(view)), 5000)


class QuestionClosure(TempCase):
    publish = memory_tests.MemoryStateOptimization.publish

    def test_stable_id_closes_long_question_and_replay_is_idempotent(self):
        journal = self.journal()
        question = 'prefix interaction? ' + 'detail ' * 200
        content = meta(1); content['next_questions'] = [question, 'unfinished H3']
        original = self.publish(journal, content)
        memory = ResearchMemory(journal)
        record = memory.current_research_state()['question_records'][0]
        self.assertEqual(record['question_id'], question_id(question))
        content = meta(1); content['next_questions'] = []; content['resolved_question_refs'] = [record['question_id']]
        self.publish(journal, content)
        self.publish(journal, content)
        self.assertEqual(ResearchMemory(journal).snapshot()['L1']['open_questions'], ['unfinished H3'])
        self.assertEqual(ResearchMemory(journal).current_research_state()['open_questions'], ['unfinished H3'])
        self.assertEqual(journal.card(original['id'])['payload']['next_questions'][0], question)
        content['resolved_question_refs'] = ['question-invented']
        with self.assertRaises(ContractError):
            self.publish(journal, content)


if __name__ == '__main__':
    unittest.main()
