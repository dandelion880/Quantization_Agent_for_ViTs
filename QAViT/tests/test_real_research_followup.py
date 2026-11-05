"""Offline boundary regressions from real tiny-ViT traces; never launches an LLM."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, make_card, PAYLOAD_SCHEMAS
from agents.roles import role_spec
from tests.support import experiment_payload, hypothesis, meta, validation_config
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as memory_delivery


class RealResearchFollowup(TempCase):
    begin = memory_delivery.DeliveryRegression.begin
    task = memory_delivery.DeliveryRegression.task

    def setUp(self):
        super().setUp()
        block = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('offline regression'))
        block.start()
        self.addCleanup(block.stop)

    def evidence(self, wf, granularity='per_tensor', source='synthetic_computation', padding=0):
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        scope = {'schema_version': 'qavit.evidence_scope/1', 'model': 'fixture',
            'model_identity_sha256': 'fixture', 'measurement_source': source,
            'measurement_kind': 'local_activation_probe', 'site': ['fc2'], 'activation_bits': [3],
            'quantization': {'fc2': {'activation': {'bits': 3, 'granularity': granularity}, 'weight': None}},
            'data': {'calibration': {'sample_ids': ['fixture-0'], 'sample_count': 1}}}
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'fixture', 'status': 'completed',
            'execution_id': 'fixture-' + granularity, 'measurement_source': source,
            'observations': {'relative_mse': .25, 'scientific_details': 'x' * padding},
            'provenance': {}, 'resources': {}, 'artifacts': [], 'scope': scope}, refs=[spec['id']])
        wf.journal.commit('fixture/' + evidence['id'], cards=[spec, evidence])
        return evidence

    def record(self, wf, refs, **kwargs):
        tid = wf.enqueue_recorder(1, incomplete=True, **kwargs)
        task, _ = self.begin(wf, wf.queue.get(tid))
        batch = task['inputs']['context']['recording_batch']
        draft = meta(1, batch['judgment_refs'])
        draft['evidence_refs'] = refs
        draft['knowledge_updates'] = [{'claim_id': 'fixture', 'text': 'unit fixture, not a scientific conclusion',
            'status': 'conditional', 'evidence_refs': refs, 'rationale': 'unit fixture'}]
        return task, draft

    def test_probe_normalizes_explicit_activation_granularity_without_guessing(self):
        wf = self.workflow(1)
        draft = experiment_payload()
        draft['execution']['resolved_choices']['granularity'] = 'per_tensor'
        before = deepcopy(draft)
        normalized = wf.broker.preflight(draft)
        self.assertEqual(normalized['execution']['resolved_choices']['granularity'], {'activation': 'per_tensor'})
        self.assertEqual(normalized['execution']['parameters'], before['execution']['parameters'])
        self.assertEqual(draft, before)
        draft['execution']['parameters']['granularity'] = 'per_channel'
        with self.assertRaisesRegex(ContractError, '不一致'):
            wf.broker.preflight(draft)
        draft = deepcopy(before)
        draft['execution']['resolved_choices']['activation_granularity'] = 'per_channel'
        with self.assertRaises(ContractError):
            wf.broker.preflight(draft)
        draft = deepcopy(before)
        draft['execution']['service'] = 'profile'
        draft['execution'].update(tool='query_structure', arguments={})
        query = wf.broker.preflight(draft)
        self.assertNotIn('calibration', query['execution'])
        self.assertNotIn('calibration_samples', query['execution']['resolved_choices'])
        self.assertEqual(query['execution']['quantization_contract'], {})

    def test_evidence_payload_boundary_keeps_all_science_without_envelope_paging(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf, padding=6000)
        size = len(canonical(evidence['payload']))
        self.assertLess(size, wf.DIRECT_CHARACTERS)
        self.assertGreater(len(canonical(evidence)), wf.DIRECT_CHARACTERS)
        task, context = self.begin(wf, self.task(wf, refs=[evidence['id']]))
        self.assertFalse(context['memory']['required_reads'])
        shown = next(c for c in context['memory']['cards'] if c.get('id') == evidence['id'])
        self.assertEqual(shown, evidence)
        self.assertIn(evidence['id'], context['memory']['read_manifest']['refs'])

    def test_omitted_scope_copies_only_identical_verified_evidence_scopes(self):
        wf = self.workflow(1)
        evidence = [self.evidence(wf), self.evidence(wf)]
        task, draft = self.record(wf, [c['id'] for c in evidence])
        raw = deepcopy(draft)
        wf.submit(task, draft, 'unit fixture')
        card = wf.result_card(wf.queue.get(task['task_id']))
        self.assertEqual(card['payload']['knowledge_updates'][0]['scope'], evidence[0]['payload']['scope'])
        self.assertEqual(draft, raw)
        self.assertNotIn('scope', draft['knowledge_updates'][0])
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])

    def test_different_scopes_publish_comparison_with_each_actual_condition(self):
        wf = self.workflow(1)
        a, b = self.evidence(wf), self.evidence(wf, 'per_channel')
        task, draft = self.record(wf, [a['id'], b['id']])
        draft['knowledge_updates'].append({**deepcopy(draft['knowledge_updates'][0]),
            'claim_id': 'valid', 'evidence_refs': [a['id']], 'scope': {'activation_bits': [3]}})
        wf.submit(task, draft, 'unit fixture')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(payload['scientific_analysis'], draft['scientific_analysis'])
        self.assertEqual([c['claim_id'] for c in payload['knowledge_updates']], ['fixture', 'valid'])
        arms = payload['knowledge_updates'][0]['scope']['comparison']
        self.assertEqual({arm['evidence_ref']: arm['scope'] for arm in arms},
                         {a['id']: a['payload']['scope'], b['id']: b['payload']['scope']})
        self.assertNotIn('pending_knowledge_updates', payload)
        self.assertEqual(set(wf.memory.snapshot()['L2']), {'fixture', 'valid'})
        status = wf.memory.recording_status()
        self.assertEqual(status['facts'], 'archived_in_journal')
        self.assertEqual(status['scientific_review'], 'completed')
        self.assertEqual(status['knowledge'], 'no_pending_updates')
        self.assertEqual(status['pending_knowledge_count'], 0)
        retrieved = wf.retriever.retrieve('generation', query={'layer': 'L2',
            'quantization.fc2.activation.granularity': 'per_channel'})
        self.assertEqual([row['claim_id'] for row in retrieved['entries']], ['fixture'])
        self.assertEqual({arm['evidence_ref']: arm['matches_requested_conditions']
                          for arm in retrieved['entries'][0]['comparison']}, {a['id']: False, b['id']: True})
        from core.workflow import Workflow
        recovered = Workflow(wf.run_dir)
        recovered.submit(recovered.queue.get(task['task_id']), draft, 'replayed fixture')
        self.assertEqual(len(recovered.journal.find_cards('MetareviewCard')), 1)
        self.assertEqual(recovered.memory.recording_status()['pending_knowledge_count'], 0)

    def test_partial_scope_binds_identity_data_and_quantization(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf)
        task, draft = self.record(wf, [evidence['id']])
        draft['knowledge_updates'][0]['scope'] = {'activation_bits': 3,
            'quantization': {'fc2': {'activation': {'bits': 3}}},
            'data': {'calibration': {'sample_count': 1}}}
        wf.submit(task, draft, 'unit fixture')
        scope = wf.result_card(wf.queue.get(task['task_id']))['payload']['knowledge_updates'][0]['scope']
        self.assertEqual(scope['model_identity_sha256'], 'fixture')
        self.assertEqual(scope['data'], evidence['payload']['scope']['data'])
        self.assertEqual(scope['quantization'], evidence['payload']['scope']['quantization'])

    def test_bad_rows_are_local_including_scope_conflict_unknown_source_and_missing_science(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf)
        task, draft = self.record(wf, [evidence['id']])
        good = deepcopy(draft['knowledge_updates'][0])
        draft['knowledge_updates'] = [
            {**good, 'scope': {'model_identity_sha256': 'wrong'}},
            {**good, 'evidence_refs': ['EvidenceCard-missing@1']},
            {**good, 'evidence_refs': [{}]},
            {**good, 'status': 'unknown'},
            'malformed row', good]
        wf.submit(task, draft, 'unit fixture')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(len(payload['knowledge_updates']), 1)
        self.assertEqual(len(payload['pending_knowledge_updates']), 5)
        self.assertIn('scope.model_identity_sha256', payload['pending_knowledge_updates'][0]['issue'])
        self.assertEqual(payload['pending_knowledge_updates'][0]['submission']['scope'], {'model_identity_sha256': 'wrong'})
        self.assertEqual(wf.memory.snapshot()['L2']['fixture']['scope']['model_identity_sha256'], 'fixture')

    def test_test_double_is_archived_but_knowledge_stays_pending(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf, source='test_double')
        task, draft = self.record(wf, [evidence['id']])
        wf.submit(task, draft, 'unit fixture')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(payload['knowledge_updates'], [])
        self.assertIn('模拟', payload['pending_knowledge_updates'][0]['issue'])
        self.assertEqual(wf.memory.snapshot()['L2'], {})
        self.assertEqual(wf.journal.card(evidence['id']), evidence)

    def test_no_evidence_does_not_invent_scope_or_knowledge(self):
        wf = self.workflow(1)
        judgment = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('unmeasured_scientific_source', cards=[judgment])
        task, draft = self.record(wf, [])
        wf.submit(task, draft, 'unit fixture')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertFalse(payload['knowledge_updates'])
        self.assertIn('明确 Evidence', payload['pending_knowledge_updates'][0]['issue'])

    def test_existing_claim_revision_identity_is_completed_and_explicit_conflict_is_local(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf)
        task, draft = self.record(wf, [evidence['id']])
        wf.submit(task, draft, 'unit fixture')
        prior = wf.result_card(wf.queue.get(task['task_id']))
        second = self.evidence(wf)
        task, revised = self.record(wf, [second['id']], key='fixture/revision')
        revised['knowledge_updates'][0]['text'] = 'Revised scientific explanation'
        normalized = wf.memory.prepare_metareview(revised, task['inputs']['context']['recording_batch'], 1)
        self.assertEqual(normalized['knowledge_updates'][0]['supersedes'], prior['id'])
        revised['knowledge_updates'][0]['supersedes'] = 'MetareviewCard-wrong@1'
        wf.submit(task, revised, 'unit fixture')
        payload = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertFalse(payload['knowledge_updates'])
        self.assertTrue(payload['pending_knowledge_updates'])
        self.assertEqual(wf.memory.snapshot()['L2']['fixture']['revision_ref'], prior['id'])

    def test_nested_repair_path_is_rejected_and_original_draft_survives_poll(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf)
        task, draft = self.record(wf, [evidence['id']])
        wf.queue.change(task['task_id'], 'running', patch={'repair_feedback': {
            'task_id': task['task_id'], 'rejected_response': {'decision': {'action': 'submit', 'content': draft}}}})
        task = wf.queue.get(task['task_id'])
        bad = {'repair_patch': {'knowledge_updates[0].scope': evidence['payload']['scope']}}
        with self.assertRaisesRegex(ContractError, '顶层字段'):
            wf.submit(task, bad, 'unit fixture')
        receipt = {'status': 'completed', 'execution_id': 'fixture', 'data': {'response': {
            'source': 'test_double', 'decision': {'action': 'submit', 'content': bad}}}}
        wf.config['purpose'] = 'infrastructure_test'
        with patch.object(wf.runtime, 'poll', return_value=[(task, receipt, 'unit fixture')]), \
             patch.object(wf.runtime, 'settle'):
            wf.poll()
        retried = wf.queue.get(task['task_id'])
        self.assertEqual(retried['state'], 'retry')
        self.assertEqual(wf.rejected_draft(retried), draft)
        wf.queue.ready(); task = wf.queue.claim(task['task_id'])
        wf.submit(task, {'repair_patch': {'knowledge_updates': [], 'extra_scientific_note': 'unit fixture'}}, 'unit fixture')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')
        self.assertEqual(wf.result_card(wf.queue.get(task['task_id']))['payload']['extra_scientific_note'], 'unit fixture')

    def test_scope_optional_only_in_recorder_draft_not_immutable_card_schema(self):
        spec = role_spec('recorder', 'record', {}, validation_config()['llm'])
        self.assertNotIn('scope', spec['context']['payload_contract']['properties']['knowledge_updates'].get('items', {}).get('required', []))
        self.assertIn('scope', PAYLOAD_SCHEMAS['MetareviewCard']['properties']['knowledge_updates']['items'].get('required', []))


if __name__ == '__main__':
    unittest.main()
