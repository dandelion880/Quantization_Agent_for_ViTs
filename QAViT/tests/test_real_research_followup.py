"""Offline boundary regressions from real tiny-ViT traces; never launches an LLM."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, make_card, PAYLOAD_SCHEMAS
from agents.roles import role_spec
from tests.support import experiment_payload, meta, validation_config
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

    def record(self, wf, refs):
        tid = wf.enqueue_recorder(1, incomplete=True)
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
        with self.assertRaisesRegex(ContractError, 'weight/activation'):
            wf.broker.preflight(draft)

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
        malformed = deepcopy(draft)
        malformed['knowledge_updates'][0]['evidence_refs'] = [{}]
        with self.assertRaisesRegex(ContractError, 'evidence_refs'):
            wf.submit(task, malformed, 'unit fixture')
        wf.submit(task, draft, 'unit fixture')
        card = wf.result_card(wf.queue.get(task['task_id']))
        self.assertEqual(card['payload']['knowledge_updates'][0]['scope'], evidence[0]['payload']['scope'])
        self.assertEqual(draft, raw)
        self.assertNotIn('scope', draft['knowledge_updates'][0])
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])

    def test_different_scopes_are_not_merged_and_explicit_bad_scope_is_not_replaced(self):
        wf = self.workflow(1)
        a, b = self.evidence(wf), self.evidence(wf, 'per_channel')
        task, draft = self.record(wf, [a['id'], b['id']])
        with self.assertRaisesRegex(ContractError, '范围不同'):
            wf.submit(task, draft, 'unit fixture')
        draft['knowledge_updates'][0]['scope'] = deepcopy(a['payload']['scope'])
        with self.assertRaises(ContractError):
            wf.submit(task, draft, 'unit fixture')
        self.assertFalse(wf.journal.find_cards('MetareviewCard'))

    def test_scope_copy_does_not_bypass_test_double_or_missing_evidence_gates(self):
        wf = self.workflow(1)
        evidence = self.evidence(wf, source='test_double')
        task, draft = self.record(wf, [evidence['id']])
        with self.assertRaisesRegex(ContractError, '模拟'):
            wf.submit(task, draft, 'unit fixture')
        draft['knowledge_updates'][0]['evidence_refs'] = []
        with self.assertRaisesRegex(ContractError, '明确 Evidence'):
            wf.submit(task, draft, 'unit fixture')

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
        self.assertNotIn('scope', spec['context']['payload_contract']['properties']['knowledge_updates']['items']['required'])
        self.assertIn('scope', PAYLOAD_SCHEMAS['MetareviewCard']['properties']['knowledge_updates']['items']['required'])


if __name__ == '__main__':
    unittest.main()
