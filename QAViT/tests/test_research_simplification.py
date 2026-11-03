"""CPU handoff, evidence delivery and ACL cost regressions; no model workers."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, fingerprint, make_card
from core.journal import atomic_json
from core.workflow import Workflow
from tests.support import experiment_payload, review, meta, validation_config
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery


class ResearchSimplification(TempCase):
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    task = delivery.DeliveryRegression.task
    science = delivery.DeliveryRegression.science
    debate_task = delivery.DeliveryRegression.debate_task
    read = delivery.DeliveryRegression.read

    def setUp(self):
        super().setUp()
        blocker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU boundary test'))
        blocker.start()
        self.addCleanup(blocker.stop)

    def request(self, wf, task):
        wf.receive_tools(task, {'tool_calls': [{'id': 'probe', 'name': 'request_experiment',
            'arguments': {'experiment': experiment_payload()}}]}, 'SCRIPTED')
        wf.queue.ready()
        return wf.queue.claim(wf.queue.get(task['task_id'])['waiting_for'][0])

    def cached_result(self, wf, request, evidence):
        # Deliberate test double only for the scheduler; real CPU services have separate integration coverage.
        with patch.object(wf.broker, 'cached', return_value=evidence):
            wf.progress_request(request)
        return wf.queue.get(request['task_id'])

    def test_waiting_judger_receives_evidence_and_can_request_again_after_restart(self):
        wf = self.workflow(1)
        evidence = self.science(wf, 'returned')[1]
        task, _ = self.begin(wf, self.debate_task(wf))
        request = self.request(wf, task)
        wf = Workflow(wf.run_dir)
        result = self.cached_result(wf, wf.queue.get(request['task_id']), evidence)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['outputs'], [evidence['id']])
        self.assertTrue(result['result']['cache_reused'])
        self.assertNotIn(request['key'] + '/interpret', wf.tasks())
        wf = Workflow(wf.run_dir)
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertIn(evidence['id'], wf.required_refs(task))
        self.assertIn(evidence['payload']['observations']['explanation'], canonical(context))
        again = self.cached_result(wf, self.request(wf, task), evidence)
        self.assertEqual(again['result']['evidence_ref'], evidence['id'])
        task, _ = self.begin(wf, wf.queue.get(task['task_id']))
        wf.submit(task, review('position', side='A', arguments=['scripted observation use'], objections=[],
            responses_to_prior=[], key_predictions=[], evidence_refs=[evidence['id']]), 'SCRIPTED')
        judgment = wf.result_card(wf.queue.get(task['task_id']))
        self.assertIn(evidence['id'], judgment['refs'])
        self.assertEqual(len(wf.journal.find_cards('EvidenceCard')), 1)
        self.assertIn(evidence['id'], wf.memory.recording_batch()['source_refs'])
        self.assertIn(judgment['id'], wf.memory.recording_batch()['judgment_refs'])

    def test_recorder_audits_returned_position_after_an_earlier_batch_and_restart(self):
        wf = self.workflow(1)
        evidence = self.science(wf, 'audit_return')[1]
        task, _ = self.begin(wf, self.debate_task(wf))
        result = self.cached_result(wf, self.request(wf, task), evidence)
        # A previous Recorder batch may have covered the Evidence before the parent judgment arrives.
        recorder = wf.enqueue_recorder(1, incomplete=True, key='fixture/audit_before_parent')
        recording, _ = self.begin(wf, wf.queue.get(recorder))
        batch = recording['inputs']['context']['recording_batch']
        wf.submit(recording, meta(1, batch['judgment_refs']), 'SCRIPTED')
        self.assertIn(result['task_card'], batch['source_refs'])
        task, _ = self.begin(wf, wf.queue.get(task['task_id']))
        wf.submit(task, review('position', side='A', arguments=['scripted observation use'], objections=[],
            responses_to_prior=[], key_predictions=[], evidence_refs=[evidence['id']]), 'SCRIPTED')
        judgment = wf.result_card(wf.queue.get(task['task_id']))
        unrelated = make_card('ReviewDecisionCard', wf.journal.run_id, review('position'))
        wf.journal.commit('unrelated/position', cards=[unrelated])
        wf = Workflow(wf.run_dir)
        batch = wf.memory.recording_batch()
        self.assertEqual(set(batch['judgment_refs']), {judgment['id'], unrelated['id']})
        self.assertIn(unrelated['id'], batch['source_refs'])
        recorder = wf.enqueue_recorder(1, incomplete=True, key='fixture/audit_after_parent')
        recording, _ = self.begin(wf, wf.queue.get(recorder))
        # Both late positions remain discoverable; no mechanical row-per-card audit is required.
        wf.submit(recording, meta(1, batch['judgment_refs']), 'SCRIPTED')
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])

    def test_generation_and_evolution_keep_scientific_approval_and_interpretation(self):
        for purpose in ('generate', 'evolve'):
            with self.subTest(purpose=purpose):
                wf = Workflow(self.root / purpose, validation_config())
                cards = self.science(wf, purpose)
                context = {} if purpose == 'generate' else {'parent_ref': cards[2]['id']}
                task, _ = self.begin(wf, self.task(wf, purpose, refs=[wf.brief['id']], **context))
                request = self.request(wf, task)
                wf.progress_request(request)
                approval = wf.tasks()[request['key'] + '/scientific_approval']
                approval, _ = self.begin(wf, approval)
                wf.submit(approval, review('experiment_approval', approved=True), 'SCRIPTED')
                wf.queue.ready()
                request = wf.queue.claim(request['task_id'])
                result = self.cached_result(wf, request, cards[1])
                self.assertEqual(result['state'], 'waiting')
                self.assertIn(request['key'] + '/interpret', wf.tasks())
                self.assertEqual(wf.queue.get(task['task_id'])['state'], 'waiting')

    def test_unbound_request_keeps_interpretation(self):
        wf = self.workflow(1)
        evidence = self.science(wf, 'no_parent')[1]
        tid = wf.queue.enqueue('fixture/unbound', 'experiment_request', inputs={
            'requester': 'judger', 'refs': [wf.brief['id']], 'draft': experiment_payload()})
        wf.queue.ready()
        request = wf.queue.claim(tid)
        result = self.cached_result(wf, request, evidence)
        self.assertEqual(result['state'], 'waiting')
        self.assertIn(request['key'] + '/interpret', wf.tasks())

    def test_plan_l3_supporting_request_keeps_judgment_card(self):
        wf = self.workflow(1)
        evidence = self.science(wf, 'planning')[1]
        task, _ = self.begin(wf, self.task(wf, 'plan_l3', refs=[wf.brief['id']]))
        request = self.request(wf, task)
        result = self.cached_result(wf, request, evidence)
        self.assertEqual(result['state'], 'waiting')
        self.assertIn(request['key'] + '/interpret', wf.tasks())

    def test_published_interpretation_survives_recovery_without_duplicate(self):
        wf = self.workflow(1)
        evidence = self.science(wf, 'legacy_interpret')[1]
        task, _ = self.begin(wf, self.debate_task(wf))
        request = self.request(wf, task)
        iid = wf.enqueue_role(request['key'] + '/interpret', 'interpret', refs=[evidence['id']],
                              parent_task=request['task_id'])
        wf = Workflow(wf.run_dir)
        result = self.cached_result(wf, wf.queue.get(request['task_id']), evidence)
        self.assertEqual(result['waiting_for'], [iid])
        interpretation, _ = self.begin(wf, wf.queue.get(iid))
        wf.submit(interpretation, review('interpretation', evidence_refs=[evidence['id']]), 'SCRIPTED')
        wf.queue.ready()
        result = self.cached_result(wf, wf.queue.claim(request['task_id']), evidence)
        self.assertEqual(result['result']['interpretation_ref'], wf.queue.get(iid)['outputs'][0])
        self.assertEqual(len([t for t in wf.tasks().values() if t['inputs'].get('purpose') == 'interpret']), 1)

    def test_failed_execution_returns_failure_without_evidence(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.debate_task(wf))
        request = self.request(wf, task)
        with patch.object(wf.broker, 'cached', return_value=None):
            wf.progress_request(request)
            execution = wf.tasks()[request['key'] + '/execute']
            wf.queue.ready(); wf.queue.claim(execution['task_id'])
            wf.queue.fail(execution['task_id'], {'code': 'CPU_injected_failure'})
            wf.queue.ready()
            wf.progress_request(wf.queue.claim(request['task_id']))
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertEqual(context['tool_results'][0]['state'], 'failed')
        self.assertEqual(wf.journal.find_cards('EvidenceCard'), [])
        self.assertNotIn(request['key'] + '/interpret', wf.tasks())

    def test_direct_return_preserves_opponent_and_reference_isolation(self):
        wf = self.workflow(1)
        own = self.debate_task(wf)
        other, _ = self.begin(wf, self.debate_task(wf, side='B'))
        request = self.request(wf, other)
        # Publish after branch creation, through its real execution task ownership.
        wf.progress_request(request)
        execution = wf.tasks()[request['key'] + '/execute']
        wf.queue.ready(); wf.queue.claim(execution['task_id'])
        spec, evidence, *_ = self.science(wf, 'private', commit=False)
        wf.queue.finish(execution['task_id'], outputs=[evidence['id']], cards=[spec, evidence])
        wf.queue.ready(); wf.progress_request(wf.queue.claim(request['task_id']))
        own, _ = self.begin(wf, own)
        with self.assertRaises(ContractError):
            self.read(wf, own, 'read_card', ref=evidence['id'])
        self.assertNotIn(str(wf.journal.card_path(evidence['id'])), wf.authorized_assets(own))
        wf.queue.finish(own['task_id'])
        ref, _ = self.begin(wf, self.task(wf, 'reference', refs=[wf.brief['id'], spec['id']], experiment_ref=spec['id']))
        with self.assertRaises(ContractError):
            self.read(wf, ref, 'read_card', ref=evidence['id'])

    def evidence_fixture(self, wf, *, large=False):
        payload = experiment_payload()
        payload['execution']['measurement_implementation'] = {'files': {'CPU_SOURCE_HASH_LIST': 'x' * 16000}}
        spec = make_card('ExperimentCard', wf.journal.run_id, payload)
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'fixture', 'status': 'completed',
            'execution_id': 'fixture', 'measurement_source': 'test_double',
            'observations': {'relative_mse': 0.125, **({'large_scientific_result': 'y' * 18000} if large else {})},
            'provenance': {'resolved_execution': deepcopy(spec['payload']['execution']),
                           'experiment_sha256': spec['content_sha256']},
            'resources': {}, 'artifacts': [], 'scope': {'model': 'CPU_fixture', 'samples': 2}}, refs=[spec['id']])
        wf.journal.commit('delivery', cards=[spec, evidence])
        return spec, evidence

    def test_numbers_and_scope_arrive_without_engineering_pages_and_original_remains_readable(self):
        wf = self.workflow(1)
        spec, evidence = self.evidence_fixture(wf)
        original = deepcopy(evidence)
        task, context = self.begin(wf, self.task(wf, refs=[evidence['id']]))
        self.assertFalse(context['memory']['required_reads'])
        shown = next(c for c in context['memory']['cards'] if c['id'] == evidence['id'])
        self.assertEqual(shown['payload']['observations'], original['payload']['observations'])
        self.assertEqual(shown['payload']['scope'], original['payload']['scope'])
        pointer = shown['payload']['provenance']['resolved_execution']
        self.assertEqual(pointer['sha256'], fingerprint(spec['payload']['execution']))
        self.assertNotIn('CPU_SOURCE_HASH_LIST', canonical(context))
        self.assertIn(evidence['id'], context['memory']['read_manifest']['projected_refs'])
        self.assertNotIn(evidence['id'], context['memory']['read_manifest']['refs'])
        task, context = self.tools(wf, task, [('read_card', {
            'ref': spec['id'], 'field': '/payload/execution/measurement_implementation'})])
        self.assertIn('CPU_SOURCE_HASH_LIST', canonical(context['tool_results']))
        self.assertEqual(wf.journal.card(evidence['id']), original)

    def test_scientific_long_results_still_require_complete_reads(self):
        wf = self.workflow(1)
        _, evidence = self.evidence_fixture(wf, large=True)
        task, context = self.begin(wf, self.task(wf, refs=[evidence['id']]))
        self.assertTrue(any(r['ref'] == evidence['id'] and r['next_offset'] == 0 for r in context['memory']['required_reads']))
        with self.assertRaisesRegex(ContractError, '未读完'):
            wf.submit(task, review('screening', eligible=True, hard_rejection_reasons=[],
                similarity_analysis='fixture', evidence_gaps=[]), 'SCRIPTED')

    def test_nonidentical_execution_is_not_folded(self):
        wf = self.workflow(1)
        _, evidence = self.evidence_fixture(wf)
        for key, value in (('experiment_sha256', 'wrong_hash'), ('resolved_execution', {'seed': 999})):
            changed = deepcopy(evidence)
            changed['payload']['provenance'][key] = value
            self.assertEqual(wf.present_card(changed)['payload']['provenance'], changed['payload']['provenance'])

    def test_acl_diamond_traversal_reads_each_ancestor_once_and_checks_integrity(self):
        wf = self.workflow(1)
        task = self.debate_task(wf)
        cards = []
        for i in range(24):
            card = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation'),
                             refs=[c['id'] for c in cards[-5:]])
            wf.journal.commit('diamond/' + str(i), cards=[card]); cards.append(card)
        with patch.object(wf.journal, 'card', wraps=wf.journal.card) as reads:
            self.assertTrue(wf.can_read_card(task, cards[-1]))
            self.assertLessEqual(reads.call_count, len(cards) - 1)
        broken = deepcopy(cards[0]); broken['payload']['rationale'] = 'tampered'
        atomic_json(wf.journal.card_path(broken['id']), broken)
        with self.assertRaises(ContractError):
            wf.can_read_card(task, cards[-1])

    def test_unrelated_task_denial_does_not_expand_history(self):
        wf = self.workflow(1)
        task = self.debate_task(wf)
        tid = wf.queue.enqueue('unrelated', 'host_tool', inputs={'name': 'read_card'})
        card = wf.journal.card(wf.queue.get(tid)['task_card'])
        with patch.object(wf.journal, 'card', wraps=wf.journal.card) as reads:
            self.assertFalse(wf.can_read_card(task, card))
            self.assertEqual(reads.call_count, 0)


if __name__ == '__main__':
    unittest.main()
