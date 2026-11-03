"""CPU normal submit → immutable refs → Recorder basis, with exact wrapped identities."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, make_card
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery, test_recorder_prior_basis as basis
from tests.support import hypothesis, review


class RecorderReferenceHandoff(TempCase):
    begin = delivery.DeliveryRegression.begin
    task = delivery.DeliveryRegression.task
    recorder = delivery.DeliveryRegression.recorder
    record_payload = delivery.DeliveryRegression.record_payload
    prior = basis.RecorderPriorBasis.prior
    new_review = basis.RecorderPriorBasis.new_review

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU only'))
        worker.start(); self.addCleanup(worker.stop)

    def test_text_wrappers_do_not_become_part_of_exact_identity(self):
        formats = [
            'claim_id={claim}, revision_ref={ref}.',
            '(claim_id={claim}, revision_ref={ref})',
            'claim_id=({claim}), revision_ref=({ref}).',
            '（claim_id=（{claim}）, revision_ref=（{ref}））。',
            '`claim_id={claim}`, `revision_ref={ref}`.',
            '`claim_id`: `{claim}`, `revision_ref`: `{ref}`.',
            '"claim_id": "{claim}", "revision_ref": "{ref}".',
        ]
        base = self.root
        for i, template in enumerate(formats):
            with self.subTest(template=template):
                self.root = base / str(i)
                wf = self.workflow(1); prior = self.prior(wf, 'condition.v1')
                self.new_review(wf, prior, text=template.format(claim='condition.v1', ref=prior['id']))
                _, context = self.begin(wf, self.recorder(wf))
                self.assertIn('OLD_EXACT_CONDITIONAL_BASIS', canonical(context))
                self.assertEqual([(e['read_card']['ref'], e['read_card']['field']) for e in context['memory']['bound_feedback']],
                                 [(prior['id'], '/payload/knowledge_updates/0')])

    def test_structured_pairs_take_precedence_over_conflicting_text(self):
        wf = self.workflow(1); prior = self.prior(wf)
        self.new_review(wf, prior,
            text='引用示例 claim_id=wrong-example, revision_ref=MetareviewCard-unknown@9.',
            extra={'prior_basis': {'claim_id': 'fixture-conditional-prior', 'revision_ref': prior['id']}})
        _, context = self.begin(wf, self.recorder(wf))
        self.assertEqual([e['read_card']['ref'] for e in context['memory']['bound_feedback']], [prior['id']])

    def test_engineering_log_examples_are_reviewed_without_becoming_scientific_prior_refs(self):
        base = self.root
        examples = [
            {'tool_results': [{'text': '兼容 claim_id=…、revision_ref=… 文本标注'}]},
            {'rejected_draft': {'claim_id': 'unpublished', 'revision_ref': 'MetareviewCard-unknown@9'}},
        ]
        for i, logged in enumerate(examples):
            with self.subTest(logged=logged):
                self.root = base / str(i)
                wf = self.workflow(1); prior = self.prior(wf)
                self.new_review(wf, prior, extra={'prior_basis': {
                    'claim_id': 'fixture-conditional-prior', 'revision_ref': prior['id']}})
                tid = wf.queue.enqueue('r001/L3', 'experiment_request')
                wf.queue.ready(); wf.queue.claim(tid)
                wf.queue.finish(tid, result={'status': 'rejected', **logged})
                engineering_ref = wf.queue.get(tid)['task_card']
                task, context = self.begin(wf, self.recorder(wf))
                batch = task['inputs']['context']['recording_batch']
                self.assertIn(engineering_ref, batch['engineering_refs'])
                self.assertIn(engineering_ref, batch['source_refs'])
                self.assertEqual([e['read_card']['ref'] for e in context['memory']['bound_feedback']], [prior['id']])
                content = self.record_payload(task)
                self.assertNotIn('reviewed_refs', content)
                wf.submit(task, content, 'SCRIPTED_RECORDER')
                self.assertEqual(wf.memory.recording_batch()['source_refs'], [])

    def test_generation_and_judger_submit_record_direct_refs_then_recorder_uses_exact_old_version(self):
        wf = self.workflow(1)
        old = self.prior(wf, text='ORIGINAL_HANDOFF_BASIS')
        latest = self.prior(wf, text='LATEST_NOT_AUTOMATIC_BASIS', previous=old)
        gen, _ = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id']],
            round=1, slot='H1', hypothesis_type='existing_method'))
        content = hypothesis(1, 'H1')
        content['mechanism'] += ' 旧条件 claim_id=fixture-conditional-prior, revision_ref=' + old['id'] + '.'
        content['feedback_used'] = [{'metareview_ref': latest['id'], 'change': 'CPU exact prior-identity fixture only'}]
        wf.submit(gen, content, 'SCRIPTED_GENERATION')
        generated = wf.result_card(wf.queue.get(gen['task_id']))
        self.assertIn(old['id'], generated['refs'])
        judger, _ = self.begin(wf, self.task(wf, 'interpret', refs=[generated['id']], round=1))
        payload = review('interpretation', rationale='Explicit old basis, CPU fixture only',
            prior_basis={'claim_id': 'fixture-conditional-prior', 'revision_ref': old['id']})
        wf.submit(judger, payload, 'SCRIPTED_JUDGER')
        judged = wf.result_card(wf.queue.get(judger['task_id']))
        self.assertIn(old['id'], judged['refs'])
        record, context = self.begin(wf, self.recorder(wf))
        self.assertEqual(canonical(context).count('ORIGINAL_HANDOFF_BASIS'), 1)
        self.assertNotIn('LATEST_NOT_AUTOMATIC_BASIS', canonical(context))
        self.assertEqual(set(record['inputs']['context']['recording_batch']['source_refs']),
                         {generated['id'], judged['id']})
        wf.submit(record, self.record_payload(record), 'SCRIPTED_RECORDER')
        final = wf.result_card(wf.queue.get(record['task_id']))['payload']
        self.assertEqual(final['knowledge_updates'], [])
        self.assertNotIn(old['id'], final['source_refs'])
        self.assertEqual(sum(e['type'] == 'recorder_batch_completed' for e in wf.journal.events), 1)

    def test_submit_rejects_missing_version_suffix_and_wrong_claim_without_publishing(self):
        base = self.root
        for mode in ('missing_version', 'suffix', 'quoted_suffix', 'wrong_claim'):
            with self.subTest(mode=mode):
                self.root = base / mode
                wf = self.workflow(1); prior = self.prior(wf)
                task, _ = self.begin(wf, self.task(wf, 'interpret', round=1))
                ref = prior['id'].split('@')[0] if mode == 'missing_version' else prior['id']
                if mode == 'suffix': ref += '.wrong'
                if mode == 'quoted_suffix': ref = '"' + prior['id'] + '."'
                claim = 'fixture-conditional-prior-prefix' if mode == 'wrong_claim' else 'fixture-conditional-prior'
                with self.assertRaises(ContractError):
                    wf.submit(task, review('interpretation', rationale='claim_id=' + claim + ', revision_ref=' + ref + '.'), 'SCRIPTED')
                self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])

    def test_structured_identity_is_not_normalized_or_repaired(self):
        wf = self.workflow(1); prior = self.prior(wf)
        task, _ = self.begin(wf, self.task(wf, 'interpret', round=1))
        with self.assertRaises(ContractError):
            wf.submit(task, review('interpretation',
                rationale='claim_id=fixture-conditional-prior, revision_ref=' + prior['id'], prior_basis={
                'claim_id': 'fixture-conditional-prior', 'revision_ref': '(' + prior['id'] + ')'}), 'SCRIPTED')
        self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])

    def test_explicit_submit_reference_obeys_snapshot_and_visibility(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.task(wf, 'interpret', round=1))
        future = self.prior(wf)
        content = review('interpretation', prior_basis={
            'claim_id': 'fixture-conditional-prior', 'revision_ref': future['id']})
        with self.assertRaisesRegex(ContractError, '权限拒绝'):
            wf.submit(task, content, 'SCRIPTED_FUTURE')
        self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])

    def test_transitive_reference_does_not_grant_a_direct_prior_handoff(self):
        wf = self.workflow(1); prior = self.prior(wf)
        bridge = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation'), refs=[prior['id']])
        wf.journal.commit('bridge', cards=[bridge])
        task, _ = self.begin(wf, self.task(wf, 'interpret', refs=[bridge['id']], round=1))
        wf.submit(task, review('interpretation', rationale='No explicit old claim identity'), 'SCRIPTED')
        published = wf.result_card(wf.queue.get(task['task_id']))
        self.assertNotIn(prior['id'], published['refs'])
        legacy = make_card('ReviewDecisionCard', wf.journal.run_id,
            review('interpretation', rationale='claim_id=fixture-conditional-prior, revision_ref=' + prior['id']),
            refs=[bridge['id']])
        wf.journal.commit('legacy_missing_direct_ref', cards=[legacy])
        with self.assertRaisesRegex(ContractError, '直接 refs'):
            wf.request_body(self.recorder(wf))

    def test_same_exchange_opponent_prior_remains_invisible_at_submit(self):
        wf = self.workflow(1)
        branch, _ = self.begin(wf, self.task(wf, 'position', round=1, pair=['H1', 'H2'],
            exchange=1, side='B', position_for='H2'))
        prior = self.prior(wf)
        wf.queue.finish(branch['task_id'], outputs=[prior['id']], result={})
        candidate, _ = self.begin(wf, self.task(wf, 'position', round=1, pair=['H1', 'H2'],
            exchange=1, side='A', position_for='H1'))
        content = review('position', side='A', arguments=[], objections=[], responses_to_prior=[], key_predictions=[],
            prior_basis={'claim_id': 'fixture-conditional-prior', 'revision_ref': prior['id']})
        with self.assertRaisesRegex(ContractError, '权限拒绝'):
            wf.submit(candidate, content, 'SCRIPTED_OPPONENT')
        self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])
