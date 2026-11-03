"""CPU Recorder prior-basis delivery through actual request/tool/submit boundaries."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, make_card
from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.support import meta, review


class RecorderPriorBasis(TempCase):
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    read = delivery.DeliveryRegression.read
    index = delivery.DeliveryRegression.index
    recorder = delivery.DeliveryRegression.recorder
    record_payload = delivery.DeliveryRegression.record_payload

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU only'))
        worker.start(); self.addCleanup(worker.stop)

    def prior(self, wf, claim_id='fixture-conditional-prior', *, text='OLD_EXACT_CONDITIONAL_BASIS',
              other=False, previous=None):
        value = meta(1)
        value['generation_guidance'] = 'UNRELATED_GLOBAL_GUIDANCE_' + 'g' * 9000
        row = {'claim_id': claim_id, 'text': text, 'status': 'hypothesis',
               'scope': {'model': 'tiny_vit', 'condition': 'FP upstream calibration subset',
                         'excluded_condition': 'quantized upstream held-out subset', 'fixture_only': True},
               'evidence_refs': [], 'rationale': 'Artificial hypothesis only, no measured support.'}
        if previous:
            row['supersedes'] = previous['id']
        value['knowledge_updates'] = [row]
        if other:
            value['knowledge_updates'].append({**row, 'claim_id': claim_id + '-unrelated',
                                               'text': 'UNRELATED_CLAIM_BODY_' + 'u' * 18000})
        card = make_card('MetareviewCard', wf.journal.run_id, value, author='recorder')
        event = wf.memory.apply_metareview(card)
        wf.journal.commit('prior/' + card['id'], cards=[card], events=[event])
        return card

    def new_review(self, wf, prior, *, claim_id=None, text=None, extra=None, refs=None):
        claim_id = claim_id or prior['payload']['knowledge_updates'][0]['claim_id']
        value = review('interpretation', rationale=text or
            ('新增人工材料需核对旧条件：claim_id=' + claim_id + '，revision_ref=' + prior['id'] + '。'))
        value.update(extra or {})
        card = make_card('ReviewDecisionCard', wf.journal.run_id, value,
                         refs=[prior['id']] if refs is None else refs)
        wf.journal.commit('batch/' + card['id'], cards=[card])
        return card

    def test_short_exact_old_basis_is_direct_without_expanding_increment_or_requiring_reread(self):
        wf = self.workflow(1)
        prior = self.prior(wf, other=True)
        source = self.new_review(wf, prior)
        task, context = self.begin(wf, self.recorder(wf))
        self.assertIn('OLD_EXACT_CONDITIONAL_BASIS', canonical(context))
        self.assertNotIn('UNRELATED_CLAIM_BODY_', canonical(context))
        self.assertNotIn('UNRELATED_GLOBAL_GUIDANCE_', canonical(context))
        self.assertFalse(context['memory']['required_reads'])
        self.assertEqual(task['inputs']['context']['recording_batch']['source_refs'], [source['id']])
        self.assertNotIn(prior['id'], task['inputs']['refs'])
        self.assertIn(prior['id'], self.index(wf, task))
        wf.submit(task, self.record_payload(task), 'SCRIPTED')
        published = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(published['knowledge_updates'], [])
        self.assertEqual(published['source_refs'], [source['id']])
        self.assertEqual(wf.memory.snapshot()['L2']['fixture-conditional-prior']['revision_ref'], prior['id'])

    def test_long_exact_item_requires_all_pages_and_survives_restart(self):
        wf = self.workflow(1)
        prior = self.prior(wf, text='LONG_PRIOR_BASIS_' + 'x' * 25000, other=True)
        self.new_review(wf, prior)
        task, context = self.begin(wf, self.recorder(wf))
        expected = {'ref': prior['id'], 'field': '/payload/knowledge_updates/0'}
        self.assertEqual([(r['ref'], r['field']) for r in context['memory']['required_reads']],
                         [(expected['ref'], expected['field'])])
        with self.assertRaisesRegex(ContractError, '未读完'):
            wf.submit(task, self.record_payload(task), 'SCRIPTED')
        task, _ = self.tools(wf, task, [('read_card', {**expected, 'offset': 0})])
        with self.assertRaisesRegex(ContractError, '未读完'):
            wf.submit(task, self.record_payload(task), 'SCRIPTED')
        wf = Workflow(wf.run_dir)
        task = wf.queue.get(task['task_id'])
        task, context = self.tools(wf, task, [('read_card', {**expected, 'offset': offset}) for offset in (12000, 24000)])
        self.assertIsNone(context['memory']['required_reads'][0]['next_offset'])
        self.assertEqual(canonical(wf.request_body(task)).count('LONG_PRIOR_BASIS_'), 1)
        self.assertNotIn('UNRELATED_CLAIM_BODY_', canonical(context))
        wf.submit(task, self.record_payload(task), 'SCRIPTED')

    def test_two_revisions_and_multiple_claims_bind_only_explicit_pairs(self):
        wf = self.workflow(1)
        first = self.prior(wf, 'first', text='FIRST_SELECTED_BASIS', other=True)
        second = self.prior(wf, 'second', text='SECOND_SELECTED_BASIS', other=True)
        self.new_review(wf, first, text='本批对应两个明确旧依据。', refs=[first['id'], second['id']],
            extra={'prior_checks': [{'claim_id': 'first', 'revision_ref': first['id']},
                                   {'claim_id': 'second', 'revision_ref': second['id']}]})
        _, context = self.begin(wf, self.recorder(wf))
        fields = [(r['read_card']['ref'], r['read_card']['field']) for r in context['memory']['bound_feedback']]
        self.assertEqual(set(fields), {(first['id'], '/payload/knowledge_updates/0'),
                                      (second['id'], '/payload/knowledge_updates/0')})
        self.assertNotIn('UNRELATED_CLAIM_BODY_', canonical(context))

    def test_existing_text_labels_pair_without_cross_revision_matching(self):
        wf = self.workflow(1)
        first = self.prior(wf, 'first', text='FIRST_TEXT_BASIS', other=True)
        second = self.prior(wf, 'second', text='SECOND_TEXT_BASIS', other=True)
        self.new_review(wf, first, refs=[first['id'], second['id']],
            text='claim_id=first, revision_ref=' + first['id'] +
                 '；claim_id=second, revision_ref=' + second['id'] + '。')
        _, context = self.begin(wf, self.recorder(wf))
        fields = [(r['read_card']['ref'], r['read_card']['field']) for r in context['memory']['bound_feedback']]
        self.assertEqual(set(fields), {(first['id'], '/payload/knowledge_updates/0'),
                                      (second['id'], '/payload/knowledge_updates/0')})
        self.assertNotIn('UNRELATED_CLAIM_BODY_', canonical(context))

    def test_explicit_old_revision_is_not_silently_replaced_by_latest(self):
        wf = self.workflow(1)
        old = self.prior(wf, text='SPECIFIC_OLD_VERSION_BODY')
        latest = self.prior(wf, text='LATEST_VERSION_NOT_SELECTED', previous=old)
        self.new_review(wf, old)
        _, context = self.begin(wf, self.recorder(wf))
        self.assertIn('SPECIFIC_OLD_VERSION_BODY', canonical(context))
        self.assertNotIn('LATEST_VERSION_NOT_SELECTED', canonical(context))
        self.assertEqual(wf.memory.snapshot()['L2']['fixture-conditional-prior']['revision_ref'], latest['id'])

    def test_missing_revision_and_unrelated_history_stay_optional(self):
        wf = self.workflow(1)
        prior = self.prior(wf)
        self.new_review(wf, prior, text='claim_id=fixture-conditional-prior；没有指定旧版本。')
        task, context = self.begin(wf, self.recorder(wf))
        self.assertNotIn('OLD_EXACT_CONDITIONAL_BASIS', canonical(context))
        self.assertFalse(context['memory']['required_reads'])
        self.assertIn(prior['id'], self.index(wf, task))

    def test_claim_prefix_and_wrong_revision_do_not_bind(self):
        base = self.root
        for mode in ('claim_prefix', 'wrong_revision'):
            with self.subTest(mode=mode):
                self.root = base / mode
                wf = self.workflow(1)
                prior = self.prior(wf)
                claim = 'fixture-conditional-prior-prefix' if mode == 'claim_prefix' else 'fixture-conditional-prior'
                revision = prior['id'] if mode == 'claim_prefix' else 'MetareviewCard-uncommitted@1'
                self.new_review(wf, prior, text='claim_id=' + claim + ', revision_ref=' + revision)
                with self.assertRaises(ContractError):
                    wf.request_body(self.recorder(wf))

    def test_explicit_prior_cannot_bypass_visibility(self):
        wf = self.workflow(1)
        prior = self.prior(wf)
        self.new_review(wf, prior)
        task = self.recorder(wf)
        visible = wf.can_read_card
        with patch.object(wf, 'can_read_card', side_effect=lambda t, c: c['id'] != prior['id'] and visible(t, c)):
            with self.assertRaisesRegex(ContractError, '权限拒绝'):
                wf.request_body(task)

    def test_prior_delivery_does_not_advance_or_duplicate_atomic_coverage(self):
        wf = self.workflow(1)
        prior = self.prior(wf)
        source = self.new_review(wf, prior)
        task, _ = self.begin(wf, self.recorder(wf))
        value = self.record_payload(task)
        original = wf.journal.commit
        def fail(key, **kwargs):
            if key.startswith('complete/'):
                def fault(stage):
                    if stage == 'cards_written': raise OSError('CPU atomic fault')
                kwargs['fault'] = fault
            return original(key, **kwargs)
        with patch.object(wf.journal, 'commit', side_effect=fail), self.assertRaises(OSError):
            wf.submit(task, value, 'SCRIPTED')
        self.assertFalse(any(e['type'] == 'recorder_batch_completed' for e in wf.journal.events))
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [source['id']])
        wf = Workflow(wf.run_dir); task = wf.queue.get(task['task_id'])
        wf.submit(task, value, 'SCRIPTED')
        wf.submit(task, value, 'SCRIPTED')
        events = [e for e in wf.journal.events if e['type'] == 'recorder_batch_completed']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['source_refs'], [source['id']])
        self.assertNotIn(prior['id'], events[0]['source_refs'])
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])
