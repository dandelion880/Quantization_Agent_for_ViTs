"""Scripted queue/receipt fixtures only; no model, numerical worker or Evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import atomic_json
from core.workflow import Workflow
from runtime.identity import code_identity
from tests.test_infrastructure import TempCase
from tests.test_single_implementation import SingleImplementation
from tools.safeguards import approval_receipt, validate_approval


OLD = '0' * 64


class StaleSafeguardRecovery(TempCase):
    begin = SingleImplementation.begin
    start = SingleImplementation.start
    coding = SingleImplementation.coding
    progress = SingleImplementation.progress

    def setUp(self):
        super().setUp()
        guard = patch('runtime.task_runtime.TaskRuntime.launch',
                      side_effect=AssertionError('control fixture must not launch workers'))
        self.launch = guard.start()
        self.addCleanup(guard.stop)

    def old_review(self):
        wf, author, request = self.start()
        with patch('core.experiment_requests.code_identity', return_value={'core_sha256': OLD}):
            request = self.coding(wf, request)
        review = wf.tasks()[request['key'] + '/safeguard/' + OLD[:16]]
        return wf, author, request, review

    def claim_old(self, wf, review, *, legacy=False):
        if legacy:
            inputs = deepcopy(review['inputs'])
            inputs['context'].pop('host_implementation_sha256')
            wf.queue.change(review['task_id'], review['state'], patch={'inputs': inputs})
        # Exact observed form: old publication key A, latest retry request core B.
        path = wf.run_dir / 'control_receipts' / review['task_id'] / 'request.json'
        atomic_json(path, {'core_sha256': code_identity()['core_sha256']})
        wf.queue.ready()
        return wf.queue.claim(review['task_id'], request_path=str(path))

    def current_reviews(self, wf, request):
        return [t for t in wf.tasks().values()
                if t['inputs'].get('purpose') == 'safeguard'
                and t['inputs'].get('parent_task') == request['task_id']
                and not wf.stale_safeguard_owner(t)]

    def test_old_key_new_retry_core_and_descendants_recover_one_current_review(self):
        wf, _, request, review = self.old_review()
        old = self.claim_old(wf, review, legacy=True)
        child = wf.queue.enqueue(review['key'] + '/read', 'host_tool', inputs={
            'parent_task': review['task_id'], 'name': 'read_asset',
            'arguments': {'asset': 'methods/uniform_full.md'}})
        grandchild = wf.queue.enqueue(review['key'] + '/read/child', 'host_tool',
                                      inputs={'parent_task': child})
        wf.wait_for(old, [child])
        with patch.object(wf, 'host_tool', side_effect=AssertionError('obsolete read must not execute')):
            wf.dispatch()
        for tid in (review['task_id'], child, grandchild):
            self.assertEqual(wf.queue.get(tid)['state'], 'cancelled')
        current = self.current_reviews(wf, request)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]['inputs']['context']['host_implementation_sha256'],
                         code_identity()['core_sha256'])
        keys = set(wf.tasks())
        for _ in range(2):
            wf = Workflow(wf.run_dir)
            wf.progress_request(wf.queue.get(request['task_id']))
            self.assertEqual(set(wf.tasks()), keys)
            self.assertEqual(len(self.current_reviews(wf, request)), 1)
        task, _ = self.begin(wf, wf.queue.get(current[0]['task_id']))
        wf.submit(task, {'approved': True, 'rationale': 'Scripted current-host review only'},
                  'SCRIPTED_CONTROL_FIXTURE')
        request = self.progress(wf, request)
        trial = wf.tasks()[request['key'] + '/execute']
        validate_approval(trial['inputs']['approval'], wf.journal.card(request['experiment_ref']))
        self.assertEqual(wf.queue.get(review['task_id'])['state'], 'cancelled')
        self.launch.assert_not_called()

    def test_late_success_and_failure_settle_usage_without_submission_or_retry(self):
        base = self.root
        cases = [('completed', {'input_tokens': 11, 'output_tokens': 3}),
                 ('failed', {'input_tokens': 7, 'output_tokens': 2}), ('failed', None)]
        for i, (status, usage) in enumerate(cases):
            with self.subTest(status=status, usage=usage):
                self.root = base / str(i)
                wf, _, _, review = self.old_review()
                task = self.claim_old(wf, review)
                execution = task['task_id'] + '-late-control'
                wf.budget.reserve(execution, 'llm', estimated_seconds=1)
                receipt = {'status': status, 'execution_id': execution,
                    'data': {'usage': usage, 'response': {'source': 'test_double', 'decision': {
                        'action': 'submit', 'content': {'approved': True,
                        'rationale': 'Obsolete scripted response must not become a ReviewCard'}}}},
                    'error': {'code': 'llm_timeout', 'message': 'Injected old-host timeout'}}
                path = wf.run_dir / 'control_receipts' / execution / 'result.json'
                atomic_json(path, receipt)
                with patch.object(wf.runtime, 'poll', return_value=[(task, receipt, path)]):
                    wf.poll()
                old = wf.queue.get(task['task_id'])
                self.assertEqual(old['state'], 'cancelled')
                self.assertEqual(old['superseded_receipt'], str(path))
                self.assertEqual(old['retry_count'], task['retry_count'])
                self.assertFalse(old.get('repair_count'))
                self.assertFalse(wf.journal.find_cards('ReviewDecisionCard'))
                self.assertEqual(wf.budget.snapshot()['settlements'][execution]['reported_usage'], usage)
                sequence = wf.journal.sequence
                with patch.object(wf.runtime, 'poll', return_value=[]):
                    wf.poll()
                self.assertEqual(sequence, wf.journal.sequence)
        self.root = base

    def test_running_old_attempt_is_not_cancelled_before_a_receipt(self):
        wf, _, _, review = self.old_review()
        task = self.claim_old(wf, review)
        with patch.object(wf.runtime, 'poll', return_value=[]):
            wf.poll()
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'running')
        wf.dispatch()
        self.launch.assert_not_called()
        execution = task['task_id'] + '-lost-control'
        wf.budget.reserve(execution, 'llm', estimated_seconds=1)
        receipt = {'status': 'interrupted', 'execution_id': execution,
                   'error': {'code': 'worker_lost', 'message': 'Confirmed lost fixture process'}}
        path = wf.run_dir / 'control_receipts' / 'lost.json'
        atomic_json(path, receipt)
        wf.accept(task, receipt, path)
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'cancelled')
        self.assertIsNone(wf.budget.snapshot()['settlements'][execution]['reported_usage'])

    def test_pending_old_clarification_intent_is_archived_at_every_entry(self):
        base = self.root
        for entry in ('progress_request', 'return_code_clarifications', 'finish_clarified_request'):
            with self.subTest(entry=entry):
                self.root = base / entry
                wf, _, request, review = self.old_review()
                task = self.claim_old(wf, review)
                clarification = {'question': 'Old-host review question', 'reason': 'Scripted handoff only'}
                wf.queue.change(task['task_id'], 'needs_clarification', patch={'clarification': clarification})
                if entry == 'return_code_clarifications':
                    # Crash after cancelling the old child but before finishing the handoff.
                    wf.queue.change(task['task_id'], 'cancelled', patch={'error': {
                        'code': 'frozen_specification_clarification'}})
                returned = {'child': task['task_id'], 'descendants': [task['task_id']],
                            'result': {'status': 'rejected', 'reason': 'frozen_specification_clarification',
                                       'clarification': clarification}}
                wf.queue.change(request['task_id'], 'waiting', patch={'clarification_return': returned})
                before = set(wf.tasks())
                if entry == 'return_code_clarifications':
                    wf.return_code_clarifications()
                else:
                    getattr(wf, entry)(wf.queue.get(request['task_id']))
                current = wf.queue.get(request['task_id'])
                self.assertIsNone(current.get('clarification_return'))
                self.assertEqual(current['superseded_clarification_returns'], [returned])
                self.assertNotEqual(current['state'], 'completed')
                wf.return_code_clarifications()
                wf.progress_request(wf.queue.get(request['task_id']))
                self.assertEqual(len(self.current_reviews(wf, request)), 1)
                self.assertEqual(wf.queue.get(request['task_id'])['request_ref'], request['request_ref'])
                new = set(wf.tasks()) - before
                self.assertTrue(all('/safeguard/' in k for k in new))
                self.assertEqual(wf.queue.get(request['task_id'])['superseded_clarification_returns'], [returned])
        self.root = base

    def test_terminal_facts_old_approval_and_nonreview_roles_keep_their_boundaries(self):
        wf, _, request, review = self.old_review()
        with patch('core.experiment_requests.code_identity', return_value={'core_sha256': OLD}):
            task, _ = self.begin(wf, review)
            wf.submit(task, {'approved': True, 'rationale': 'Scripted historical review'}, 'SCRIPTED')
        task = wf.queue.get(review['task_id'])
        verdict = wf.journal.card(task['outputs'][0])
        experiment = wf.journal.card(request['experiment_ref'])
        candidate = task['inputs']['context']['candidate']
        with patch('runtime.identity.code_identity', return_value={'core_sha256': OLD}):
            old_approval = approval_receipt(experiment, candidate, verdict)
        failed = wf.enqueue_role('fixture/unreturned/safeguard/' + OLD[:16], 'safeguard',
            refs=review['inputs']['refs'], context=review['inputs']['context'])
        wf.queue.ready(); wf.queue.claim(failed)
        wf.queue.fail(failed, {'code': 'fixture_failure', 'message': 'Historical failed review'})
        ordinary = [wf.enqueue_role('fixture/' + purpose, purpose, refs=[wf.brief['id']], context={})
                    for purpose in ('generate', 'evolve', 'code')]
        before = {tid: wf.queue.get(tid) for tid in [review['task_id'], failed, *ordinary]}
        wf.retire_stale_safeguards()
        wf.retry_failed('Control fixture: preserve consumed/superseded work')
        for tid, value in before.items():
            self.assertEqual(wf.queue.get(tid), value)
        with self.assertRaisesRegex(ContractError, '核心实现'):
            validate_approval(old_approval, experiment)
        with self.assertRaisesRegex(ContractError, '核心实现'):
            wf.cached_experiment(experiment, request, old_approval)
        with self.assertRaisesRegex(ContractError, '过期'):
            wf.request_body(task)
        with self.assertRaisesRegex(ContractError, '过期'):
            wf.validate_review(task, {'approved': True})
        self.assertEqual(wf.journal.card(verdict['id']), verdict)
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))
