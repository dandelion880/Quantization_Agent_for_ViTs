"""Explicit intervention control fixtures; no research models or workers."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from runtime.identity import code_identity, measurement_implementation
from tests.test_infrastructure import TempCase
from tests import test_single_implementation as single
from tools.experiments import asset, verify_assets
from tools.safeguards import validate_approval


class ExecutionBindingRecovery(TempCase):
    begin = single.SingleImplementation.begin
    start = single.SingleImplementation.start
    coding = single.SingleImplementation.coding
    progress = single.SingleImplementation.progress
    approve = single.SingleImplementation.approve

    def stale_binding(self, *, trial=True):
        wf, author, request = self.start()
        with patch('tools.experiments.measurement_implementation', return_value={'files': {'old_worker': '0' * 64}}):
            request = self.coding(wf, request)
        if trial:
            request = self.approve(wf, request)
        return wf, author, request

    def intervention(self, wf):
        wf.event('implementation_intervention', 'fixture/intervention', previous_core_sha256='0' * 64,
                 current_core_sha256=code_identity()['core_sha256'], reason='Scripted host repair')

    def test_pending_trial_rebinds_before_dispatch_and_requires_new_review(self):
        wf, _, request = self.stale_binding()
        old = wf.journal.card(request['experiment_ref'])
        original_request = wf.journal.card(request['request_ref'])
        trial = wf.tasks()[request['key'] + '/execute']
        old_review = next(
            t for t in wf.tasks().values() if t['inputs'].get('purpose') == 'safeguard')
        self.intervention(wf)
        with patch.object(wf.runtime, 'launch') as launch:
            wf.dispatch()
            self.assertFalse(any(call.args[0]['kind'] == 'experiment' for call in launch.call_args_list))
            wf.dispatch()
            self.assertFalse(any(call.args[0]['kind'] == 'experiment' for call in launch.call_args_list))
        rebound = wf.queue.get(request['task_id'])
        new = wf.journal.card(rebound['experiment_ref'])
        self.assertNotEqual(new['id'], old['id'])
        self.assertEqual(new['payload']['execution']['measurement_implementation'], measurement_implementation())
        self.assertEqual(wf.journal.card(original_request['id']), original_request)
        before, after = deepcopy(old['payload']), deepcopy(new['payload'])
        before['execution'].pop('measurement_implementation')
        after['execution'].pop('measurement_implementation')
        self.assertEqual(before, after)
        self.assertEqual(rebound.get('code_revision', 0), request.get('code_revision', 0))
        self.assertEqual(wf.queue.get(trial['task_id'])['inputs'], trial['inputs'])
        self.assertEqual(wf.queue.get(old_review['task_id']), old_review)
        reviews = [t for t in wf.tasks().values() if t['inputs'].get('purpose') == 'safeguard']
        self.assertEqual(len(reviews), 2)
        pending = next(t for t in reviews if t['state'] != 'completed')
        self.assertIn('/binding/', pending['key'])
        self.assertEqual(pending['inputs']['context']['experiment_ref'], new['id'])
        # A stop after publishing the new binding must not leave the request
        # waiting forever for the still-held old numerical input.
        wf.queue.change(rebound['task_id'], 'waiting', patch={'waiting_for': [trial['task_id']]})
        self.assertIn(trial['task_id'], wf.wake_pending_execution_rebinds())
        self.assertEqual(wf.queue.get(rebound['task_id'])['state'], 'ready')
        rebound = self.approve(wf, rebound)
        current = wf.queue.get(trial['task_id'])
        self.assertEqual(current['inputs']['experiment_ref'], new['id'])
        self.assertEqual(current['attempt'], trial['attempt'])
        self.assertEqual(current['superseded_execution_inputs'][0]['inputs'], trial['inputs'])
        validate_approval(current['inputs']['approval'], new)
        verify_assets(new['payload']['execution'])
        self.assertEqual(wf.request_body(current)['experiment']['id'], new['id'])
        count = len(wf.journal.find_cards('ExperimentCard'))
        wf.refresh_pending_execution(wf.queue.get(request['task_id']))
        self.assertEqual(len(wf.journal.find_cards('ExperimentCard')), count)

    def test_interrupted_attempt_keeps_consumption_attempt_error_and_new_inputs(self):
        wf, _, request = self.stale_binding()
        trial = wf.tasks()[request['key'] + '/execute']
        wf.queue.ready()
        wf.queue.claim(trial['task_id'], input_versions={request['experiment_ref']: 'old'}, request_path='old-request.json')
        error = {'code': 'InterruptedError', 'message': 'task stopped by signal 15', 'origin': 'host'}
        wf.queue.fail(trial['task_id'], error, retryable=True)
        prior = wf.queue.get(trial['task_id'])
        wf.budget.reserve('settled-before-intervention', 'experiment', estimated_seconds=1)
        wf.budget.settle('settled-before-intervention', status='completed', resources={'wall_seconds': 1})
        before_budget = wf.budget.snapshot()
        self.intervention(wf)
        held = wf.wake_pending_execution_rebinds()
        self.assertIn(trial['task_id'], held)
        request = self.progress(wf, request)
        request = self.approve(wf, request)
        current = wf.queue.get(trial['task_id'])
        self.assertEqual(current['task_id'], prior['task_id'])
        for key in ('attempt', 'retry_count', 'max_retries', 'error', 'request_path'):
            self.assertEqual(current.get(key), prior.get(key))
        self.assertEqual(current['input_versions'], {})
        previous = current['superseded_execution_inputs'][0]
        self.assertEqual(previous['input_versions'], prior['input_versions'])
        self.assertEqual(previous['error'], error)
        self.assertEqual(previous['attempt'], 1)
        self.assertEqual(wf.budget.snapshot()['settlements'], before_budget['settlements'])
        validate_approval(current['inputs']['approval'], wf.journal.card(request['experiment_ref']))

    def test_missing_trial_can_refresh_but_no_intervention_cannot(self):
        wf, _, request = self.stale_binding(trial=False)
        self.assertFalse(wf.pending_execution_rebind(request))
        self.assertEqual(wf.refresh_pending_execution(request)['experiment_ref'], request['experiment_ref'])
        self.intervention(wf)
        current = wf.refresh_pending_execution(request)
        self.assertNotEqual(current['experiment_ref'], request['experiment_ref'])
        self.assertEqual(current['request_ref'], request['request_ref'])
        self.assertEqual(current['bound_code_id'], request['bound_code_id'])

    def test_new_coding_revision_defers_refresh_of_older_bound_candidate(self):
        wf, _, request = self.stale_binding()
        self.intervention(wf)
        wf.queue.change(request['task_id'], 'ready', patch={
            'code_revision': 1, 'code_revision_feedback': {'instruction': 'Scripted pending revision'},
            'waiting_for': []})
        request = self.progress(wf, request)
        code = wf.tasks()[request['key'] + '/code/revision1']
        count = len(wf.journal.find_cards('ExperimentCard'))
        for state in ('pending', 'running'):
            if state == 'running':
                wf.queue.ready(); wf.queue.claim(code['task_id'])
            with self.subTest(state=state):
                self.assertEqual(wf.queue.get(code['task_id'])['state'], state)
                current = wf.queue.get(request['task_id'])
                self.assertFalse(wf.pending_execution_rebind(current))
                wf.progress_request(current)
                current = wf.queue.get(request['task_id'])
                self.assertEqual(current['state'], 'waiting')
                self.assertEqual(current['waiting_for'], [code['task_id']])
                self.assertEqual(current['experiment_ref'], request['experiment_ref'])
                self.assertEqual(len(wf.journal.find_cards('ExperimentCard')), count)
                self.assertNotIn('execution_binding_history', current)

    def test_completed_failed_and_returned_requests_are_not_reopened(self):
        base = self.root
        for state in ('completed', 'failed', 'returned'):
            with self.subTest(state=state):
                self.root = base / state
                wf, _, request = self.stale_binding()
                trial = wf.tasks()[request['key'] + '/execute']
                wf.queue.ready(); wf.queue.claim(trial['task_id'])
                if state == 'completed':
                    wf.queue.finish(trial['task_id'], result={'historical_fact': True})
                else:
                    wf.queue.fail(trial['task_id'], {'code': 'ValueError', 'message': 'actual execution failed'})
                if state == 'returned':
                    wf.queue.fail(request['task_id'], {'code': 'service_failed', 'message': 'already returned'})
                self.intervention(wf)
                task = wf.queue.get(request['task_id'])
                trial_before = wf.queue.get(trial['task_id'])
                self.assertFalse(wf.pending_execution_rebind(task))
                self.assertEqual(wf.refresh_pending_execution(task), task)
                self.assertEqual(wf.queue.get(trial['task_id']), trial_before)
        self.root = base

    def test_rebinding_still_rejects_changed_frozen_assets(self):
        wf, _, request = self.stale_binding()
        original = wf.journal.card(request['experiment_ref'])
        source = self.root / 'frozen-data.json'
        source.write_text('original')
        payload = deepcopy(original['payload'])
        payload['execution']['test_asset'] = asset(source)
        frozen = make_card('ExperimentCard', wf.journal.run_id, payload, refs=original['refs'], author=original['author'])
        wf.queue.change(request['task_id'], request['state'], patch={'experiment_ref': frozen['id']}, cards=[frozen])
        source.write_text('changed')
        self.intervention(wf)
        with self.assertRaisesRegex(ContractError, '资产已变化'):
            wf.refresh_pending_execution(wf.queue.get(request['task_id']))
        self.assertEqual(wf.queue.get(request['task_id'])['experiment_ref'], frozen['id'])
