"""Targeted queue/receipt regressions; scripted decisions are not research evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, make_card
from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests import test_generation_loop as generation, test_memory_delivery as delivery
from tests import test_structured_output_repair as structured
from tests import test_code_repair as code_repair
from tests.test_structured_output_repair import BAD_FINALS, final_for
from tests.test_inner_json_repair import envelope
from tests.support import experiment_payload, hypothesis, review


class ClarificationReturn(TempCase):
    begin = delivery.DeliveryRegression.begin
    start = generation.GenerationLoop.start
    ask = generation.GenerationLoop.ask

    def setup_request(self, purpose='code', parent=True):
        wf = self.workflow(1)
        task, _ = self.start(wf)
        draft = experiment_payload()
        draft['execution']['requires_code'] = True
        draft['validation'] = {'mathematics': 'fixture specification', 'atol': 1e-6,
                              'rtol': 1e-6, 'cases': [{'seed': 1, 'values': [0., 1.]}]}
        request = self.ask(wf, task, draft)
        wf.progress_request(request)
        if not parent:
            inputs = wf.queue.get(request['task_id'])['inputs']
            inputs.pop('parent_task')
            wf.queue.change(request['task_id'], 'waiting', patch={'inputs': inputs})
        children = {t['inputs'].get('purpose'): t for t in wf.tasks().values()
                    if t['inputs'].get('parent_task') == request['task_id']}
        if purpose == 'safeguard':
            # This fixture tests return routing, not numerical approval.
            for child in children.values():
                wf.queue.ready(); wf.queue.claim(child['task_id'])
                wf.queue.finish(child['task_id'], result={'fixture': True})
            tid = wf.enqueue_role(request['key'] + '/safeguard', 'safeguard',
                context={'experiment_ref': wf.queue.get(request['task_id']).get('experiment_ref') or wf.queue.get(request['task_id'])['request_ref']},
                parent_task=request['task_id'])
            wf.queue.change(request['task_id'], 'waiting', patch={'waiting_for': [tid]})
            children[purpose] = wf.queue.get(tid)
        wf.queue.ready()
        child = wf.queue.claim(children[purpose]['task_id'])
        clarification = {'question': 'Specify exact rounding boundary',
                         'reason': 'Frozen mathematics needs a new draft'}
        wf.queue.change(child['task_id'], 'needs_clarification', patch={'clarification': clarification})
        return wf, task, request, children

    def test_code_and_safeguard_return_to_same_parent_and_preserve_original_request(self):
        root = self.root
        for purpose in ('code', 'safeguard'):
            with self.subTest(purpose=purpose):
                self.root = root / purpose
                wf, parent, request, children = self.setup_request(purpose)
                frozen = wf.journal.card(wf.queue.get(request['task_id']).get('experiment_ref') or wf.queue.get(request['task_id'])['request_ref'])
                wf = Workflow(wf.run_dir)
                wf.return_code_clarifications()
                parent, context = self.begin(wf, wf.queue.get(parent['task_id']))
                result = wf.queue.get(request['task_id'])['result']
                self.assertEqual(result['clarification_purpose'], purpose)
                self.assertIn('Specify exact rounding boundary', canonical(context['tool_results']))
                self.assertEqual(wf.journal.card(frozen['id']), frozen)
                self.assertEqual(wf.journal.find_cards('EvidenceCard'), [])
                self.assertEqual(len(wf.journal.find_cards('ExperimentCard')), 1)
                self.assertNotIn('reference', children)
                seq = wf.journal.sequence
                wf.return_code_clarifications()
                self.assertEqual(wf.journal.sequence, seq)

    def test_no_waiting_parent_remains_explicit_clarification(self):
        wf, _, request, children = self.setup_request('code', parent=False)
        seq = wf.journal.sequence
        wf.return_code_clarifications()
        self.assertEqual(wf.journal.sequence, seq)
        self.assertEqual(wf.queue.get(children['code']['task_id'])['state'], 'needs_clarification')
        self.assertEqual(wf.queue.get(request['task_id'])['state'], 'waiting')

    def test_return_intent_recovers_after_child_cancellation_and_claim(self):
        wf, parent, request, _ = self.setup_request('code')
        original_finish = wf.queue.finish
        def interrupted(tid, **kwargs):
            if tid == request['task_id']:
                raise RuntimeError('fixture interruption before return commit')
            return original_finish(tid, **kwargs)
        with patch.object(wf.queue, 'finish', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                wf.return_code_clarifications()
        wf = Workflow(wf.run_dir)
        wf.poll()  # Interrupted host operation becomes retry, without a worker call.
        wf.return_code_clarifications(); wf.queue.promote()
        self.assertEqual(wf.queue.get(parent['task_id'])['state'], 'ready')
        self.assertEqual(wf.queue.get(request['task_id'])['result']['clarification_purpose'], 'code')


class StepRepair(TempCase):
    attempt = structured.StructuredOutputRepair.attempt

    def test_independent_errors_repair_after_accepted_steps_and_restart(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role('r001/generate/H1', 'generate', refs=[wf.brief['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        deadline = wf.budget.snapshot()['deadline_unix']
        for index in range(3):
            self.attempt(wf, tid, BAD_FINALS[0][1]); wf.poll()
            current = wf.queue.get(tid)
            self.assertEqual(current['repair_count'], 1)
            self.assertEqual(current['contract_error_count'], index + 1)
            self.attempt(wf, tid, final_for('read_card', json.dumps({'ref': wf.brief['id']})))
            wf.poll(); wf.dispatch(); wf.queue.promote()
            wf = Workflow(wf.run_dir)
            current = wf.queue.get(tid)
            self.assertEqual(current['state'], 'ready')
            self.assertEqual(current['step'], index + 1)
            self.assertEqual(current['repair_count'], 0)
            self.assertEqual(wf.request_body(current)['spec']['context']['repair_feedback'], {})
            self.assertEqual(wf.budget.snapshot()['deadline_unix'], deadline)
        self.attempt(wf, tid, envelope(json.dumps({'action': 'submit', 'content': hypothesis(1, 'H1')})))
        wf.poll()
        self.assertEqual(wf.queue.get(tid)['state'], 'completed')
        self.assertEqual(wf.queue.get(tid)['contract_error_count'], 3)
        self.assertEqual(len(wf.budget.snapshot()['reservations']), 7)
        rejected = [e for e in wf.journal.events if e['type'] == 'task_snapshot'
                    and e['task'].get('repair_feedback', {}).get('rejected_receipt')]
        self.assertGreaterEqual(len(rejected), 3)

    def test_three_consecutive_errors_still_exhaust_same_step(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role('fixture/generate', 'generate', refs=[wf.brief['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        for _ in range(3):
            self.attempt(wf, tid, BAD_FINALS[0][1]); wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual((task['state'], task['repair_count'], task['contract_error_count']), ('failed', 2, 3))
        self.assertEqual(task['step'], 0)


class FrozenIdentityReturn(TempCase):
    begin = delivery.DeliveryRegression.begin
    start = generation.GenerationLoop.start
    ask = generation.GenerationLoop.ask

    def test_changed_implementation_returns_failure_without_launch_or_rewrite(self):
        root = self.root
        for kind in ('experiment',):
            with self.subTest(kind=kind):
                self.root = root / kind
                wf = self.workflow(1)
                parent, _ = self.start(wf)
                draft = experiment_payload()
                request = self.ask(wf, parent, draft)
                wf.progress_request(request)
                request = wf.queue.get(request['task_id'])
                frozen = wf.journal.card(request['experiment_ref'])
                child = next(t for t in wf.tasks().values()
                             if t['inputs'].get('parent_task') == request['task_id'])
                # The real experiment worker checks the bound identity before launch.
                wf.queue.change(child['task_id'], child['state'], patch={'kind': kind})
                with patch('tools.experiments.measurement_implementation', return_value={'files': {'changed': 'identity'}}), \
                        patch.object(wf.runtime, 'launch') as launch:
                    wf.dispatch()
                    launch.assert_not_called()
                self.assertEqual(wf.queue.get(child['task_id'])['state'], 'failed')
                wf = Workflow(wf.run_dir)
                wf.dispatch()
                parent, context = self.begin(wf, wf.queue.get(parent['task_id']))
                self.assertIn('测量实现已变化', canonical(context['tool_results']))
                self.assertEqual(wf.journal.card(frozen['id']), frozen)
                self.assertEqual(wf.journal.find_cards('EvidenceCard'), [])


class LongEvidence(TempCase):
    begin = delivery.DeliveryRegression.begin

    def test_oversized_projection_is_bounded_without_raw_read_gate(self):
        wf = self.workflow(1)
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        observations = {'rows': [{f'metric_{i}': n - 30 for i in range(900)} for n in range(60)],
                        'comparison': {'candidate': 2, 'baseline': 1},
                        'failure_reason': 'Keep exact failure reason', 'definition': 'descriptive fixture only'}
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'fixture', 'status': 'completed',
            'observations': observations, 'provenance': {'source': 'fixture only'}, 'resources': {},
            'artifacts': [], 'measurement_source': 'test_double', 'execution_id': 'fixture'}, refs=[spec['id']])
        wf.journal.commit('fixture/evidence', cards=[spec, evidence])
        view = wf.present_card(evidence)
        self.assertLess(len(canonical(view['payload'])), wf.PAGE_CHARACTERS)
        tid = wf.enqueue_role('fixture/generate', 'generate', refs=[wf.brief['id'], evidence['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        wf.queue.ready(); wf.queue.claim(tid)
        returned = wf.queue.enqueue('fixture/returned', 'experiment_request', inputs={
            'parent_task': tid, 'requester': 'generation', 'draft': experiment_payload()})
        wf.wait_for(wf.queue.get(tid), [returned])
        wf.queue.ready(); wf.queue.claim(returned)
        wf.queue.finish(returned, outputs=[evidence['id']], result={'status': 'completed'})
        task, context = self.begin(wf, wf.queue.get(tid))
        shown = next(c for c in context['memory']['cards'] if c['id'] == evidence['id'])
        self.assertEqual(shown, view)
        self.assertFalse(any(r['ref'] == evidence['id'] for r in context['memory']['required_reads']))
        stats = shown['payload']['observations']['rows']['numeric_summary']
        self.assertLess(len(stats), 900)
        coverage = shown['payload']['observations']['rows']['display_coverage']
        self.assertEqual(coverage['numeric_metrics_total'], 900)
        self.assertEqual(coverage['mixed_sign_metrics_total'], 900)
        from core.evidence import selected_leaves
        exact = list(selected_leaves(observations, '/payload/observations', {'metric': 'metric_899'}))
        self.assertEqual(sum(r['value'] < 0 for r in exact), 30)
        self.assertEqual(max(exact, key=lambda r: r['value'])['field'], '/payload/observations/rows/59/metric_899')
        self.assertEqual(wf.journal.card(evidence['id']), evidence)
        wf.submit(task, hypothesis(1, 'H1'), 'test_double')
        self.assertEqual(wf.queue.get(tid)['state'], 'completed')
