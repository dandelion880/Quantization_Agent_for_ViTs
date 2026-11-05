"""科研服务的真实队列交接；脚本角色/合同计算夹具，不调用 LLM 或模型。"""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical
from core.workflow import Workflow
from agents.roles import role_spec
from tests.test_infrastructure import TempCase
from tests import test_minimal_fixes as minimal
from tests import test_generation_loop as generation
from tests import test_infrastructure as infrastructure
from tests.support import experiment_payload, validation_config


class RequestContinuation(TempCase):
    planned = minimal.L3Regression.planned
    evidence = minimal.L3Regression.evidence
    measured = generation.GenerationLoop.measured

    def start_tool_request(self, purpose):
        config = validation_config()
        config.update(purpose='infrastructure_test')
        config['budget'].update(max_rounds=1, min_free_bytes=0)
        wf = Workflow(self.root / purpose, config)
        parent = wf.enqueue_role('fixture/' + purpose, purpose, refs=[wf.brief['id']], context={'round': 1})
        wf.queue.ready(); wf.queue.claim(parent)
        tid = wf.queue.enqueue('fixture/request', 'experiment_request', inputs={
            'draft': experiment_payload(), 'requester': wf.queue.get(parent)['role'],
            'refs': [wf.brief['id']], 'parent_task': parent})
        wf.wait_for(wf.queue.get(parent), [tid])
        wf.queue.ready()
        return wf, parent, wf.queue.claim(tid)

    def test_every_research_output_kind_receives_measurement_without_extra_interpretation(self):
        # plan_l3's ExperimentCard output was the overlooked distinct contract.
        for purpose in ('generate', 'evolve', 'screen', 'plan_l3'):
            with self.subTest(purpose=purpose):
                wf, parent, request = self.start_tool_request(purpose)
                evidence = self.measured(wf, request, 1)
                wf.queue.ready()
                resumed = wf.queue.get(parent)
                self.assertEqual(resumed['state'], 'ready')
                self.assertEqual(resumed['tool_results'][-1]['result']['evidence_ref'], evidence['id'])
                self.assertFalse(any(t['inputs'].get('purpose') == 'interpret' for t in wf.tasks().values()))

    def test_final_l3_clarification_returns_to_new_plan_and_survives_recovery(self):
        for purpose in ('code', 'safeguard'):
            with self.subTest(purpose=purpose):
                wf, trial = self.planned('clarify-' + purpose, requires_code=True)
                original_plan = deepcopy(wf.tasks()['r001/plan_l3'])
                wf.progress_request(trial)
                child = wf.tasks()['r001/L3/code']
                if purpose == 'safeguard':
                    # Route a real Safeguard task result through the same request;
                    # numerical/approval execution itself is covered by code_repair.
                    wf.queue.ready(); wf.queue.claim(child['task_id'])
                    wf.queue.finish(child['task_id'], result={'fixture': 'routing only'})
                    sid = wf.enqueue_role('r001/L3/safeguard/routing', 'safeguard',
                        refs=[wf.brief['id']], context={}, parent_task=trial['task_id'])
                    wf.queue.change(trial['task_id'], 'waiting', patch={'waiting_for': [sid]})
                    child = wf.queue.get(sid)
                wf.queue.ready(); wf.queue.claim(child['task_id'])
                clarification = {'question': 'Which already available statistic defines this axis?',
                                 'reason': 'Different axes test different mechanisms'}
                wf.queue.change(child['task_id'], 'needs_clarification', patch={'clarification': clarification})
                wf = Workflow(wf.run_dir)
                wf.advance(); wf.queue.ready()
                resumed = wf.tasks()['r001/plan_l3/revision1']
                self.assertEqual(resumed['state'], 'ready')
                self.assertEqual(wf.queue.get(trial['task_id'])['result']['clarification'], clarification)
                self.assertEqual(wf.queue.get(child['task_id'])['state'], 'cancelled')
                self.assertEqual(wf.tasks()['r001/plan_l3'], original_plan)
                self.assertNotIn('parent_task', wf.queue.get(trial['task_id'])['inputs'])
                self.assertEqual(wf.queue.get(trial['task_id'])['inputs']['origin_task'], original_plan['task_id'])
                self.assertIn('request_feedback', resumed['inputs']['context'])
                before = set(wf.tasks()); wf.advance()
                self.assertEqual(set(wf.tasks()), before)
                # A revised scientist submission creates a new immutable service request.
                revised = wf.queue.claim(resumed['task_id'])
                draft = deepcopy(trial['inputs']['draft'])
                draft['question'] = 'Revised question after resolving statistical axis from available sources'
                wf.submit(revised, draft, 'scripted fixture')
                wf.advance()
                next_trial = wf.tasks()['r001/L3/revision1']
                self.assertEqual(next_trial['inputs']['previous_request_task'], trial['task_id'])
                self.assertEqual(next_trial['inputs']['original_request']['question'], draft['question'])
                self.assertEqual(wf.queue.get(trial['task_id'])['inputs']['draft'], trial['inputs']['draft'])

    def test_final_l3_execution_and_budget_failures_resume_author_and_can_defer_truthfully(self):
        for code in ('RequestBudgetInsufficient', 'TypeError', 'implementation_unchanged_after_failure'):
            with self.subTest(code=code):
                wf, trial = self.planned('failure-' + code)
                if code == 'implementation_unchanged_after_failure':
                    wf.queue.finish(trial['task_id'], result={'status': 'rejected', 'reason': code})
                else:
                    wf.queue.fail(trial['task_id'], {'code': code, 'message': 'Specific unavailable execution condition'})
                wf.advance(); wf.queue.ready()
                wf.retry_failed('resume remaining work, retain consumed failure')
                self.assertEqual(wf.queue.get(trial['task_id'])['state'], 'completed' if code == 'implementation_unchanged_after_failure' else 'failed')
                author = wf.tasks()['r001/plan_l3/revision1']
                self.assertIn(code, canonical(author['inputs']['context']['request_feedback']))
                wf.submit(wf.queue.claim(author['task_id']), {'defer_reason': 'Current budget cannot complete the required protocol'}, 'fixture')
                wf.advance()
                self.assertIn('r001/record', wf.tasks())
                outcome = wf.l3_outcome(1)
                self.assertEqual(outcome['status'], 'deferred')
                self.assertFalse(outcome['formal_validation_success'])
                self.assertFalse(wf.journal.find_cards('EvidenceCard'))

    def test_program_request_budget_check_returns_actual_refusal_to_author(self):
        wf, trial = self.planned('actual-budget', estimated_seconds=10**9, claim_request=False)
        wf.dispatch()
        self.assertEqual(wf.queue.get(trial['task_id'])['state'], 'failed')
        self.assertEqual(wf.queue.get(trial['task_id'])['error']['code'], 'RequestBudgetInsufficient')
        wf.advance(); wf.queue.ready()
        author = wf.tasks()['r001/plan_l3/revision1']
        self.assertEqual(author['state'], 'ready')
        self.assertIn('RequestBudgetInsufficient', canonical(author['inputs']['context']['request_feedback']))
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))

    def test_program_and_cached_results_are_immediately_available_and_not_independent_repeats(self):
        wf, trial = self.planned('immediate')
        experiment, evidence = self.evidence(wf, trial)
        wf.progress_request(wf.queue.get(trial['task_id']))
        first = wf.queue.get(trial['task_id'])
        self.assertEqual(first['state'], 'completed')
        self.assertFalse(first['result']['cache_reused'])
        worker = wf.tasks()['r001/L3/execute']
        # The host/worker cache path must resolve the program request's author too.
        worker_view = {**worker, 'inputs': {**worker['inputs'], 'parent_task': trial['task_id']}}
        self.assertEqual(wf.request_author(worker_view)['task_id'], wf.tasks()['r001/plan_l3']['task_id'])
        def visible_cache(spec, *, visible, **kwargs):
            self.assertIsNotNone(visible)
            return evidence if visible(evidence) else None
        with patch.object(wf.broker, 'cached', side_effect=visible_cache):
            self.assertEqual(wf.cached_experiment(experiment, worker_view)['id'], evidence['id'])
        cached = wf.queue.enqueue('fixture/cached', 'experiment_request', inputs=trial['inputs'])
        wf.queue.ready(); cached = wf.queue.claim(cached)
        with patch.object(wf.broker, 'cached', return_value=evidence):
            wf.progress_request(cached)
        self.assertTrue(wf.queue.get(cached['task_id'])['result']['cache_reused'])
        self.assertEqual(wf.queue.get(cached['task_id'])['outputs'], [evidence['id']])
        self.assertEqual(len([t for t in wf.tasks().values() if t['kind'] == 'experiment']), 1)
        self.assertFalse(any(t['inputs'].get('purpose') == 'interpret' for t in wf.tasks().values()))
        self.assertTrue(wf.summary('running')['formal_validation_success'])
        self.assertFalse(wf.summary('running')['workflow_completed'])

    def test_legacy_waiting_interpretation_recovers_without_rerun_or_wait(self):
        wf, trial = self.planned('legacy-interpret')
        experiment, evidence = self.evidence(wf, trial)
        interpretation = wf.enqueue_role('r001/L3/interpret', 'interpret', refs=[experiment['id'], evidence['id']],
                                         context={}, parent_task=trial['task_id'])
        wf.queue.ready(); wf.queue.claim(interpretation)
        wf.wait_for(wf.queue.get(trial['task_id']), [interpretation])
        wf = Workflow(wf.run_dir)
        wf.release_measured_requests()
        request = wf.queue.claim(trial['task_id'])
        wf.progress_request(request)
        self.assertEqual(wf.queue.get(trial['task_id'])['outputs'], [evidence['id']])
        wf.queue.fail(interpretation, {'code': 'fixture_interpretation_failed'})
        self.assertTrue(wf.l3_outcome(1)['formal_validation_success'])
        self.assertEqual(wf.l3_outcome(1)['interpretation_state'], 'failed')
        self.assertNotIn('r001/plan_l3/revision1', wf.tasks())
        self.assertEqual(len([t for t in wf.tasks().values() if t['kind'] == 'experiment']), 1)

    def test_unstarted_legacy_interpret_is_cancelled_after_valid_evidence(self):
        wf, trial = self.planned('legacy-unstarted')
        experiment, evidence = self.evidence(wf, trial)
        interpretation = wf.enqueue_role('r001/L3/interpret', 'interpret', refs=[experiment['id'], evidence['id']],
                                         context={}, parent_task=trial['task_id'])
        wf.wait_for(wf.queue.get(trial['task_id']), [interpretation])
        wf.release_measured_requests()
        self.assertEqual(wf.queue.get(interpretation)['state'], 'cancelled')
        wf.progress_request(wf.queue.claim(trial['task_id']))
        self.assertTrue(wf.l3_outcome(1)['formal_validation_success'])
        self.assertFalse(any(t['inputs'].get('purpose') == 'interpret' for t in wf.queue.ready()))
        wf.advance(); wf.queue.ready()
        recorder = wf.tasks()['r001/record']
        wf.queue.claim(recorder['task_id'])
        wf.queue.fail(recorder['task_id'], {'code': 'fixture_recorder_failed'})
        report = wf.summary('running')
        self.assertTrue(report['formal_validation_success'])
        self.assertFalse(report['scientific_review_complete'])
        # Recorder remains independently recoverable; settled experiment children do not reopen.
        wf.retry_failed('recover deferred scientific review')
        self.assertEqual(wf.queue.get(recorder['task_id'])['state'], 'ready')
        self.assertEqual(wf.queue.get(trial['task_id'])['state'], 'completed')

    def test_two_round_full_scientific_structure_with_l3_revision_and_resume(self):
        config = validation_config()
        config.update(purpose='infrastructure_test', enable_l3=True)
        config['budget'].update(max_rounds=2, min_free_bytes=0)
        wf = Workflow(self.root / 'two-rounds', config)
        revised, restored, measurements = False, False, []
        for _ in range(160):
            if wf.advance():
                break
            for ready in wf.queue.ready():
                task = wf.queue.claim(ready['task_id'])
                if task['kind'] == 'experiment_request':
                    if task['key'] == 'r001/L3' and not revised:
                        wf.queue.fail(task['task_id'], {'code': 'RequestBudgetInsufficient',
                            'message': 'Scripted request refusal; a revised feasible estimate is needed'})
                        revised = True
                    else:
                        _, evidence = self.evidence(wf, task)
                        measurements.append(evidence['id'])
                        wf.progress_request(wf.queue.get(task['task_id']))
                elif task['inputs']['purpose'] == 'plan_l3':
                    draft = experiment_payload('L3')
                    draft['execution'].update(service='evaluate_artifact', parameters={}, evaluations=[{
                        'kind': 'classification', 'split': 'final', 'count': 2}])
                    if task['inputs']['context'].get('plan_revision'):
                        self.assertIn('RequestBudgetInsufficient', canonical(task['inputs']['context']['request_feedback']))
                        draft['question'] += ' (revised after explicit feasibility feedback)'
                    wf.submit(task, draft, 'scripted fixture')
                else:
                    self.assertEqual(task['kind'], 'llm')
                    wf.submit(task, infrastructure.WorkflowSimulation.scripted(self, wf, task), 'scripted fixture')
            if revised and not restored and 'r001/plan_l3/revision1' in wf.tasks():
                wf = Workflow(wf.run_dir)
                wf.retry_failed('continue revised plan without reopening superseded failures')
                self.assertEqual(wf.tasks()['r001/L3']['state'], 'failed')
                restored = True
        else:
            self.fail('two-round full structure did not finish')
        self.assertTrue(restored)
        self.assertEqual(len(measurements), 2)
        for number in (1, 2):
            hypotheses = wf.round_hypotheses(number)
            self.assertEqual(set(hypotheses), {'H1', 'H2', 'H3', 'H4'})
            self.assertEqual(len(wf.matches(number)), 4)
            parent = hypotheses['H4']['parents'][0]
            slot = next(k for k, h in hypotheses.items() if h['id'] == parent)
            self.assertIn('H4-' + slot, wf.matches(number))
            self.assertTrue(wf.l3_outcome(number)['formal_validation_success'])
        self.assertTrue(all(h['payload']['feedback_used'] for h in wf.round_hypotheses(2).values()))
        self.assertFalse(any(t['inputs'].get('purpose') == 'interpret' for t in wf.tasks().values()))
        self.assertTrue(wf.summary('completed')['scientific_review_complete'])
        self.assertEqual(wf.tasks()['r001/L3']['state'], 'failed')
        self.assertEqual(wf.tasks()['r001/L3/revision1']['state'], 'completed')

    def test_defer_is_visible_optional_and_never_mixed_with_a_fabricated_plan(self):
        spec = role_spec('judger', 'plan_l3', {}, {})
        self.assertIn('defer_reason', spec['context']['payload_contract']['alternative']['properties'])
        wf, trial = self.planned('defer-contract')
        wf.queue.fail(trial['task_id'], {'code': 'fixture_failure'})
        wf.advance(); wf.queue.ready()
        plan = wf.queue.claim(wf.tasks()['r001/plan_l3/revision1']['task_id'])
        with self.assertRaises(ContractError):
            wf.submit(plan, {'defer_reason': 'cannot execute', 'level': 'L3'}, 'fixture')
        self.assertEqual(wf.queue.get(plan['task_id'])['state'], 'running')
