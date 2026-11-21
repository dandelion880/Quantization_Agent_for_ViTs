"""G-only CPU lifecycle tests. Scripted decisions are not scientific measurements."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cards import ContractError, make_card, fingerprint
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from runtime.configuration import config_check, model_defaults
from runtime.model_router import client_for
from agents.tasks import TASKS, RETIRED_PURPOSES
from tests.support import advance_with_topic_fixture
from tests.support import validation_config, hypothesis, review, meta
from tests.test_infrastructure import TempCase


class GOnlyRegression(TempCase):
    def setUp(self):
        super().setUp()
        guard = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU fixture cannot call LLM'))
        guard.start()
        self.addCleanup(guard.stop)

    def workflow_case(self, name='research', rounds=1, l3=True):
        config = validation_config()
        config.update(purpose='infrastructure_test', enable_l3=l3)
        config['budget'].update(max_rounds=rounds, min_free_bytes=0)
        return Workflow(self.root / name, config)

    def drive(self, wf, *, choice='H1', stop=None, reload=False):
        for _ in range(60):
            if advance_with_topic_fixture(wf):
                return wf, None
            ready = wf.queue.ready()
            self.assertTrue(ready, 'G-only stalled')
            for candidate in ready:
                task = wf.queue.claim(candidate['task_id'])
                c, purpose = task['inputs']['context'], task['inputs']['purpose']
                self.assertNotIn(purpose, RETIRED_PURPOSES)
                if stop and stop(task):
                    return wf, task
                if purpose == 'generate':
                    value = hypothesis(c['round'], c['slot'])
                    latest = wf.memory.snapshot()['L3']
                    if latest:
                        value['feedback_used'] = [{'metareview_ref': latest[-1], 'change': 'revise from previous feedback'}]
                elif purpose == 'select':
                    value = review('ranking', selected_slot=choice, selection_basis='CPU fixture evidence preselection')
                elif purpose == 'plan_l3':
                    value = {'defer_reason': 'CPU fixture; real full-dataset L3 is outside test scope'}
                elif purpose == 'record':
                    value = meta(c['round'], c['recording_batch']['judgment_refs'])
                else:
                    self.fail('unexpected task ' + purpose)
                wf.submit(task, value, 'SCRIPTED_CPU_ONLY')
            if reload:
                wf = Workflow(wf.run_dir)
        self.fail('G-only did not finish')

    def test_every_candidate_can_reach_l3_and_recorder_without_debate(self):
        for choice in ('H1', 'H2', 'H3'):
            with self.subTest(choice=choice):
                wf, plan = self.drive(self.workflow_case(choice), choice=choice,
                                      stop=lambda t: t['inputs']['purpose'] == 'plan_l3' and t['inputs']['context'].get('slot') == choice)
                selected = wf.selection_card(1)
                self.assertEqual(plan['role'], 'generation')
                self.assertEqual(selected['author'], 'generation')
                self.assertTrue(selected['payload']['provisional'])
                self.assertEqual(plan['inputs']['context']['hypothesis_ref'], wf.round_hypotheses(1)[choice]['id'])
                self.assertEqual(plan['inputs']['context']['selection_ref'], selected['id'])
                self.assertEqual(set(wf.round_hypotheses(1)), {'H1', 'H2', 'H3'})
                self.assertEqual(wf.matches(1), {})
                wf.submit(plan, {'defer_reason': 'CPU test budget'}, None)
                wf, _ = self.drive(wf)
                record = wf.tasks()['r001/record']
                self.assertEqual(record['inputs']['context']['selection_ref'], selected['id'])
                self.assertFalse(wf.summary()['formal_validation_success'])

    def test_preselection_freezes_candidate_versions_and_is_idempotent_after_resume(self):
        wf, task = self.drive(self.workflow_case(), stop=lambda t: t['inputs']['purpose'] == 'select')
        candidates = deepcopy(task['inputs']['context']['candidate_refs'])
        original = wf.journal.card(candidates['H2'])
        changed = make_card('HypothesisCard', wf.journal.run_id, {**original['payload'], 'title': 'later version'}, previous=original)
        wf.journal.commit('fixture/later', cards=[changed])
        for value in ({'selected_slot': 'H4'}, {'selected_slot': 'H2', 'selected_hypothesis_ref': changed['id']},
                      {'selected_slot': 'H2', 'provisional': False}):
            with self.subTest(value=value), self.assertRaises(ContractError):
                wf.submit(task, {'rationale': 'fixture', **value}, None)
        wf.submit(task, {'selected_slot': 'H2', 'rationale': 'fixture'}, None)
        selected = wf.selection_card(1)
        self.assertEqual(selected['payload']['selected_hypothesis_ref'], original['id'])
        with self.assertRaises(ContractError):
            wf.submit(task, {'selected_slot': 'H1', 'rationale': 'rewrite'}, None)
        wf = Workflow(wf.run_dir)
        advance_with_topic_fixture(wf)
        keys, sequence = set(wf.tasks()), wf.journal.sequence
        for _ in range(3):
            self.assertFalse(advance_with_topic_fixture(wf))
        self.assertEqual((keys, sequence), (set(wf.tasks()), wf.journal.sequence))
        self.assertEqual(wf.tasks()[wf.l3_request_keys(1, 'H2', original['id'])[0]]['inputs']['context']['hypothesis_ref'], original['id'])

    def test_generation_completion_orders_join_once_with_all_three_versions(self):
        from itertools import permutations
        for order in permutations(('H1', 'H2', 'H3')):
            wf = self.workflow_case(''.join(order), l3=False)
            advance_with_topic_fixture(wf)
            for i, slot in enumerate(order):
                wf.queue.ready()
                task = wf.queue.claim(wf.tasks()['r001/generate/' + slot]['task_id'])
                wf.submit(task, hypothesis(1, slot), None)
                wf = Workflow(wf.run_dir)
                advance_with_topic_fixture(wf)
                self.assertEqual('r001/select' in wf.tasks(), i == 2)
            self.assertEqual(set(wf.tasks()['r001/select']['inputs']['context']['candidate_refs']), set(order))

    def test_failed_h2_does_not_block_completed_h1_h3_and_selection(self):
        wf, failed = self.drive(self.workflow_case(l3=False), stop=lambda t: t['key'] == 'r001/generate/H2')
        wf.queue.fail(failed['task_id'], {'code': 'fixture_local_failure'})
        wf, select = self.drive(wf, stop=lambda t: t['inputs']['purpose'] == 'select')
        self.assertEqual(select['inputs']['context']['incomplete_slots'], ['H2'])
        self.assertEqual(set(select['inputs']['context']['candidate_refs']), {'H1', 'H3'})
        self.assertIn(wf.queue.get(failed['task_id'])['task_card'], select['inputs']['refs'])
        wf.submit(select, {'selected_slot': 'H1', 'rationale': 'retain mature fallback'}, None)
        wf, _ = self.drive(wf)
        wf.retry_failed('explicit recovery after completed partial round')
        self.assertEqual(wf.queue.get(failed['task_id'])['state'], 'failed')
        self.assertEqual(wf.summary()['completed_rounds'], [1])

    def test_failed_selection_keeps_candidates_and_library_without_selecting_h1(self):
        wf, select = self.drive(self.workflow_case(), stop=lambda t: t['inputs']['purpose'] == 'select')
        library = wf.solutions.snapshot()
        wf.queue.fail(select['task_id'], {'code': 'fixture_selection_failure'})
        wf = Workflow(wf.run_dir)
        self.assertFalse(wf.advance())
        self.assertEqual(wf.summary()['completed_rounds'], [])
        self.assertTrue(any(e['type'] == 'synthesis_pending' for e in wf.journal.events))
        self.assertIsNone(wf.selection_card(1))
        self.assertNotIn('r001/plan_l3', wf.tasks())
        self.assertEqual(set(wf.round_hypotheses(1)), {'H1', 'H2', 'H3'})
        self.assertEqual(wf.solutions.snapshot(), library)
        self.assertFalse(wf.summary()['formal_validation_success'])

    def test_all_generation_failures_keep_library_and_finish_without_fake_hypotheses(self):
        wf = self.workflow_case()
        original = wf.solutions.snapshot()
        advance_with_topic_fixture(wf)
        for t in wf.queue.ready():
            wf.queue.claim(t['task_id'])
            wf.queue.fail(t['task_id'], {'code': 'fixture_failure'})
        wf, _ = self.drive(wf)
        self.assertFalse(wf.round_hypotheses(1))
        self.assertFalse(wf.selection_card(1))
        self.assertNotIn('r001/plan_l3', wf.tasks())
        self.assertEqual(wf.solutions.snapshot(), original)
        recorder = wf.tasks()['r001/record']
        self.assertEqual(recorder['state'], 'completed')
        self.assertEqual(len(recorder['inputs']['context']['recording_batch']['scientific_failure_refs']), 3)

    def test_null_preference_preserves_all_three_l3_candidates(self):
        wf, _ = self.drive(self.workflow_case(), choice=None)
        self.assertIsNone(wf.selection_card(1)['payload']['selected_slot'])
        self.assertEqual(set(wf.l3_outcome(1)['candidates']), {'H1', 'H2', 'H3'})
        self.assertTrue(all(r['status'] == 'deferred' for r in wf.l3_outcome(1)['candidates'].values()))
        self.assertEqual(wf.tasks()['r001/record']['state'], 'completed')

    def test_two_rounds_and_restores_use_preselection_feedback_without_legacy_tasks(self):
        wf, _ = self.drive(self.workflow_case(rounds=2, l3=False), choice='H3', reload=True)
        previous = wf.result_card(wf.tasks()['r001/record'])
        for slot in ('H1', 'H2', 'H3'):
            task = wf.tasks()['r002/generate/' + slot]
            self.assertIn(previous['id'], task['inputs']['refs'])
            self.assertTrue(wf.round_hypotheses(2)[slot]['payload']['feedback_used'])
        self.assertEqual(wf.memory.snapshot()['L1']['round_selection']['review_ref'], wf.selection_card(2)['id'])
        self.assertFalse(RETIRED_PURPOSES.intersection(t['inputs'].get('purpose') for t in wf.tasks().values()))

    def test_generation_route_defaults_overrides_and_direct_router_cannot_downgrade(self):
        default = model_defaults()
        self.assertEqual(default['generation'], ['gpt-6.2-astra', 'high'])
        self.assertNotIn('judger', default)
        self.assertNotIn('evolution', default)
        for purpose in ('generate', 'select', 'plan_l3'):
            self.assertEqual(TASKS[purpose]['role'], 'generation')
        config_check(validation_config())
        for route in (['gpt-6.1-sol', 'high'], ['gpt-6-astra', 'medium']):
            config = validation_config()
            config['llm']['agent_models'] = {**default, 'generation': route}
            with self.assertRaisesRegex(ContractError, 'Astra / high'):
                config_check(config)
            with self.assertRaisesRegex(ContractError, 'Astra / high'):
                client_for('generation', config['llm'])

    def test_preselection_can_experiment_resume_and_pass_evidence_to_l3(self):
        from tests import test_generation_loop as loop
        from tests import test_memory_delivery as delivery
        from tests.support import experiment_payload
        wf, task = self.drive(self.workflow_case(), stop=lambda t: t['inputs']['purpose'] == 'select')
        # drive() has already claimed this task; freeze its request before tools.
        body = wf.request_body(task)
        path = wf.run_dir / 'selection_request.json'
        atomic_json(path, body)
        wf.queue.change(task['task_id'], 'running', patch={'request_path': str(path)})
        task, context = wf.queue.get(task['task_id']), body['spec']['context']
        self.assertIn('skill_catalog', context)
        self.assertIn('complete_solutions', context)
        self.assertIn('current_research_state', context['memory'])
        results = []
        for level in ('L1', 'L2'):
            request = loop.GenerationLoop.ask(self, wf, task, experiment_payload(level),
                                             note='CPU fixture: revise decision after contrary result')
            results.append(loop.GenerationLoop.measured(self, wf, request, len(results) + 1))
            wf = Workflow(wf.run_dir)
            task, context = delivery.DeliveryRegression.begin(self, wf, wf.queue.get(task['task_id']))
            self.assertIn(results[-1]['id'], wf.required_refs(task))
            self.assertIn('contrary result', context['memory']['research_continuity']['current_note'])
        wf.submit(task, {'selected_slot': 'H2', 'rationale': 'CPU fixture: revised choice after two observations',
                         'evidence_refs': [r['id'] for r in results]}, None)
        advance_with_topic_fixture(wf)
        plan = wf.tasks()[wf.l3_request_keys(1, 'H2', wf.candidate_refs(1)['H2'])[0]]
        context = wf.context(plan)
        from core.cards import canonical
        for result in results:
            self.assertIn(result['id'], canonical(context))
        self.assertEqual(wf.solutions.snapshot()[0]['status'], 'pending_evaluation')
        self.assertFalse(RETIRED_PURPOSES.intersection(t['inputs'].get('purpose') for t in wf.tasks().values()))

    def test_l3_is_considered_before_recorder_and_on_budget_shutdown(self):
        from tests import test_minimal_fixes as l3
        for stop in (False, True):
            wf, trial = l3.L3Regression.planned(self, 'stop-' + str(stop))
            experiment, evidence = l3.L3Regression.evidence(self, wf, trial)
            wf.queue.finish(trial['task_id'], outputs=[evidence['id']], result={'status': 'completed'})
            with patch.object(wf.solutions, 'consider_round', wraps=wf.solutions.consider_round) as considered:
                if stop:
                    from runtime.budget import BudgetExhausted
                    with patch.object(wf.budget, 'check', side_effect=BudgetExhausted('CPU fixture stop')):
                        wf.run()
                else:
                    advance_with_topic_fixture(wf)
                    self.assertNotEqual(wf.tasks()['r001/record']['state'], 'completed')
                self.assertTrue(considered.called)
                self.assertEqual(considered.call_args.args[1]['id'], evidence['id'])
            # This is synthetic subset L3; it must not replace a real dual-task solution.
            self.assertTrue(all(row['status'] == 'pending_evaluation' for row in wf.solutions.snapshot()))

    def test_full_history_is_readable_but_cannot_be_silently_resumed_as_gonly(self):
        wf = self.workflow_case(l3=False)
        path = wf.run_dir / 'research_run.json'
        saved = read_json(path)
        saved['config'].pop('architecture')
        saved['config']['llm']['agent_models'].update(generation=['gpt-6.1-sol', 'high'],
                                                    judger=['gpt-6.1-sol', 'medium'], evolution=['gpt-6.1-sol', 'high'])
        saved['config_sha256'] = fingerprint(saved['config'])
        atomic_json(path, saved)
        with self.assertRaisesRegex(ContractError, 'Full'):
            Workflow(wf.run_dir)
        self.assertEqual(Workflow(wf.run_dir, read_only=True).summary()['architecture'], 'full')


if __name__ == '__main__':
    unittest.main()
