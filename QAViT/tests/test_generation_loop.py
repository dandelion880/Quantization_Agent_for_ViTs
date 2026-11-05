"""Generation's real queue/host loop; scripted decisions, never real scientific claims."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, fingerprint, make_card
from core.journal import atomic_json
from core.workflow import Workflow
from tests.support import experiment_payload, hypothesis
from tests.test_infrastructure import TempCase
from tests import test_infrastructure as infrastructure
from tests import test_memory_delivery as delivery


class GenerationLoop(TempCase):
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    read = delivery.DeliveryRegression.read
    scripted = infrastructure.WorkflowSimulation.scripted

    def test_two_rounds_with_pre_h_experiments_preserve_debates_h4_and_recorder(self):
        wf = self.workflow(2)
        experimented = set()
        for tick in range(150):
            if wf.advance():
                break
            for ready in wf.queue.ready():
                self.assertEqual(ready['kind'], 'llm')
                if ready['inputs']['purpose'] == 'generate' and ready['inputs']['context']['slot'] == 'H1':
                    task, _ = self.begin(wf, ready)
                    key = task['key']
                    if key not in experimented:
                        evidence = self.measured(wf, self.ask(wf, task), len(experimented) + 1)
                        experimented.add(key)
                        task, _ = self.begin(wf, wf.queue.get(task['task_id']))
                        self.assertIn(evidence['id'], wf.required_refs(task))
                else:
                    task = wf.queue.claim(ready['task_id'])
                if task['inputs']['purpose'] == 'position':
                    self.assertTrue({'H1', 'H2', 'H3'} <= wf.round_hypotheses(task['inputs']['context']['round']).keys())
                wf.submit(task, self.scripted(wf, task), 'test_double')
            if tick == 4:
                wf = Workflow(wf.run_dir)
        else:
            self.fail('two-round workflow did not finish')
        self.assertEqual(len(experimented), 2)
        self.assertEqual(len(wf.memory.snapshot()['L3']), 2)
        for number in (1, 2):
            hs = wf.round_hypotheses(number)
            self.assertEqual(set(hs), {'H1', 'H2', 'H3', 'H4'})
            self.assertEqual(len(wf.matches(number)), 4)
            parent = hs['H4']['parents'][0]
            parent_slot = next(k for k, h in hs.items() if h['id'] == parent)
            self.assertIn('H4-' + parent_slot, wf.matches(number))
        self.assertTrue(all(h['payload']['feedback_used'] for h in wf.round_hypotheses(2).values()))
        self.assertFalse(wf.memory.recording_batch()['source_refs'])

    def start(self, wf):
        tid = wf.enqueue_role('r001/generate/H1', 'generate', refs=[wf.brief['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        return self.begin(wf, wf.queue.get(tid))

    def ask(self, wf, task, draft=None, note='Scripted test: use feedback to choose next measurement'):
        wf.receive_tools(task, {'tool_calls': [{'id': 'probe', 'name': 'request_experiment',
            'arguments': {'experiment': draft or experiment_payload(), 'research_note': note}}]}, 'test_double')
        wf.queue.ready()
        return wf.queue.claim(wf.queue.get(task['task_id'])['waiting_for'][0])

    def measured(self, wf, request, index):
        with patch.object(wf.broker, 'cached', return_value=None):
            wf.progress_request(request)
        execution = wf.tasks()[request['key'] + '/execute']
        wf.queue.ready(); wf.queue.claim(execution['task_id'])
        spec = wf.journal.card(wf.queue.get(request['task_id'])['experiment_ref'])
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'fixture', 'status': 'completed',
            'observations': {'mse': index / 10}, 'provenance': {}, 'resources': {}, 'artifacts': [],
            'measurement_source': 'test_double', 'execution_id': 'fixture-' + str(index)}, refs=[spec['id']])
        wf.queue.finish(execution['task_id'], outputs=[evidence['id']], cards=[evidence])
        wf.queue.ready(); wf.progress_request(wf.queue.claim(request['task_id']))
        return evidence

    def test_direct_submission_needs_no_experiment_or_judger(self):
        wf = self.workflow(1)
        task, _ = self.start(wf)
        wf.submit(task, hypothesis(1, 'H1'), 'test_double')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')
        self.assertEqual(len(wf.tasks()), 1)
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))

    def test_multiple_experiments_before_h_resume_and_selected_support(self):
        wf = self.workflow(1)
        task, _ = self.start(wf)
        evidence = []
        for index in (1, 2, 3):
            request = self.ask(wf, task, note='Interpretation before step ' + str(index))
            evidence.append(self.measured(wf, request, index))
            self.assertFalse(wf.journal.find_cards('HypothesisCard'))
            self.assertFalse(any(t['inputs'].get('purpose') in ('experiment_approval', 'interpret') for t in wf.tasks().values()))
            if index == 2:
                wf = Workflow(wf.run_dir)
            task, context = self.begin(wf, wf.queue.get(task['task_id']))
            self.assertIn(evidence[-1]['id'], canonical(context))
            self.assertIn('Interpretation before step ' + str(index), canonical(context))
            self.assertEqual(task['step'], index)
        content = hypothesis(1, 'H1')
        content['evidence_refs'] = [wf.retriever.handle(evidence[1]['id'])]
        content['sources'] = [{'ref': wf.retriever.handle(evidence[1]['id']), 'supports': 'scripted use only'}]
        wf.submit(task, content, 'test_double')
        final = wf.result_card(wf.queue.get(task['task_id']))
        self.assertEqual(final['payload']['evidence_refs'], [evidence[1]['id']])
        self.assertIn(evidence[1]['id'], final['refs'])
        self.assertTrue({e['id'] for e in evidence} <= set(wf.memory.recording_batch()['source_refs']))

    def test_failed_experiment_returns_and_researcher_can_change_plan(self):
        wf = self.workflow(1); task, _ = self.start(wf)
        request = self.ask(wf, task)
        with patch.object(wf.broker, 'cached', return_value=None): wf.progress_request(request)
        worker = wf.tasks()[request['key'] + '/execute']
        wf.queue.ready(); wf.queue.claim(worker['task_id'])
        wf.queue.fail(worker['task_id'], {'code': 'missing_site', 'message': 'No numerical measurement'})
        wf.queue.ready(); wf.progress_request(wf.queue.claim(request['task_id']))
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertIn('missing_site', canonical(context['tool_results']))
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))
        other = experiment_payload(); other['execution']['site'] = 'blocks.0.attn.proj'
        other['execution']['resolved_choices']['quantization_scope']['sites'] = ['blocks.0.attn.proj']
        other['scope']['site'] = 'blocks.0.attn.proj'
        evidence = self.measured(wf, self.ask(wf, task, other), 1)
        task, _ = self.begin(wf, wf.queue.get(task['task_id']))
        content = hypothesis(1, 'H1'); content['evidence_refs'] = [evidence['id']]
        wf.submit(task, content, 'test_double')

    def test_request_budget_refusal_and_hard_termination_do_not_fabricate_h(self):
        wf = self.workflow(1); task, _ = self.start(wf)
        draft = experiment_payload(); draft['resources']['estimated_seconds'] = 10**9
        wf.receive_tools(task, {'tool_calls': [{'id': 'over-budget', 'name': 'request_experiment',
            'arguments': {'experiment': draft}}]}, 'test_double')
        wf.dispatch()
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertIn('RequestBudgetInsufficient', canonical(context['tool_results']))
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))
        self.assertFalse(wf.journal.find_cards('HypothesisCard'))
        deadline = wf.budget.snapshot()['deadline_unix']
        # The researcher here is scripted; it has no actual worker receipt to poll.
        with patch.object(wf.budget, 'clock', return_value=deadline + 1), patch.object(wf.runtime, 'poll', return_value=[]):
            result = wf.run()
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['workflow_completed'])
        self.assertFalse(wf.journal.find_cards('HypothesisCard'))
        self.assertEqual(wf.budget.snapshot()['deadline_unix'], deadline)

    def test_new_code_needs_no_reference_or_fixtures_and_clarification_returns(self):
        wf = self.workflow(1); task, _ = self.start(wf)
        draft = {'question': 'Measure a precisely specified statistic', 'measurements': ['activation_mse']}
        request = self.ask(wf, task, draft); wf.progress_request(request)
        children = {t['inputs'].get('purpose'): t for t in wf.tasks().values() if t['inputs'].get('parent_task') == request['task_id']}
        self.assertEqual(set(children), {'code'})
        original = wf.journal.card(wf.queue.get(request['task_id'])['request_ref'])
        self.assertEqual(original['payload']['original_request']['question'], draft['question'])
        wf.queue.ready(); wf.queue.claim(children['code']['task_id'])
        wf.queue.change(children['code']['task_id'], 'needs_clarification', patch={
            'clarification': {'question': 'Missing boundary definition', 'reason': 'Statistical population ambiguous'}})
        wf = Workflow(wf.run_dir); wf.return_code_clarifications()
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertIn('Missing boundary definition', canonical(context['tool_results']))
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))
        self.assertEqual(wf.queue.get(children['code']['task_id'])['state'], 'cancelled')


class EvidenceRetrieval(TempCase):
    begin = delivery.DeliveryRegression.begin
    read = delivery.DeliveryRegression.read

    def test_experiment_cache_cannot_cross_current_debate_branch(self):
        wf = self.workflow(1)
        own = delivery.DeliveryRegression.debate_task(self, wf)
        other = delivery.DeliveryRegression.debate_task(self, wf, side='B')
        evidence = self.evidence(wf)
        worker = wf.queue.enqueue('opposite/execute', 'experiment', inputs={
            'parent_task': other['task_id'], 'experiment_ref': evidence['payload']['experiment_ref']})
        wf.queue.ready(); wf.queue.claim(worker)
        wf.queue.finish(worker, outputs=[evidence['id']])
        spec = wf.journal.card(evidence['payload']['experiment_ref'])
        # The numerical cache gate must receive the same ACL as the original researcher.
        def candidate(card, *, visible, **kwargs):
            return evidence if visible(evidence) else None
        with patch.object(wf.broker, 'cached', side_effect=candidate):
            self.assertIsNone(wf.cached_experiment(spec, {'inputs': {'parent_task': own['task_id']}}))
            self.assertEqual(wf.cached_experiment(spec, {'inputs': {'parent_task': other['task_id']}}), evidence)
        with patch.object(wf.broker, 'validate_evidence_for') as validate:
            self.assertIsNone(wf.broker.cached(spec, visible=lambda c: False))
            validate.assert_not_called()

    def test_module_keyed_coverage_summary_groups_metrics_but_preserves_extreme_paths(self):
        from core.evidence import project_observations, selected_leaves
        data = {'passes': [{'phase': p, 'calls': {'blocks.' + str(i) + '.attn.qkv': i for i in range(200)}}
                           for p in ('fp', 'quantized')]}
        shown = project_observations(data, 'fixture')
        summary = shown['passes']['numeric_summary']['/*/calls/*site*']
        self.assertEqual(summary['count'], 400)
        self.assertEqual(summary['maximum']['value'], 199)
        rows = list(selected_leaves(data, '/payload/observations', {'site': 'blocks.199.attn.qkv'}))
        self.assertEqual([r['value'] for r in rows], [199, 199])
        self.assertLess(len(canonical(shown)), len(canonical(data)) / 3)

    def evidence(self, wf):
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        rows = [{'site': 'blocks.' + str(i) + '.mlp.fc2', 'token_type': 'patch',
                 'nmse': .1, 'kl': .2, 'sign_delta': 0.01} for i in range(120)]
        rows[-1].update(nmse=999, kl=-3, sign_delta=-.5)
        obs = {'arms': {'baseline': {'nmse': 1, 'kl': 2}, 'candidate': {'nmse': 2, 'kl': 1}},
               'rows': rows, 'dense': list(range(9000))}
        ev = make_card('EvidenceCard', wf.journal.run_id, {'experiment_ref': spec['id'],
            'execution_fingerprint': 'fixture', 'status': 'completed', 'observations': obs,
            'provenance': {}, 'resources': {}, 'artifacts': [], 'measurement_source': 'test_double',
            'execution_id': 'fixture'}, refs=[spec['id']])
        wf.journal.commit('evidence', cards=[spec, ev]); return ev

    def test_overview_preserves_conflicting_metrics_tail_extrema_and_full_query(self):
        wf = self.workflow(1); ev = self.evidence(wf)
        tid = wf.enqueue_role('fixture/interpret', 'interpret', refs=[ev['id']])
        task, context = self.begin(wf, wf.queue.get(tid))
        shown = next(c for c in context['memory']['cards'] if c['id'] == ev['id'])
        obs = shown['payload']['observations']
        self.assertEqual(obs['arms'], ev['payload']['observations']['arms'])
        summary = obs['rows']['numeric_summary']
        self.assertEqual(summary['/*/nmse']['maximum']['value'], 999)
        self.assertEqual(summary['/*/sign_delta']['negative'], 1)
        self.assertEqual(obs['dense']['numeric_summary']['/*']['count'], 9000)
        self.assertLess(len(canonical(obs)), len(canonical(ev['payload']['observations'])) / 3)
        result = self.read(wf, task, 'read_card', ref=ev['id'], select={
            'site': 'blocks.119.mlp.fc2', 'token_type': 'patch', 'metric': 'nmse'})
        self.assertEqual(result['total'], 1); self.assertEqual(result['entries'][0]['value'], 999)
        extreme = self.read(wf, task, 'read_card', ref=ev['id'], field=summary['/*/nmse']['maximum']['field'])
        self.assertEqual(extreme['text'], '999')
        fields = []
        for offset in (0, 50, 100):
            result = self.read(wf, task, 'read_card', ref=ev['id'], select={'metric': 'nmse', 'token_type': 'patch'}, offset=offset, limit=50)
            self.assertEqual(result['total'], 120); fields += [r['field'] for r in result['entries']]
        self.assertEqual(len(set(fields)), 120)
        self.assertEqual(wf.journal.card(ev['id']), ev)

    def test_selection_uses_same_snapshot_and_reference_permissions(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role('fixture/generate', 'generate', refs=[wf.brief['id']], context={'round': 1})
        task, _ = self.begin(wf, wf.queue.get(tid)); ev = self.evidence(wf)
        with self.assertRaises(ContractError): self.read(wf, task, 'read_card', ref=ev['id'], select={'metric': 'nmse'})

    def test_read_session_reuses_verification_without_mutation_or_tamper_bypass(self):
        import core.journal as module
        wf = self.workflow(1); ev = self.evidence(wf)
        with patch.object(module, 'validate_card', wraps=module.validate_card) as validate:
            with wf.journal.read_session():
                value = wf.journal.card(ev['id']); value['payload']['observations']['rows'][0]['nmse'] = -99
                with wf.journal.read_session():
                    self.assertEqual(wf.journal.card(ev['id']), ev)
                self.assertEqual(validate.call_count, 1)
                changed = deepcopy(ev); changed['payload']['status'] = 'failed'
                atomic_json(wf.journal.card_path(ev['id']), changed)
                with self.assertRaises(ContractError): wf.journal.card(ev['id'])
            self.assertIsNone(wf.journal._read_cache)
        with self.assertRaises(ContractError): wf.journal.audit()


if __name__ == '__main__':
    unittest.main()
