"""Display cost controls preserve original evidence and research continuation; CPU only."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, fingerprint, make_card
from core.evidence import project_observations, selected_leaves
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery


class ContextCostRegression(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    science = delivery.DeliveryRegression.science
    recorder = delivery.DeliveryRegression.recorder

    def test_wide_projection_is_bounded_and_tail_contradiction_stays_queryable(self):
        rows = [{'site': 'block.' + str(i), 'token_type': 'patch',
                 **{'metric_' + str(j): j + i / 100 for j in range(240)},
                 'contradiction': -5 if i == 39 else 1} for i in range(40)]
        raw = {'rows': rows, 'summary': {'accuracy': 0.72, 'samples': 40},
               'limitation': 'subset only; no independent replication'}
        shown = project_observations(raw, 'evidence')
        self.assertLess(len(canonical(shown)), 7000)
        self.assertEqual(shown['summary'], raw['summary'])
        self.assertEqual(shown['limitation'], raw['limitation'])
        summary = shown['rows']
        self.assertGreater(summary['display_coverage']['omitted_metric_groups'], 0)
        self.assertEqual(summary['numeric_summary']['/*/contradiction']['minimum']['value'], -5)
        self.assertEqual(summary['display_coverage']['mixed_sign_metrics_total'], 1)
        tail = list(selected_leaves(rows, '/payload/observations/rows',
                                   {'site': 'block.39', 'token_type': 'patch', 'metric': 'contradiction'}))
        self.assertEqual([r['value'] for r in tail], [-5])
        self.assertEqual(summary['sha256'], fingerprint(rows))

    def test_current_state_and_directory_do_not_repeat_bound_evidence(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'ONE_COPY')
        task, context = self.begin(wf, self.task(wf, refs=[cards[1]['id']]))
        self.assertIn('current_research_state', context['memory'])
        self.assertEqual(canonical(context).count('FIXTURE observation ONE_COPY/0'), 1)
        self.assertTrue(all('body' not in row for row in context['memory']['directory']['entries']))
        document = context['method_catalog'][0]
        self.assertTrue(document['title'])
        self.assertTrue(document['applicable_questions'])
        self.assertTrue(document['read_asset']['asset'].startswith('methods/'))
        read = wf.host_tool({'inputs': {'name': 'read_asset', 'arguments': document['read_asset'],
                                       'parent_task': task['task_id']}})
        self.assertTrue(read['text'])
        task, context = self.tools(wf, task, [('retrieve_memory', {})])
        self.assertTrue(any(row.get('body') for row in context['tool_results'][-1]['result']['entries']))

    def test_recorder_gets_purpose_result_conditions_and_source_without_large_engineering(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'recording', commit=False)
        payload = deepcopy(cards[1]['payload'])
        payload['provenance'] = {'data_identity': {'rows': 'LARGE_IDENTITY' * 3000},
                                 'limitations': ['small subset'], 'warnings': ['no repeat']}
        evidence = make_card('EvidenceCard', wf.journal.run_id, payload, refs=[cards[0]['id']])
        wf.journal.commit('recording', cards=[cards[0], cards[1], evidence, cards[2], cards[3]])
        _, context = self.begin(wf, self.recorder(wf))
        rendered = canonical(context)
        self.assertNotIn('LARGE_IDENTITY', rendered)
        self.assertIn('small subset', rendered)
        self.assertIn('no repeat', rendered)
        self.assertIn('FIXTURE recording statistics', rendered)
        body = next(c for c in context['memory']['cards'] if c['id'] == evidence['id'])
        self.assertEqual(body['payload']['status'], 'completed')
        self.assertIn('provenance_source', body['payload'])
        self.assertEqual(wf.card_field(evidence, body['payload']['provenance_source']['read_card']['field']), payload['provenance'])

    def test_explicit_note_compacts_old_tool_body_and_keeps_failure_and_source(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'notes')
        first = {'task_id': 'old', 'task_card': cards[1]['id'], 'kind': 'host_tool', 'state': 'completed',
                 'inputs': {'name': 'read_card', 'arguments': {'ref': cards[1]['id']}}}
        second = {'task_id': 'note', 'task_card': cards[0]['id'], 'kind': 'experiment_request', 'state': 'completed',
                  'inputs': {'draft': {'question': 'targeted repeat'}, 'research_note': 'retain contradictory tail result'}}
        rows = [({'task': 'old'}, first, cards[1]),
                ({'task': 'note'}, second, {'status': 'failed', 'reason': 'measurement unavailable'})]
        with patch.object(wf, 'result_values', side_effect=lambda task: iter(rows)):
            result = wf.compact_results({}, {}, complete_bodies=[])
        self.assertNotIn('FIXTURE observation', canonical(result))
        self.assertEqual(result[0]['result']['read_card']['field'], '/payload/result')
        self.assertIn('measurement unavailable', canonical(result))
        self.assertIn('memory.research_continuity', canonical(result))

    def test_many_small_tables_share_a_budget_and_scientific_limits_survive(self):
        raw = {'table_' + str(i): [j / 10 for j in range(600)] for i in range(24)}
        shown = project_observations(raw, 'source')
        self.assertLess(len(canonical(shown)), 7000)
        self.assertEqual(shown['read_card']['field'], '/payload/observations')
        self.assertEqual(shown['sha256'], fingerprint(raw))
        with_scalar = {**{'group_' + str(i): list(range(900)) for i in range(20)}, 'version': 1}
        compact = project_observations(with_scalar, 'source', max_characters=2400)
        self.assertLess(len(canonical(compact)), 3500)
        self.assertEqual(compact['recorded_context']['version'], 1)
        self.assertEqual(compact['sha256'], fingerprint(with_scalar))
        wide = {**{'metric_' + str(i): i for i in range(500)},
                'limitations': ['Only one seed'], 'counterevidence': ['held-out accuracy decreased'],
                'summary': {'accuracy_delta': -0.02}}
        shown = project_observations(wide, 'source', max_characters=2400)
        self.assertEqual(shown['recorded_context']['limitations'], wide['limitations'])
        self.assertEqual(shown['recorded_context']['counterevidence'], wide['counterevidence'])
        self.assertEqual(shown['recorded_context']['summary'], wide['summary'])
        self.assertLess(len(canonical(shown)), 3500)

    def test_research_note_moves_previous_experiment_to_short_history(self):
        wf = self.workflow(1)
        old, new = self.science(wf, 'old'), self.science(wf, 'new')
        task = self.task(wf, 'generate', round=1, slot='H1', hypothesis_type='existing_method')
        rows = [({}, {'kind': 'experiment_request', 'outputs': [old[0]['id'], old[1]['id']],
                     'inputs': {}}, {}),
                ({}, {'kind': 'experiment_request', 'outputs': [new[0]['id'], new[1]['id']],
                     'inputs': {'research_note': 'test whether prior loss persists'}}, {})]
        with patch.object(wf, 'result_values', side_effect=lambda task: iter(rows)):
            refs = wf.required_refs(task)
            self.assertNotIn(old[1]['id'], refs)
            self.assertIn(new[1]['id'], refs)
            self.assertIn(old[1]['id'], wf.historical_experiment_refs(task))
            task['inputs']['refs'].append(old[1]['id'])
            self.assertIn(old[1]['id'], wf.required_refs(task))

    def test_projection_cache_is_immutable_across_presentations(self):
        wf = self.workflow(1)
        card = self.science(wf, 'cache')[1]
        with patch('agents.interaction.project_observations', wraps=project_observations) as project:
            a = wf.present_card(card)
            a['payload']['observations']['mean'] = 99
            b = wf.present_card(card)
            self.assertEqual(project.call_count, 1)
            self.assertEqual(b['payload']['observations']['mean'], 0)
            self.assertEqual(card['payload']['observations']['mean'], 0)
