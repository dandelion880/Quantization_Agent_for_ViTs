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

    def test_large_condition_sample_lists_keep_metadata_hashes_and_original_pages(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'condition_samples', commit=False)
        payload = deepcopy(cards[1]['payload'])
        ids = ['validation/sample-' + str(i).zfill(6) for i in range(2000)]
        name = 'final/a~b'
        scope = {'model': 'fixture-official', 'weight_bits': [4], 'activation_bits': [4],
                 'model_identity_sha256': 'model-hash', 'implementation': {'code': 'frozen'},
                 'quantization': {'head': {'weight': {'bits': 4}, 'activation': {'bits': 4}}},
                 'data': {name: {'split': 'final', 'sample_count': 2000, 'full_dataset': True,
                                 'sample_ids': ids}}}
        payload['scope'] = deepcopy(scope)
        payload['provenance'] = {'seed': 42, 'rng_seeds': {'python': 42, 'torch': 42, 'numpy': 42},
            'protocol_hash': 'protocol-hash', 'model_unchanged': True,
            'execution_facts': {'data': deepcopy(scope['data']), 'quantization': deepcopy(scope['quantization'])},
            'sample_selection': {name: list(ids), 'small': ['a', 'b']}}
        evidence = make_card('EvidenceCard', wf.journal.run_id, payload, refs=[cards[0]['id']])
        wf.journal.commit('large_condition_lists', cards=[cards[0], evidence])
        task, context = self.begin(wf, self.task(wf, 'code', refs=[evidence['id']],
            experiment_ref=cards[0]['id'], request_ref=cards[0]['id']))
        shown = next(v for v in context['memory']['cards'] if v['id'] == evidence['id'])['payload']
        self.assertLess(len(canonical(shown)), 16000)
        self.assertFalse(any(r['ref'] == evidence['id'] for r in context['memory']['required_reads']))
        self.assertIn(evidence['id'], context['memory']['read_manifest']['projected_refs'])
        self.assertNotIn(evidence['id'], context['memory']['read_manifest']['refs'])
        for key in scope.keys() - {'data'}:
            self.assertEqual(shown['scope'][key], scope[key])
        for key in ('seed', 'rng_seeds', 'protocol_hash', 'model_unchanged'):
            self.assertEqual(shown['provenance'][key], payload['provenance'][key])
        self.assertEqual(shown['provenance']['execution_facts']['quantization'], scope['quantization'])
        self.assertEqual(shown['provenance']['sample_selection']['small'], ['a', 'b'])
        views = [shown['scope']['data'][name]['sample_ids'],
                 shown['provenance']['execution_facts']['data'][name]['sample_ids'],
                 shown['provenance']['sample_selection'][name]]
        for view in views:
            self.assertEqual(view['items'], 2000)
            self.assertEqual(view['sha256'], fingerprint(ids))
            self.assertNotEqual(view['sha256'], fingerprint(list(reversed(ids))))
            self.assertEqual(wf.card_field(evidence, view['read_card']['field']), ids)
            self.assertIn('final~1a~0b', view['read_card']['field'])
            result = wf.host_tool({'inputs': {'name': 'read_card', 'parent_task': task['task_id'],
                'arguments': {**view['read_card'], 'offset': len(canonical(ids)) - 200, 'limit': 200}}})
            self.assertEqual(result['text'], canonical(ids)[-200:])
            self.assertEqual(result['sha256'], fingerprint(ids))
            self.assertIsNone(result['next_offset'])
        for row in (shown['scope']['data'][name], shown['provenance']['execution_facts']['data'][name]):
            self.assertEqual({k: v for k, v in row.items() if k != 'sample_ids'},
                             {k: v for k, v in scope['data'][name].items() if k != 'sample_ids'})
        self.assertEqual(wf.journal.card(evidence['id'])['payload'], payload)

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
