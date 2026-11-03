"""Scientific submission, deferred synthesis and numeric-intent compilation; no LLM/GPU."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, make_card
from core.workflow import Workflow
from agents.roles import role_spec
from tools.experiments import expand_plan
from tests.support import hypothesis, review, experiment_payload, validation_config
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery, test_parent_evolution as parent_evolution
from tests import test_real_research_followup as followup


class ResearchRefactor(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    read = delivery.DeliveryRegression.read
    evidence = followup.RealResearchFollowup.evidence
    drive = parent_evolution.ParentEvolutionRegression.drive
    workflow_case = parent_evolution.ParentEvolutionRegression.workflow_case

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('offline regression'))
        worker.start(); self.addCleanup(worker.stop)

    @staticmethod
    def synthesis():
        return {'scientific_analysis': 'Scripted scientific summary: no external validity claim.',
                'generation_guidance': 'Distinguish local NMSE from whole-model output and task performance.',
                'next_questions': ['Which controlled measurement would reverse the current judgment?']}

    def test_three_fields_archive_all_sources_without_claiming_exhaustive_review(self):
        wf = self.workflow(1)
        ev = self.evidence(wf)
        tid = wf.enqueue_recorder(1, incomplete=True)
        task, context = self.begin(wf, wf.queue.get(tid))
        batch = task['inputs']['context']['recording_batch']
        self.assertNotIn('source_refs', context['task']['recording_batch'])
        self.assertNotIn('required_judgment_refs', context['task'])
        schema = role_spec('recorder', 'record', {}, validation_config()['llm'])['context']['payload_contract']
        self.assertEqual(set(schema['required']), set(self.synthesis()))
        wf.submit(task, self.synthesis(), 'test_double')
        card = wf.result_card(wf.queue.get(tid))
        self.assertEqual(card['payload']['source_refs'], batch['source_refs'])
        self.assertIn(ev['id'], card['refs'])
        self.assertNotIn('reviewed_refs', card['payload'])
        self.assertEqual(card['payload']['judgment_audit'], [])
        self.assertIn('not_exhaustive_audit', card['payload']['analysis_scope'])
        self.assertEqual(wf.memory.recording_status()['scientific_review'], 'completed')
        self.assertEqual(wf.memory.snapshot()['L2'], {})

    def test_short_refs_bind_real_knowledge_scope_and_next_generation_feedback(self):
        wf = self.workflow(1); ev = self.evidence(wf)
        task, context = self.begin(wf, wf.queue.get(wf.enqueue_recorder(1, incomplete=True)))
        handle = wf.retriever.handle(ev['id'])
        self.assertEqual(context['memory']['source_handles'][handle], ev['id'])
        content = self.synthesis()
        content['knowledge_updates'] = [{'text': 'Synthetic fixture only', 'status': 'observed_once',
            'evidence_refs': [handle], 'rationale': 'Single measured synthetic condition, not task accuracy.'}]
        wf.submit(task, content, 'test_double')
        meta = wf.result_card(wf.queue.get(task['task_id']))
        claim = meta['payload']['knowledge_updates'][0]
        self.assertTrue(claim['claim_id'].startswith('knowledge-'))
        self.assertEqual(claim['scope'], ev['payload']['scope'])
        self.assertEqual(claim['evidence_refs'], [ev['id']])
        gen, ctx = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id'], meta['id']],
            round=2, slot='H1', hypothesis_type='existing_method'))
        self.assertIn(meta['payload']['generation_guidance'], canonical(ctx))
        value = hypothesis(2, 'H1')
        for k in ('round', 'slot', 'hypothesis_type'): value.pop(k)
        value['predictions'][0].pop('id')
        value['feedback_used'] = [{'change': 'Use matched global and local measurements.'}]
        value['intervention']['model_ref'] = 'scientific-asset-name-not-a-card'
        wf.submit(gen, value, 'test_double')
        stored = wf.result_card(wf.queue.get(gen['task_id']))['payload']
        self.assertEqual(stored['feedback_used'][0]['metareview_ref'], meta['id'])
        self.assertEqual(stored['predictions'][0]['id'], 'p1')
        self.assertEqual(stored['slot'], 'H1')

    def test_aliases_cannot_read_unknown_future_or_private_evidence(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.task(wf, 'interpret', round=1))
        future = self.evidence(wf)
        value = {'rationale': 'Fixture', 'evidence_refs': [wf.retriever.handle(future['id'])]}
        with self.assertRaisesRegex(ContractError, '权限拒绝'):
            wf.submit(task, value, 'test_double')
        with self.assertRaises(ContractError): self.read(wf, task, 'read_card', ref='C999999')
        wf.queue.fail(task['task_id'], {'code': 'fixture_done'})
        own = delivery.DeliveryRegression.debate_task(self, wf)
        other = delivery.DeliveryRegression.debate_task(self, wf, side='B')
        wf.queue.ready(); wf.queue.claim(other['task_id'])
        secret = make_card('ReviewDecisionCard', wf.journal.run_id, review('position',
            round=1, pair=['H1', 'H2'], exchange=1, side='B'))
        wf.queue.finish(other['task_id'], cards=[secret], outputs=[secret['id']])
        own, _ = self.begin(wf, own)
        with self.assertRaisesRegex(ContractError, '权限拒绝'):
            self.read(wf, own, 'read_card', ref=wf.retriever.handle(secret['id']))

    def test_fixed_identity_conflicts_rejected_and_visible_card_read_by_handle(self):
        wf = self.workflow(1); ev = self.evidence(wf)
        task, _ = self.begin(wf, self.task(wf, 'interpret', refs=[ev['id']], round=1))
        returned = self.read(wf, task, 'read_card', ref=wf.retriever.handle(ev['id']))
        self.assertEqual(returned, ev)
        with self.assertRaisesRegex(ContractError, '固定身份'):
            wf.submit(task, {'phase': 'verdict', 'rationale': 'Wrong task'}, 'test_double')
        with self.assertRaisesRegex(ContractError, '固定身份'):
            wf.submit(task, {'round': 2, 'rationale': 'Wrong round'}, 'test_double')
        wf.submit(task, {'rationale': 'Visible measured fixture', 'evidence_refs': [wf.retriever.handle(ev['id'])]}, 'test_double')
        stored = wf.result_card(wf.queue.get(task['task_id']))['payload']
        self.assertEqual(stored['phase'], 'interpretation')
        self.assertEqual(stored['evidence_refs'], [ev['id']])

    def test_failed_recorder_preserves_pending_facts_allows_next_round_and_later_synthesis(self):
        wf, record = self.drive(self.workflow_case(rounds=2, l3=False),
                                stop=lambda t: t['inputs']['purpose'] == 'record')
        old_sources = set(record['inputs']['context']['recording_batch']['source_refs'])
        wf.queue.fail(record['task_id'], {'code': 'scientific_synthesis_failed'})
        self.assertFalse(wf.advance())
        self.assertFalse(wf.journal.find_cards('MetareviewCard'))
        self.assertEqual(wf.memory.recording_status()['scientific_review'], 'pending')
        self.assertTrue(old_sources <= set(wf.memory.recording_batch()['source_refs']))
        gen = wf.tasks()['r002/generate/H1']
        self.assertTrue(gen['inputs']['context']['previous_review_pending'])
        body = wf.request_body(gen)
        self.assertTrue(any(c.get('kind') == 'HypothesisCard' for c in body['spec']['context']['memory']['cards']))
        self.assertFalse(wf.summary('completed')['scientific_review_complete'])
        # Restoring cannot silently mark the missing synthesis complete.
        wf = Workflow(wf.run_dir)
        wf, later = self.drive(wf, stop=lambda t: t['inputs']['purpose'] == 'record')
        self.assertEqual(later['inputs']['context']['round'], 2)
        self.assertTrue(old_sources <= set(later['inputs']['context']['recording_batch']['source_refs']))
        wf.submit(later, self.synthesis(), 'test_double')
        self.assertTrue(wf.advance())
        self.assertTrue(wf.summary('completed')['scientific_review_complete'])
        self.assertEqual(wf.memory.recording_status()['deferred_task_refs'], [])

    def test_failed_last_recorder_does_not_repeat_same_batch_or_fake_science(self):
        wf, record = self.drive(self.workflow_case(rounds=1, l3=False),
                                stop=lambda t: t['inputs']['purpose'] == 'record')
        wf.queue.fail(record['task_id'], {'code': 'scientific_synthesis_failed'})
        self.assertTrue(wf.advance())
        keys = set(wf.tasks()); count = len(wf.journal.events)
        for _ in range(3): self.assertTrue(wf.advance())
        self.assertEqual(keys, set(wf.tasks())); self.assertEqual(count, len(wf.journal.events))
        summary = wf.summary('completed')
        self.assertTrue(summary['workflow_completed'])
        self.assertFalse(summary['scientific_review_complete'])
        self.assertFalse(summary['formal_validation_success'])
        self.assertFalse(wf.memory.snapshot()['L3'])
        wf.retry_failed('Explicit fixture retry; same budget and batch')
        self.assertFalse(wf.advance())
        self.assertEqual(keys, set(wf.tasks()))
        retry = wf.queue.claim(record['task_id'])
        wf.submit(retry, self.synthesis(), 'test_double')
        self.assertTrue(wf.advance())
        self.assertTrue(wf.summary('completed')['scientific_review_complete'])

    def test_experiment_tool_resolves_short_source_before_preflight_and_private_queries_filter(self):
        wf = self.workflow(1); ev = self.evidence(wf)
        task, _ = self.begin(wf, self.task(wf, 'interpret', refs=[ev['id']], round=1))
        draft = experiment_payload()
        draft['execution']['trial_evidence_refs'] = [wf.retriever.handle(ev['id'])]
        with patch.object(wf.broker, 'preflight', side_effect=lambda x: x) as preflight:
            wf.receive_tools(task, {'tool_calls': [{'id': 'test', 'name': 'request_experiment',
                'arguments': {'experiment': draft}}]}, 'test_double')
        self.assertEqual(preflight.call_args.args[0]['execution']['trial_evidence_refs'], [ev['id']])
        parent = wf.queue.get(task['task_id'])
        child = wf.queue.get(parent['waiting_for'][0])
        self.assertEqual(child['inputs']['draft']['execution']['trial_evidence_refs'], [ev['id']])
        # Unknown handles in a query return no match; direct reads remain strict.
        query = self.read(wf, task, 'retrieve_memory', query={'evidence_ids': ['C999999']})
        self.assertEqual(query['entries'], [])

    def test_grouped_builtin_plan_expands_exactly_and_mirrors_only_actual_choices(self):
        wf = self.workflow(1)
        spec = {'format': 'INT4', 'bits': 4, 'granularity': 'per_tensor', 'axis': None,
                'group_size': None, 'symmetric': True, 'scale': 'minmax'}
        row = {'modules': ['blocks.0.mlp.fc2', 'blocks.0.attn.proj'], 'kind': 'linear',
               'weight': spec, 'activation': deepcopy(spec), 'weight_method': 'rtn', 'activation_method': 'rtn',
               'reason': 'Fixture', 'evidence': ['Fixture']}
        plan = {'schema_version': 3, 'unlisted_modules': 'keep_fp', 'preprocess': [], 'unresolved': [],
                'execution': {'mode': 'reference_qdq', 'compute_dtype': 'FP32', 'activation_calibration': 'static', 'order': 'joint'},
                'layers': [row]}
        raw = deepcopy(plan)
        expanded = expand_plan(plan)
        self.assertEqual(plan, raw)
        self.assertEqual([r['module'] for r in expanded['layers']], row['modules'])
        self.assertTrue(all(r['activation'] == spec for r in expanded['layers']))
        draft = experiment_payload('L2')
        for k in ('scope', 'validation', 'replication'): draft.pop(k)
        draft['resources'] = {'estimated_seconds': 60}
        draft['execution'] = {'service': 'quantize_evaluate', 'plan': plan, 'seed': 42,
            'calibration': {'count': 2}, 'evaluations': [{'kind': 'classification', 'split': 'search', 'count': 2}]}
        frozen = wf.broker.preflight(draft)
        self.assertEqual(frozen['execution']['plan'], expanded)
        self.assertEqual(frozen['execution']['resolved_choices']['activation_bits'], 4)
        self.assertFalse(frozen['execution']['requires_code'])
        draft['execution']['resolved_choices'] = {'activation_bits': 8}
        with self.assertRaises(ContractError): wf.broker.preflight(draft)
        for names in ([], ['blocks.0.mlp.fc2'] * 2):
            bad = deepcopy(plan); bad['layers'][0]['modules'] = names
            with self.assertRaises(ContractError): expand_plan(bad)

