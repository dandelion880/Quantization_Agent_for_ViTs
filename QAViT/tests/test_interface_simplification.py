"""CPU-only production-boundary regressions; scripted roles are never research evidence."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.roles import role_spec
from core.cards import ContractError, canonical, make_card
from core.debate import validate_verdict
from core.journal import atomic_json
from runtime.paths import QAVIT
from ptq_agent.schemas import LLMRequest
from runtime.subscription import StructuredOutputError, output_schema
from tests.support import experiment_payload, hypothesis, meta, review, validation_config
from tests.test_infrastructure import TempCase
from tests.test_inner_json_repair import transport, events, event_text
from tools.experiments import preflight_experiment


def batch_final(calls):
    return json.dumps({'name': '__tool_batch__', 'arguments_json': json.dumps({'calls': calls})})


def decode(spec, final):
    request = LLMRequest(role=spec['role'], system_prompt=spec['system_prompt'],
        messages=spec['messages'], tools=spec['tools'], context=spec['context'], response_schema=spec['response_schema'])
    return transport().decode(event_text(events(final)), final, request)[0]


class InterfaceSimplification(TempCase):
    def setUp(self):
        super().setUp()
        blocker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('No LLM/GPU launch in boundary tests'))
        blocker.start()
        self.addCleanup(blocker.stop)

    def task(self, wf, purpose='screen', context=None, refs=()):
        tid = wf.enqueue_role('fixture/' + purpose, purpose, refs=refs, context=context or {})
        wf.queue.ready()
        return wf.queue.claim(tid)

    def test_transport_batch_reaches_queue_and_failures_are_isolated(self):
        wf = self.workflow(1)
        task = self.task(wf)
        spec = wf.request_body(task)['spec']
        calls = [{'name': 'read_card', 'arguments': {'ref': wf.brief['id']}},
                 {'name': 'read_asset', 'arguments': {'asset': str(QAVIT / 'runtime/driver.py')}}]
        response = asdict(decode(spec, batch_final(calls)))
        self.assertEqual(len(response['tool_calls']), 2)
        wf.receive_tools(task, response, 'cpu-scripted-receipt')
        wf.dispatch()
        children = [wf.queue.get(i) for i in wf.queue.get(task['task_id'])['waiting_for']]
        self.assertEqual([t['state'] for t in children], ['completed', 'failed'])
        wf.queue.ready()
        next_body = wf.request_body(wf.queue.get(task['task_id']))
        self.assertIn('资产读取限', canonical(next_body))
        self.assertIn(wf.brief['id'], canonical(next_body))
        self.assertEqual(len(next_body['spec']['messages'][0]['content']['tool_calls']), 2)

    def test_batch_schema_failure_is_located_before_any_tool_is_queued(self):
        wf = self.workflow(1)
        task = self.task(wf)
        spec = wf.request_body(task)['spec']
        calls = [{'name': 'read_card', 'arguments': {'ref': wf.brief['id']}},
                 {'name': 'read_card', 'arguments': {'ref': 17}}]
        with self.assertRaises(StructuredOutputError) as caught:
            decode(spec, batch_final(calls))
        self.assertEqual(caught.exception.stage, 'tool_schema')
        self.assertEqual(caught.exception.location['path'], '$.arguments_json.calls[1].arguments.ref')
        self.assertEqual(len(wf.tasks()), 1)
        for bad in ([], [{'name': '__agent_decision__', 'arguments': {}}],
                    [{'name': 'not_permitted', 'arguments': {}}]):
            with self.subTest(calls=bad), self.assertRaises(StructuredOutputError):
                decode(spec, batch_final(bad))

    def test_existing_capabilities_accept_one_canonical_slim_draft(self):
        wf = self.workflow(1)
        draft = experiment_payload()
        original = deepcopy(draft)
        del draft['execution']['requires_code']
        del draft['resources']['cpu_threads']
        del draft['replication']
        del draft['validation']
        del draft['execution']['resolved_choices']['training']
        value = wf.broker.preflight(draft)
        self.assertEqual(value['resources']['cpu_threads'], wf.config['runtime']['cpu_threads'])
        self.assertEqual(value['replication'], original['replication'])
        self.assertFalse(value['execution']['requires_code'])
        self.assertFalse(value['execution']['resolved_choices']['training'])
        self.assertEqual(value, wf.broker.preflight(value))
        self.assertNotIn('replication', draft)
        # The single tool and plan_l3 share the same draft/preflight, not new scientific interfaces.
        task = self.task(wf)
        args = {'experiment': draft}
        response = decode(wf.request_body(task)['spec'], json.dumps({
            'name': 'request_experiment', 'arguments_json': json.dumps(args)}))
        wf.receive_tools(task, asdict(response), 'cpu-draft')
        child = wf.queue.get(wf.queue.get(task['task_id'])['waiting_for'][0])
        self.assertEqual(child['inputs']['draft'], value)

    def test_no_scientific_guess_or_conflict_override(self):
        brief = validation_config()['research_brief']
        draft = experiment_payload()
        del brief['hard_constraints']['training']
        del draft['execution']['resolved_choices']['training']
        with self.assertRaisesRegex(ContractError, 'training'):
            preflight_experiment(draft, brief)
        for key in ('bits', 'signed', 'granularity', 'axis', 'group_size'):
            bad = experiment_payload()
            del bad['execution']['parameters'][key]
            with self.subTest(field=key), self.assertRaises(ContractError):
                preflight_experiment(bad, validation_config()['research_brief'])
        bad = experiment_payload()
        bad['execution']['resolved_choices']['training'] = True
        with self.assertRaisesRegex(ContractError, 'training'):
            preflight_experiment(bad, validation_config()['research_brief'])

    def test_cuda_resource_uses_only_the_unique_authorized_device(self):
        wf = self.workflow(1)
        draft = experiment_payload()
        draft['resources'].pop('gpu', None)
        resolved = deepcopy(draft['execution'])
        resolved['backend'] = {'kind': 'server', 'config': {'backend': {'device': 'cuda:0'}}}
        with patch('tools.experiments.resolve_execution', return_value=resolved):
            wf.broker.runtime = {**wf.config['runtime'], 'allowed_gpus': [3]}
            card = wf.broker.freeze(draft, requester='judger')
            self.assertEqual(card['payload']['resources']['gpu'], 3)
            self.assertIn('gpu:3', wf.experiment_resources(card))
            self.assertNotIn('gpu', draft['resources'])
            config_path = self.root / 'backend.json'
            atomic_json(config_path, json.loads((QAVIT / 'configs/deitb_backend.json').read_text()))
            resolved['backend']['config_asset'] = {'path': str(config_path)}
            self.assertEqual(wf.broker.freeze(draft, requester='judger')['payload']['resources']['gpu'], 3)
            resolved['backend'].pop('config_asset')
            for allowed in ([], [0, 3]):
                wf.broker.runtime['allowed_gpus'] = allowed
                with self.assertRaisesRegex(ContractError, 'resources.gpu'):
                    wf.broker.freeze(draft, requester='judger')
            explicit = deepcopy(draft); explicit['resources']['gpu'] = 0
            self.assertEqual(wf.broker.freeze(explicit, requester='judger')['payload']['resources']['gpu'], 0)
            resolved['backend']['config']['backend']['device'] = 'cpu'
            self.assertNotIn('gpu', wf.broker.freeze(draft, requester='judger')['payload']['resources'])

    def test_failed_experiment_returns_actual_cause_to_original_requester(self):
        wf = self.workflow(1); task = self.task(wf)
        response = {'tool_calls': [{'id': 'science', 'name': 'request_experiment',
                    'arguments': {'experiment': experiment_payload()}}]}
        wf.receive_tools(task, response, 'CPU_ONLY')
        request_id = wf.queue.get(task['task_id'])['waiting_for'][0]
        wf.queue.ready(); request = wf.queue.claim(request_id)
        child = wf.queue.enqueue(request['key'] + '/execute', 'experiment', inputs={'parent_task': request_id})
        wf.queue.ready(); wf.queue.claim(child)
        cause = {'code': 'invalid_config', 'message': 'CUDA_DEVICE_NOT_VISIBLE', 'retryable': False}
        wf.queue.fail(child, cause)
        wf.wait_children_or_fail(request, [child])
        self.assertEqual(wf.queue.get(request_id)['error']['causes'][0]['error'], cause)
        wf.queue.ready()
        body = wf.request_body(wf.queue.get(task['task_id']))
        self.assertIn('CUDA_DEVICE_NOT_VISIBLE', canonical(body))
        self.assertEqual(wf.queue.get(request_id)['state'], 'failed')
        self.assertEqual(wf.journal.find_cards('EvidenceCard'), [])

    def test_frozen_count_offset_and_fixture_capacity_checked_before_coding(self):
        draft = experiment_payload()
        execution = draft['execution']
        execution.update(service='custom_experiment', interface='experiment_v1', requires_code=True)
        execution['calibration']['count'] = 8
        execution['resolved_choices']['calibration_samples'] = 8
        draft['validation']['interface_cases'] = [{'seed': 1, 'shape': [1, 4, 8],
            'sites': [execution['site']], 'counts': {'calibration': 2, 'search': 4, 'final': 2}}]
        with self.assertRaisesRegex(ContractError, r'interface_cases\[0\].*calibration.*8.*2'):
            preflight_experiment(draft, validation_config()['research_brief'])
        draft['validation']['interface_cases'][0]['counts']['calibration'] = 8
        good = preflight_experiment(draft, validation_config()['research_brief'])
        self.assertEqual(good['execution']['calibration']['count'], 8)
        execution['calibration']['offset'] = 1
        with self.assertRaisesRegex(ContractError, 'interface_cases'):
            preflight_experiment(draft, validation_config()['research_brief'])

    def test_public_executor_is_readable_only_at_engineering_boundary(self):
        wf = self.workflow(1)
        experiment = wf.broker.freeze(experiment_payload(), requester='judger')
        wf.journal.commit('fixture/spec', cards=[experiment])
        public = QAVIT / 'tools/ptq_agent/execution/pipeline.py'
        for purpose in ('code', 'reference', 'screen'):
            with self.subTest(purpose=purpose):
                task = self.task(wf, purpose, {'experiment_ref': experiment['id']}, [wf.brief['id'], experiment['id']])
                body = wf.request_body(task)
                permissions = body['read_permissions']
                self.assertEqual(str(public) in permissions, purpose != 'screen')
                self.assertNotIn(str(QAVIT / 'runtime/driver.py'), permissions)
                self.assertNotIn(str(QAVIT / 'agents/interaction.py'), permissions)
                if purpose == 'reference':
                    self.assertNotIn('directory', body['spec']['context']['memory'])
                if purpose != 'screen':
                    wf.receive_tools(task, {'tool_calls': [{'id': purpose, 'name': 'read_asset',
                        'arguments': {'asset': str(public), 'offset': 0, 'limit': 12000}}]}, 'cpu-public')
                    wf.dispatch()
                    child = wf.queue.get(wf.queue.get(task['task_id'])['waiting_for'][0])
                    self.assertEqual(child['state'], 'completed')
                    self.assertIn('class QuantizationExecutor', child['result']['text'])
                wf.queue.change(task['task_id'], 'cancelled')

    def test_undetermined_needs_explicit_actionability_assessment(self):
        payload = review('verdict', outcome='undetermined', decisive_reason='CPU fixture', next_questions=[])
        payload.pop('continuation_assessment')
        with self.assertRaisesRegex(ContractError, 'continuation_assessment'):
            validate_verdict(payload, 1)
        payload['continuation_assessment'] = {'experiment_available': False,
            'next_dispute_available': False, 'reason': 'No discriminating path in this CPU fixture.'}
        validate_verdict(payload, 1)
        payload['continuation_assessment']['experiment_available'] = True
        with self.assertRaisesRegex(ContractError, '取证'):
            validate_verdict(payload, 1)
        # At the final exchange no sixth debate is allowed, but an executable experiment
        # can still be requested before this turn submits its final verdict.
        with self.assertRaisesRegex(ContractError, '取证'):
            validate_verdict(payload, 5)
        payload['continuation_assessment'].update(experiment_available=False, next_dispute_available=True)
        validate_verdict(payload, 5)

    def test_prompts_match_real_memory_and_capability_inputs(self):
        from agents.tasks import TASKS
        for purpose, item in TASKS.items():
            spec = role_spec(item['role'], purpose, {}, {})
            self.assertNotIn('memory.scientific_index', spec['system_prompt'])
            self.assertIn('memory.directory', spec['system_prompt'])
        wf = self.workflow(1)
        task = self.task(wf)
        context = wf.request_body(task)['spec']['context']
        self.assertIn('pdf_text_extraction', context['literature_capabilities'])

    def test_assigned_cross_round_guidance_is_delivered_without_full_metareview(self):
        wf = self.workflow(2)
        payload = meta(1)
        payload['scientific_analysis'] = 'OPTIONAL_WHOLE_REVIEW_' * 1000
        payload['generation_guidance'] = 'Bound preceding-round guidance'
        card = make_card('MetareviewCard', wf.journal.run_id, payload)
        wf.journal.commit('fixture/meta', cards=[card])
        task = self.task(wf, 'generate', {'round': 2, 'slot': 'H1'}, [wf.brief['id'], card['id']])
        body = wf.request_body(task)
        text = canonical(body)
        self.assertEqual(text.count('Bound preceding-round guidance'), 1)
        self.assertNotIn(payload['scientific_analysis'], text)  # A bounded directory title is permitted.
        self.assertFalse(any(row['ref'] == card['id'] and row['field'] == '/payload'
                             for row in body['spec']['context']['memory']['required_reads']))

    def test_batch_pages_reach_next_request_once_and_allow_submit(self):
        wf = self.workflow(1)
        payload = hypothesis(1, 'H1')
        payload['mechanism'] = 'UNIQUE_EXPANDED_MECHANISM_' + 'x' * 26000
        card = make_card('HypothesisCard', wf.journal.run_id, payload)
        wf.journal.commit('fixture/long', cards=[card])
        task = self.task(wf, context={'round': 1, 'hypothesis_ref': card['id']}, refs=[card['id']])
        body = wf.request_body(task)
        path = wf.run_dir / 'jobs/pages/request.json'
        atomic_json(path, body)
        wf.queue.change(task['task_id'], 'running', patch={'request_path': str(path)})
        task = wf.queue.get(task['task_id'])
        with self.assertRaisesRegex(ContractError, '未读完'):
            wf.submit(task, review('screening', eligible=True, hard_rejection_reasons=[],
                                 similarity_analysis='CPU', evidence_gaps=[]), 'CPU')
        calls = [{'name': 'read_card', 'arguments': {'ref': card['id'], 'field': '/payload', 'offset': offset}}
                 for offset in range(0, len(canonical(payload)), 12000)]
        wf.receive_tools(task, asdict(decode(body['spec'], batch_final(calls))), 'CPU-pages')
        wf.dispatch()
        wf.queue.ready()
        next_body = wf.request_body(wf.queue.get(task['task_id']))
        self.assertTrue(all(r['next_offset'] is None for r in next_body['spec']['context']['memory']['required_reads']))
        self.assertEqual(canonical(next_body['spec']).count('UNIQUE_EXPANDED_MECHANISM_'), 1)
        atomic_json(path, next_body)
        task = wf.queue.claim(task['task_id'], request_path=str(path))
        wf.submit(task, review('screening', eligible=True, hard_rejection_reasons=[],
                              similarity_analysis='CPU', evidence_gaps=[]), 'CPU')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')

    def test_batch_boundary_shares_existing_two_repairs_and_failure_usage(self):
        from tests import test_inner_json_repair as inner
        wf, tid = inner.InnerJSONRepairChain.start(self)
        calls = [{'name': 'read_card', 'arguments': {'ref': 17}}]
        finals = [batch_final(calls), inner.envelope(inner.BAD_INNER), batch_final(calls)]
        for final in finals:
            with patch('tests.test_inner_json_repair.envelope', return_value=final):
                receipt, _, _ = inner.InnerJSONRepairChain.attempt(self, wf, tid, 'unused')
            self.assertEqual(receipt['data']['usage'], inner.USAGE)
            wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual((task['state'], task['repair_count'], task['retry_count']), ('failed', 2, 0))
        settled = [e for e in wf.journal.events if e['type'] == 'budget_settled']
        self.assertEqual(len(settled), 3)
        self.assertTrue(all(e['reported_usage'] == inner.USAGE for e in settled))

    def test_disabled_tools_cannot_be_restored_by_batch_envelope(self):
        spec = role_spec('judger', 'screen', {'remaining_budget': {'tools': 0}}, {})
        with self.assertRaises(StructuredOutputError) as caught:
            decode(spec, batch_final([{'name': 'read_card', 'arguments': {'ref': 'not-used'}}]))
        self.assertEqual(caught.exception.stage, 'outer_schema')

    def test_import_capabilities_match_actual_gate_without_expanding_it(self):
        from tools.safeguards import inspect_source, NUMERIC_IMPORTS
        wf = self.workflow(1)
        experiment = wf.broker.freeze(experiment_payload(), requester='judger')
        wf.journal.commit('fixture/spec', cards=[experiment])
        task = self.task(wf, 'reference', {'experiment_ref': experiment['id']}, [wf.brief['id'], experiment['id']])
        context = wf.request_body(task)['spec']['context']
        self.assertEqual(set(context['numeric_imports']), NUMERIC_IMPORTS)
        for module in ('time', 'hashlib', 'os', 'subprocess'):
            with self.subTest(module=module), self.assertRaises(ContractError):
                inspect_source('import ' + module + '\ndef execute(p,c): return {}', 'experiment_v1')
        inspect_source('import math\ndef execute(p,c): return {}', 'experiment_v1')

    def test_builtin_plan_reuses_execution_without_coding_and_unknown_plan_rejected(self):
        from tests.test_execution_contracts import full_fixture
        brief, draft, plan = full_fixture()
        draft['execution'].pop('interface')
        draft['execution'].pop('requires_code')
        draft['execution']['plan'] = plan
        value = preflight_experiment(draft, brief)
        self.assertFalse(value['execution']['requires_code'])
        for mutation in ('code', 'interface', 'schema', 'unresolved'):
            bad = deepcopy(draft)
            if mutation == 'code': bad['execution']['requires_code'] = True
            elif mutation == 'interface': bad['execution']['interface'] = 'executor_v1'
            elif mutation == 'schema': del bad['execution']['plan']['layers'][0]['activation']
            else: bad['execution']['plan']['unresolved'] = ['missing science']
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                preflight_experiment(bad, brief)
        # Explicit new implementation cannot silently become an unaudited service.
        draft['execution'].pop('plan')
        draft['execution']['requires_code'] = False
        with self.assertRaisesRegex(ContractError, '缺少 execution.plan'):
            preflight_experiment(draft, brief)

    def test_pdf_capability_exposes_installed_state_without_fetching(self):
        wf = self.workflow(1)
        task = self.task(wf)
        for available in (False, True):
            with self.subTest(available=available), patch('agents.interaction.find_spec', return_value=object() if available else None):
                body = wf.request_body(task)
                self.assertEqual(body['spec']['context']['literature_capabilities']['pdf_text_extraction'], available)


if __name__ == '__main__':
    unittest.main()
