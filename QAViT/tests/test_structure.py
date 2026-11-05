"""本次移动/解耦的剩余风险：Prompt、完整工具初始化、传输、递归身份。

仅确定性 CPU 输入及伪造传输事件；没有真实 LLM、下载或任务精度验证。
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.paths import ROOT, QAVIT
from runtime.configuration import config_check, model_defaults
from runtime import identity
from runtime.subscription import SubscriptionTransport, DECISION
from runtime.codex_cli import ISOLATION_NOTICE
from core.cards import ContractError, fingerprint
from core.journal import read_json
from agents.tasks import TASKS
from agents.roles import role_spec
from ptq_agent.schemas import LLMRequest, PTQError, ToolCall
from tests.support import validation_config, experiment_payload


class StructureContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'EXP/cache/tmp', prefix='structure-')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_final_prompts_match_baseline_or_explicit_fixes(self):
        golden = read_json(QAVIT / 'tests/data/contract_golden.json')
        # 保留原基线；仅叠加单列 diff 中授权的流程提示词变化。
        baseline = read_json(QAVIT / 'tests/data/prompt_readiness_baseline.json')
        expected_prompts = {**baseline['assembled_prompts'], **baseline.get('explicit_prompt_fixes', {})}
        config = config_check(validation_config())
        # The historical golden includes the removed routine scientific approval task.
        self.assertEqual(set(TASKS), set(golden['prompts']) - {'experiment_approval', 'reference'})
        self.assertEqual(config['llm']['agent_models'], model_defaults())
        self.assertNotIn('agent_models', read_json(QAVIT / 'configs/research.json')['llm'])
        self.assertNotIn('llm', read_json(QAVIT / 'configs/deitb_backend.json'))
        for purpose, task in TASKS.items():
            spec = role_spec(task['role'], purpose, {}, config['llm'])
            self.assertEqual(hashlib.sha256(spec['system_prompt'].encode()).hexdigest(), expected_prompts[purpose])
            self.assertEqual([t['name'] for t in spec['tools']], task['tools'])

    def test_model_schemas_expose_science_without_persistent_identity_fields(self):
        from agents.roles import submission_schema, TOOLS
        from core.cards import validate_schema
        draft = TOOLS['request_experiment']['parameters']['properties']['experiment']
        validate_schema({'question': 'Measure the declared attention statistic',
                         'measurements': ['paired statistic under unchanged baseline']}, draft, 'request')
        for field in ('stage', 'request_ref', 'request_sha256', 'scope', 'validation'):
            self.assertNotIn(field, draft['properties'])
        self.assertNotIn('resolved_choices', draft['properties']['execution']['properties'])
        self.assertNotIn('cpu_threads', draft['properties']['resources']['properties'])
        self.assertNotIn('round', submission_schema('generate')['properties'])
        self.assertNotIn('phase', submission_schema('safeguard')['properties'])
        validate_schema({'approved': False, 'rationale': 'The implementation samples the wrong site'},
                        submission_schema('safeguard'), 'review')
        recorder = submission_schema('record')
        self.assertIn('knowledge_updates', recorder['properties'])
        self.assertNotIn('pending_knowledge_updates', recorder['properties'])
        self.assertNotIn('round', recorder['properties'])
        self.assertIn('execution_choices', submission_schema('code')['properties'])
        self.assertNotIn('source_sha256', submission_schema('code')['properties'])

    def test_backend_config_keeps_numerical_resource_validation(self):
        from copy import deepcopy
        from ptq_agent.configuration import validate_config
        config = read_json(QAVIT / 'configs/deitb_backend.json')
        validate_config(config)
        for key in ('max_weight_bytes_ratio', 'max_latency_ms', 'max_peak_memory_mib'):
            invalid = deepcopy(config)
            invalid['loop'][key] = 0
            with self.assertRaises(PTQError, msg=key):
                validate_config(invalid)

    def test_full_profile_initialization_and_program_worker(self):
        from tools.experiment_worker import backend_and_data
        from tools.experiments import resolve_execution
        from tools.profiling import profile_host, PROFILE_NAMES
        import torch
        torch.set_num_threads(2)
        config = config_check(validation_config())
        execution = resolve_execution(experiment_payload()['execution'], config['research_brief'])
        backend, cfg, _ = backend_and_data(execution)
        host = profile_host(backend, cfg, self.root / 'profile')
        expected = read_json(QAVIT / 'tests/data/contract_golden.json')['schemas']
        self.assertEqual(set(host.registry.definitions), set(expected))
        self.assertTrue(set(PROFILE_NAMES) <= set(expected))
        for name, tool in host.registry.definitions.items():
            self.assertEqual(fingerprint(tool.schema()), expected[name], name)
        for obsolete in ('agents', 'client', 'memory', 'policy'):
            self.assertFalse(hasattr(host, obsolete), obsolete)
        host.config['budget']['measurements'] = 0
        host.consume('measurements', 1)
        self.assertEqual(host.state['budget']['measurements'], 1)
        # 动态 worker 路径和完整工具闭包绑定；代码只读取固定结构，不运行模型。
        code = "def run(inputs):\n    return {'status': tool('query_structure', {})['status']}\n"
        written = host.registry.invoke('profiling', ToolCall('write', 'write_task_program',
            {'name': 'structure', 'source': code, 'purpose': 'offline path check'}))
        self.assertEqual(written.status, 'success', written.error)
        result = host.registry.invoke('profiling', ToolCall('run', 'run_task_program',
            {'program': written.data['program'], 'inputs': {}}))
        self.assertEqual(result.status, 'success', result.error)
        self.assertEqual(result.data['result'], {'status': 'success'})

    def test_recursive_identity_snapshot_and_tamper_detection(self):
        for name, data in {'agents/prompts/tasks/new.md': 'prompt', 'tools/new/numeric.py': 'x=1',
                           'runtime/nested/worker.py': 'x=2', 'configs/models.json': '{}',
                           '.runtime_pkgs/cache.py': 'excluded', 'tools/__pycache__/bad.py': 'excluded'}.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        with patch.object(identity, 'QAVIT', self.root):
            before = identity.code_identity()
            self.assertEqual(len(before['core_files']), 4)
            self.assertIn('runtime/nested/worker.py', identity.measurement_implementation()['files'])
            snapshot = identity.code_snapshot(before)
            self.assertEqual(snapshot['agents/prompts/tasks/new.md'], 'prompt')
            (self.root / 'tools/new/numeric.py').write_text('x=3')
            self.assertNotEqual(before['core_sha256'], identity.code_identity()['core_sha256'])
            with self.assertRaises(ContractError):
                identity.code_snapshot(before)

    def test_executor_export_reload_preserves_fixed_cpu_output(self):
        from tools.experiment_worker import backend_and_data
        from tools.experiments import resolve_execution
        from ptq_agent.execution.pipeline import QuantizationExecutor
        from ptq_agent.execution.evaluation import model_digest
        from ptq_agent.execution.model import forward
        import torch
        torch.set_num_threads(2)
        config = config_check(validation_config())
        execution = resolve_execution(experiment_payload()['execution'], config['research_brief'])
        backend, cfg, data = backend_and_data(execution)
        original = model_digest(backend.model)
        spec = {'format': 'INT4', 'bits': 4, 'granularity': 'per_tensor', 'axis': None,
                'group_size': None, 'scale': 'minmax', 'symmetric': True}
        plan = {'schema_version': 3, 'unlisted_modules': 'keep_fp', 'preprocess': [], 'unresolved': [],
                'execution': {'mode': 'reference_qdq', 'compute_dtype': 'FP32',
                              'activation_calibration': 'static', 'order': 'joint'},
                'layers': [{'module': 'blocks.0.mlp.fc2', 'kind': 'linear', 'weight': spec,
                            'activation': spec, 'weight_method': 'rtn', 'activation_method': 'rtn',
                            'reason': 'fixed offline migration fixture', 'evidence': ['offline_fixture']}]}
        executor = QuantizationExecutor(backend, cfg, data)
        model, weights, activations, audit = executor.prepare(plan, {})
        directory = self.root / 'model'
        executor.export(model, plan, weights, activations, directory, audit)
        loaded, manifest = executor.load(directory)
        with torch.inference_mode():
            x, _ = next(data.batches('search'))
            left, right = forward(backend, model, x), forward(backend, loaded, x)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.isfinite(right).all())
        self.assertEqual(model_digest(backend.model), original)
        self.assertEqual(manifest['plan'], plan)

    def test_subscription_decode_audit_without_cli_calls(self):
        from runtime.model_router import call_once
        from runtime import codex_cli
        spec = role_spec('supervisor', 'supervise', {}, {'agent_models': model_defaults()})
        decision = {'action': 'submit', 'content': {'status_analysis': 'offline fixture'}}
        final = json.dumps({'name': DECISION, 'arguments_json': json.dumps(decision)})
        events = [
            {'type': 'item.completed', 'item': {'type': 'error', 'message': ISOLATION_NOTICE}},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': final}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 7, 'output_tokens': 3}},
        ]
        raw = '\n'.join(json.dumps(event) for event in events)
        def run(command, directory, timeout, prompt=None):
            self.assertIn('--sandbox', command)
            self.assertIn('read-only', command)
            (directory / 'answer.json').write_text(final)
            return 0, raw, ''
        with patch.object(codex_cli, 'executable', return_value='/offline/codex'), \
             patch.object(SubscriptionTransport, 'preflight', return_value={'source': 'test_double'}), \
             patch.object(codex_cli, 'run', side_effect=run):
            import time
            result = call_once(spec, self.root / 'call', time.time() + 30)
            audit = read_json(result['audit'])
            self.assertTrue(audit['model_call_started'])  # 伪传输的审计分支，不是真实调用。
            self.assertEqual(result['response']['decision'], decision)
            self.assertIsNone(audit['actual']['model'])
            self.assertEqual(audit['requested']['model'], model_defaults()['supervisor'][0])
            transport = SubscriptionTransport(*model_defaults()['supervisor'], {})
            request = LLMRequest('supervisor', spec['system_prompt'], [], spec['tools'], {}, spec['response_schema'])
            bad = json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution'}})
            with self.assertRaises(PTQError):
                transport.decode(bad, final, request)
            mismatch = raw + '\n' + json.dumps({'type': 'turn.started', 'model': 'wrong-model'})
            with self.assertRaises(PTQError):
                transport.decode(mismatch, final, request)


if __name__ == '__main__':
    unittest.main()
