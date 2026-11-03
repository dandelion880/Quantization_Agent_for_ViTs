"""固定 CPU 输入的重构契约快照；不调用 LLM、不加载外部权重。

输出包含最终 Prompt、有效配置、注册工具及实际局部 QDQ/特殊位点证据。
两次输出可逐字段比较；路径/源码身份变化由调用者显式归一化。
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.paths import ROOT
from core.journal import atomic_json
from core.workflow import config_check
from agents.roles import role_spec
from tools.experiments import resolve_execution
from tools.profiling import profile_host
from tools.experiment_worker import backend_and_data
from tests.support import validation_config, experiment_payload
from ptq_agent.schemas import ToolCall

PURPOSE_ROLES = {
    'generate': 'generation', 'evolve': 'evolution', 'screen': 'judger',
    'position': 'judger', 'arbitrate': 'judger', 'rank': 'judger',
    'experiment_approval': 'judger', 'plan_l3': 'judger', 'record': 'recorder',
    'code': 'coding', 'reference': 'safeguard', 'safeguard': 'safeguard',
    'supervise': 'supervisor', 'interpret': 'judger',
}


def snapshot(directory):
    import torch
    torch.set_num_threads(2)
    config = config_check(validation_config())
    execution = resolve_execution(experiment_payload()['execution'], config['research_brief'])
    backend, cfg, _ = backend_and_data(execution)
    host = profile_host(backend, cfg, directory / 'profile')
    specs = {purpose: role_spec(role, purpose, {'fixed_input': '相同科研上下文'}, config['llm'])
             for purpose, role in PURPOSE_ROLES.items()}
    # 来源哈希随文件迁移变化，最终 Prompt 本文必须保持不变。
    for spec in specs.values():
        spec['context'].pop('prompt_source_sha256')
    result = {'config': config, 'backend_config': cfg, 'role_specs': specs,
              'registry': {n: d.schema() for n, d in host.registry.definitions.items()}, 'measurements': {}}
    spec = {'format': 'INT4', 'bits': 4, 'granularity': 'per_tensor', 'axis': None,
            'group_size': None, 'scale': 'minmax', 'symmetric': True}
    calls = [
        ('query_structure', {}), ('query_ptq_plugins', {}),
        ('query_ptq_sites', {'prefix': 'blocks.0.attn', 'offset': 0, 'limit': 16}),
        ('profile_distribution', {'modules': ['blocks.0.mlp.fc2'], 'target': 'activation',
            'views': ['tensor', 'channel'], 'tokens': 'all', 'statistics': ['range', 'mean', 'std'],
            'histogram_bins': 8, 'top_k': 2, 'count': 2, 'seed': 42, 'offset': 0}),
        ('profile_sensitivity', {'modules': ['blocks.0.mlp.fc2'],
            'variants': [{'name': 'fixed_rtn', 'method': 'rtn', 'weight': spec, 'activation': spec}],
            'count': 2, 'seed': 42, 'offset': 0}),
        ('profile_plugin_sites', {'sites': [{'module': 'blocks.0.attn', 'kind': 'post_softmax'}],
            'count': 2, 'seed': 42, 'offset': 0}),
    ]
    for index, (name, args) in enumerate(calls):
        value = asdict(host.registry.invoke('profiling', ToolCall(str(index), name, args)))
        if value['status'] != 'success':
            raise AssertionError((name, value))
        result['measurements'][name] = value
        host.state['tool_results'].append({'arguments': args, 'result': value, 'epoch': 0})
        if name.startswith('profile_'):
            before = deepcopy(host.state['budget']['measurements'])
            cached = host.registry.invoke('profiling', ToolCall(str(index), name, args))
            assert cached.status == 'success' and cached.data['cached'] is True
            assert host.state['budget']['measurements'] == before
    result['evidence'] = {p.name: __import__('json').loads(p.read_text())
                          for p in sorted((directory / 'profile/evidence').glob('*.json'))}
    result['measurements_used'] = host.state['budget']['measurements']
    atomic_json(directory / 'contracts.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if ROOT / 'EXP' not in output.parents or output.exists():
        parser.error('输出必须为 EXP 下的新目录')
    output.mkdir(parents=True)
    snapshot(output)
    print(str(output / 'contracts.json'))
