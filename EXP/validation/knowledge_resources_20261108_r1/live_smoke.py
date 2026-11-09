"""Bounded real resource-organization smoke; hypothetical inputs, no measured Evidence.

Uses the real Workflow queue, workers, host tools, subscription and submit path.
The only ablation is in this harness: hide optional Skill assets and discovery.
No runtime router, new experiment, GPU work, or automatic scenario retries.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'QAViT'))

from core.workflow import Workflow
from core.journal import atomic_json, read_json
from tests.support import validation_config

BASE = Path(__file__).parent
CASES = {
    'resource_choice': ('generate', '题设已通过同条件干预定位：LN 后通道幅度差异使 shared-scale A4 吞没小通道，具体残差/分支是否允许等价吸收还未知。研究组希望从仓库成熟的完整量化起点中选择一个可复用种子作为机制参照，再提出一个可证伪的新候选；不要求忠实论文复现。请按需查阅本地策略、组件或方案说明，区分文档支持的定义、待测种子与本题尚缺的测量。只需简洁假说与最有信息价值的下一步。'),
    'engineering_only': ('interpret', '测量因 shape mismatch 失败，工程人员已确切定位为 reshape 的一个维度写错，修正位置与预期 shape 均明确，尚无有效数值结果。请简洁判断这次失败能否改变算法结论及该沿什么路径继续；不需要重新诊断已确定的原因。'),
}


def run_case(case, use_skills, deadline):
    purpose, scenario = CASES[case]
    label = case + ('_skills' if use_skills else '_without_skills')
    config = validation_config()
    config.update(name=label, purpose='validation', enable_l3=False)
    config['budget'].update(max_rounds=1, wall_seconds=420, parent_deadline_unix=deadline)
    config['runtime'].update(task_timeout_seconds=240, llm_concurrency=1, allowed_gpus=[])
    config['llm']['max_transport_retries'] = 0
    brief = config['research_brief']
    brief['scientific_question'] = scenario
    brief['goals'] = {'scope': '低成本真实角色文本烟测；题设是评估情境，不是本运行的测量或 Evidence。形成一个简洁、可复核的科学决定即可。'}
    brief['execution_guidance'] = ('本次仅完成当前角色文本任务。可自主查阅相关本地资料，也可直接提交；'
        '不执行数值实验、外部文献联网或完整科研轮次。需要的新实验写入后续设计。'
        '不要编造实测或来源身份，不要求证明候选有效。')
    brief.pop('experiment_example', None)
    run_dir = BASE / 'live' / label
    workflow = Workflow(run_dir, config)
    if not use_skills:
        original_context, original_assets = workflow.context, workflow.authorized_assets
        def context_without_skills(task):
            value = original_context(task)
            value.pop('skill_catalog', None)
            locations = value['memory'].get('asset_locations', {})
            for key in list(locations):
                if '/docs/skills/' in key:
                    del locations[key]
            return value
        workflow.context = context_without_skills
        workflow.authorized_assets = lambda task: {
            k: v for k, v in original_assets(task).items() if '/docs/skills/' not in k}
    task_context = {'round': 1, 'component_validation': True, 'requirements': brief['execution_guidance']}
    if purpose == 'generate':
        task_context.update(slot='H3', hypothesis_type='new_algorithm')
    tid = workflow.enqueue_role('smoke/' + case, purpose, refs=[workflow.brief['id']], context=task_context)
    result = {'case': case, 'skills_available': use_skills, 'scenario': scenario, 'purpose': purpose,
              'real_llm': True, 'measurement_source': 'none; hypothetical scenario', 'status': 'incomplete'}
    started = time.monotonic()
    try:
        while True:
            workflow.budget.check()
            workflow.poll()
            task = workflow.queue.get(tid)
            if task['state'] == 'completed':
                result.update(status='completed', output=workflow.journal.card(task['outputs'][0]))
                break
            if task['state'] in ('failed', 'needs_clarification', 'cancelled'):
                result['failure'] = {k: task.get(k) for k in ('state', 'error', 'clarification')}
                break
            audits = list(run_dir.rglob('llm_audit.json'))
            if len(audits) >= 4 and task['state'] != 'running':
                raise RuntimeError('Smoke harness cost cap: at most four model calls per case')
            if any(t['kind'] in ('experiment_request', 'experiment', 'literature')
                   for t in workflow.queue.snapshot().values()):
                raise RuntimeError('Smoke text scope exceeded; execution stopped before numerical/network dispatch')
            workflow.dispatch()
            time.sleep(.25)
    except Exception as exc:
        result['error'] = {'code': type(exc).__name__, 'message': str(exc)}
    finally:
        workflow.runtime.stop()
        result['elapsed_seconds'] = time.monotonic() - started
        calls = []
        for path in sorted(run_dir.rglob('llm_audit.json')):
            audit = read_json(path)
            calls.append({k: audit.get(k) for k in ('requested', 'actual', 'usage', 'error', 'elapsed_seconds')})
        result['model_calls'] = calls
        result['tool_calls'] = [{**{k: t.get(k) for k in ('key', 'state', 'error')},
                                'name': t['inputs']['name'], 'arguments': t['inputs']['arguments']}
            for t in workflow.queue.snapshot().values() if t['kind'] == 'host_tool']
        result['numeric_tasks'] = sum(t['kind'] == 'experiment' for t in workflow.queue.snapshot().values())
        atomic_json(BASE / (label + '.json'), result)
    print(json.dumps({k: result[k] for k in ('case', 'skills_available', 'status', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=list(CASES))
    parser.add_argument('--without-skills', action='store_true')
    args = parser.parse_args()
    todo = [(args.case, not args.without_skills)] if args.case else [
        ('resource_choice', True), ('engineering_only', True)]
    deadline = time.time() + 900
    results = []
    for case, enabled in todo:
        if time.time() >= deadline:
            break
        results.append(run_case(case, enabled, deadline))
        # Do not silently repeat failed calls or switch model/provider.
        if results[-1].get('error') or results[-1].get('failure'):
            break
    summary = {'scope': 'real production-role text smoke; no measured scientific gain or full-loop claim',
               'cases': results, 'planned_cases': len(todo)}
    atomic_json(BASE / ('live_summary.json' if not args.case else 'live_' + args.case + '_summary.json'), summary)


if __name__ == '__main__':
    main()
