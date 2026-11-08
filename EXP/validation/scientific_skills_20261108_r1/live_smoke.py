"""Bounded real production-role smoke; scenarios are NOT measured Evidence.

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
    'A1': ('generate', '同一模型、样本和冻结协议下，题设 W4A32 接近 FP、joint W4A4 明显退化；尚无同前缀 A-only。主导机制未知，不能预先认定是 activation outlier。围绕这一缺口提出一个可证伪候选及最有信息价值的下一项研究。'),
    'B1': ('interpret', '题设同一 GELU 位点新网格局部 SSE 更好，但同时把拟合输入从 FP 改为 quantized prefix。网格能力和前缀适配都能解释观测；现在需要判断这个观察支持什么机制、如何用最小对照区分解释。'),
    'C1': ('generate', '题设已通过同条件干预定位：LN 后通道幅度差异使 shared-scale A4 吞没小通道。现在希望构造可部署的 static W4A4 改进，具体 LN 残差/分支是否允许等价吸收还未知。提出机制明确的可证伪改进，说明推理态与最简单替代解释。'),
    'D1': ('interpret', '题设一个冻结量化前缀下的 AV 局部干预，相对同 quantized peer 的 SSE 降低 5%，相对原 FP 目标 SSE 降低 1%；仅 calibration/search，缺独立 holdout、logits 和任务评测。完整评测昂贵。解释它能支持的最强结论，以及下一项值得投入的研究。'),
    'A4': ('interpret', '当前测量因 shape mismatch 失败，工程人员已确切定位为 reshape 的一个维度写错，修正位置与预期 shape 均明确，尚无任何有效数值结果。请说明这次失败能否改变算法判断、该沿什么路径继续。无需重新讨论已定位的原因。'),
    'D_policy': ('interpret', '题设：一个局部补偿在 calibration/search 相对 peer 的 SSE 小幅改善，但对 FP 的改善更小，未获得独立 holdout 或下游传播证据；已耗多轮调参，完整评测需数小时。研究组分别建议立即 L3、补传播实验、停止方向。希望复用项目已有的研究投入决策准则，而非只凭一个误差百分比投票：请按需取用仓库中的策略参考来支持选择，区分采用的准则和本题仍缺的证据，说明什么观测会让你停止或升级。'),
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
            if len(audits) >= 6 and task['state'] != 'running':
                raise RuntimeError('Smoke harness cost cap: at most six model calls per case')
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
        ('A1', False), ('A1', True), ('B1', True), ('C1', True), ('D1', True), ('A4', True)]
    deadline = time.time() + 1800
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
