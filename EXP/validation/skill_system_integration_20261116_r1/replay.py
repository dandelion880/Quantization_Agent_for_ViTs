"""Bounded real Generation/host-read replay, synthetic cases, no numerical dispatch.

Uses production Workflow, subscription, read_asset and submission. Review guide and
package author conclusions are outside the researcher ACL. No expected answer or
required skill name is put into the request. Not a measure of algorithmic benefit.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json, read_json
from runtime.identity import code_identity
from tests.support import validation_config

BASE = Path(__file__).resolve().parent
CASE_SOURCE = BASE / 'source_package/qavit_skill_system_20261116/evaluation/CASES.md'
CASES = {m[0]: m[1].strip() for m in re.findall(r'## (C\d+)[^\n]*\n(.*?)(?=\n## |\Z)', CASE_SOURCE.read_text(), re.S)}


def run_case(case, attempt=None):
    artifact_name = case + ('_' + attempt if attempt else '')
    config = validation_config()
    config.update(name=artifact_name, purpose='validation', enable_l3=False)
    deadline = time.time() + 480
    config['budget'].update(max_rounds=1, wall_seconds=480, min_free_bytes=0, parent_deadline_unix=deadline)
    config['runtime'].update(task_timeout_seconds=240, llm_concurrency=1, allowed_gpus=[])
    config['llm']['max_transport_retries'] = 0
    brief = config['research_brief']
    brief['scientific_question'] = CASES[case]
    brief['goals'] = {'scope': '人工构造的文本科研情境；只评当前判断与可接续提案，没有本运行数值测量。'}
    if case == 'C01':
        brief['hard_constraints'].pop('training')
        brief['hard_constraints']['training_allowed'] = True
        brief['search_space']['reference_scope'] = '允许隔离 scale-only STE 研究参考；最终算法独立 training-free，源码不继承参考实例答案。'
    requirements = ('仅完成这项科研文本任务，案例给定信息是假想情境而非本运行Evidence。'
                    '可以按需读取已授权资料，也可以直接提交。'
                    '所需的新数值实验写入后续设计，本次不执行数值、外部联网、完整科研轮次或Skill沉淀。'
                    '按当前任务形成可检验提案，准确说明假设与未验证范围。')
    brief['execution_guidance'] = {'notes': requirements}
    brief.pop('experiment_example', None)
    run_dir = BASE / 'live' / artifact_name
    if run_dir.exists():
        raise FileExistsError(f'Preserve prior replay: {run_dir}; use a new --attempt label')
    wf = Workflow(run_dir, config)
    tid = wf.enqueue_role('replay/' + case, 'generate', refs=[wf.brief['id']],
        context={'round': 1, 'slot': 'H3', 'hypothesis_type': 'new_algorithm',
                 'component_validation': True, 'requirements': requirements})
    result = {'case': case, 'attempt': attempt, 'input': CASES[case], 'status': 'incomplete',
              'real_llm': True, 'measurement_source': 'none; hypothetical scenario',
              'source_identity': code_identity()['core_sha256'], 'model_requested': ['gpt-6-astra', 'high'],
              'review_guide_delivered': False, 'old_new_comparison': False}
    started = time.monotonic()
    try:
        while True:
            wf.budget.check()
            wf.poll()
            task = wf.queue.get(tid)
            if task['state'] == 'completed':
                result.update(status='completed', output=wf.journal.card(task['outputs'][0]))
                break
            if task['state'] in ('failed', 'needs_clarification', 'cancelled'):
                result['failure'] = {k: task.get(k) for k in ('state', 'error', 'clarification')}
                break
            audits = list(run_dir.rglob('llm_audit.json'))
            if len(audits) >= 6 and task['state'] != 'running':
                raise RuntimeError('Replay-only cost cap: six model calls')
            if any(t['kind'] in ('experiment_request', 'experiment', 'literature') for t in wf.queue.snapshot().values()):
                raise RuntimeError('Text replay scope exceeded; stopped before numerical/network dispatch')
            wf.dispatch()
            time.sleep(.25)
    except Exception as exc:
        result['error'] = {'code': type(exc).__name__, 'message': str(exc)}
    finally:
        wf.runtime.stop()
        result['elapsed_seconds'] = time.monotonic() - started
        result['model_calls'] = [{k: a.get(k) for k in ('requested', 'actual', 'usage', 'error', 'elapsed_seconds')}
            for a in (read_json(p) for p in sorted(run_dir.rglob('llm_audit.json')))]
        result['tool_calls'] = [{**{k: t.get(k) for k in ('key', 'state', 'error')},
            'name': t['inputs']['name'], 'arguments': t['inputs']['arguments']}
            for t in wf.queue.snapshot().values() if t['kind'] == 'host_tool']
        result['numeric_tasks'] = sum(t['kind'] == 'experiment' for t in wf.queue.snapshot().values())
        result['evidence_count'] = len(wf.journal.find_cards('EvidenceCard'))
        # Check actual serialized LLM inputs, not merely the harness setup.
        requests = [read_json(p) for p in run_dir.glob('jobs/*/*/request.json')]
        result['review_guide_delivered'] = any('行为评价要点：不进入被测上下文' in json.dumps(r.get('spec', {}), ensure_ascii=False) for r in requests)
        atomic_json(BASE / (artifact_name + '_replay.json'), result)
    print(json.dumps({k: result.get(k) for k in ('case', 'status', 'elapsed_seconds', 'error', 'failure')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--attempt', help='New artifact suffix; never overwrite an existing replay')
    args = parser.parse_args()
    if args.attempt and not re.fullmatch(r'[A-Za-z0-9_-]+', args.attempt):
        parser.error('--attempt must contain only letters, digits, underscores or hyphens')
    results = []
    for case in ([args.case] if args.case else ['C01', 'C04', 'C07', 'C09']):
        results.append(run_case(case, args.attempt))
        if results[-1].get('failure') or results[-1].get('error'):
            break
    summary_name = (args.case or 'replay') + ('_' + args.attempt if args.attempt else '') + '_summary.json'
    atomic_json(BASE / summary_name, {
        'scope': 'real LLM text behavior only; no numerical or scientific benefit claims', 'cases': results})
