"""Bounded real G-only research. Real LLM + actual torch, synthetic tiny ViT only.

Uses unmodified Workflow.advance/run, workers, host tools and evidence contracts.
No scripted scientific decisions and no L3 or dataset-scale execution.
"""
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


def summarize(wf, report):
    tasks = list(wf.queue.snapshot().values())
    calls = []
    for path in sorted(wf.run_dir.glob('jobs/*/llm_audit.json')):
        audit = read_json(path)
        calls.append({k: audit.get(k) for k in ('role', 'requested', 'actual', 'model_call_started', 'usage', 'error', 'elapsed_seconds')})
    evidence = wf.journal.find_cards('EvidenceCard')
    result = {
        'status': report['status'], 'real_llm': True, 'model_data': 'seeded tiny ViT / synthetic RGB32',
        'formal_l3_executed': False, 'report': report,
        'hypotheses': [{'ref': c['id'], **{k: c['payload'].get(k) for k in ('round', 'slot', 'title', 'evidence_refs', 'feedback_used')}}
                       for c in wf.journal.find_cards('HypothesisCard')],
        'measurements': [{'ref': c['id'], 'level': wf.journal.card(c['payload']['experiment_ref'])['payload']['level'],
                          **{k: c['payload'].get(k) for k in ('measurement_source', 'status', 'execution_id', 'observations')}} for c in evidence],
        'tasks': [{k: t.get(k) for k in ('key', 'kind', 'role', 'state', 'error')} for t in tasks],
        'calls': calls,
        'no_retired_research_tasks': not any(t['role'] in ('judger', 'evolution') for t in tasks),
        'all_generation_routes_astra_high': all(c['requested'] == {'model':'gpt-6-astra', 'reasoning_effort':'high'}
                                               for c in calls if c['role'] == 'generation'),
    }
    atomic_json(BASE / (wf.run_dir.name + '_result.json'), result)
    print(json.dumps({k: result[k] for k in ('status', 'no_retired_research_tasks', 'all_generation_routes_astra_high')}
                     | {'hypotheses': len(result['hypotheses']), 'measurements':len(evidence), 'calls':len(calls)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='live_gonly_r1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--h3-only', action='store_true', help='Bounded independent H3 follow-up, no complete round')
    args = parser.parse_args()
    directory = BASE / args.name
    if args.resume:
        wf = Workflow(directory)
    else:
        config = validation_config()
        config.update(name=args.name, scope_target={'round':2, 'stage':'generation'})
        config['budget'].update(max_rounds=2, wall_seconds=900, closeout_reserve_seconds=90,
                                shutdown_reserve_seconds=15, min_free_bytes=1024**3, max_artifact_bytes=1024**3)
        config['runtime'].update(cpu_threads=2, allowed_gpus=[], llm_concurrency=1, task_timeout_seconds=180,
                                 numeric_task_timeout_seconds={'L1':90,'L2':120,'L3':120})
        config['llm']['max_transport_retries'] = 0
        brief = config['research_brief']
        brief['scientific_question'] = (
            '这是900秒内的小规模G-only科研闭环验证。研究 seeded tiny ViT 的 blocks.0.mlp.fc2 激活静态A4：'
            '全局per_tensor与channel-wise的误差差异，是否源于通道尺度失衡，局部输入MSE能否代表算子输出误差？'
            'H1优先保留完整成熟种子并清楚标明待正式评价；H2根据真实tensor_probe诊断提出改进；'
            'H3提出与竞争解释可区分的新机制。请自主选择能改变方案的L1/L2实验，分析正负结果后修订或提交。'
            '当前只是随机小ViT和合成输入，不能外推ImageNet/ADE20K精度。请将研究保持在已支持的局部tensor_probe能力与小样本范围内，'
            '避免文献联网和大型实现。校准2、search4；FP源保持不变、不训练、不读final。'
            '每个任务依据已有证据及时提交，不用反复实验耗尽预算；第二轮优先使用前轮Memory修订三个假说。')
        brief['goals'] = {'validation': 'H1/H2/H3、真实实验申请/反馈、证据修订与预选、Recorder、第二轮Memory交接；L3关闭。'}
        brief.pop('experiment_example', None)
        brief['execution_guidance'] = {'notes': ('仅用CPU局部tensor_probe；per_channel使用axis=-1，granularity与resolved_choices保持一致。'
            '已有方法可直接提交execution.service=tensor_probe、method=rtn和parameters。L1/L2自由选择，不要求固定实验次数。')}
        if args.h3_only:
            config.pop('scope_target')
            config['budget'].update(max_rounds=1, wall_seconds=300, closeout_reserve_seconds=0)
            config['runtime']['task_timeout_seconds'] = 300
            config['llm']['max_transport_retries'] = 0
            brief['goals'] = {'validation': 'Independent H3 capability follow-up; no complete round or L3.'}
            brief['scientific_question'] = (
                '这是300秒内的独立H3能力补测：请提出区别于单纯按通道尺度或按局部输出误差选粒度的开放创新机制。'
                '给定同任务 live_gonly_r2 已实测的合成tiny ViT FC2静态A4局部诊断（这些是Brief转述的历史数据，'
                '不属于本次Journal的Evidence，不能伪造本次实验引用）：2张校准时per_tensor/per_channel的'
                '算子输出NMSE分别0.03187111017/0.00525990191，同4张search整网输出NMSE为'
                '0.00148388759/0.00022766676；只将校准数改为1时，局部NMSE为'
                '0.00853767002/0.00204406054，search整网输出NMSE却为0.00032487156/0.00035894608。'
                'FP模型seed42、search样本及其顺序相同，权重和其它位点保持FP；输入为合成RGB32，不是任务准确率。'
                'H1保留RepQ完整种子加局部per_channel适配；H2按局部算子误差选择粒度，额外收益未证实。'
                '请用该排序反转收窄主张，形成有明确算法干预和可区分竞争机制预测的新假说，说明不同结果会怎样修订。'
                '保留静态W4A4、无训练和既有完整方案，不外推ImageNet/ADE20K。可按需读取静态知识/Skill；'
                '本次不联网检索，不宣称世界首创。'
                '只需简洁的可检验提案，不要求本次求解最优算法；正文控制在约1200中文字以内。'
                '实验选择自主，已有资料足以形成可检验假说时可直接提交，不必再次复现这四个条件。'
                '若新增测量，只限已有CPU tensor_probe和小样本，不读取final或启动完整模型评估。')
        atomic_json(BASE / (args.name + '_input.json'), config)
        wf = Workflow(directory, config)
    if not args.h3_only:
        summarize(wf, wf.run())
        return
    parent = wf.enqueue_role('smoke/generate/H3', 'generate', refs=[wf.brief['id']],
                             context={'round': 1, 'slot': 'H3', 'hypothesis_type': 'new_algorithm'})
    status, error = 'validation_completed', None
    try:
        while True:
            wf.budget.check()
            wf.poll()
            task = wf.queue.get(parent)
            if task['state'] == 'completed':
                break
            if task['state'] in ('failed', 'cancelled', 'needs_clarification'):
                raise RuntimeError(str(task.get('error') or task['state']))
            wf.dispatch()
            time.sleep(.2)
    except BaseException as exc:
        status, error = 'validation_incomplete', {'code': type(exc).__name__, 'message': str(exc)}
    finally:
        wf.runtime.stop()
        wf.poll()
    summarize(wf, wf.summary(status, error=error))


if __name__ == '__main__':
    main()
