"""Single real Generation→Coding→Safeguard→Evidence→Generation smoke, no research loop.

Reuses the explicitly selected input's model/data/resources. No scripted model decisions,
reference implementation or numerical oracle. Output must be a new EXP directory.
"""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import ROOT
from core.journal import read_json, atomic_json
from core.workflow import Workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--wall-seconds', type=int, default=1800)
    parser.add_argument('--question-file', help='UTF-8 scientific request for this one-experiment validation')
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    if directory.exists() or ROOT / 'EXP' not in directory.parents:
        parser.error('output must be a new EXP directory')
    config = read_json(args.config)
    config.update(name=directory.name, purpose='validation', enable_l3=False)
    config['budget'].update(wall_seconds=args.wall_seconds, max_rounds=1,
                            closeout_reserve_seconds=0, shutdown_reserve_seconds=15)
    config['budget'].pop('parent_deadline_unix', None)
    brief = config['research_brief']
    brief['scientific_question'] = (
        '本次只验证一次真实局部研究链，不开启完整研究轮次。请在生成H1前调用一次request_experiment：'
        '以真实DeiT的blocks.0.attn.proj输入为统计位点，新增一份quantizer_v1实现：'
        '用规定32张calibration图像捕获的激活绝对值的0.995分位数拟合对称per_tensor静态4-bit scale，'
        'scale=max(quantile(abs(x),0.995)/7,1e-12)，round ties-to-even后裁剪到[-8,7]再反量化。'
        '用同一FP32模型和相同输入测量输入及局部算子输出relative MSE，不训练、不读final、不做整网评估。'
        '这次明确要求Coding实现该fit/apply算法以检验新代码链。请求只需question、measurements和'
        '必要科学约束，execution可仅指定service=tensor_probe、requires_code=true、site；'
        '参数选择交Coding补全，不填内部身份、scope、资源、validation、预测或fixture。'
        '拿到真实Evidence后，回到当前Generation任务分析它，提交一个范围严格限于该局部测量的H1；'
        '不得继续申请第二个实验，不声称量化全模型精度或硬件收益。')
    if args.question_file:
        brief['scientific_question'] = Path(args.question_file).read_text(encoding='utf-8')
    brief['goals'] = {'validation': 'one local implementation chain and original researcher interpretation only'}
    brief.pop('experiment_example', None)
    wf = Workflow(directory, config)
    parent = wf.enqueue_role('smoke/generate', 'generate', refs=[wf.brief['id']],
        context={'round':1,'slot':'H1','hypothesis_type':'existing_method'})
    result = {'status':'incomplete','real_llm_used':True,'full_research_loop_started':False}
    try:
        while True:
            wf.budget.check()
            wf.poll()
            task = wf.queue.get(parent)
            if task['state'] == 'completed':
                break
            blocked = [t for t in wf.queue.snapshot().values() if t['state']=='needs_clarification']
            if task['state'] in ('failed','cancelled') or blocked:
                raise RuntimeError(str({'parent':task, 'clarifications':blocked}))
            requests = [t for t in wf.queue.snapshot().values() if t['kind']=='experiment_request']
            if len(requests)>1:
                raise RuntimeError('smoke budget allows only one research request')
            wf.dispatch()
            time.sleep(.2)
        tasks = list(wf.queue.snapshot().values())
        evidence = wf.journal.find_cards('EvidenceCard')
        assert len(evidence)==1 and evidence[0]['payload']['measurement_source']=='real', 'missing real Evidence'
        assert not any(t['kind']=='verification' or t['inputs'].get('purpose')=='reference' for t in tasks)
        researcher = wf.journal.card(task['outputs'][0])
        assert evidence[0]['id'] in researcher['refs'], 'researcher did not cite returned Evidence'
        reviews = [c for c in wf.journal.find_cards('ReviewDecisionCard') if c['payload'].get('phase')=='safeguard']
        assert reviews and reviews[-1]['payload']['approved']
        result.update(status='passed',evidence_refs=[e['id'] for e in evidence],
                      researcher_ref=researcher['id'],safeguard_refs=[r['id'] for r in reviews],
                      tasks=[{k:t.get(k) for k in ('key','kind','state','outputs')} for t in tasks],
                      observations=evidence[0]['payload']['observations'])
    except BaseException as exc:
        result['error']={'code':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        wf.runtime.stop()
        atomic_json(directory/'service_validation.json',result)
        print(json.dumps(result,ensure_ascii=False))

if __name__ == '__main__': main()
