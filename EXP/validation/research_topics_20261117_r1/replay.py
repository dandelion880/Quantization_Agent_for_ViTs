"""Three bounded real Generation topic replays; zero numerical dispatch.

Seed topics/ideas/method/code are explicit state fixtures. Historical scalar
metrics are copied read-only with file hashes and never imported as new Evidence.
Each case changes only the incoming observation/uncertainty, not an expected label.
"""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json, read_json, file_sha256
from tests.support import validation_config, topic_decision, hypothesis
from tests.test_inspirations import InspirationTests

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
OLD=ROOT/'EXP/runs/deitb_static_w4a4_gonly_20261115_r2'
CASES={
 'no_gain': '假想新增观察：同一想法的初版没有提高指标，但局部误差与传播后表征误差排序不一致的配对检查尚未完成；已有实现中存在一项可定位、可回退的裁剪修复，预算足以做一个小型配对测量。尚无证据证明机制命题被反驳。请据这些条件作当前方向决定，不因已投入而默认继续。',
 'new_evidence': '假想新增证据：相同冻结背景下的配对检查发现，在当前候选范围内尺度干预被后续归一化抵消，实际码器未改变；旧进展把代码安装完成误记为机制支持。同时原有传播表征问题仍未解决，尚缺扩大候选生成或改变干预位置所需诊断。本次请根据该条件重新决定问题和范围，区分命题适用条件失败与工程未测，不必为延续而保留原设定。',
 'explain_simplify': '假想当前研究状态：可比较实验确认 ADE 主指标有小幅收益，但同时改变 metric、候选分组和前瞻步骤，来源尚不能归因；更多复杂度提高了校准成本，没有组件必要性的证据。你可以选择继续提高指标、解释收益、检验必要性、共享参数或简化，说明哪种新信息最有决策价值。该假想确认不应与下面不受控的历史成绩混为一份真实证据。'
}


def history():
    result=[]
    for p in sorted((OLD/'solution_library').glob('*.json')):
        row=read_json(p)
        if row.get('status')!='evaluated':continue
        ref=row.get('evidence_ref')
        result.append({'origin':'historical numerical result, not a new/current measurement',
            'file':str(p),'sha256':file_sha256(p), 'original_evidence_id':ref,
            'model_ref':row['id'],'metrics':row['metrics'],
            'scope':'original run/protocol only; comparisons are not paired causal controls'})
    return result


def run_case(name,attempt='r1'):
    run=BASE/'live'/f'{name}_{attempt}'
    if run.exists():raise FileExistsError(run)
    cfg=validation_config()
    cfg.update(name=name,purpose='validation',enable_l3=False)
    cfg['budget']={'max_rounds':2,'wall_seconds':360,'llm_call_limit':4,'min_free_bytes':0}
    cfg['runtime'].update(allowed_gpus=[],llm_concurrency=1,task_timeout_seconds=180)
    cfg['llm']['max_transport_retries']=0
    requirements=('隔离的动态主题决策回放。上轮主题/灵感/方法/代码是明确脚本状态夹具，'
        '情境新增观察是假想输入；historical_numeric 是真实历史数值摘要，有原路径与hash，但不是本运行的新测量、'
        '不是同背景受控对照，不能支持假想机制的实证结论。当前 Brief 是 tiny 状态验证，历史大模型指标不可外推。'
        '请先使用宿主恢复信息，按需读取原件，再根据当前输入作一次共同主题决定；'
        '没有唯一正确选题标签，不指定必须用哪个决策。只检查决策及可恢复下一步，不开展数值或外部网络实验。'
        '历史原 ID 不是本运行 Journal 引用，不放入 source_refs。未获得的测量保持未知。')
    cfg['research_brief']['execution_guidance']={'notes':requirements}
    wf=Workflow(run,cfg)
    helper=InspirationTests()
    wf.advance();wf.queue.ready();task=wf.queue.claim(wf.tasks()['r001/focus']['task_id'])
    seed={**topic_decision(wf),'question':'尺度选择何时不能保持传播后的表征？',
        'mutable_dimensions':['metric定义','前瞻程度','候选生成','分组规则'],
        'held_conditions':['ResearchBrief硬约束与校准划分','相同FP背景与原评测协议'],
        'objective':'解释误差传播并改善量化表示',
        'next_step':'在保留源码上修复裁剪并完成FP/Q同背景传播表征误差配对检查',
        'reconsider_when':'若实际码器未改变，或配对结果排除传播误差解释，则重新选择干预范围',
        'rationale':'明确脚本初始化，仅用于验证恢复和真实Generation决策；不是历史LLM决定'}
    wf.submit(task,seed,'SCRIPTED_SEED_ONLY');wf.advance()
    # Prevent accidental launch of the unused candidate slots in this component replay.
    for slot in ('H1','H2'):
        wf.queue.change(wf.tasks()['r001/generate/'+slot]['task_id'],'cancelled',patch={'error':{'code':'component_fixture_only'}})
    wf.queue.ready();parent=wf.queue.claim(wf.tasks()['r001/generate/H3']['task_id'])
    identifier=helper.idea(wf,parent,status='active',reason='scripted continuity fixture, not measured support')
    request,artifact,_=helper.code_request(wf,parent)
    wf.queue.ready();parent=wf.queue.claim(parent['task_id'])
    value=hypothesis(1,'H3');value.update(title='脚本占位实现，仅检查精确接续',
        mechanism='状态夹具；当前代码不是经过测量的量化算法，不支持机制性能主张。',
        method_lineage={'name':'state-fixture-method','change':'seed original code/config for recovery'})
    wf.submit(parent,value,'SCRIPTED_SEED_ONLY')
    refs=[wf.brief['id'],*wf.candidate_refs(1).values()]
    tid=wf.enqueue_role('replay/focus','focus',refs=refs,context={
        'round':2,'component_validation':True,'requirements':requirements,
        'incoming_observation':{'kind':'hypothetical_case','text':CASES[name]},
        'historical_numeric':history()})
    # Demonstrate restart before the first real call, with source identity unchanged.
    wf=Workflow(run)
    initial=wf.request_body(wf.queue.get(tid))
    atomic_json(BASE/f'{name}_{attempt}_input_summary.json',{
        'scenario':CASES[name],'requirements':requirements,
        'restored_topic':initial['spec']['context']['research_topic'],
        'restored_idea':initial['spec']['context']['inspiration_continuation'],
        'method_directory':initial['spec']['context']['method_catalog_state'],
        'historical_numeric':history(),'original_code':artifact})
    result={'case':name,'attempt':attempt,'status':'incomplete','real_generation':True,
        'seed_state':'scripted fixture','observations':'hypothetical conditions',
        'historical_numeric':'read-only original scalars with hashes; no new Evidence',
        'before_topic_ref':wf.topics.active()['ref'],'idea_id':identifier,
        'method_id':wf.journal.card(wf.candidate_refs(1)['H3'])['payload']['method_id']}
    started=time.monotonic()
    try:
        while True:
            wf.budget.check();wf.poll();task=wf.queue.get(tid)
            if task['state']=='completed':
                result.update(status='completed',decision=wf.journal.card(task['outputs'][0]));break
            if task['state'] in ('failed','cancelled','needs_clarification'):
                result['failure']={k:task.get(k) for k in ('state','error','clarification')};break
            pending=[t for t in wf.queue.snapshot().values() if t['state'] not in ('completed','cancelled','failed')]
            if any(t['kind'] in ('experiment_request','experiment','literature') for t in pending):
                raise RuntimeError('Replay scope violation; no numerical/network task dispatched')
            wf.dispatch();time.sleep(.25)
    except Exception as exc:
        result['error']={'code':type(exc).__name__,'message':str(exc)}
    finally:
        wf.runtime.stop()
        result['elapsed_seconds']=time.monotonic()-started
        audits=[read_json(p) for p in sorted(run.rglob('llm_audit.json'))]
        result['model_calls']=[{k:a.get(k) for k in ('requested','actual','usage','elapsed_seconds','error','model_call_started')} for a in audits]
        result['tools']=[{'name':t['inputs']['name'],'arguments':t['inputs']['arguments'],'state':t['state']}
            for t in wf.queue.snapshot().values() if t['kind']=='host_tool' and t['inputs'].get('parent_task')==tid]
        result['numeric_worker_count']=sum(t['kind']=='experiment' for t in wf.queue.snapshot().values())
        result['new_evidence_count']=len(wf.journal.find_cards('EvidenceCard'))
        result['retained_idea']=wf.inspirations.snapshot().get(identifier)
        result['budget']=wf.budget.input_receipt()
        atomic_json(BASE/f'{name}_{attempt}_result.json',result)
    print(json.dumps({k:result.get(k) for k in ('case','status','elapsed_seconds','failure','error')},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES);parser.add_argument('--attempt',default='r1')
    args=parser.parse_args()
    for case in ([args.case] if args.case else CASES):
        result=run_case(case,args.attempt)
        if result['status']!='completed':break
