"""Bounded real Generation replay over exact historical code/results, no GPU jobs.

The replay's topic/action/idea state is explicitly scripted. Historical code and
numerical results are originals with hashes, never new Evidence or causal proof.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'QAViT'))
from core.workflow import Workflow
from core.journal import read_json, atomic_json, file_sha256
from core.cards import fingerprint
from tests.support import topic_decision

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
OLD=ROOT/'EXP/runs/deitb_static_w4a4_gonly_20261115_r2'
H3='0abd9a8021d7acda42bb230953d9f736ee8a74a7eef486fc6f92433f94cb3eab'
H3_REF='HypothesisCard-8cb31620665b4e71b3384774221c1d2c@1'
H2=['22bb57be0494f08bd7a1cfe852f1c3ffa558c70c575f0a8b9fc0b0bcaf15c54d',
    'acaf7ebee8b31658a824b04a53f0a4cd6e47fe6e4df7a0efa025b1f7db513141']
CASES={
 'unranked_continuation':'generate',
 'explain_simplify':'focus',
 'deferred_resume':'focus',
}

def original(path):
    return {'path':str(path),'sha256':file_sha256(path),'text':path.read_text(),
            'evidence_class':'read-only historical original; not a new measurement'}

def card(ref):
    path=OLD/'cards'/(hashlib.sha256(ref.encode()).hexdigest()+'.json')
    return original(path),read_json(path)

def material(name):
    h3_manifest=read_json(OLD/'code'/H3/'manifest.json')
    h3_source=original(Path(h3_manifest['path']))
    if h3_source['sha256']!=h3_manifest['source_sha256']:raise ValueError('historical code changed')
    h3_hyp,hyp=card(H3_REF)
    h3_spec,_=card(h3_manifest['experiment_ref'])
    result={'h3_actual_source':h3_source,'h3_original_hypothesis':h3_hyp,
            'h3_original_request':h3_spec,
            'boundary':'Historical H3 source is substantial code; its load authorization failed before mechanism measurement. No H3 performance score is available.'}
    if name=='explain_simplify':
        result={'boundary':'Actual archived dual-task values, not paired single-factor causal evidence. No new measurements.'}
        for i,filename in enumerate(H2):
            path=OLD/'solution_library'/(filename+'.json');value=read_json(path)
            result[f'h2_solution_{i}']={'path':str(path),'sha256':file_sha256(path),
                'title':value['title'],'metrics':value['metrics'], 'configuration':value['execution'],
                'evidence_class':'historical original under its own protocol'}
            result[f'h2_actual_source_{i}']=original(Path(value['code']['path']))
            result[f'h2_original_hypothesis_{i}']=card(value['hypothesis_ref'])[0]
            result[f'h2_original_evidence_{i}']=card(value['evidence_ref'])[0]
    return result,hyp

def seed_decision(wf,body,key,number=1):
    tid=wf.enqueue_role(key,'focus',refs=[wf.brief['id']],context={'round':number})
    wf.queue.ready();wf.submit(wf.queue.claim(tid),body,'EXPLICIT_SCRIPTED_STATE')
    return wf.topics.active()

def run_case(name,attempt,deadline):
    run=BASE/'live'/f'{name}_{attempt}'
    if run.exists():raise FileExistsError(run)
    cfg=deepcopy(read_json(OLD/'research_run.json')['config'])
    cfg.update(name=name,purpose='validation',enable_l3=False)
    cfg.pop('scope_target',None)
    cfg['runtime'].update(allowed_gpus=[],experiment_concurrency=1,llm_concurrency=1,task_timeout_seconds=240)
    for key in ('gpu_bindings','gpu_selection'):cfg['runtime'].pop(key,None)
    cfg['budget']={'max_rounds':2,'wall_seconds':480,'parent_deadline_unix':deadline,
                   'llm_call_limit':4,'min_free_bytes':0}
    cfg['llm']['max_transport_retries']=0
    requirements=('这是隔离的组件行为回放，topic/idea/action 由脚本建立，不是先前真实模型决定。'
        'historical_material 是实际历史实现与记录，保留原路径/SHA256；不是本次实验或新 Evidence，'
        '不得把工程失败当机制反证，也不要声称未执行的新修复/方案已经提高精度。'
        '历史卡 ID 不属于此运行，来源可在科学正文按原路径说明，不能放入本运行 evidence_refs/source_refs。'
        '当前只基于已有材料形成待验证候选或方向决定，不启动新数值、外部网络或代码生成工作；'
        '没有最低实验数或 Skill 阅读要求，材料足够即可直接提交，必要原件可按现有 read_card/read_asset 深读。')
    cfg['research_brief']['execution_guidance']={'notes':requirements}
    wf=Workflow(run,cfg)
    historical,hyp=material(name)
    seed={**topic_decision(wf),'question':'量化收益或局部误差的真实来源与必要条件是什么？',
          'mutable_dimensions':['概率编码/配对规则','尺度选择与共享','解释与必要性检验'],
          'held_conditions':['冻结 static W4A4','原 calibration/search/final 数据边界','ADE 冻结迁移'],
          'objective':'依据已有成果形成可回退的下一步',
          'rationale':'Scripted starting state for component replay; no fabricated previous research decision',
          'next_step':'在确切原源码与冻结来源上核实空间配对机制，再决定修复或简化',
          'reconsider_when':'来源绑定修复且可形成有效配对测量后复议；若实际码器没有改变则调整问题'}
    topic=seed_decision(wf,seed,'seed/focus')
    seed_task=wf.queue.get_by_key('seed/focus')
    idea,events=wf.inspirations.update_events({'title':hyp['payload']['title'],
        'idea':hyp['payload']['mechanism'],'basis':'actual historical H3 hypothesis, still unvalidated',
        'progress':'Actual source exists. Historical host load rejection prevented mechanism measurement.',
        'next_step':seed['next_step'],'status':'active','reason':'scripted retained idea state'},
        seed_task,key='replay-seed-idea')
    wf.journal.commit('seed/idea',events=events)
    if name=='deferred_resume':
        seed_decision(wf,{**topic_decision(wf),'decision':'pause','deferred_action_ids':[topic['actions'][0]['id']],
            'rationale':'Scripted pause while exact source selection was unavailable'},'seed/pause')
    messages={
        'unranked_continuation':('H3 历史未进入 Top-3，已有完整实际源码而缺少有效机制测量；'
            '另外给定一个明确假想条件：同一灵感的早期局部得分低于当前备选，但配对反证尚未完成。'
            '请沿保留灵感形成本轮 H3 候选，是否修复、调参、组合、简化或换机制由你判断；'
            '不要把假想低分当真实数值，不要求重新发明机制。'),
        'explain_simplify':('原 H2 轮2/轮3 的确切源码、配置与双任务结果已提供。'
            '请据这些实际材料决定下一研究问题与可修改范围，可以把已有收益转为解释、必要性检验或简化。'
            '跨轮成绩不是单因素控制试验，不补造因果归因。'),
        'deferred_resume':('脚本上轮主题已暂停，原动作 deferred，原源码保留。'
            '当前宿主精确 source selector 和 request-origin 回送已在 CPU 接口回归中修复，尚未重新运行该数值实验。'
            '请决定恢复、继续暂存或换题。若选择恢复同一工作，请准确接续已有 action ID；无需重建同一动作。')}
    purpose=CASES[name]
    context={'round':2,'component_validation':True,'requirements':requirements,
             'incoming_observation':messages[name],'historical_material':historical}
    if purpose=='generate':context.update(slot='H3',hypothesis_type='new_algorithm',topic_ref=topic['ref'])
    tid=wf.enqueue_role('replay/'+purpose,purpose,refs=[wf.brief['id']],context=context)
    wf=Workflow(run)
    before={'source_identity':read_json(run/'research_run.json')['code_identity'],
            'topic':wf.topics.snapshot(),'inspiration':wf.inspirations.snapshot(),
            'originals':{k:{f:v[f] for f in ('path','sha256') if f in v}
                         for k,v in historical.items() if isinstance(v,dict)}}
    result={'case':name,'attempt':attempt,'status':'incomplete','evidence_classes':{
        'topic_idea_action':'scripted state','material':'actual historical source and results',
        'new_conditions':'explicit scenario; low score is hypothetical','decision':'real Generation',
        'new_numerical_evidence':False},'before':before}
    start=time.monotonic()
    try:
        while True:
            wf.budget.check();wf.poll();task=wf.queue.get(tid)
            if task['state']=='completed':
                result.update(status='completed',decision=wf.journal.card(task['outputs'][0]));break
            if task['state'] in ('failed','cancelled','needs_clarification'):
                result['failure']={k:task.get(k) for k in ('state','error','clarification')};break
            if any(t['kind'] in ('experiment_request','experiment','literature') and t['state'] not in ('completed','failed','cancelled')
                   for t in wf.queue.snapshot().values()):
                raise RuntimeError('Replay requested work outside the bounded read/decision scope; nothing dispatched')
            wf.dispatch();time.sleep(.25)
    except Exception as exc:
        result['error']={'code':type(exc).__name__,'message':str(exc)}
    finally:
        wf.runtime.stop()
        result['elapsed_seconds']=time.monotonic()-start
        audits=[read_json(p) for p in sorted(run.rglob('llm_audit.json'))]
        result['model_calls']=[{k:a.get(k) for k in ('requested','actual','usage','elapsed_seconds','error','model_call_started')} for a in audits]
        result['tools']=[{'name':t['inputs']['name'],'arguments':t['inputs']['arguments'],'state':t['state']}
            for t in wf.queue.snapshot().values() if t['kind']=='host_tool' and t['inputs'].get('parent_task')==tid]
        result.update(topics=wf.topics.snapshot(),inspirations=wf.inspirations.snapshot(),
            new_evidence_count=len(wf.journal.find_cards('EvidenceCard')),
            numerical_tasks=sum(t['kind']=='experiment' for t in wf.queue.snapshot().values()),
            budget=wf.budget.input_receipt())
        result['original_hashes_unchanged']=all(file_sha256(v['path'])==v['sha256'] for v in before['originals'].values())
        atomic_json(BASE/f'{name}_{attempt}_result.json',result)
    print(json.dumps({k:result.get(k) for k in ('case','status','elapsed_seconds','failure','error')},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES);parser.add_argument('--attempt',default='r1')
    args=parser.parse_args();deadline=time.time()+1200
    for name in ([args.case] if args.case else CASES):
        if run_case(name,args.attempt,deadline)['status']!='completed':break
