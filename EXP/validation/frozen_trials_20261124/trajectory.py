"""Matched three-decision research rehearsal, isolated from production scheduling and the held-out partition."""
import argparse
from copy import deepcopy
from pathlib import Path
from replay import OUT,EVALUATIONS,act,get_request,projection,read
from core.cards import canonical,fingerprint
from core.journal import read_json,atomic_json,file_sha256
from agents.working_materials import material_selection
from agents.roles import role_spec

BUDGET=('一次局部研究回放。在已授权的 P64 同状态下，block 8 位于已讨论的 block 11 上游。'
        '这条短支线考察合法 P/V 网格响应，原 block 11 旁路问题保留自己的测量状态。'
        '每条轨迹的预算为一次选臂、一项增量或回退、一次复议。24 张 ImageNet dev 与 6 张 ADE dev 可供配对确认。'
        '可按需 read_card/read_asset，仍用原始来源；不调终评。其余自由建议作为未测。')
def facts(path):
    r=read_json(path)
    return {'source_sha256':file_sha256(path),'parent':r['parent_model_ref'],'split':r['split'],'samples':len(r['sample_ids']),
            'view_units':len(r['views']),'ids_sha256':fingerprint(r['sample_ids']),
            'actions':{x['id']:x['receipt']['scales'] for x in r['candidates']},
            'scores':{x['id']:x['scores'] for x in r['candidates']},'winners':r.get('winners'),
            'observations':r['observations'],'reduction':r['reduction']}
def tasks(rec):
    return {k:{'samples':v['sample_count'],'ids_sha256':fingerprint(v['sample_ids']),
               'measurements':{m:z for m,z in v['metrics'].items() if m in ('top1','baseline_top1','mIoU','baseline_mIoU','fp_mIoU','delta_mIoU_pp')}}
            for k,v in rec['tasks'].items()}
def main(variant,phase):
    req=get_request('h1_block')
    dest=EVALUATIONS/('frozen_trials_20261124_chain_'+variant)
    if phase=='begin':
        dest.mkdir(exist_ok=False)
    s=req['spec']
    current=projection(req) if variant=='new' else deepcopy(s['context'])
    trace=read_json(dest/'trace.json') if phase=='finish' else []
    state=read_json(dest/'state.json') if phase=='finish' else {}
    cal=OUT/'paired/calibration.json'; confirm=OUT/'paired/confirmation.json'
    table={'calibration':facts(cal),'confirmation':facts(confirm)}
    # One prescribed first exposure to a genuine authorized page in both arms.
    # Later calls follow the delivery policy. It is never counted as a voluntary read.
    page=read(req,{'name':'read_asset','arguments':{'asset':'skills/diagnose-ptq-failure/references/frozen-propagation.md'}})
    assets={page['asset']:page}
    visible=[]
    def decide(stage,conditions,question,fields):
        c=deepcopy(current)
        compact=variant=='new' and stage>1
        old_reads=c.get('tool_results',[])
        if compact:
            records=[(r,{'kind':'host_tool','task_id':r.get('task'),'state':r.get('state','completed'),
                          'inputs':{'name':r['name']}},r.get('result')) for r in old_reads]
            records.append(({}, {'kind':'experiment_request','inputs':{}},None))
            keep,_=material_selection({'conversation':[{'content':{'tool_calls':[{'id':'branch-feedback'}]}}]},records)
            for i,r in enumerate(old_reads):
                if i not in keep and r.get('source_ref') and r.get('result') and 'body_location' not in r['result']:
                    r['result']={'delivery':'previously_delivered_body_omitted','read_card':{'ref':r['source_ref'],'field':'/payload/result'}}
        c['local_study']={'scope':BUDGET,'stage':stage,'prior_actions':deepcopy(state),'paired_facts':table,
            'current_measurement':conditions,'question':question,'original_page':
                {'asset':page['asset'],'sha256':page['sha256'],'delivery':'previously_delivered_body_omitted',
                 'read_asset':{'asset':page['asset']}} if compact else page}
        c['payload_contract']=fields
        spec=role_spec('generation','generate',c,s['settings'],messages=s['messages']) if variant=='new' else deepcopy(s)
        spec['context']=c
        spec['tools']=[t for t in spec['tools'] if t['name'] in ('read_card','read_asset')]
        spec['system_prompt']+='\n本次在原 Generation 任务的一个独立局部支线作决定。只按 context.local_study.question 和 payload_contract 提交本步 JSON。保留原问题的证据状态，明确区分校准、search、dev 与官方结果。'
        for attempt in range(3):
            job=dest/('s%d-%d'%(stage,attempt+1))
            r,row=act(spec,job);row.update(stage=stage,job=job.name)
            trace.append(row);atomic_json(dest/'trace.json',trace)
            if not r['tool_calls']:
                if r['decision']['action']!='submit':raise ValueError('clarified; no implied scientific action')
                return r['decision']['content']
            for call in r['tool_calls']:
                try: val=read(req,call)
                except (ValueError,NotImplementedError) as exc: val={'error':str(exc)}
                spec['context'].setdefault('tool_results',[]).append({'name':call['name'],'call_id':call['id'],'result':val})
            spec['messages'].append({'role':'assistant','content':{'tool_calls':[{'id':x['id'],'name':x['name']} for x in r['tool_calls']]}})
        raise RuntimeError('bounded read budget reached')
    if phase=='begin':
        state['selection']=decide(1,{},'从实际不同目标提名的四个候选中，选一个在开发子集与确切 P64 作配对反馈。选择 parent 表示保持不改。说明取舍、当前反证及独立性。',{'candidate_id':'parent|P08|P12|both','reason':'text'})
        chosen=state['selection']['candidate_id']
        assert chosen in ('parent','P08','P12','both')
        actual=read_json(OUT/'dev/results.json')['results']
        paired={k:{'state_sha256':actual[k]['candidate']['child_state_sha256'],'model_ref':actual[k]['model_ref'],'tasks':tasks(actual[k])}
                for k in dict.fromkeys(('parent',chosen))}
        state['paired_feedback']=paired
        state['revision']=decide(2,paired,'依据刚回传的真实双任务结果，选择下一个起点与一项可辨别的修订。预算支持只改一个已存在的激活 scale：block 8 的 P 或 V，倍率 (0.5,1.5)；也可以停止。相对 base 的当前参数。给出确切 base、理由和复议条件。',{'base_id':'parent or chosen candidate id','target':'P|V|stop','factor':'number or 1 for stop','reason':'text','reconsider_when':'text'})
        assert state['revision']['base_id'] in ('parent',chosen)
        assert state['revision']['target'] in ('P','V','stop')
        assert .5 < state['revision']['factor'] < 1.5
    else:
        feedback=read_json(OUT/('revision_'+variant)/'result.json')
        state['disposition']=decide(3,feedback,'依据刚完成的修订、冷载和配对反馈，决定在这条支线保留哪个状态，以及下一项尚未测的行动。给出确切状态身份和收益边界。',{'decision':'retain|revert|defer','state_sha256':'actual hash','reason':'text','next_step':'text'})
    atomic_json(dest/'state.json',state)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['old','new']);p.add_argument('phase',choices=['begin','finish'])
    a=p.parse_args();main(a.variant,a.phase)
