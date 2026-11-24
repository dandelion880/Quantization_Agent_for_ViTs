"""Paired development breakpoints. Call cap and elapsed deadline apply to the entire arm."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import time

from agents.interaction import RoleInteraction
from agents.roles import role_spec, submission_schema
from agents.source_workspace import asset_text_page
from agents.working_materials import review_subject
from core.cards import ContractError, canonical, fingerprint, validate_schema, validate_card
from core.journal import atomic_json, file_sha256, read_json
from runtime.configuration import model_defaults
from runtime.model_router import call_once
from runtime.paths import ROOT
from tools.evaluation.breakpoint import start_request
from tools.evaluation.judge import evaluation_manifest

OUT=Path(__file__).resolve().parent
G=ROOT.with_name('Quantization_Agent_for_ViTs_gonly')
COL=read_json(G/'EXP/validation/engineering_replay_20261122/collection.json')
ARCHIVE=G/'EXP/evaluations'/COL['evaluation']


def original(number):
    if number==2:
        return start_request(ARCHIVE,COL['cases']['synthesis'])
    manifest=evaluation_manifest(ARCHIVE)
    tid='task-'+fingerprint({'run':COL['evaluation'],'key':'r001/select'})[:24]+'-a0001'
    row=next(r for r in manifest['source_files'] if r['path']==f'jobs/{tid}/request.json')
    path=ARCHIVE/row['object']
    assert path.resolve().is_relative_to((ARCHIVE/'sources').resolve()) and file_sha256(path)==row['sha256']
    req=read_json(path)
    assert req['execution_id']==tid and req['spec']['purpose']=='select'
    return req


def card(req,ref):
    handles=req['spec']['context']['memory']['source_handles']
    ref=handles.get(ref,ref)
    filename=hashlib.sha256(ref.encode()).hexdigest()+'.json'
    path=next(Path(p) for p in req['read_permissions'] if p.endswith('/cards/'+filename))
    if file_sha256(path)!=req['read_permissions'][str(path)]:
        raise ContractError('source changed')
    v=read_json(path);validate_card(v)
    assert v['id']==ref
    return v


def packet(req, calls_cap=3):
    old=req['spec']['context']
    candidates=old['task']['candidate_refs']
    subjects=[review_subject(card(req,ref)) for ref in candidates.values()]
    # Freeze the exact direct projections used for these dated, paired comparisons.
    for subject in subjects:
        p = subject['payload']
        for field in ('research_question','evidence_interpretation','validation_design'):
            p.pop(field, None)
        intervention = p.get('intervention')
        if calls_cap == 3 and isinstance(intervention, dict) and intervention.get('complete') is False:
            intervention.pop('fields', None)
    facts=[]
    for v in old['memory']['cards']:
        if v['kind']!='EvidenceCard':continue
        p=v.get('payload',{})
        key=p.get('key_facts',{})
        if not key: continue
        exp=card(req,key['experiment_ref'])
        facts.append({'id':v['id'],'kind':'EvidenceCard','question':exp['payload']['question'],
            'key_facts':{k:key[k] for k in ('status','measurement_source','evaluations','limitations','protocol_hash') if k in key},
            'read_card':{'ref':v['id']}})
    brief=old['research_brief']
    b=brief['payload']; topic=old.get('research_topic') or {}
    return {'task':{'round':old['task']['round'],'candidate_refs':candidates},
            'research_brief':{k:b[k] for k in ('scientific_question','goals','hard_constraints','evaluation_protocol')},
            'brief_ref':brief['id'],'research_topic':{k:topic[k] for k in ('ref','question','mutable_dimensions','objective','next_step') if k in topic},
            'candidate_research_bindings':old.get('candidate_research_bindings',{}),
            'fallback_B':{k:v for k,v in (old.get('fallback_B',{}).get('candidate') or {}).items() if k in ('id','title','metrics','configuration_asset','status')},
            'memory':{'cards':subjects+facts,'source_handles':{k:v for k,v in old['memory']['source_handles'].items() if v in candidates.values() or v in {f['id'] for f in facts}}},
            'read_view':old['read_view'],'validation_budget':'同一断点每侧至多三次调用、420 秒、零新增数值作业。可以提交必要的修改或实验请求，未执行时保留待测状态。',
            'tool_results':[]}


def on_read(req,call):
    args=call['arguments']
    if call['name']=='read_asset':
        asset=Path(args['asset'])
        options=[asset] if asset.is_absolute() else [G/asset,G/'QAViT'/asset,G/'QAViT/docs'/asset]
        path=next(p for p in options if str(p.resolve()) in req['read_permissions']).resolve()
        if RoleInteraction.archive_path(path) or file_sha256(path)!=req['read_permissions'][str(path)]:raise ValueError('unavailable source')
        return asset_text_page(path.read_text(),args,12000)
    if call['name']!='read_card':return {'status':'unavailable_in_frozen_replay'}
    value=card(req,args['ref'])
    return asset_text_page(canonical(RoleInteraction.card_field(value,args.get('field',''))),args,12000)


def compare(number, enabled, calls_cap=3, output=OUT):
    req=original(number)
    common=packet(req,calls_cap)
    directory=output.resolve()/f'r{number}-{"critic" if enabled else "self"}'
    directory.mkdir(parents=True, exist_ok=False)
    if calls_cap != 3:
        common['validation_budget'] = f'同一断点每侧总计最多 {calls_cap} 次角色调用，420 秒，零新增数值作业。可提交方案修订、可检验建议或必要实验请求。'
    settings=deepcopy(req['spec']['settings']);settings['agent_models']['critic']=model_defaults(critic=True)['critic']
    phase='critique' if enabled else 'select'
    ctx=deepcopy(common)
    trace=[];deadline=time.time()+420
    for i in range(calls_cap):
        ctx['remaining_role_calls']=calls_cap-i
        if calls_cap != 3:
            turns = calls_cap - 2 - i if phase == 'critique' else calls_cap - i
            ctx['current_role_calls_available'] = turns
            if turns == 1:
                ctx['breakpoint_closeout'] = '这是该角色在本断点的最后一次调用。按已有事实完成科学决定，保留未验证处；请求新测量将记为待执行。'
        spec=role_spec('critic' if phase=='critique' else 'generation',phase,ctx,settings)
        where=directory/str(i+1);start=time.monotonic()
        response=call_once(spec,where,deadline)['response'];atomic_json(where/'response.json',response)
        d=response.get('decision');calls=response.get('tool_calls') or []
        row={'role':spec['role'],'seconds':time.monotonic()-start,'usage':response['usage'],'read_chars':0,
             'action':d or calls,'context_sha256':fingerprint(ctx),'requested_route':settings['agent_models'][spec['role']]}
        trace.append(row);atomic_json(directory/'trace.json',trace)
        if d and d['action']=='submit':
            content=d['content'];validate_schema(content,submission_schema(phase),phase)
            if phase=='select':break
            refs=common['task']['candidate_refs']
            for note in content.get('candidate_notes',[]):
                assert common['memory']['source_handles'].get(note['hypothesis_ref'],note['hypothesis_ref']) in refs.values()
            ctx=deepcopy(common)
            ctx['task']['critic_feedback']={'review_ref':'isolated-review-'+fingerprint(content),'candidate_refs':refs}
            ctx['memory']['cards'].append({'kind':'ReviewDecisionCard','payload':content})
            phase='select';continue
        if any(c['name']=='revise_candidate' for c in calls) or (d and d['action']=='clarify'):break
        values=[]
        for call in calls:
            if call['name']=='request_experiment':v={'status':'not_executed','reason':'no numerical budget in paired breakpoint'}
            else:
                try:v=on_read(req,call)
                except (OSError,ValueError,StopIteration,KeyError) as exc:v={'status':'unavailable_in_frozen_replay','reason':str(exc)}
            row['read_chars']+=len(v.get('text',''))
            values.append({'name':call['name'],'arguments':call['arguments'],'result':v})
        ctx['tool_results']=values
        if phase=='critique' and i==(1 if calls_cap==3 else calls_cap-3):
            phase='select';ctx=deepcopy(common)
            ctx['task']['critic_feedback']={'status':'no_review_within_replay_budget'}
        atomic_json(directory/'trace.json',trace)
    return trace

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('round',type=int);parser.add_argument('arm',choices=['self','critic'])
    parser.add_argument('--calls', type=int, choices=(3,5), default=3)
    parser.add_argument('--output', type=Path, default=OUT)
    args=parser.parse_args();compare(args.round,args.arm=='critic',args.calls,args.output)
