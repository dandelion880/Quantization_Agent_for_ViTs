"""Isolated cut-off replay. Only frozen request grants and measurements from this branch reach the actor."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
import hashlib

from runtime.paths import ROOT, QAVIT
from core.cards import canonical, fingerprint, validate_schema
from core.journal import atomic_json, read_json, file_sha256
from agents.roles import role_spec, submission_schema
from agents.interaction import RoleInteraction
from agents.working_materials import research_card, research_recovery, material_selection
from agents.source_workspace import asset_text_page
from tools.evaluation.breakpoint import start_request
from tools.evaluation.common import EVALUATIONS
from runtime.model_router import call_once
from runtime.identity import code_identity

OUT=Path(__file__).resolve().parent
COLLECTION=read_json(OUT.parent/'engineering_replay_20261122/collection.json')
SKILL='skills/select-calibration-objective/references/ranking-validation.md'

def get_request(case):
    case=COLLECTION['cases'][case]
    assert case['partition']=='development'
    return start_request(EVALUATIONS/COLLECTION['evaluation'], case)

def grant(req,path):
    path=Path(path).resolve()
    grants=req['read_permissions']
    if str(path) not in grants or not grants[str(path)] or RoleInteraction.archive_path(path):
        raise ValueError('unavailable at frozen read view')
    if path.suffix not in ('.py','.json','.md','.txt') or file_sha256(path)!=grants[str(path)]:
        raise ValueError('pinned bytes unavailable')
    return path

def read(req,call):
    a=call['arguments'];name=call['name']
    if name=='read_asset' and 'snapshot_field' not in a:
        path=grant(req,RoleInteraction.asset_path(a['asset']))
        b=path.read_text()
        return {'asset':str(path),'sha256':req['read_permissions'][str(path)],**asset_text_page(b,a,12000)}
    if name=='read_card':
        ref=req['spec']['context']['memory']['source_handles'].get(a['ref'],a['ref'])
        filename=hashlib.sha256(ref.encode()).hexdigest()+'.json'
        path=next((p for p in req['read_permissions'] if p.endswith('/cards/'+filename)),None)
        if not path: raise ValueError('card outside frozen view')
        c=read_json(grant(req,path))
        if ref!=c['id']: raise ValueError('wrong source')
        field=a.get('field','/payload/observations' if 'select' in a else '')
        val=RoleInteraction.card_field(c,field)
        if 'select' in a:
            from core.evidence import selected_leaves
            rows=list(selected_leaves(val,field,a['select']))
            off=a.get('offset',0); n=min(50,a.get('limit',20))
            return {'ref':ref,'field':field,'sha256':fingerprint(val),'entries':rows[off:off+n],'total':len(rows),'offset':off}
        if not field and not ('offset' in a or 'limit' in a) and len(canonical(val))<=12000: return c
        return {'ref':ref,'field':field,'sha256':fingerprint(val),**asset_text_page(canonical(val),a,12000)}
    raise NotImplementedError(name)

def act(spec,path):
    before=time.monotonic()
    out=call_once(spec,path,time.time()+240)['response']
    atomic_json(path/'response.json',out)
    return out,{'seconds':time.monotonic()-before,'usage':out['usage'],'action':out['tool_calls'] or out['decision']}

def projection(req):
    c=deepcopy(req['spec']['context'])
    for card in c['memory'].get('cards',[]):
        if card.get('kind') in ('ExperimentCard','EvidenceCard') and card.get('payload'):
            prior=deepcopy(card)
            research_card(card)
            manifest=c['memory'].get('read_manifest',{})
            if prior!=card and card['id'] in manifest.get('refs',[]):
                manifest['refs'].remove(card['id'])
                manifest.setdefault('projected_refs',[]).append(card['id'])
    research_recovery(c)
    # Compact only past read bodies. A pinned TaskCard contains the complete result
    # even when a later file/host version changes. No future task result is opened.
    rows=[]
    for r in c.get('tool_results',[]):
        child={'kind':'host_tool' if r['name'].startswith(('read','retrieve')) else 'experiment_request',
               'task_id':r['task'],'task_card':r['source_ref'],'state':r['state'],'inputs':{'name':r['name']}}
        rows.append((r,child,r.get('result')))
    keep,_=material_selection({'conversation':req['spec']['messages']},rows)
    for i,(r,_,result) in enumerate(rows):
        if i not in keep and result and 'body_location' not in result:
            r['result']={'delivery':'previously_delivered_body_omitted','read_card':{'ref':r['source_ref'],'field':'/payload/result'},
                         'sha256':fingerprint(result)}
    if 'experiment_interface' in c:
        path=QAVIT/'docs/EXPERIMENT_INPUTS.md'
        c['experiment_interface']={'read_asset':{'asset':str(path)},'sha256':req['read_permissions'].get(str(path))}
        c['memory'].get('asset_locations',{}).pop(str(path),None)
    return c

def focus(variant,dest):
    req=get_request('initial')
    s=req['spec']
    ctx=projection(req) if variant!='old' else deepcopy(s['context'])
    if variant in ('direct','skill'):
        hand=ctx['research_brief']['payload']['research_handoff']
        # Source is explicitly bound in the Brief even for old calls whose path ACL missed it.
        facts={}
        for k in ('b_development','baseline_plan'):
            path=(ROOT/hand[k]['path']).resolve()
            assert file_sha256(path)==hand[k]['sha256']
            obj=read_json(path)
            if k=='baseline_plan':
                obj=obj['plan']
                obj={'execution':obj['execution'],'preprocess':obj['preprocess'],
                     'first_actual_layers':obj['layers'][:2],'layer_count':len(obj['layers'])}
            facts[k]={'source':hand[k],'body':obj}
        ctx['bound_parent_facts']=facts
    if variant=='skill':
        ctx['strategy_page']=read(req,{'name':'read_asset','arguments':{'asset':SKILL}})
    spec=deepcopy(s) if variant=='old' else role_spec('generation','focus',ctx,s['settings'])
    trace=[]
    for i in range(3):
        r,row=act(spec,dest/str(i+1));trace.append(row)
        atomic_json(dest/'trace.json',trace)
        if not r['tool_calls']:
            validate_schema(r['decision']['content'], submission_schema('focus'),'focus')
            break
        feedback=[]
        for call in r['tool_calls']:
            if call['name']=='request_experiment':
                from tools.experiments import preflight_experiment
                try:
                    prepared=preflight_experiment(call['arguments']['experiment'],ctx['research_brief']['payload'])
                    row['preflight']=prepared['execution']['service']
                except Exception as exc:
                    row['preflight_error']=repr(exc)
                row['numerical_status']='unmeasured'
                atomic_json(dest/'trace.json',trace)
                return
            try: body=read(req,call)
            except ValueError as exc: body={'error':str(exc)}
            except NotImplementedError:
                row['unserved']=call['name'];atomic_json(dest/'trace.json',trace);return
            feedback.append({'name':call['name'],'call_id':call['id'],'result':body})
        spec['context']['tool_results']=[*spec['context'].get('tool_results',[]),*feedback]
        spec['messages'].append({'role':'assistant','content':{'tool_calls':[{'id':c['id'],'name':c['name']} for c in r['tool_calls']]}})

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('variant');args=a.parse_args()
    dest=EVALUATIONS/('frozen_trials_20261124_focus_'+args.variant+'_complete')
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('choose a fresh replay output')
    dest.mkdir(exist_ok=True)
    focus(args.variant,dest)
