"""Index the original receipts and summarize observed results; no generated research submissions."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'QAViT'))
from runtime.paths import QAVIT
from core.journal import read_json, file_sha256
from core.cards import fingerprint
from core.workflow import Workflow
def write(path,body):
    path.write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n')
def asset(path):
    return {'path':str(path.resolve().relative_to(ROOT)), 'sha256':file_sha256(path)}
def first(run,name):
    return next(f for f in run.glob('cards/*.json') if read_json(f)['id']==name)
def metrics(p):
    return {k:{name:v for name,v in row['metrics'].items()
               if name in ('top1','top5','mIoU','mAcc','aAcc','baseline_top1','baseline_top5','baseline_mIoU','baseline_mAcc','baseline_aAcc')}
            for k,row in p['observations']['evaluations'].items()}
def main():
    run=HERE/'roles_direct'
    wf=Workflow(run,read_only=True)
    attempts=[]
    for path in sorted(run.glob('jobs/*/llm_audit.json'),key=lambda p:p.stat().st_mtime):
        a=read_json(path); request=read_json(path.parent/'request.json')
        files=[path.parent/'request.json',path,path.parent/'cli/instructions.txt',path.parent/'cli/answer.json',path.parent/'result.json']
        attempts.append({'task':request['task_id'],'role':a['role'],'requested':a.get('requested'),'actual':a.get('actual'),
            'model_call_started':a.get('model_call_started'),'error':a.get('error'),
            'files':[asset(f) for f in files if f.is_file()]})
    code_dir=HERE/'research_code'
    code_dir.mkdir(exist_ok=True)
    code_archive=[]
    for manifest in sorted((run/'code').glob('*/manifest.json')):
        m=read_json(manifest); source=Path(m['path'])
        assert file_sha256(source)==m['source_sha256']
        dest=code_dir/(m['source_sha256']+'.py')
        dest.write_bytes(source.read_bytes())
        code_archive.append({'id':m['id'],'source':asset(dest),'binding':asset(manifest),'request':m['experiment_ref']})
    num=[]
    for ev in wf.journal.find_cards('EvidenceCard'):
        p=ev['payload']; exp=wf.journal.card(p['experiment_ref'])
        num.append({'evidence':ev['id'],'experiment':exp['id'],'question':exp['payload']['question'],
            'source':p['measurement_source'],'code':p['provenance'].get('code_artifact'),
            'frozen_experiment':asset(first(run,exp['id'])),
            'execution_inputs':{k:v for k,v in exp['payload']['execution'].items() if k in
                ('parameters','calibration','resolved_choices','evaluations','seed','service','interface','source_ref','training')},

            'data':{s:{'count':r.get('sample_count'),'ids_sha256':fingerprint(r.get('sample_ids',[]))}
                    for s,r in p['scope'].get('data',{}).items()},
            'resources':p['resources'],'receipt':asset(first(run,ev['id']))})
    active=wf.inspirations.active(); recovery=wf.inspirations.recovery(active['id'])
    history=[{'code':row['candidate']['id'],'configuration':row['configuration'],'executions':row['execution_refs'],
              'evidence':row['evidence_refs'],'implementation':asset(Path(row['candidate']['path']))}
             for row in recovery['implementation_history']]
    components=[read_json(run/f'component-{i}.json') for i in (1,2)]
    assert all(c['topic'].split('@')[0]==components[0]['topic'].split('@')[0] for c in components)
    assert components[1]['restored_idea']==active['id'] and len(components[1]['evidence'])>len(components[0]['evidence'])
    assert not any(e['type']=='round_finished' for e in wf.journal.events)
    code_tasks=[t for t in wf.tasks().values() if t['inputs'].get('purpose')=='code' and t['key'].startswith('component/2/')]
    previous={row['code'] for row in history if set(row['evidence']).intersection(components[0]['evidence'])}
    delivered={r['candidate']['id'] for t in code_tasks for r in t['inputs']['context'].get('implementation_history',[])}
    assert previous & delivered
    lineage={'topic_versions':[c['topic'] for c in components],'inspiration':active['id'],'recovery':active['next_step'],
             'previous_exact_code_delivered':sorted(previous & delivered),'implementation_history':history}
    base=HERE/'baseline_partition'
    baseline=read_json(base/'baseline_receipt.json')
    ev=read_json(first(base,baseline['reference']['evidence_ref']))['payload']
    model=read_json(base/'model_metadata.json')
    handoff=read_json(HERE/'prepared_handoff/handoff_check.json')
    old=run/'research_run.json'; new=base/'research_run.json'
    interventions=[{'reason':e['reason'],'previous':e['previous_core_sha256'],'current':e['current_core_sha256']}
                   for e in wf.journal.events if e['type']=='implementation_intervention']
    write(HERE/'source_index.json',{'role_manifest':asset(old),'baseline_manifest':asset(new),'interventions':interventions,
        'llm_attempts':attempts,'coding_sources':code_archive,'experiments':num,'continuation':lineage,'baseline_evidence':asset(first(base,baseline['reference']['evidence_ref']))})
    pilots=[]
    for p in [HERE/'pilot_gptaq',HERE/'pilot_repq',HERE/'pilot_group32_r2',HERE/'role_source/EXP/pilot_partition']:
        d=read_json(p/'pilot.json')
        pilots.append({'seed':d['seed'],'wall_seconds':d['seconds'],'metrics':metrics(d),'receipt':asset(p/'pilot.json')})
    resources={}
    for name in ('dinov3b_preparation_20261118_r1','dinov3b_prepared_baseline_20261118_r1'):
        evaluation=ROOT/'EXP/evaluations'/name
        report=sorted(evaluation.glob('reports/report-*.json'))[-1]
        d=read_json(report)
        resources[name]={**d['research_resources'],'report':asset(report)}
    write(HERE/'RESULTS.json',{'formal_rounds_started':0,'model':model['summary'],'bound_assets':model['asset_binding'],
        'pilots':pilots,'baseline':baseline,'static_boundaries':len(ev['observations']['static_w4a4']['boundaries']),
        'actual_protocol_hash':handoff['protocol'],'historical_B':handoff['B'],'reconstruction':read_json(HERE/'independent_rebuild/rebuild_check.json'),
        'reference_reuse':read_json(HERE/'fp_reference_check/receipt.json'),'DeiT':asset(HERE/'deit/summary.json'),
        'components':components,'continuation':lineage,'resources_eval_L0':resources,'roles_index':asset(HERE/'source_index.json'),
        'launch_input':asset(ROOT/handoff['prepared_input'])})
if __name__=='__main__': main()
