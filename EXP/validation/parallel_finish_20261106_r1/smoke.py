"""Bounded real DeiT and real Coding/Safeguard smoke, never a research round."""
from pathlib import Path
from copy import deepcopy
import json, sys, time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json, read_json
from runtime.identity import code_identity
from tests.support import validation_config, experiment_payload
from tests.test_plan_execution_consistency import numeric_plan

out=Path(__file__).resolve().parent
name=sys.argv[1] if len(sys.argv)>1 else 'gpu_smoke_r1'
config=validation_config()
config['budget']={'wall_seconds':300,'max_rounds':1,'clock_mode':'active_segments','exclude_numeric_wait':True,
                  'min_free_bytes':1024**3}
config['runtime'].update(allowed_gpus=[0,1,2,3,4],experiment_concurrency=5,task_timeout_seconds=120,cpu_threads=2)
config['llm']['cli_path']='/root/.codex/packages/standalone/releases/0.166.0-x86_64-unknown-linux-musl/bin/codex'
b=config['research_brief']
b['scientific_question']='有界工程验证：局部激活QDQ及冻结包分类分片一致性；不做科研排名或正式L3。'
b['goals']={'validation':'bounded real GPU integration smoke only'}
b['model']={'name':'DeiT-B','scope':{'model':'deit_base_patch16_224'},'adapter':{
    'kind':'server','config_path':'QAViT/configs/deitb_backend.json','backend_overrides':{'device':'cuda:0'}}}
b['hard_constraints']={'training':False}
b['search_space']={}
b['evaluation_protocol']={'kind':'bounded_local_and_classification_subset','formal_incumbent_allowed':False}
b.pop('experiment_example',None)
wf=Workflow(out/name,config)
wf.budget.begin_segment()
started=time.time(); cap=started+900
report={'kind':'real_DeiT_ImageNet_subset_and_real_Coding_Safeguard','formal_L3':False,'stages':[],
        'source_identity':code_identity(),'config':config}
runtimes=[wf.runtime]

def enqueue(key,payload):
    card=wf.broker.freeze(payload,requester='user')
    wf.journal.commit(key+'/spec',cards=[card])
    return wf.queue.enqueue(key,'experiment',inputs={'experiment_ref':card['id'],'approval':None},
                            resource_keys=wf.experiment_resources(card))

def drive(ids,restore=False):
    global wf
    restored=False
    while not all(wf.queue.get(t)['state']=='completed' for t in ids):
        if time.time()>cap: raise TimeoutError('bounded smoke overall 900 seconds')
        wf.budget.check(allow_numeric_drain=wf.numeric_running())
        wf.poll(); wf.update_numeric_wait()
        for t in wf.queue.snapshot().values():
            if t['state'] in ('failed','cancelled','needs_clarification'):
                raise RuntimeError(str({'key':t['key'],'state':t['state'],'error':t.get('error'),'clarification':t.get('clarification')}))
        wf.dispatch(); wf.update_numeric_wait()
        if restore and not restored and sum(t['state']=='running' and t['kind']=='experiment' for t in wf.queue.snapshot().values())>=2:
            wf=Workflow(wf.run_dir);runtimes.append(wf.runtime);restored=True
        time.sleep(.1)
    wf.budget.numeric_wait(False)
    report['stages'].append({'tasks':ids,'live_restore':restored,'budget':wf.budget.input_receipt()})

def evidence(tid):return wf.journal.card(wf.queue.get(tid)['outputs'][0])

try:
    probes=[]
    for index,site in enumerate(['blocks.0.mlp.fc2','blocks.0.attn.proj']):
        p=experiment_payload('L2');p['scope']={'model':'deit_base_patch16_224'}
        p['execution'].update(site=site,calibration={'split':'calibration','count':2},evaluation={'split':'search','count':32},resolved_choices={'training':False})
        p['resources'].update(cpu_threads=2,estimated_seconds=360)
        probes.append(enqueue('probes/'+str(index),p))
    draft={'question':'实现本次局部静态4位对称QDQ：在blocks.0.attn.proj，2张calibration输入的abs最大值/7，clamp_min=1e-12；round ties-to-even并clamp[-8,7]。不训练，仅同一FP基准的输入和算子输出误差；不要额外实验。',
           'measurements':['input_relative_mse','operator_output_relative_mse'],
           'execution':{'service':'tensor_probe','requires_code':True,'site':'blocks.0.attn.proj',
                        'calibration':{'split':'calibration','count':2},'seed':42,
                        'method':'rtn','parameters':{'bits':4,'signed':True,'granularity':'per_tensor','axis':None,'group_size':None},
                        'resolved_choices':{'training':False}},
           'resources':{'estimated_seconds':360,'cpu_threads':2}}
    coded=wf.queue.enqueue('coded_probe','experiment_request',role='supervisor',inputs={
        'draft':draft,'original_request':deepcopy(draft),'requester':'user','refs':[wf.brief['id']]})
    drive(probes+[coded],restore=True)
    assert wf.queue.get(coded)['result']['status']=='completed',wf.queue.get(coded)['result']
    assert evidence(coded)['payload']['measurement_source']=='real'
    report['code_evidence']=evidence(coded)['id']
    report['code_request_result']=wf.queue.get(coded)['result']
    # Calibrate once and evaluate the exported/reloaded frozen model on 17 images.
    p=experiment_payload('L2');p['scope']={'model':'deit_base_patch16_224'}
    p['execution']={'service':'quantize_evaluate','requires_code':False,'plan':numeric_plan(3,granularity='per_tensor'),
        'resolved_choices':{'training':False},'seed':42,'calibration':{'split':'calibration','count':2},
        'evaluations':[{'kind':'classification','split':'search','count':17}]}
    p['resources'].update(cpu_threads=2,estimated_seconds=360)
    single=enqueue('single_frozen',p);drive([single])
    ev=evidence(single)
    model_asset=next(a for a in ev['payload']['artifacts'] if Path(a['path']).name=='weights.pt')
    model_dir=str(Path(model_asset['path']).parent)
    p['execution']={'service':'evaluate_artifact','requires_code':False,'model_dir':model_dir,
                   'resolved_choices':{'training':False},'seed':42,
                   'evaluations':[{'kind':'classification','split':'search','count':17}]}
    p['resources']['gpu_count']=5
    multi=enqueue('five_gpu_same_frozen',p);drive([multi])
    def raw(tid):
        ev=evidence(tid)
        return read_json(next(a['path'] for a in ev['payload']['artifacts'] if Path(a['path']).name=='classification.json'))
    one,many=raw(single),raw(multi)
    assert one['sample_ids']==many['sample_ids']
    assert one['classification_counts']==many['classification_counts']
    errors=[]
    for key,sums in one['metrics']['raw_error_sums'].items():
        for field,value in sums.items():
            actual=many['metrics']['raw_error_sums'][key][field]
            relative=abs(actual-value)/max(1.,abs(value));errors.append(relative)
            assert relative < 1e-12,(key,field,value,actual)
    report['frozen_comparison']={'single_task':single,'multi_task':multi,'model_dir':model_dir,
        'sample_count':len(one['sample_ids']),'classification_counts':many['classification_counts'],
        'maximum_accumulator_relative_difference':max(errors),'parallelism':many['parallelism']}
    assert len(many['parallelism']['physical_gpus'])==5
    # A stopped segment resumes with the same charged time, no reset or restart.
    before=wf.budget.snapshot()['elapsed_seconds']
    for runtime in runtimes:runtime.stop()
    wf.budget.end_segment(processes_stopped=True,reason='bounded smoke pause')
    paused=wf.budget.snapshot()['elapsed_seconds'];time.sleep(.3)
    wf=Workflow(wf.run_dir);runtimes.append(wf.runtime)
    assert abs(wf.budget.snapshot()['elapsed_seconds']-paused)<1e-6
    wf.budget.begin_segment();wf.budget.end_segment(processes_stopped=True,reason='resume check only')
    report['pause_resume']={'before_stop':before,'paused_consumption':paused,'after_resume':wf.budget.snapshot()['elapsed_seconds']}
    report['status']='passed'
except BaseException as exc:
    report.update(status='failed',error={'code':type(exc).__name__,'message':str(exc)})
    raise
finally:
    for runtime in runtimes:runtime.stop()
    wf.poll();wf.budget.numeric_wait(False)
    wf.budget.end_segment(processes_stopped=True,reason='bounded smoke ended')
    report['wall_seconds']=time.time()-started
    report['budget']=wf.budget.snapshot()
    report['tasks']=[{k:t.get(k) for k in ('key','task_id','kind','role','state','attempt','request_path','outputs','allocated_gpus')} for t in wf.queue.snapshot().values()]
    intervals=[]
    tasks={t['task_id']:t for t in wf.queue.snapshot().values()}
    for path in (wf.run_dir/'jobs').glob('*/result.json'):
        receipt=read_json(path);t=tasks[receipt['task_id']];directory=path.parent
        marker=read_json(directory/'started.json')
        intervals.append({'kind':t['kind'],'role':t['role'],'key':t['key'],'start':marker['started_unix'],
                          'end':receipt['finished_unix'],'gpus':receipt['resources'].get('physical_gpus')})
    report['intervals']=intervals
    numerics=[v for v in intervals if v['key'].startswith('probes/')]
    coding=[v for v in intervals if v['role']=='coding']
    report['independent_numeric_overlap_seconds']=max(0,min(v['end'] for v in numerics)-max(v['start'] for v in numerics)) if len(numerics)==2 else None
    report['coding_numeric_overlap_seconds']=max([max(0,min(a['end'],b['end'])-max(a['start'],b['start'])) for a in numerics for b in coding] or [0])
    report['no_owned_processes']=not wf.runtime.live_processes()
    report['source_unchanged']=report['source_identity']==code_identity()
    atomic_json(out/(name+'_RESULT.json'),report)
    print(json.dumps({k:report.get(k) for k in ('status','error','wall_seconds','frozen_comparison','independent_numeric_overlap_seconds','coding_numeric_overlap_seconds','no_owned_processes','source_unchanged')},ensure_ascii=False),flush=True)
