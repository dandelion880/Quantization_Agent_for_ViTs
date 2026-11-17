"""Repeat only the same frozen subset after CUDA discovery and teardown settle."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import read_json,atomic_json
from runtime.identity import code_identity
from runtime.gpu_resources import gpu_bindings,externally_busy
OUT=Path(__file__).resolve().parent
previous=read_json(OUT/'gpu_candidates_r2_report.json')
config=read_json(OUT/'gpu_candidates_r2/research_run.json')['config']
config['runtime'].update(allowed_gpus=[2,3],experiment_concurrency=1)
config['runtime'].pop('gpu_bindings',None)
config['runtime'].pop('gpu_selection',None)
name=sys.argv[1] if len(sys.argv)>1 else 'gpu_same_frozen_parallel_r1'
wf=Workflow(OUT/name,config)
wf.budget.begin_segment()
report={'scope':'same H1 frozen package, classification3/ADE1 engineering subset only',
        'source_identity':code_identity(),'full_research_loop_started':False}
started=time.time()
try:
    # CUDA enumeration itself can briefly register as external GPU activity.
    # Wait for both authorized cards only for this explicit two-card comparison;
    # production scheduling intentionally permits immediate one-card fallback.
    wf.runtime._gpu_bindings=gpu_bindings(wf.config['runtime']['allowed_gpus'],wf.config['runtime'])
    identities={r['uuid'] for r in wf.runtime._gpu_bindings}
    while externally_busy() & identities:
        if time.time()-started>60: raise TimeoutError('two-card validation capacity unavailable')
        time.sleep(1.1)
    time.sleep(1.1)
    p={'level':'L2','question':'Same frozen H1 package serial versus two-card dual-task subset',
       'measurements':['top1','mIoU'],
       'execution':{'service':'evaluate_artifact','requires_code':False,
          'model_dir':previous['tasks']['H1']['model_dir'],'seed':42,'resolved_choices':{'training':False},
          'evaluations':[{'kind':'classification','split':'search','count':3},{'kind':'ade_frozen','count':1}]},
       'resources':{'cpu_threads':2,'estimated_seconds':300,'gpu_count':2},
       'replication':{'requested':True,'purpose':'bounded engineering validation','configuration_difference':{'parallel':True}}}
    card=wf.broker.freeze(p,requester='user');wf.journal.commit('spec',cards=[card])
    tid=wf.queue.enqueue('same_frozen','experiment',inputs={'experiment_ref':card['id']})
    while wf.queue.get(tid)['state']!='completed':
        if time.time()-started>420:raise TimeoutError('bounded comparison cap')
        wf.poll();wf.update_numeric_wait()
        task=wf.queue.get(tid)
        if task['state'] in ('failed','retry','cancelled'):raise RuntimeError(str(task.get('error')))
        wf.dispatch();wf.update_numeric_wait();time.sleep(.1)
    wf.budget.numeric_wait(False)
    task=wf.queue.get(tid);directory=Path(task['request_path']).parent
    ev=wf.journal.card(task['outputs'][0]);wf.broker.validate_evidence_for(card,ev)
    request=read_json(task['request_path']);assert len(request['gpus'])==2,request['parallelism']
    a=previous['tasks']['H1'];classification=read_json(directory/'classification.json');segmentation=read_json(directory/'segmentation.json')
    assert a['classification']['sample_ids']==classification['sample_ids']
    assert a['classification']['classification_counts']==classification['classification_counts']
    assert a['classification']['metrics']['raw_error_sums']==classification['metrics']['raw_error_sums']
    assert a['segmentation']['sample_ids']==segmentation['sample_ids']
    assert a['segmentation']['metrics']==segmentation['metrics']
    dual=read_json(directory/'dual_task_receipt.json');assert dual['overlap_seconds']>0
    report.update(status='completed',task_id=tid,evidence_ref=ev['id'],allocated_gpus=request['gpus'],
                  same_frozen_metrics_equal=True,dual_task_receipt=dual)
    wf.runtime.stop();wf.budget.end_segment(processes_stopped=True,reason='bounded comparison pause')
    elapsed=wf.budget.snapshot()['elapsed_seconds'];wf=Workflow(wf.run_dir)
    assert wf.budget.snapshot()['elapsed_seconds']==elapsed
    wf.budget.begin_segment();wf.budget.end_segment(processes_stopped=True,reason='resume checked; no new experiment')
    report.update(pause_resume_verified=True,budget=wf.budget.snapshot())
except BaseException as exc:
    report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    raise
finally:
    wf.runtime.stop()
    if wf.budget.snapshot().get('active_segment'):wf.budget.end_segment(processes_stopped=True,reason='cleanup')
    report['wall_seconds']=time.time()-started
    atomic_json(OUT/(name+'_report.json'),report)
    print(json.dumps({'status':report['status']}))
