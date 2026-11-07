"""Single and five GPU evaluation of the exact retained real frozen package."""
from pathlib import Path
from copy import deepcopy
import sys,time,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json,read_json
from runtime.identity import code_identity
out=Path(__file__).resolve().parent
source=out/'gpu_smoke_r3'
config=read_json(source/'research_run.json')['config']
config['budget'].update(wall_seconds=20)
model_dir=next(p.parent for p in (source/'jobs').glob('*/model/weights.pt'))
name=sys.argv[1] if len(sys.argv)>1 else 'frozen_compare_r1'
wf=Workflow(out/name,config);wf.budget.begin_segment()
report={'source_identity':code_identity(),'model_dir':str(model_dir),'real_model':'DeiT-B pretrained',
        'scope':'17 ImageNet search samples, static W8A8 on blocks.0.mlp.fc1/fc2 only; not L3',
        'full_research_loop_started':False,'attempts':[]}
started=time.time();runtimes=[wf.runtime]
try:
    for count in (1,5):
        payload={'level':'L2','question':'单卡与多卡对同一冻结模型和样本的数值一致性','measurements':['top1','relative_mse'],
          'execution':{'service':'evaluate_artifact','requires_code':False,'model_dir':str(model_dir),
                       'resolved_choices':{'training':False},'seed':42,
                       'evaluations':[{'kind':'classification','split':'search','count':17}]},
          'resources':{'cpu_threads':2,'estimated_seconds':240,'gpu_count':count},
          'replication':{'requested':True,'purpose':'independent single/multi validation','configuration_difference':{'gpu_count':count}}}
        card=wf.broker.freeze(payload,requester='user');wf.journal.commit('spec/'+str(count),cards=[card])
        tid=wf.queue.enqueue('eval/'+str(count),'experiment',inputs={'experiment_ref':card['id']})
        while wf.queue.get(tid)['state']!='completed':
            if time.time()-started>480:raise TimeoutError('bounded frozen comparison')
            wf.budget.check(allow_numeric_drain=wf.numeric_running());wf.poll();wf.update_numeric_wait()
            t=wf.queue.get(tid)
            if t['state'] in ('failed','cancelled'):raise RuntimeError(str(t.get('error')))
            wf.dispatch();wf.update_numeric_wait();time.sleep(.1)
        wf.budget.numeric_wait(False)
        ev=wf.journal.card(wf.queue.get(tid)['outputs'][0])
        path=next(a['path'] for a in ev['payload']['artifacts'] if Path(a['path']).name=='classification.json')
        report['attempts'].append({'task':tid,'evidence':ev['id'],'classification':read_json(path),'resources':ev['payload']['resources']})
    one,many=[a['classification'] for a in report['attempts']]
    assert one['sample_ids']==many['sample_ids'];assert one['classification_counts']==many['classification_counts']
    differences=[]
    for key,values in one['metrics']['raw_error_sums'].items():
        for field,value in values.items():
            other=many['metrics']['raw_error_sums'][key][field]
            relative=abs(other-value)/max(1,abs(value));differences.append(relative)
            assert relative<1e-12,(key,field,value,other)
    assert many['parallelism']['shards']==5
    report['max_relative_accumulator_difference']=max(differences)
    wf.runtime.stop();wf.budget.end_segment(processes_stopped=True,reason='pause check')
    paused=wf.budget.snapshot()['elapsed_seconds'];time.sleep(.4)
    wf=Workflow(wf.run_dir);runtimes.append(wf.runtime)
    assert wf.budget.snapshot()['elapsed_seconds']==paused
    wf.budget.begin_segment();wf.budget.end_segment(processes_stopped=True,reason='resume without resetting')
    report['paused_consumption']=paused;report['status']='passed'
except BaseException as exc:
    report.update(status='failed',error={'code':type(exc).__name__,'message':str(exc)});raise
finally:
    for r in runtimes:r.stop()
    wf.poll();wf.budget.numeric_wait(False);wf.budget.end_segment(processes_stopped=True,reason='bounded check complete')
    report.update(budget=wf.budget.snapshot(),wall_seconds=time.time()-started,
                  no_owned_processes=not wf.runtime.live_processes(),source_unchanged=report['source_identity']==code_identity())
    atomic_json(out/(name+'_RESULT.json'),report)
    print(json.dumps({k:report.get(k) for k in ('status','error','wall_seconds','max_relative_accumulator_difference','paused_consumption','no_owned_processes','source_unchanged')},ensure_ascii=False),flush=True)
