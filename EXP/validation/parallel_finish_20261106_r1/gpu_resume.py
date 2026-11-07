"""Interrupt real two-GPU evaluation with live shards, then resume the same task."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json,read_json
from runtime.identity import code_identity
from runtime.gpu_resources import acquire
import os
base=Path(__file__).resolve().parent
source=base/'gpu_smoke_r3'
config=read_json(source/'research_run.json')['config'];config['budget'].update(wall_seconds=30)
model_dir=next(p.parent for p in (source/'jobs').glob('*/model/weights.pt'))
wf=Workflow(base/'gpu_resume',config);wf.budget.begin_segment()
report={'source_identity':code_identity(),'kind':'real_GPU_interruption_recovery_only','formal_L3':False}
started=time.time();runtimes=[wf.runtime]
payload={'level':'L2','question':'中断释放与恢复同一冻结包评测','measurements':['top1','relative_mse'],
'execution':{'service':'evaluate_artifact','requires_code':False,'model_dir':str(model_dir),'resolved_choices':{'training':False},'seed':42,
'evaluations':[{'kind':'classification','split':'search','count':17}]},'resources':{'cpu_threads':2,'estimated_seconds':240,'gpu_count':2}}
try:
 card=wf.broker.freeze(payload,requester='user');wf.journal.commit('spec',cards=[card])
 tid=wf.queue.enqueue('eval','experiment',inputs={'experiment_ref':card['id']})
 wf.dispatch();wf.update_numeric_wait()
 task=wf.queue.get(tid);directory=Path(task['request_path']).parent
 while not list((directory/'processes').glob('shard-*.json')):
  if time.time()-started>180:raise TimeoutError('no live shard before smoke bound')
  wf.poll();wf.update_numeric_wait()
  if wf.queue.get(tid)['state']!='running':raise RuntimeError('worker ended before interruption')
  time.sleep(.1)
 report['live_before_stop']=wf.runtime.live_processes()
 assert len(report['live_before_stop'])>=2
 wf.runtime.stop();wf.poll();wf.budget.numeric_wait(False)
 interrupted=wf.queue.get(tid);assert interrupted['state']=='retry',interrupted
 wf.budget.end_segment(processes_stopped=True,reason='manual validation interruption')
 report['first_attempt']=read_json(directory/'result.json')
 assert report['first_attempt']['status']=='interrupted'
 assert not wf.runtime.live_processes()
 # Prove the kernel leases themselves released, not just the queue state.
 selected,fds=acquire([0,1],2)
 for fd in fds:os.close(fd)
 assert selected==[0,1]
 paused=wf.budget.snapshot()['elapsed_seconds'];time.sleep(.3)
 wf=Workflow(wf.run_dir);runtimes.append(wf.runtime)
 assert wf.budget.snapshot()['elapsed_seconds']==paused
 wf.budget.begin_segment()
 while wf.queue.get(tid)['state']!='completed':
  if time.time()-started>420:raise TimeoutError('bounded resume')
  wf.budget.check(allow_numeric_drain=wf.numeric_running());wf.poll();wf.update_numeric_wait()
  t=wf.queue.get(tid)
  if t['state']=='failed':raise RuntimeError(str(t.get('error')))
  wf.dispatch();wf.update_numeric_wait();time.sleep(.1)
 final=wf.queue.get(tid);assert final['attempt']==2
 ev=wf.journal.card(final['outputs'][0]);assert ev['payload']['experiment_ref']==card['id']
 cls=read_json(next(a['path'] for a in ev['payload']['artifacts'] if Path(a['path']).name=='classification.json'))
 assert cls['sample_count']==17 and cls['classification_counts']['top1']==16
 report.update(status='passed',task_id=tid,attempts=final['attempt'],retry_count=final['retry_count'],evidence_ref=ev['id'],
               experiment_ref=card['id'],paused_consumption=paused,classification_counts=cls['classification_counts'])
except BaseException as exc:
 report.update(status='failed',error={'code':type(exc).__name__,'message':str(exc)});raise
finally:
 for runtime in runtimes:runtime.stop()
 wf.poll();wf.budget.numeric_wait(False);wf.budget.end_segment(processes_stopped=True,reason='validation complete')
 report.update(wall_seconds=time.time()-started,budget=wf.budget.snapshot(),no_owned_processes=not wf.runtime.live_processes(),
               source_unchanged=report['source_identity']==code_identity())
 atomic_json(base/'GPU_RESUME_RESULT.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','error','wall_seconds','attempts','retry_count','paused_consumption','no_owned_processes','source_unchanged')},ensure_ascii=False),flush=True)
