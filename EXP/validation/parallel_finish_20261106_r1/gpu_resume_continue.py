"""Continue the already interrupted task; NVML utilization decay is resource wait."""
from pathlib import Path
import json,sys,time,os
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json,read_json
from runtime.identity import code_identity
from runtime.gpu_resources import acquire,ResourceBusy
base=Path(__file__).resolve().parent
wf=Workflow(base/'gpu_resume');tid=wf.tasks()['eval']['task_id']
prior=read_json(base/'GPU_RESUME_RESULT.json')
report={'source_identity':code_identity(),'kind':'continue_same_interrupted_real_GPU_task','formal_L3':False,
        'previous_probe_report':'GPU_RESUME_RESULT.json','first_attempt':prior['first_attempt'],
        'live_before_stop':prior['live_before_stop']}
started=time.time();paused=wf.budget.snapshot()['elapsed_seconds']
try:
 assert wf.queue.get(tid)['state']=='retry' and wf.queue.get(tid)['attempt']==1
 assert wf.budget.snapshot()['active_segment'] is None
 while True:
  if time.time()-started>15:raise TimeoutError('GPU release wait exceeded smoke bound')
  try:
   selected,fds=acquire([0,1],2)
  except ResourceBusy:
   time.sleep(.2);continue
  for fd in fds:os.close(fd)
  if selected==[0,1]:break
  time.sleep(.2)
 assert not wf.runtime.live_processes()
 assert wf.budget.snapshot()['elapsed_seconds']==paused
 wf.budget.begin_segment()
 while wf.queue.get(tid)['state']!='completed':
  if time.time()-started>300:raise TimeoutError('bounded same-task resume')
  wf.budget.check(allow_numeric_drain=wf.numeric_running());wf.poll();wf.update_numeric_wait()
  if wf.queue.get(tid)['state']=='failed':raise RuntimeError(str(wf.queue.get(tid).get('error')))
  wf.dispatch();wf.update_numeric_wait();time.sleep(.1)
 final=wf.queue.get(tid);assert final['attempt']==2
 ev=wf.journal.card(final['outputs'][0]);assert ev['payload']['experiment_ref']==final['inputs']['experiment_ref']
 cls=read_json(next(a['path'] for a in ev['payload']['artifacts'] if Path(a['path']).name=='classification.json'))
 assert cls['sample_count']==17 and cls['classification_counts']['top1']==16
 report.update(status='passed',task_id=tid,attempts=final['attempt'],retry_count=final['retry_count'],
   evidence_ref=ev['id'],experiment_ref=ev['payload']['experiment_ref'],paused_consumption=paused,
   classification_counts=cls['classification_counts'],physical_gpus=ev['payload']['resources']['physical_gpus'])
except BaseException as exc:
 report.update(status='failed',error={'code':type(exc).__name__,'message':str(exc)});raise
finally:
 wf.runtime.stop();wf.poll();wf.budget.numeric_wait(False);wf.budget.end_segment(processes_stopped=True,reason='same-task resume complete')
 report.update(wall_seconds=time.time()-started,budget=wf.budget.snapshot(),no_owned_processes=not wf.runtime.live_processes(),
               source_unchanged=report['source_identity']==code_identity())
 atomic_json(base/'GPU_RESUME_COMPLETED.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','error','wall_seconds','attempts','retry_count','paused_consumption','no_owned_processes','source_unchanged')},ensure_ascii=False),flush=True)
