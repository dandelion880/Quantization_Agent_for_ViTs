"""Replay one historical implementation request without advancing a research round."""
import os,time,sys
from pathlib import Path
from runtime.paths import ROOT
from core.workflow import Workflow
from core.journal import read_json,atomic_json
from runtime.task_queue import process_identity
out=Path(__file__).resolve().parent
old=next((ROOT/'EXP/validation').glob('dinov3b_preparation*/roles_direct/jobs/task-0a3*/request.json'))
prior=read_json(old)['experiment']['payload']
cfg=read_json(old.parents[2]/'research_run.json')['config']
rerun = '--matched-timeout' in sys.argv
label = 'engineering_matched' if rerun else 'engineering_trial'
cfg['name']=label
cfg['llm'].update(subscription_max_used_percent=96,coding_native_tools=True)
cfg['budget'].update(max_rounds=1,wall_seconds=1000 if rerun else 600)
cfg['runtime'].update(allowed_gpus=[0],task_timeout_seconds=cfg['runtime']['task_timeout_seconds'] if rerun else 180)
cfg['research_brief']['model'].pop('binding',None)
draft=prior['original_request'].copy()
# Original-run topic and inspiration handles are administrative bindings, not scientific inputs.
for key in ('topic_ref','inspiration_id'):
    draft.pop(key,None)
wf=Workflow(out/label,cfg)
wf.budget.begin_segment(owner=process_identity(os.getpid()))
tid=wf.queue.enqueue('component/replay','experiment_request',role='supervisor',inputs={
    'draft':draft,'original_request':draft,'requester':'generation','refs':[wf.brief['id']]})
try:
    while True:
        wf.budget.check(allow_numeric_drain=wf.numeric_running())
        wf.poll()
        t=wf.queue.get(tid)
        if t['state'] in ('completed','failed','cancelled','needs_clarification') or wf.transport_blocked:
            break
        wf.dispatch()
        wf.update_numeric_wait()
        time.sleep(.5)
finally:
    wf.runtime.stop()
    wf.budget.end_segment(processes_stopped=True,reason='bounded engineering replay')
    atomic_json(out/(label+'_result.json'),dict(task=wf.queue.get(tid),budget=wf.budget.input_receipt(),
        evidence=[c['id'] for c in wf.journal.find_cards('EvidenceCard')],historical_request=str(old)))
