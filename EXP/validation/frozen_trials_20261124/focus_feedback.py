"""Fulfil the unsolicited focus experiment, seal its Evidence and wake the original focus task."""
import os
import subprocess
import sys
import time
from core.workflow import Workflow
from core.journal import atomic_json, read_json
from tools.evaluation.common import EVALUATIONS
from runtime.identity import code_identity
from runtime.paths import QAVIT
from replay import act, get_request
def main():
    dest=EVALUATIONS/'frozen_trials_20261124_focus_host_r3'
    wf=Workflow(dest/'branch',read_only=True)
    original=get_request('initial')
    # Restore the historical budget clock at the fork; all actual grants and evaluation paths are production.
    t=time.monotonic()
    prior=original['spec']['context']['budget_receipt']['as_of_unix']
    wf.budget.clock=lambda:prior+time.monotonic()-t
    tid=read_json(dest/'unmeasured.json')['task']
    child=wf.queue.get(tid)
    job=dest/'numeric'
    if not (job/'evidence.json').is_file():
        if not wf.queue.get_by_key(child['key']+'/execute'):
            wf.progress_request(child)
        sub=wf.queue.get_by_key(child['key']+'/execute')
        wf.queue.ready()
        if sub['state']!='running': sub=wf.queue.claim(sub['task_id'])
        card=wf.journal.card(sub['inputs']['experiment_ref'])
        e=card['payload']['execution']
        assert e['service']=='frozen_helper' and e['tool']=='profile_frozen_propagation'
        assert e['evaluation']=={'split':'search','count':16,'offset':0} and len(e['arguments']['interventions'])==5
        uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        req={'kind':'experiment','experiment':card,'approval':None,'task_id':sub['task_id'],'execution_id':'focus-measure',
             'core_sha256':code_identity()['core_sha256'],'resource_keys':['gpu:'+uuid],'gpu':2,'gpus':[2],
             'gpu_bindings':[{'index':2,'uuid':uuid,'device_node':'/dev/nvidia2'}],'deadline_unix':time.time()+480}
        atomic_json(job/'request.json',req)
        with (job/'worker.log').open('w') as f:
            subprocess.run([sys.executable,'-B',str(QAVIT/'runtime/task_worker.py'),str(job/'request.json')],stdout=f,stderr=f,check=True)
        receipt=read_json(job/'result.json')
        if receipt['status']!='completed':raise RuntimeError(receipt['error'])
        ev=wf.broker.evidence(card,receipt,job/'result.json')
        wf.broker.validate_evidence_for(card,ev)
        wf.queue.finish(sub['task_id'],outputs=[ev['id']],cards=[ev])
        atomic_json(job/'evidence.json',{'ref':ev['id'],'experiment_ref':card['id'],'wall_seconds':receipt['resources']['wall_seconds']})
    child=wf.queue.get(tid)
    if child['state']!='completed':
        wf.queue.ready();wf.progress_request(wf.queue.claim(tid))
    wf.queue.ready()
    focus=wf.tasks()['r001/focus']
    first=max(int(p.name) for p in dest.iterdir() if p.name.isdigit())+1
    for step in range(first,first+2):
        body=wf.request_body({**focus,'request_path':None});body['spec']['settings']=original['spec']['settings']
        at=dest/str(step);atomic_json(at/'request.json',body)
        focus=wf.queue.claim(focus['task_id'],request_path=str(at/'request.json'))
        response,row=act(body['spec'],at)
        trace=read_json(dest/'trace.json'); trace.append(row);atomic_json(dest/'trace.json',trace)
        if not response['tool_calls']:
            if response['decision']['action']=='submit':
                wf.submit(focus,response['decision']['content'],str(at/'response.json'))
                atomic_json(dest/'final_topic.json',wf.topics.active())
            return
        wf.receive_tools(focus,response,str(at/'response.json'));wf.queue.ready()
        for t_id in wf.queue.get(focus['task_id'])['waiting_for']:
            sub=wf.queue.claim(t_id)
            if sub['kind']!='host_tool':
                atomic_json(dest/'next_unmeasured.json',{'kind':sub['kind'],'task':t_id})
                return
            wf.queue.finish(t_id,result=wf.host_tool(sub))
        wf.queue.ready();focus=wf.queue.get(focus['task_id'])
if __name__=='__main__':main()
