"""Build the production focus input on the real seq-14 prefix. All writes are isolated."""
from pathlib import Path
import hashlib
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from tools.evaluation.judge import evaluation_manifest
from tools.evaluation.common import confined, digest
from replay import EVALUATIONS, COLLECTION, get_request, act

def main():
    req=get_request('initial')
    source=EVALUATIONS/COLLECTION['evaluation']
    import sys
    dest=EVALUATIONS/('frozen_trials_20261124_focus_host'+('' if len(sys.argv)==1 else '_'+sys.argv[1]))
    dest.mkdir(exist_ok=False)
    branch=dest/'branch'
    objects={r['path']:r for r in evaluation_manifest(source)['source_files']}
    def copy(name):
        row=objects[name]
        raw=confined(source/row['object'],source/'sources').read_bytes()
        assert digest(raw)==row['sha256']
        f=branch/name
        f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes(raw)
    copy('research_run.json')
    for index in range(1,1+req['spec']['context']['read_view']['sequence']):
        name='commits/'+str(index).zfill(9)+'.json'
        copy(name)
        for card in read_json(branch/name)['cards']:
            copy('cards/'+hashlib.sha256(card['ref'].encode()).hexdigest()+'.json')
    wf=Workflow(branch,read_only=True)
    task=wf.tasks()['r001/focus']
    trace=[]
    for i in range(3):
        body=wf.request_body({**task,'request_path':None})
        spec=body['spec']
        # Wall time spent since the archived run is unrelated to this decision.
        for field in ('budget_receipt','available_seconds'):
            spec['context'][field]=req['spec']['context'][field]
        spec['settings']=req['spec']['settings']
        job=dest/str(i+1)
        atomic_json(job/'request.json',body)
        if task['state']!='running':
            wf.queue.ready()
            task=wf.queue.claim(task['task_id'],request_path=str(job/'request.json'))
        task={**task,'request_path':str(job/'request.json')}
        response,row=act(spec,job)
        trace.append(row);atomic_json(dest/'trace.json',trace)
        if not response['tool_calls']:
            if response['decision']['action']=='submit':
                wf.submit(task,response['decision']['content'],str(job/'response.json'))
                atomic_json(dest/'final_topic.json',wf.topics.active())
            return
        wf.receive_tools(task,response,str(job/'response.json'))
        wf.queue.ready()
        for tid in wf.queue.get(task['task_id'])['waiting_for']:
            child=wf.queue.claim(tid)
            if child['kind']!='host_tool':
                atomic_json(dest/'unmeasured.json',{'kind':child['kind'],'task':tid,'budget':'bounded focus replay'})
                return
            wf.queue.finish(tid,result=wf.host_tool(child))
        wf.queue.ready()  # Wake appends the completed tool result before composing the next call.
        task=wf.queue.get(task['task_id'])
if __name__=='__main__':
    main()
