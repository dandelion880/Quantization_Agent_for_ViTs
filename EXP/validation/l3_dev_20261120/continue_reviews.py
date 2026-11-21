"""Complete bounded review replays through the historical read ACL and view."""
from pathlib import Path
import time
from runtime.paths import ROOT
from core.workflow import Workflow
from agents.roles import role_spec
from core.journal import read_json, atomic_json
from runtime.model_router import call_once
from core.cards import ContractError
out=Path(__file__).resolve().parent
wf=Workflow(next((ROOT/'EXP/runs').glob('dino*')),read_only=True)
for path in sorted(out.glob('review_replay/*/comparison.json')):
    record=read_json(path)
    request=read_json(record['old'])
    base=request['spec']
    from agents.source_workspace import engineering_context
    base={**base,'context':engineering_context(base['context'])}
    spec=role_spec(base['role'],base['purpose'],base['context'],base['settings'],messages=base['messages'])
    response=record['response']
    for turn in range(1,5):
        if not response.get('tool_calls'):
            break
        for call in response['tool_calls']:
            args=call['arguments']
            assert call['name'] in ('read_card','read_asset')
            inputs=dict(read_view={**base['context']['read_view'],'assets':request['read_permissions']},
                        parent_task=request['task_id'],name=call['name'],arguments=args)
            try:
                result=wf.host_tool({'inputs':inputs})
            except (KeyError,ContractError, OSError) as e:
                result={'error':str(e)}
            spec['context']['tool_results'].append({'name':call['name'],'arguments':args,'result':result})
        spec['messages'].append({'role':'assistant','content':{'tool_calls':[
            {'id':c['id'],'name':c['name']} for c in response['tool_calls']]}})
        dest=path.parent/('followup'+str(turn))
        if (dest/'response.json').exists():
            response=read_json(dest/'response.json')
            continue
        response=call_once(spec,dest,time.time()+150)['response']
        atomic_json(dest/'response.json',response)
    atomic_json(path.parent/'terminal.json',response)
