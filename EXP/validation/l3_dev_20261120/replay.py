"""Paired transport replay at the historical decision point, without publishing a review."""
from pathlib import Path
import time, sys
from runtime.paths import ROOT
from core.journal import read_json,atomic_json
from agents.roles import role_spec
from runtime.model_router import call_once
out = Path(__file__).resolve().parent
jobs = next((ROOT/'EXP/runs').glob('dino*'))/'jobs'
for prefix in sys.argv[1:]:
    old = sorted(jobs.glob(prefix+'*/request.json'))[-1]
    spec = read_json(old)['spec']
    audit = read_json(old.parent/'llm_audit.json')
    dest = out/'review_replay'/old.parent.name
    if (dest/'comparison.json').exists():
        continue
    from agents.source_workspace import engineering_context
    spec = {**spec,'context':engineering_context(spec['context'])}
    modern = role_spec(spec['role'],spec['purpose'],spec['context'],spec['settings'],messages=spec['messages'])
    try:
        result = call_once(modern,dest,time.time()+150)
        output = result['response']
    except Exception as exc:
        output = {'error':str(exc)}
    new = read_json(dest/'llm_audit.json')
    atomic_json(dest/'comparison.json',dict(old=str(old),role=spec['role'],settings=spec['settings']['agent_models'][spec['role']],
        original_usage=audit['usage'],original_elapsed=audit['elapsed_seconds'],
        current_usage=new['usage'],current_elapsed=new['elapsed_seconds'],response=output,
        original_answer=read_json(old.parent/'cli/answer.json'),
        scope='one decision point, historical preceding tool calls held fixed; no new experiment evidence'))
