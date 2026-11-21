"""Count every matched attempt; a missing usage receipt is not zero tokens."""
from pathlib import Path
from core.journal import read_json,atomic_json
out=Path(__file__).resolve().parent


def summarize(audits):
    rows=[read_json(p) for p in audits]
    return dict(attempts=len(rows),missing_usage=sum(not isinstance(r.get('usage'),dict) for r in rows),
                **{k:sum((r.get('usage') or {}).get(k,0) for r in rows) for k in
                   ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},
                model_call_seconds=sum(r.get('elapsed_seconds',0) for r in rows))


def chain(jobs, numeric):
    successful=read_json(numeric/'request.json')['experiment']
    target={successful['id'],successful['payload']['request_ref']}
    audits=[]
    for p in jobs.glob('*/llm_audit.json'):
        audit=read_json(p)
        c=audit['request']['context']
        ids={c.get(k,{}).get('id') for k in ('original_request','frozen_experiment') if isinstance(c.get(k),dict)}
        ids.update(c.get('task',{}).get(k) for k in ('request_ref','experiment_ref'))
        if audit['role'] in ('coding','safeguard') and ids & target:
            audits.append(p)
    numeric_tasks=[]
    for f in jobs.glob('*/request.json'):
        req=read_json(f)
        exp=req.get('experiment',{})
        if exp and (exp['id'] in target or exp['payload'].get('request_ref') in target):
            numeric_tasks.append(f.parent)
    return {**summarize(audits),'numeric':[
        {'task':p.name,'status':r['status'],'worker_wall_seconds':r['resources']['wall_seconds'],
         'gpus':r['resources'].get('physical_gpus',r['resources'].get('physical_gpu'))}
        for p in numeric_tasks if (p/'result.json').exists() for r in [read_json(p/'result.json')]]}


comparisons={}
for report in out.glob('review_replay/*/comparison.json'):
    old=Path(read_json(report)['old']).parent
    terminal=read_json(report.parent/'terminal.json')
    comparisons[old.name]={'historical_final_call':summarize([old/'llm_audit.json']),
        'modern_from_same_decision_point':summarize(report.parent.rglob('llm_audit.json')),
        'historical_prefix_common_to_both':summarize(p for p in old.parent.glob(old.name[:-5]+'*/llm_audit.json') if p.parent!=old),
        'historical_decision':read_json(old/'cli/answer.json'),'decision':terminal.get('decision'),
        'tool_calls_pending':len(terminal.get('tool_calls') or [])}
old=next(out.parent.glob('dinov3b_preparation*/roles_direct/jobs/task-0a3*/result.json')).parent
def trial_cost(label):
    trial = out/label
    audits = list(trial.glob('jobs/*/llm_audit.json'))
    values = [read_json(p) for p in audits]
    numeric = []
    for req in trial.glob('jobs/*/request.json'):
        if read_json(req)['kind']=='experiment' and req.with_name('result.json').is_file():
            result=read_json(req.with_name('result.json'))
            numeric.append({'task':req.parent.name,'status':result['status'],'resources':result['resources']})
    result_path = out/(label+'_result.json')
    summary = read_json(result_path) if result_path.is_file() else {}
    return {**summarize(audits), 'numeric':numeric,'valid_evidence_refs':summary.get('evidence',[]),
            'model_calls_started':sum(v.get('model_call_started') is True for v in values),
            'preflight_stops':[v.get('error') for v in values if not v.get('model_call_started')],
            'unknown_inflight_usage':sum(v.get('model_call_started') is True and v.get('usage') is None for v in values),
            'complete': result_path.is_file(),'budget':{k:summary.get('budget',{}).get(k) for k in
               ('elapsed_seconds','excluded_numeric_wait_seconds','configured_total_wall_seconds')},
            'coding_channels':[v.get('coding_channel') for v in values if v.get('coding_channel')]}

atomic_json(out/'costs.json',{'safeguard_replay':comparisons,
    'engineering_trial':{'old':chain(old.parent,old),'new':trial_cost('engineering_trial'),
                         'matched_timeout':trial_cost('engineering_matched')},
    'scope':'All attempts including interrupted attempts, bounded historical L2 workflow and four decision-point replays.'})
