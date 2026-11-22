"""Read-only comparison of the server run with the existing Eval-L0 release."""
from collections import Counter
from pathlib import Path
import time
from core.journal import file_sha256, read_json, atomic_json
from runtime.paths import ROOT
from tools.evaluation.judge import evaluation_manifest

here = Path(__file__).resolve().parent
collection = read_json(here/'collection.json')
evaluation = ROOT/'EXP/evaluations'/collection['evaluation']
manifest = evaluation_manifest(evaluation)
if any(c['manifest_sha256'] != manifest['manifest_sha256'] for c in collection['cases'].values()):
    raise ValueError('collection belongs to a different release')
run = Path(manifest['source_run'])
d = read_json(evaluation/'run.json')
started = time.monotonic()
changed, empty = [], Counter()
for row in manifest['source_files']:
    path = run/row['path']
    if not path.is_file() or file_sha256(path)!=row['sha256']:
        changed.append(row['path'])
    if row['bytes']==0:
        empty[path.name]+=1
tasks = {t['task_id']:t for t in d['tasks']}
picked = []
release_cases = {c['case_id']:c for c in manifest['cases']}
for name, case in collection['cases'].items():
    req = read_json(run/case['source'])
    release_case = release_cases[case['case_id']]
    versions = case['source_versions']
    if (file_sha256(run/case['source']) != case['request_sha256'] or req['execution_id'] != case['execution_id']
            or release_case['task_id'] != req['task_id'] or release_case['case_sha256'] != versions['case_sha256']
            or release_case['role'] != case['role'] or req['spec']['role'] != case['role']
            or req['spec']['context']['read_view'] != case['read_view'] or req['input_versions'] != case['input_versions']
            or req['spec']['context']['prompt_source_sha256'] != versions['prompt_source_sha256']
            or req['core_sha256'] != versions['core_sha256']
            or [t['name'] for t in req['spec']['tools']] != case['tool_capabilities']):
        raise ValueError('invalid role breakpoint metadata: ' + name)
    picked.append(name)
sources = []
reads = Counter()
for t in tasks.values():
    if t['kind']!='host_tool':
        continue
    card = read_json(evaluation/t['source']['object'])
    name = card['payload']['inputs']['name']
    args = card['payload']['inputs']['arguments']
    result = card['payload'].get('result') or {}
    if name=='read_asset':
        requested = args['asset']
        if 'skill' in requested.lower() or 'snapshot_field' in args:
            kind = ('snapshot member' if args.get('member') else 'snapshot directory') if 'snapshot_field' in args else requested
            reads[(kind,t['state'])]+=1
    if t['state']=='failed' or name in ('retrieve_memory','update_inspiration') or (
            name=='read_asset' and 'skills/' in args['asset']):
        sources.append({'id':t['task_id'],'owner':t['parent_task'],'key':t['key'],'source':t['source']['origin'],
            'sha256':t['source']['sha256'],'tool':name,'state':t['state'],'error':t.get('error'),
            'arguments':{k:v for k,v in args.items() if k in ('ref','asset','member','snapshot_field','field','base')},
            'result_fields':list(result) if isinstance(result,dict) else None})
inputs = ROOT/'EXP/inputs'/run.name
external = {n:file_sha256(inputs/n) for n in ('input.json','initial_state.json','baseline_development.json',
    'trajectory_index.json','budget_reconciliation.json','operator_interventions.jsonl','closeout_reconciliation.json',
    'final_selection.json','official_summary.json','closeout_status.json','forward_smoke.json','numeric_preflight.json',
    'role_preflight.json','round1_checkpoint.json','round2_checkpoint.json','round3_checkpoint.json','runtime_costs.json',
    'git_head.txt','host_changes.patch')}
official = []
for row in read_json(inputs/'official_summary.json')['artifacts']:
    path = Path(row['path'])
    if not path.is_file() or file_sha256(path) != row['sha256'] or path.stat().st_size != row['bytes']:
        changed.append(str(path))
    official.append({**row, 'verified':str(path) not in changed})
empty_logs = sum(1 for log in (run/'jobs').glob('*/worker.log') if log.stat().st_size == 0)
atomic_json(here/'audit.json',{'run':run.name,'manifest':manifest['manifest_sha256'],'server_sha_mismatch':changed,
    'verified_count':len(manifest['source_files'])-sum(not Path(p).is_absolute() for p in changed),
    'verified_role_breakpoints':picked,'server_empty_worker_logs':empty_logs,'external_official_artifacts':official,'empty_frozen_files':dict(empty),
    'server_jobs':len(list((run/'jobs').iterdir())),'task_cases':len(manifest['cases']),'tools':d['tools'],
    'research_cost':d['metrics'],'posthoc_sources':external,'elapsed_seconds':time.monotonic()-started,
    'reads':[{'resource':k,'state':s,'count':v} for (k,s),v in reads.items()],'host_events':sources,
    'numerical_jobs':[{k:j.get(k) for k in ('execution_id','status','error','resources','receipt_path')}
                      for j in d['jobs'] if j['kind']=='experiment']})
if changed:
    raise SystemExit('server sources differ; see audit.json')
