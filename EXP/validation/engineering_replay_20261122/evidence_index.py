"""Materialize assessor-only provenance and measured replay results from durable receipts."""
import hashlib
from pathlib import Path
from core.journal import atomic_json, file_sha256, read_json
from runtime.paths import ROOT
from tools.evaluation.judge import evaluation_manifest

HERE = Path(__file__).resolve().parent
collection = read_json(HERE/'collection.json')
audit = read_json(HERE/'audit.json')
evaluation = ROOT/'EXP/evaluations'/collection['evaluation']
manifest = evaluation_manifest(evaluation)
run = Path(manifest['source_run'])
inputs = ROOT/'EXP/inputs'/run.name
data = read_json(evaluation/'run.json')
by_task = {t['task_id']:t for t in data['tasks']}
source = {s['path']:s for s in manifest['source_files']}


def pointer(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':file_sha256(path)}


def raw(rel):
    row = source[rel]
    assert file_sha256(run/rel) == row['sha256']
    return {'origin':rel,'object':row['object'],'sha256':row['sha256']}


# Expected responsibilities live on the assessor side. A worker gets only its own request.
cases = {
    '3a5b25': ('host data view','bind declared search range before decode and verify actual reads'),
    '7353e9': ('host frozen loader','verify original state; recover the exact approved custom loader'),
    '46fe84': ('final frozen evaluation','verify independent preparation provenance for an L3 submission'),
    '48d24b': ('parent implementation','attribute missing development_view to the executing candidate'),
    '6bef7a': ('Coding revision','return device mismatch to author, invalidate review of modified source'),
    '2bef01': ('review and execute revision','bind and measure the corrected H3 while retaining parent failure'),
    '4067de': ('worker recovery','recover one lost worker without duplicating its completed Evidence'),
    'b5565e': ('numeric drain','register late H2 while preserving pending H1/H3 and unclosed Recorder'),
    'c8bdc0': ('measurement','reject missing classification field that the frozen request explicitly required'),
    'f3d0ef': ('measurement','retain independent rejection and exact request scope'),
    '34987a': ('frozen construction','export then independently cold-load the original dual-task H2'),
}
numerics = []
for job in data['jobs']:
    prefix = job['task_id'][5:11]
    if job['kind'] != 'experiment' or prefix not in cases:
        continue
    spec = read_json(run/job['request_path'])
    exp = spec['experiment']
    task = by_task[job['task_id']]
    numerics.append({'responsibility':cases[prefix][0], 'acceptance':cases[prefix][1],
        'task_key':task['key'],'request_task':task['parent_task'],'execution_id':job['execution_id'],
        'start_and_visible_input':raw(job['request_path']),'receipt_for_assessor':raw(job['receipt_path']),
        'source_version':job['core_sha256'],'experiment':exp['id'],'experiment_sha256':exp['content_sha256'],
        'service':exp['payload']['execution']['service'],'level':exp['payload']['level'],
        'attempt':job['attempt'],'historical_status':job['status'],'resources':job['resources'],
        'elapsed_seconds':job['elapsed_seconds'],'tokens':0,'live_status':'see COVERAGE.md'})
skill = next(x for x in audit['host_events'] if x['arguments'].get('asset','').endswith('/frozen-propagation.md'))
reply = read_json(run/skill['source'])['payload']['result']
following = f"jobs/{skill['owner']}-a0003/request.json"
receipt = next(t for t in read_json(run/following)['spec']['context']['tool_results'] if t.get('task') == skill['id'])
assert receipt['result']['text'] == reply['text'] == ''
restore = collection['cases']['restore']['source']
feedback = []
for entry in read_json(run/restore)['spec']['context']['memory']['bound_feedback']:
    handle = entry['read_card']
    path = 'cards/' + hashlib.sha256(handle['ref'].encode()).hexdigest()+'.json'
    value = read_json(run/path)['payload'][handle['field'].split('/')[-1]]
    assert value == entry['value']
    feedback.append({'source':raw(path),'ref':handle,'delivered_value_sha256':entry['sha256'],'equal':True})
atomic_json(HERE/'engineering_cases.json',{'manifest_sha256':manifest['manifest_sha256'],
    'rule_version':manifest['rules_fingerprint'],'numeric_breakpoints':numerics,
    'skill':{'host_task':raw(skill['source']),'following_call':raw(following),
            'delivered_chars':len(reply['text']),'offset':reply['offset'],'size':reply['total_characters']},
    'recorder':{'following_call':raw(restore),'bound_fields':feedback},
    'closeout_sources':[pointer(inputs/name) for name in audit['posthoc_sources']]})

edit = ROOT/'EXP/evaluations/independent_20261122_edit'
chain = ROOT/'EXP/evaluations/independent_20261122_chain_r2'
calls = []
h1 = ROOT/'EXP/evaluations/independent_20261122_h1'
for directory in (edit/'steps/1', edit/'steps/2',chain/'safeguard',chain/'generation',h1/'steps/1'):
    log = read_json(directory/'llm_audit.json')
    answer = read_json(directory/'response.json')
    calls.append({'source':pointer(directory/'llm_audit.json'),'response':pointer(directory/'response.json'),
                  'requested':log['requested'],'actual':log['actual'],'request_sha256':log['request_sha256'],
                  'prompt_source_sha256': log['request']['context']['prompt_source_sha256'],
                  'usage':answer['usage'],'elapsed_seconds':log['elapsed_seconds'],
                  'decision':(answer.get('decision') or {}).get('action'),'tools':[x['name'] for x in answer['tool_calls']]})
run_result = read_json(chain/'result.json')
trace = read_json(edit/'trace.json')
evref = run_result['evidence']['read_card']['ref']
ev = chain/'branch/cards'/(hashlib.sha256(evref.encode()).hexdigest()+'.json')
numeric = read_json(chain/'numeric/result.json')
feedback = read_json(chain/'generation/llm_audit.json')['request']['context']['tool_results'][-1]['result']
evidence = read_json(ev)
assert evidence['content_sha256'] == run_result['evidence']['sha256']
assert feedback['observations'] == numeric['data']['observations'] == evidence['payload']['observations']
assert feedback['evidence']['source'] == run_result['evidence']
totals = {k:sum(x['usage'][k] for x in calls) for k in calls[0]['usage']}
req_id = collection['cases']['code_repair']['execution_id'].rsplit('-a',1)[0]
baseline = []
for job in data['jobs']:
    if job['task_id'] != req_id:
        continue
    result = read_json(run/job['receipt_path'])['data']['response']
    decision = result.get('decision') or {}
    baseline.append({'source':raw(job['receipt_path']),'usage':result['usage'],'elapsed_seconds':job['elapsed_seconds'],
                     'tool_arguments':[{'name':t['name'],'fields':list(t['arguments'])} for t in result['tool_calls']],
                     'decision':decision.get('action'),'submitted_fields':list(decision.get('content',{}))})
from tools.experiments import prepare_request
h1_trace = read_json(h1/'trace.json')
h1_call = read_json(h1/'steps/1/response.json')['tool_calls'][0]
h1_ctx = read_json(h1/'steps/1/llm_audit.json')['request']['context']
prepared = prepare_request(h1_call['arguments']['experiment'],h1_ctx['research_brief']['payload'],read_json(inputs/'input.json')['runtime'])
assert h1_call['name'] == 'request_experiment' and prepared['stage'] == 'request'
decoded = read_json(HERE/'data_check/receipt.json')
forward = read_json(HERE/'package_check/receipt.json')
assert decoded['ordering_sha256'] == decoded['decoder_seen_sha256'] == forward['bound_ids_sha256']
assert decoded['first_id'] == forward['sample_id'] and decoded['count'] == 64
atomic_json(HERE/'behavior_results.json',{'source_manifest_sha256':manifest['manifest_sha256'],
    'original_breakpoint':collection['cases']['code_repair']['execution_id'],
    'historical_coding_baseline_for_assessor':baseline,'coding_trace':pointer(edit/'trace.json'),'coding_core':trace['current_core'],'calls':calls,'usage':totals,
    'router_elapsed_sum_seconds':sum(c['elapsed_seconds'] for c in calls),
    'subscription_cost':None,'cli_internal_retries':None,'observed_additional_model_attempts':0,
    'branch_setup_retries_before_review':1,'setup_failure':pointer(HERE/'chain.log'),
    'numeric':{'receipt':pointer(chain/'numeric/result.json'),'scope':{k:len(v) for k,v in numeric['data']['sample_selection'].items()},
               'resources':numeric['resources'],'core_sha256':numeric['core_sha256'],'observation_fields':list(feedback['observations'])},
    'evidence':pointer(ev),'delivered_observations_equal_new_measurement':True,'branch_result':run_result,
    'h1':{'trace':pointer(h1/'trace.json'),'start':h1_trace['start'],'core_sha256':h1_trace['current_core'],
          'scope':'real generation decision + production request preflight; no coding, new numerical result or review',
          'decision_note':h1_call['arguments']['research_note'],'question':prepared['question'],
          'level':prepared['level'],'view':prepared['execution']['parameters']['diagnostic_view'],
          'request_preflight':'accepted','execution':h1_trace['status']},
    'new_generation_experiment_measured':False,'package_probe':pointer(HERE/'package_check/receipt.json'),
    'complete_view_decode':pointer(HERE/'data_check/receipt.json')})
