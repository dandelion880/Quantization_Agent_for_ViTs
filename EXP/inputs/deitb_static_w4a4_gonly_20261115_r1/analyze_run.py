"""Run-specific posthoc accounting over frozen Eval-L0 and original receipts.

No model calls, research writes, source changes, or scientific decisions.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'QAViT/tools'))
from evaluation.metrics import call_metrics, union_seconds


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(value):
    """Derived index only: raw lists stay intact in original/frozen sources."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 100:
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            return {'omitted_from_derived_index': True, 'length': len(value),
                    'sha256': hashlib.sha256(raw).hexdigest(),
                    'retained_in': 'original run and immutable Eval-L0 source objects'}
        return [compact(v) for v in value]
    return value


parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', required=True)
parser.add_argument('--output', default=str(HERE / 'ANALYSIS.json'))
parser.add_argument('--results-output', default=str(HERE / 'RESULTS.json'))
args = parser.parse_args()
ev = Path(args.evaluation).resolve()
frozen = read(ev / 'run.json')
run = Path(frozen['source_run'])
manifest = read(ev / 'manifest.json')
summary = read(run / 'summary.json')

# Use the immutable TaskCard objects already captured by independent extraction.
tasks = {}
for t in frozen['tasks']:
    source = t.get('source')
    if source:
        card = read(ev / source['object'])
        tasks[t['task_id']] = card['payload']


def lineage(task):
    seen = set()
    while task:
        if task['task_id'] in seen:
            raise ValueError('cycle')
        seen.add(task['task_id'])
        if task['kind'] == 'llm':
            return {k: task.get(k) for k in ('task_id', 'key', 'role')} | {
                'purpose': task['inputs'].get('purpose')}
        task = tasks.get(task['inputs'].get('parent_task') or task['inputs'].get('origin_task'))
    return None


role_calls, purpose_calls = defaultdict(list), defaultdict(list)
for call in frozen['calls']:
    task = tasks.get(call['task_id'], {})
    inputs = task.get('inputs', {})
    role = task.get('role', 'unknown')
    purpose = inputs.get('purpose', 'unknown')
    slot = inputs.get('context', {}).get('slot') if purpose == 'generate' else None
    role_calls[role].append(call)
    purpose_calls[(role, purpose, slot)].append(call)

roles = [{'role': role, 'task_count': len({c['task_id'] for c in calls}),
          **call_metrics(calls)} for role, calls in sorted(role_calls.items())]
purposes = [{'role': k[0], 'purpose': k[1], 'slot': k[2],
             'task_count': len({c['task_id'] for c in calls}), **call_metrics(calls)}
            for k, calls in sorted(purpose_calls.items(), key=lambda x: str(x[0]))]
routes, catalog_deliveries = Counter(), Counter()
for call in frozen['calls']:
    task = tasks.get(call['task_id'], {})
    requested, actual = call.get('requested') or {}, call.get('actual') or {}
    routes[(task.get('role'), requested.get('model'), requested.get('reasoning_effort'),
            actual.get('model'), actual.get('reasoning_effort'))] += 1
    if call.get('audit_path'):
        audit = read(run / call['audit_path'])
        if 'skill_catalog' in audit.get('request', {}).get('context', {}):
            catalog_deliveries[task.get('role')] += 1

numeric = []
for job in frozen['jobs']:
    if job.get('kind') != 'experiment':
        continue
    directory = run / 'jobs' / job['execution_id']
    receipt = read(directory / 'result.json') if (directory / 'result.json').exists() else {}
    started = read(directory / 'started.json') if (directory / 'started.json').exists() else {}
    req = read(directory / 'request.json') if (directory / 'request.json').exists() else {}
    exp_ref = receipt.get('experiment_ref') or req.get('experiment_ref')
    exp_path = run / 'cards' / (hashlib.sha256(exp_ref.encode()).hexdigest() + '.json') if exp_ref else None
    exp = read(exp_path)['payload'] if exp_path and exp_path.exists() else {}
    task = tasks.get(job['task_id'], {})
    numeric.append({'key': task.get('key'), 'execution_id': job['execution_id'],
                    'level': exp.get('level'), 'experiment_ref': exp_ref,
                    'status': receipt.get('status', 'unknown'), 'error': receipt.get('error'),
                    'resources': receipt.get('resources'),
                    'started_unix': started.get('started_unix'),
                    'finished_unix': receipt.get('finished_unix'),
                    'result': relative(directory / 'result.json')})
numeric.sort(key=lambda n: n.get('started_unix') or 0)
intervals = [(n['started_unix'], n['finished_unix']) for n in numeric
             if isinstance(n.get('started_unix'), (float, int)) and isinstance(n.get('finished_unix'), (float, int))]
edges = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals], key=lambda x: (x[0], x[1]))
current = peak = 0
for _, delta in edges:
    current += delta
    peak = max(peak, current)

tools, assets, reads, memory, rejected = Counter(), Counter(), [], [], []
for t in tasks.values():
    if t['kind'] != 'host_tool':
        continue
    inputs = t['inputs']
    name = inputs.get('name', 'unknown')
    arguments = inputs.get('arguments', {})
    tools[(name, t['state'])] += 1
    parent = lineage(t)
    if name == 'read_asset':
        path = arguments.get('path') or arguments.get('asset') or arguments.get('ref')
        row = {'key': t['key'], 'path': path, 'arguments': arguments,
               'state': t['state'], 'parent': parent}
        reads.append(row)
        if t['state'] == 'completed':
            assets[str(path)] += 1
    if name == 'retrieve_memory':
        memory.append({'key': t['key'], 'arguments': arguments, 'state': t['state'],
                       'parent': parent, 'result': t.get('result')})
    if t['state'] == 'failed':
        rejected.append({'key': t['key'], 'tool': name, 'arguments': arguments,
                         'parent': parent, 'error': t.get('error')})

briefs, hypotheses, reviews, recorder = [], [], [], []
for file in sorted((run / 'cards').glob('*.json')):
    c = read(file)
    kind = c.get('kind')
    p = c.get('payload', {})
    if kind == 'ResearchBrief':
        briefs.append({'ref': c['id'], 'payload': p})
    elif kind == 'HypothesisCard':
        hypotheses.append({'ref': c['id'], **{k: p.get(k) for k in ('round', 'slot', 'title', 'hypothesis_type', 'evidence_refs', 'feedback_used')}})
    elif kind == 'ReviewDecisionCard':
        reviews.append({'ref': c['id'], 'payload': p})
    elif kind == 'MetareviewCard':
        recorder.append({'ref': c['id'], 'payload': p})
hypotheses.sort(key=lambda h: (h.get('round') or 0, h.get('slot') or ''))

llm_errors = [{'execution_id': c['execution_id'], 'task': tasks.get(c['task_id'], {}).get('key'),
               'error': c.get('error'), 'status': c.get('status')} for c in frozen['calls'] if c.get('error')]
resource_summary = {
    'execution_count': len(numeric), 'states': dict(Counter(n['status'] for n in numeric)),
    'wall_seconds_sum': sum((n.get('resources') or {}).get('wall_seconds', 0) for n in numeric),
    'failed_wall_seconds_sum': sum((n.get('resources') or {}).get('wall_seconds', 0) for n in numeric if n['status'] != 'completed'),
    'closed_execution_interval_union_seconds': union_seconds(intervals),
    'peak_simultaneous_numeric_workers': peak,
    'gpu_assignments': [((n.get('resources') or {}).get('physical_gpus')) for n in numeric],
    'note': 'Worker wall and active-budget excluded waits have different boundaries; do not add or equate them.'}

formal = []
for n in numeric:
    if n['level'] != 'L3' or n['status'] != 'completed':
        continue
    directory = run / 'jobs' / n['execution_id']
    c, s, r = [read(directory / name) for name in ('classification.json', 'segmentation.json', 'result.json')]
    formal.append({'key': n['key'], 'execution_id': n['execution_id'],
        'classification': {'sample_count': c['sample_count'], 'status': c['status'],
            'full_dataset_evaluation': c['full_dataset_evaluation'],
            'metrics': {k: c['metrics'].get(k) for k in ('top1', 'top5', 'baseline_top1', 'baseline_top5', 'top1_drop_pp', 'latency_ms', 'baseline_latency_ms')},
            'health': c.get('health'), 'counts': c.get('classification_counts')},
        'segmentation': {k: s.get(k) for k in ('images', 'full_dataset_evaluation', 'delta_mIoU_pp', 'trained', 'recalibrated', 'head_unchanged', 'backbones_unchanged')} | {
            'metrics': {side: {k: v.get(k) for k in ('mIoU', 'mAcc', 'aAcc')} for side, v in s['metrics'].items()}},
        'reload': r['data']['observations'].get('reload'), 'result': n['result'],
        'file_hashes': {name: sha(directory / name) for name in ('classification.json', 'segmentation.json', 'result.json', 'model/manifest.json', 'model/weights.pt')}})

summary_index = {k: summary.get(k) for k in ('run_id', 'architecture', 'status', 'completed_rounds',
    'workflow_completed', 'formal_validation_success', 'numerical_validation_status', 'scientific_review_complete',
    'state_saved', 'monitor_cleanup_completed', 'finalization_errors', 'supervisor_exit_code', 'current_implementation_matches')}
summary_index['budget'] = {k: summary['budget'].get(k) for k in ('config', 'started_unix', 'as_of_unix',
    'elapsed_seconds', 'excluded_numeric_wait_seconds', 'remaining_seconds', 'active_segment', 'extensions', 'segments', 'numeric_waits')}
summary_index['recording_status'] = compact(summary.get('recording_status'))
summary_index['memory_sizes'] = {k: len(summary.get('memory', {}).get(k, {})) for k in ('L2', 'L3')}
if (HERE/'managed_terminal.json').exists():
    managed_receipt = read(HERE/'managed_terminal.json')
    for key in ('state_saved', 'monitor_cleanup_completed', 'supervisor_exit_code', 'finalization_errors'):
        summary_index[key] = managed_receipt['terminal'][key]
    summary_index['managed_terminal_provenance'] = relative(HERE/'managed_terminal.json')
segments = summary['budget']['segments']
run_intervals = {'segment_union_seconds': frozen['timing']['active_segment_wall_union_seconds'],
    'first_segment_started_unix': min(s['started_unix'] for s in segments),
    'last_segment_stopped_unix': max(s['stopped_unix'] for s in segments)}
run_intervals['end_to_end_seconds_including_pause'] = run_intervals['last_segment_stopped_unix'] - run_intervals['first_segment_started_unix']
run_intervals['pause_gap_seconds'] = run_intervals['end_to_end_seconds_including_pause'] - run_intervals['segment_union_seconds']

latest = read(ev / 'latest_report.json')
eval_report = read(ev / latest['json'])
result = {
    'schema': 'qavit.gonly.run_analysis/1', 'created_at': datetime.now(timezone.utc).isoformat(),
    'run': relative(run), 'evaluation': relative(ev), 'journal': frozen['journal'],
    'summary': summary_index, 'research_metrics': frozen['metrics'], 'run_timing': frozen['timing'],
    'run_intervals': run_intervals,
    'role_metrics': roles, 'purpose_metrics': purposes,
    'model_routes': [{'role': k[0], 'requested_model': k[1], 'requested_effort': k[2],
                      'actual_model': k[3], 'actual_effort': k[4], 'calls': v} for k,v in sorted(routes.items(), key=lambda x:str(x[0]))],
    'skill_catalog_deliveries_per_call': dict(catalog_deliveries),
    'skill_body_or_reference_reads': sum(r['state']=='completed' and 'skills/' in str(r['path']) for r in reads),
    'numeric_summary': resource_summary, 'numeric_executions': numeric, 'formal_l3': formal,
    'tool_counts': [{'tool': k[0], 'state': k[1], 'count': v} for k,v in sorted(tools.items())],
    'successful_asset_reads': [{'path': k, 'count': v} for k,v in sorted(assets.items())],
    'asset_read_ledger': reads, 'memory_retrievals': compact(memory), 'rejected_tools': rejected,
    'llm_errors': llm_errors, 'hypotheses': hypotheses, 'review_decisions': reviews,
    'recorder': compact(recorder), 'evidence': compact(frozen['evidence']),
    'eval_l0': {'report': relative(ev / latest['json']), 'groups': eval_report['groups'],
                'judge_overhead': eval_report['judge_overhead'],
                'case_status_counts': dict(Counter(c['judge_status'] for c in eval_report['cases'])),
                'excluded_tasks': eval_report['excluded_tasks'],
                'run_data_coverage': eval_report['run_data_coverage']},
    'hashes': {relative(p): sha(p) for p in [HERE/'input.json', HERE/'experiment_identity.json',
               run/'research_run.json', run/'summary.json', ev/'manifest.json', ev/'run.json', ev/latest['json']]},
    'limitations': ['Eval-L0 is observational and uses task-time delivered evidence.',
        'Requested and actual model identities are distinct; missing actual identity remains unknown.',
        'Cached input and reasoning output are subfields, not additional totals.',
        'One external core repair and internal code revisions preclude an unassisted-autonomy claim.',
        'No paired Full-vs-G-only causal comparison or independent scientific replication.']}
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
library = read(HERE / 'solution_library_final.json') if (HERE / 'solution_library_final.json').exists() else None
compact_result = {
    'schema': 'qavit.gonly.completed_experiment/1', 'run_id': frozen['run_id'],
    'run': relative(run), 'evaluation': relative(ev), 'journal': frozen['journal'],
    'conclusion': 'Two-round evidence-driven workflow completed with one external engineering intervention. Valid L3 results exist for both rounds; novel H3 algorithms remain untested. No causal Full-vs-G-only claim.',
    'terminal': {k:v for k,v in summary_index.items() if k!='budget'},
    'budget': {k:summary_index['budget'][k] for k in ('config','elapsed_seconds','excluded_numeric_wait_seconds','remaining_seconds','extensions','active_segment')},
    'run_intervals': run_intervals, 'source_identity': {
        'initial_commit': 'b6fd75e2689e15add23611005639bcbd5f250cc6',
        'repair_commit': '42cbdd293fd2c39e19ca9e7fcc2a157d9a4f935f',
        'main_unchanged_commit': '8800006b687c56205ad624bae2ad7156c326d873'},
    'formal_l3': formal, 'research_metrics': frozen['metrics'],
    'role_metrics': roles, 'purpose_metrics': purposes, 'model_routes': result['model_routes'],
    'numeric_summary': resource_summary,
    'numeric_executions': [{k:n[k] for k in ('key','execution_id','level','status','error','result')} | {
        'wall_seconds': (n.get('resources') or {}).get('wall_seconds')} for n in numeric],
    'hypotheses': [{k:h.get(k) for k in ('ref','round','slot','title','hypothesis_type','evidence_refs')} for h in hypotheses],
    'components': {'skill_catalog_deliveries_per_call': result['skill_catalog_deliveries_per_call'],
        'skill_body_or_reference_reads': result['skill_body_or_reference_reads'],
        'method_reads': sum(v for k,v in assets.items() if k.startswith('methods/')),
        'method_documents': len([k for k in assets if k.startswith('methods/')]),
        'solution_document_reads': sum(v for k,v in assets.items() if k.startswith('solutions/')),
        'solution_asset_reads': sum(v for k,v in assets.items() if '/solution_library/' in k),
        'memory_retrieval_count': len(memory), 'tool_counts': frozen['tools'],
        'library_final': [{k:e.get(k) for k in ('id','title','status','metrics','evidence_ref')} for e in library['snapshot']['entries']] if library else None,
        'issues': ['No Skill body/reference read; catalog exposure does not establish use.',
            'FP ADE metrics exist in raw artifacts but were not retrieved by either Recorder.',
            'R2 resolved prefix-PV question remains in final L1 open_questions.',
            'Pending and measured GPTAQ recipes occupy two final library positions.']},
    'eval_l0': result['eval_l0'] | {
        'prepared_tasks': len(eval_report['cases']), 'effective_task_coverage': sum(c['quality'] is not None for c in eval_report['cases'])/len(eval_report['cases']),
        'cases': [{k:c[k] for k in ('case_id','task_key','role','purpose','judge_status','quality')} for c in eval_report['cases']]},
    'external_monitor_usage_as_of': read(HERE/'external_monitor_usage.json') if (HERE/'external_monitor_usage.json').exists() else None,
    'isolation_check': read(HERE/'eval_source_isolation.json') if (HERE/'eval_source_isolation.json').exists() else None,
    'hashes': result['hashes'], 'artifact_inventory': relative(HERE/'ARTIFACTS.json'),
    'limitations': result['limitations'] + ['Judge task coverage is partial; ungraded tasks are never assigned zero or synthetic scores.',
        'External monitor usage is a timestamped local counter, excludes later work, and is not a subscription invoice.']}
Path(args.results_output).write_text(json.dumps(compact_result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'output': relative(args.output), 'roles': len(roles), 'hypotheses': len(hypotheses),
                  'numeric': resource_summary, 'eval': result['eval_l0']['case_status_counts']}, ensure_ascii=False))
