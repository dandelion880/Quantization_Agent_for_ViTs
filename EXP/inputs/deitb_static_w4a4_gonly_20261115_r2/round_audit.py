#!/usr/bin/env python3
"""Read-only run-specific provenance/accounting index. Never feeds research roles.

Checks are observations, not additional workflow gates or scientific choices.
Large original arrays remain intact; this derived index records their hashes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'QAViT'))
sys.path.insert(0, str(ROOT / 'QAViT/tools'))
from core.journal import Journal, atomic_json
from runtime.budget import BudgetLedger
from runtime.task_queue import GlobalTaskQueue, process_alive
from evaluation.metrics import call_metrics, union_seconds


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 64:
            return {'count': len(value), 'sha256': digest(value),
                    'retained_in': 'original immutable run artifact'}
        return [compact(v) for v in value]
    return value


def pick(value, keys):
    return {k: value.get(k) for k in keys}


class Audit:
    def __init__(self, run, output):
        if not (run / 'research_run.json').is_file():
            raise FileNotFoundError('research run not created; audit cannot create it')
        self.run, self.output = run, output
        self.journal = Journal(run, run.name)
        self.cache, self.hashes = {}, {}

    def read(self, path):
        if not path.is_file():
            return {}
        st = path.stat()
        sig = (st.st_size, st.st_mtime_ns)
        if path not in self.cache or self.cache[path][0] != sig:
            self.cache[path] = (sig, json.loads(path.read_text()))
        return self.cache[path][1]

    def sha(self, path):
        if not path.is_file():
            return None
        st = path.stat()
        sig = (st.st_size, st.st_mtime_ns)
        if path not in self.hashes or self.hashes[path][0] != sig:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            self.hashes[path] = (sig, h.hexdigest())
        return self.hashes[path][1]

    def snapshot(self):
        j = self.journal
        j.refresh()
        tasks = GlobalTaskQueue(j).snapshot()
        cards = {ref: j.card(ref) for ref, item in j.cards.items()
                 if item['kind'] != 'TaskCard'}
        bykind = defaultdict(list)
        for c in cards.values():
            bykind[c['kind']].append(c)
        for values in bykind.values():
            values.sort(key=lambda c: c['created_at'])
        ev_by_exec = {c['payload']['execution_id']: c for c in bykind['EvidenceCard']}
        calls, numeric, failures, tool_reads, memory_reads = [], [], [], [], []
        catalog, tools, routes = Counter(), Counter(), Counter()

        def lineage(task):
            seen = set()
            while task and task['task_id'] not in seen:
                seen.add(task['task_id'])
                if task['kind'] == 'llm':
                    return {**pick(task, ('task_id', 'key', 'role')),
                            'purpose': task['inputs'].get('purpose')}
                task = tasks.get(task['inputs'].get('parent_task') or task['inputs'].get('origin_task'))
            return None

        for task in tasks.values():
            if task.get('state') in ('failed', 'needs_clarification'):
                failures.append(pick(task, ('task_id', 'key', 'kind', 'role', 'state', 'error')))
            if task['kind'] != 'host_tool':
                continue
            inp = task['inputs']
            name = inp.get('name', 'unknown')
            tools[(name, task['state'])] += 1
            row = {**pick(task, ('task_id', 'key', 'state', 'error')),
                   'tool': name, 'arguments': inp.get('arguments'), 'parent': lineage(task),
                   'result': task.get('result')}
            if name in ('read_asset', 'read_card'):
                tool_reads.append(row)
            if name == 'retrieve_memory':
                memory_reads.append(row)

        for directory in sorted((self.run / 'jobs').iterdir()):
            if not directory.is_dir():
                continue
            req = self.read(directory / 'request.json')
            rec = self.read(directory / 'result.json')
            if not req:
                continue
            task = tasks.get(req.get('task_id'), {})
            if req.get('kind') == 'llm':
                a = self.read(directory / 'llm_audit.json')
                if not a:
                    continue
                usage = a.get('usage') or rec.get('data', {}).get('usage')
                if usage is None and len(a.get('reported_usage') or []) == 1:
                    usage = a['reported_usage'][0]
                call = {**pick(a, ('requested', 'actual', 'elapsed_seconds', 'started_unix',
                                   'finished_unix', 'model_call_started', 'error')),
                        'execution_id': directory.name, 'task_id': req.get('task_id'),
                        'role': task.get('role', a.get('role')),
                        'purpose': task.get('inputs', {}).get('purpose'),
                        'status': rec.get('status', 'unfinished'), 'usage': usage,
                        'response_observed': bool(a.get('response')),
                        'audit_path': str(directory / 'llm_audit.json')}
                calls.append(call)
                context = a.get('request', {}).get('context', {})
                call['fallback_B_delivered'] = compact(context.get('fallback_B'))
                call['read_view'] = compact(context.get('read_view'))
                call['delivered_memory'] = compact(context.get('memory'))
                if 'skill_catalog' in context:
                    catalog[call['role']] += 1
                rq, actual = call.get('requested') or {}, call.get('actual') or {}
                routes[(call['role'], rq.get('model'), rq.get('reasoning_effort'),
                        actual.get('model'), actual.get('reasoning_effort'))] += 1
                continue
            if req.get('kind') != 'experiment':
                continue
            exp = req.get('experiment') or {}
            ep = exp.get('payload', {})
            approval = req.get('approval') or {}
            candidate = approval.get('candidate') or req.get('candidate') or {}
            evidence = ev_by_exec.get(directory.name)
            binding_issues = []

            def equal(label, left, right):
                if left != right or left is None:
                    binding_issues.append(label)

            if req.get('candidate') is not None:
                equal('approved_candidate_mismatch', req['candidate'], approval.get('candidate'))
            equal('experiment_ref_mismatch', exp.get('id'), approval.get('experiment_ref'))
            equal('experiment_hash_mismatch', exp.get('content_sha256'), approval.get('experiment_sha256'))
            source = Path(candidate['path']) if candidate.get('path') else None
            equal('source_hash_mismatch', self.sha(source) if source else None, candidate.get('source_sha256'))
            equal('approved_source_hash_mismatch', candidate.get('source_sha256'), approval.get('source_sha256'))
            saf = cards.get(approval.get('safeguard_ref'), {})
            equal('safeguard_hash_mismatch', saf.get('content_sha256'), approval.get('safeguard_sha256'))
            sb = saf.get('payload', {}).get('reviewed_binding', {})
            for key in ('experiment_ref', 'experiment_sha256', 'request_ref', 'request_sha256', 'source_sha256'):
                equal('safeguard_binding_' + key, approval.get(key), sb.get(key))
            if saf.get('payload', {}).get('approved') is not True or approval.get('status') != 'approved':
                binding_issues.append('safeguard_not_approved')
            if rec:
                equal('receipt_experiment_ref_mismatch', rec.get('experiment_ref'), exp.get('id'))
                equal('receipt_experiment_hash_mismatch', rec.get('experiment_sha256'), exp.get('content_sha256'))
                equal('receipt_request_hash_mismatch', rec.get('request_sha256'), self.sha(directory / 'request.json'))
            if evidence:
                equal('evidence_experiment_mismatch', evidence['payload'].get('experiment_ref'), exp.get('id'))
                equal('evidence_real_source_missing', evidence['payload'].get('measurement_source'), 'real')
            started = self.read(directory / 'started.json')
            observations = rec.get('data', {}).get('observations', {})
            provenance = rec.get('data', {}).get('provenance', {})
            row = {'execution_id': directory.name, 'task': task.get('key'),
                   'round': int(task['key'][1:4]) if str(task.get('key', '')).startswith('r') else None,
                   'level': ep.get('level'), 'experiment_ref': exp.get('id'),
                   'request_ref': approval.get('request_ref'), 'source_sha256': candidate.get('source_sha256'),
                   'core_sha256': req.get('core_sha256'), 'status': rec.get('status', 'unfinished'),
                   'error': rec.get('error'), 'binding_issues': binding_issues,
                   'evidence_ref': evidence['id'] if evidence else None,
                   'started_unix': started.get('started_unix'), 'finished_unix': rec.get('finished_unix'),
                   'resources': rec.get('resources'), 'gpu_bindings': req.get('gpu_bindings'),
                   'parallelism': req.get('parallelism'), 'result': str(directory / 'result.json')}
            if ep.get('level') == 'L3':
                cls, seg = self.read(directory / 'classification.json'), self.read(directory / 'segmentation.json')
                reload = observations.get('reload') or {}
                frozen = observations.get('static_w4a4') or {}
                row['formal'] = {
                    'requested_calibration': ep.get('execution', {}).get('calibration'),
                    'requested_evaluations': ep.get('execution', {}).get('evaluations'),
                    'classification': {**pick(cls, ('sample_count', 'status', 'full_dataset_evaluation')),
                        'metrics': pick(cls.get('metrics', {}), ('top1', 'top5', 'baseline_top1', 'baseline_top5'))},
                    'segmentation': {**pick(seg, ('images', 'full_dataset_evaluation', 'trained', 'recalibrated',
                                                 'head_unchanged', 'backbones_unchanged')),
                        'metrics': {k: pick(v, ('mIoU', 'mAcc', 'aAcc')) for k,v in seg.get('metrics', {}).items()}},
                    'reload': reload, 'static_w4a4': pick(frozen, ('protocol', 'frozen', 'task_modules')),
                    'model_identity': pick(provenance, ('model_ref', 'protocol_hash', 'model_unchanged',
                                                       'quantizer_state_sha256', 'baseline_state_sha256')),
                    'scope_observations': compact(provenance.get('execution_facts')),
                    'artifacts': [{**pick(a, ('path', 'bytes', 'sha256')),
                        'current_sha256': self.sha(Path(a['path']))}
                        for a in (evidence or {}).get('payload', {}).get('artifacts', [])],
                    'full_dual_task_observed': bool(rec.get('status') == 'completed' and evidence and
                        cls.get('sample_count') == 50000 and cls.get('full_dataset_evaluation') is True and
                        seg.get('images') == 2000 and seg.get('full_dataset_evaluation') is True and
                        reload.get('host_verified') is True and reload.get('passed') is True and
                        reload.get('evaluated_instance') == 'reloaded' and frozen.get('frozen') is True and
                        not binding_issues)}
            numeric.append(row)

        hypotheses = [{**pick(c, ('id', 'parents', 'refs', 'content_sha256', 'created_at')),
                       'payload': compact(c['payload'])} for c in bykind['HypothesisCard']]
        selections = [{**pick(c, ('id', 'refs', 'content_sha256')), 'payload': compact(c['payload'])}
                      for c in bykind['ReviewDecisionCard'] if c['payload'].get('selection_source') == 'generation']
        recorder = [{**pick(c, ('id', 'refs', 'content_sha256')), 'payload': compact(c['payload'])}
                    for c in bykind['MetareviewCard']]
        libraries = []
        memory_projections, round_finishes = [], []
        for e in j.events:
            if e['type'] == 'solution_library':
                libraries.append({'sequence': e['commit_sequence'], 'entries': [
                    pick(r, ('id', 'title', 'status', 'metrics', 'evidence_ref', 'experiment_ref',
                             'selection_ref', 'hypothesis_ref', 'run_dir', 'source_version')) for r in e['entries']]})
            elif e['type'] == 'memory_projection':
                v = e['value']
                memory_projections.append({'sequence': e['commit_sequence'], 'L1': compact(v.get('L1')),
                    'L2_count': len(v.get('L2', [])), 'L3_count': len(v.get('L3', []))})
            elif e['type'] == 'round_finished':
                round_finishes.append(e)
        intervals = [(x['started_unix'], x['finished_unix']) for x in numeric
                     if isinstance(x.get('started_unix'), (int,float)) and isinstance(x.get('finished_unix'), (int,float))]
        edges = sorted([(a,1) for a,b in intervals] + [(b,-1) for a,b in intervals], key=lambda v:(v[0],v[1]))
        concurrent = peak = 0
        for _, change in edges:
            concurrent += change
            peak = max(peak, concurrent)
        role_calls = defaultdict(list)
        for call in calls:
            role_calls[call['role']].append(call)
        budget = BudgetLedger(j).snapshot()
        status_events = [e for e in j.events if e['type'] == 'run_status']
        state = {'schema': 'qavit.gonly.monitor_audit/1', 'observed_at': datetime.now(timezone.utc).isoformat(),
                 'run': str(self.run), 'sequence': j.sequence,
                 'status': status_events[-1].get('status') if status_events else 'created',
                 'head': j.commits[-1]['sha256'] if j.commits else None,
                 'budget': pick(budget, ('config', 'elapsed_seconds', 'remaining_seconds',
                     'excluded_numeric_wait_seconds', 'extensions', 'segments', 'active_segment', 'rounds_started')),
                 'round_finishes': round_finishes, 'hypotheses': hypotheses, 'selections': selections,
                 'experiments': [{**pick(c, ('id','refs','content_sha256','author')),
                     'payload': {**pick(c['payload'], ('level','question','controls','decision_impact','evidence_refs')),
                         'execution': pick(c['payload'].get('execution', {}),
                                           ('interface','calibration','evaluations','result_protocol'))}}
                     for c in bykind['ExperimentCard']],
                 'safeguard_reviews': [{**pick(c, ('id','refs','content_sha256')),
                     'payload': compact(c['payload'])} for c in bykind['ReviewDecisionCard']
                     if c['payload'].get('phase') == 'safeguard'],
                 'implementation_revisions': [{**pick(t, ('task_id','key','state','role','attempt')),
                     'feedback': compact(t['inputs']['context']['revision_feedback'])}
                     for t in tasks.values() if t['kind']=='llm' and
                     t.get('inputs', {}).get('context', {}).get('revision_feedback')],
                 'experiment_request_outcomes': [{**pick(t, ('task_id','key','state','error','outputs')),
                     'result': pick(t.get('result') or {}, ('status','cache_reused','evidence_ref','experiment_ref'))}
                     for t in tasks.values() if t['kind']=='experiment_request'],
                 'recorder': recorder, 'numeric': numeric, 'calls': calls,
                 'role_costs': [{'role': role, **call_metrics(rows)} for role,rows in role_calls.items()],
                 'all_agent_costs': call_metrics(calls),
                 'numeric_costs': {'worker_wall_seconds_sum': sum((x.get('resources')or{}).get('wall_seconds',0) for x in numeric),
                     'failed_worker_wall_seconds_sum': sum((x.get('resources')or{}).get('wall_seconds',0)
                                                           for x in numeric if x['status'] == 'failed'),
                     'closed_worker_interval_union_seconds': union_seconds(intervals),
                     'allocated_gpu_wall_seconds_sum': sum((x.get('resources')or{}).get('wall_seconds',0) *
                         len((x.get('resources')or{}).get('physical_gpus') or x.get('gpu_bindings') or []) for x in numeric),
                     'peak_numeric_workers': peak},
                 'model_routes': [{'role':k[0], 'requested_model':k[1], 'requested_effort':k[2],
                     'actual_model':k[3], 'actual_effort':k[4], 'count':v} for k,v in routes.items()],
                 'skill_catalog_deliveries': dict(catalog),
                 'skill_body_or_reference_reads': sum(x['state']=='completed' and
                     ('skills/' in str(x['arguments'])) for x in tool_reads),
                 'tool_counts': [{'tool':k[0], 'state':k[1], 'count':v} for k,v in sorted(tools.items())],
                 'read_ledger': [compact(x) for x in tool_reads],
                 'memory_retrievals': [compact(x) for x in memory_reads],
                 'memory_projections': memory_projections, 'solution_library_history': libraries,
                 'solution_considered': [e for e in j.events if e['type']=='solution_considered'],
                 'interventions': [pick(e, ('commit_sequence','type','reason','note','intervention'))
                                  for e in j.events if e['type']=='implementation_intervention'],
                 'failed_tasks': [compact(x) for x in failures],
                 'limitations': ['Read-only derived index; no additional workflow gates or scientific decisions.',
                     'Requested and actual model identities are separate; absent actual stays unknown.',
                     'Cached input and reasoning output are subfields; subscription monetary cost unknown.',
                     'Worker wall/GPU allocation and excluded budget waits have distinct accounting boundaries.',
                     'Machine checks require human review of candidate changes, rationale and feedback use.']}
        atomic_json(self.output, state)
        return state


def brief(state):
    return {**pick(state, ('observed_at', 'sequence', 'status')),
            'completed_rounds': [e['round'] for e in state['round_finishes']],
            'hypotheses': len(state['hypotheses']), 'selections': len(state['selections']),
            'recorder': len(state['recorder']),
            'numeric': [{**pick(n, ('execution_id', 'round', 'level', 'status', 'evidence_ref', 'binding_issues')),
                         'full_dual_task_observed': n.get('formal', {}).get('full_dual_task_observed')}
                        for n in state['numeric']],
            'active_seconds': state['budget']['elapsed_seconds'],
            'remaining_seconds': state['budget']['remaining_seconds'],
            'excluded_seconds': state['budget']['excluded_numeric_wait_seconds']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT / 'EXP/runs' / HERE.name)
    p.add_argument('--output', type=Path, default=HERE / 'round_audit_latest.json')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--interval', type=float, default=180)
    args = p.parse_args()
    audit = Audit(args.run.resolve(), args.output.resolve())
    signature = None
    while True:
        state = audit.snapshot()
        summary = brief(state)
        meaningful = {k:v for k,v in summary.items() if k not in
                      ('observed_at','sequence','active_seconds','remaining_seconds','excluded_seconds')}
        if digest(meaningful) != signature:
            print(json.dumps(summary, ensure_ascii=False, separators=(',', ':')), flush=True)
            signature = digest(meaningful)
        segment = state['budget'].get('active_segment')
        terminal = state['status'] in ('completed','budget_exhausted','failed','failed_recoverable',
            'blocked','interrupted','recovery_required','needs_clarification','storage_blocked')
        if not args.watch or (terminal and not (segment and process_alive(segment.get('owner')))):
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
