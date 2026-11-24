#!/usr/bin/env python3
"""Run-specific read-only observer; no LLM calls and no research-state writes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'QAViT'))
from core.journal import Journal, atomic_json, read_json
from runtime.budget import BudgetLedger
from runtime.task_queue import GlobalTaskQueue, process_alive

STOPPED = {'completed', 'budget_exhausted', 'budget_stopped', 'scope_target_reached',
           'blocked', 'failed', 'failed_recoverable', 'needs_clarification',
           'storage_blocked', 'interrupted', 'recovery_required'}
MILESTONES = {'round_started', 'round_finished', 'match_closed', 'round_selection',
              'solution_considered', 'implementation_intervention', 'run_status',
              'scientific_review_deferred', 'recorder_deferred', 'optional_analysis_issue',
              'generation_preselection', 'generation_selection', 'l3_deferred', 'solution_updated'}


def short(value, limit=350):
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return text[:limit]


def compact_task(task, now):
    value = {k: task.get(k) for k in ('task_id', 'key', 'kind', 'role', 'state', 'attempt', 'step')}
    value['age_seconds'] = round(max(0, now - task.get('started_unix', task['created_unix'])))
    if task.get('error'):
        value['error'] = short(task['error'])
    if task.get('allocated_gpus'):
        value['gpus'] = task['allocated_gpus']
    return value


def task_signature(task):
    return {k: task.get(k) for k in ('state', 'attempt', 'step', 'error', 'outputs')}


def gpu_snapshot():
    try:
        call = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
        if call.returncode:
            return {'error': short(call.stderr)}
        rows = []
        for line in call.stdout.splitlines():
            fields = [x.strip() for x in line.split(',')]
            if len(fields) == 5:
                rows.append(dict(zip(('index', 'uuid', 'memory_used_mib', 'memory_free_mib', 'utilization_pct'), fields)))
        return rows
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': short(str(exc))}


class Observer:
    def __init__(self, run):
        self.run = run
        # Guard Journal's existing-root constructor: never create a research run.
        if not run.is_dir() or not (run / 'research_run.json').is_file():
            raise FileNotFoundError('research run has not been created')
        self.journal = Journal(run, run.name)
        self.queue = GlobalTaskQueue(self.journal)
        self.budget = BudgetLedger(self.journal)
        prior = read_json(HERE / 'monitor_latest.json') if (HERE / 'monitor_latest.json').is_file() else {}
        self.prior_sequence = prior.get('sequence', 0) if prior.get('run') == str(run) else 0
        self.event_index = 0
        self.task_signatures = {}
        self.log_state = {}
        self.prior = prior

    def tail(self, path):
        try:
            stat = path.stat()
            known = self.log_state.get(str(path))
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if known and known[0] == signature:
                return known[1]
            with path.open('rb') as stream:
                stream.seek(max(0, stat.st_size - 4096))
                lines = stream.read().decode('utf-8', 'replace').splitlines()
            value = {'path': str(path.relative_to(self.run)), 'bytes': stat.st_size,
                     'modified_unix': stat.st_mtime, 'tail': [short(x, 240) for x in lines[-3:] if x.strip()]}
            self.log_state[str(path)] = (signature, value)
            return value
        except FileNotFoundError:
            return None

    def sample(self):
        now = time.time()
        self.journal.refresh()
        events = self.journal.events[self.event_index:]
        self.event_index = len(self.journal.events)
        milestones = [{k: v for k, v in e.items() if k in (
            'type', 'commit_sequence', 'round', 'pair', 'outcome', 'status', 'selected_slot',
            'selection_ref', 'scientific_review', 'selected', 'evidence_ref', 'error', 'reason')}
            for e in events if e['type'] in MILESTONES and e['commit_sequence'] > self.prior_sequence]
        new_task_ids = {e['task']['task_id'] for e in events
                        if e['type'] == 'task_snapshot' and e['commit_sequence'] > self.prior_sequence}
        tasks = self.queue.snapshot()
        budget = self.budget.snapshot()
        statuses = [e for e in self.journal.events if e['type'] == 'run_status']
        status = statuses[-1].get('status') if statuses else 'created'
        active, progress, changes, faults = [], [], [], []
        had_task_snapshot = bool(self.task_signatures)
        for task in tasks.values():
            signature = task_signature(task)
            if signature != self.task_signatures.get(task['task_id']):
                row = compact_task(task, now)
                # A restart summarizes current tasks; it does not replay completed history.
                if task['task_id'] in new_task_ids and (had_task_snapshot or task['state'] not in ('completed', 'cancelled', 'ineligible')):
                    changes.append({k: v for k, v in row.items() if k != 'age_seconds'})
                if task['task_id'] in new_task_ids and task['state'] in ('failed', 'needs_clarification'):
                    faults.append({k: v for k, v in row.items() if k != 'age_seconds'})
                self.task_signatures[task['task_id']] = signature
            if task['state'] in ('running', 'ready', 'retry', 'waiting', 'needs_clarification'):
                row = compact_task(task, now)
                if task['state'] == 'running':
                    row['worker_alive'] = process_alive(task.get('process'))
                    if task.get('request_path'):
                        directory = Path(task['request_path']).parent
                        log = self.tail(directory / 'worker.log')
                        if log:
                            progress.append({'task': task['key'], **log})
                        for name in ('ade_progress.json', 'evaluation_progress.json', 'calibration_progress.json'):
                            path = directory / name
                            if path.is_file():
                                try:
                                    progress.append({'task': task['key'], 'path': str(path.relative_to(self.run)),
                                                     'progress': read_json(path)})
                                except (OSError, ValueError):
                                    pass
                active.append(row)
        gpu = gpu_snapshot()
        storage = {name: shutil.disk_usage(path).free for name, path in (
            ('project', self.run), ('root', Path('/')), ('backup', Path('/root/data-tmp')))}
        limits = {'project': 1024**3, 'root': 128 * 1024**2}
        storage_alerts = [dict(volume=name, free_bytes=storage[name]) for name, limit in limits.items()
                          if storage[name] < limit and self.prior.get('storage', {}).get(name, limit) >= limit]
        state = {'timestamp': datetime.fromtimestamp(now, timezone.utc).isoformat(), 'sample_unix': now,
                 'monitor_pid': os.getpid(), 'run': str(self.run), 'sequence': self.journal.sequence,
                 'status': status, 'budget': {k: budget.get(k) for k in (
                     'elapsed_seconds', 'remaining_seconds', 'research_remaining_seconds',
                     'excluded_numeric_wait_seconds', 'deadline_unix', 'worker_deadline_unix', 'rounds_started')},
                 'completed_rounds': sorted({e['round'] for e in self.journal.events if e['type'] == 'round_finished'}),
                 'task_counts': dict(Counter(t['state'] for t in tasks.values())), 'active_tasks': active,
                 'recent_faults': (faults or self.prior.get('recent_faults', []))[-6:],
                 'gpu': gpu, 'progress': progress, 'new_milestones': milestones}
        state.update(storage=storage, storage_alerts=storage_alerts)
        if statuses and statuses[-1].get('error'):
            state['run_error'] = short(statuses[-1]['error'], 1000)
        segment = budget.get('active_segment')
        state['active_segment'] = segment.get('segment_id') if segment else None
        state['segment_owner_alive'] = process_alive(segment.get('owner')) if segment else None
        if status in STOPPED and (self.run / 'summary.json').is_file():
            report = read_json(self.run / 'summary.json')
            state['final_result'] = {k: report.get(k) for k in (
                'workflow_completed', 'completed_rounds', 'formal_validation_success',
                'numerical_validation_status', 'scientific_review_complete', 'state_saved', 'finalization_errors')}
        meaningful = bool(changes or milestones or state['status'] != self.prior.get('status')
                          or progress != self.prior.get('progress') or gpu != self.prior.get('gpu') or storage_alerts)
        if meaningful:
            delta = {'timestamp': state['timestamp'], 'sequence': state['sequence'], 'status': status,
                     'budget': state['budget'], 'task_counts': state['task_counts'],
                     'changed_tasks': changes, 'milestones': milestones}
            if storage_alerts:
                delta.update(storage=storage, storage_alerts=storage_alerts)
            if progress != self.prior.get('progress'):
                delta['progress'] = progress
            if gpu != self.prior.get('gpu'):
                delta['gpu'] = gpu
            with (HERE / 'monitor_changes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(delta, ensure_ascii=False, separators=(',', ':')) + '\n')
        atomic_json(HERE / 'monitor_latest.json', state)
        self.prior = state
        self.prior_sequence = self.journal.sequence
        return state, milestones, faults


def brief(state, milestones=(), faults=()):
    return {'timestamp': state['timestamp'], 'sequence': state['sequence'], 'status': state['status'],
            'active_seconds': round(state['budget']['elapsed_seconds'], 1),
            'excluded_seconds': round(state['budget']['excluded_numeric_wait_seconds'], 1),
            'remaining_seconds': round(state['budget']['remaining_seconds'], 1),
            'tasks': state['task_counts'], 'running': [t['key'] for t in state['active_tasks'] if t['state'] == 'running'],
            'milestones': milestones, 'faults': faults,
            'storage': state.get('storage'), 'storage_alerts': state.get('storage_alerts', [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'EXP/runs' / HERE.name)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='sample once and print a compact snapshot (default)')
    mode.add_argument('--watch', action='store_true', help='sample every interval until the research run stops')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    observer = None
    first = True
    while True:
        try:
            if observer is None:
                observer = Observer(args.run.resolve())
            state, milestones, faults = observer.sample()
            if first or not args.watch or milestones or faults or state.get('storage_alerts') or state['status'] in STOPPED:
                print(json.dumps(brief(state, milestones, faults), ensure_ascii=False, separators=(',', ':')), flush=True)
            first = False
            # Active segment cleanup may still be receiving final receipts. Wait for it.
            if not args.watch or (state['status'] in STOPPED and not state['segment_owner_alive']):
                break
        except Exception as exc:
            error = {'timestamp': datetime.now(timezone.utc).isoformat(), 'monitor_error': type(exc).__name__,
                     'message': short(str(exc), 600), 'run': str(args.run)}
            with (HERE / 'monitor_changes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(error, ensure_ascii=False) + '\n')
            print(json.dumps(error, ensure_ascii=False), flush=True)
            if not args.watch or not isinstance(exc, FileNotFoundError):
                raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
