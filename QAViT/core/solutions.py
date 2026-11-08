"""Three reusable complete solutions, separate from scientific Memory and match ranking.

Journal snapshots are the run's read view. Shared protocol buckets are atomic
and locked; source configurations remain available even after replacement.
"""
from copy import deepcopy
from pathlib import Path
import math
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, read_json, file_lock, file_sha256
from runtime.paths import ROOT


def protocol_key(brief):
    return fingerprint({k: brief.get(k) for k in ('model', 'evaluation_protocol', 'quantization_contract', 'hard_constraints')})


def rank(rows):
    def key(row):
        m = row['metrics']
        return (row.get('status') == 'evaluated', m.get('mIoU') if m.get('mIoU') is not None else -math.inf,
                m.get('top1') if m.get('top1') is not None else -math.inf)
    # Stable ties preserve the incumbent; provisional seeds have no fabricated scores.
    return sorted(rows, key=key, reverse=True)[:3]


class SolutionLibrary:
    def __init__(self, journal, brief, *, shared=True):
        self.journal, self.brief = journal, brief
        self.key = protocol_key(brief)
        self.root = (ROOT / 'EXP/solutions' if shared else journal.root / 'solutions') / self.key

    def snapshot(self, sequence=None):
        events = [e for e in self.journal.events if e['type'] == 'solution_library'
                  and (sequence is None or e['commit_sequence'] <= sequence)]
        return deepcopy(events[-1]['entries']) if events else []

    def source_record(self, sequence=None):
        path = self.journal.root / 'research_run.json'
        saved, field = read_json(path), 'source_snapshot'
        for event in self.journal.events:
            if event['type'] == 'implementation_intervention' and (sequence is None or event['commit_sequence'] <= sequence):
                path = self.journal.root / 'commits' / f"{event['commit_sequence']:09d}.json"
                commit = self.journal.commits[event['commit_sequence'] - 1]
                index = next(i for i, item in enumerate(commit['events']) if item['type'] == 'implementation_intervention')
                saved, field = event, f'events.{index}.source_snapshot'
        return {'source_version': saved['code_identity']['core_sha256'],
                'source_snapshot': {'path': str(path), 'sha256': file_sha256(path), 'field': field}}

    def initialize(self):
        if self.snapshot():
            self.materialize()
            return
        source = self.source_record()
        seeds = read_json(ROOT / 'QAViT/docs/solutions/seeds.json')['solutions']
        entries = [{**s, 'protocol_hash': self.key, 'configuration': deepcopy(self.brief),
                    **source,
                    'status': 'pending_evaluation', 'metrics': {'top1': None, 'mIoU': None}} for s in seeds]
        with file_lock(self.root / 'library.lock', timeout=10):
            path = self.root / 'top3.json'
            if path.exists():
                entries = read_json(path)['entries']
            else:
                atomic_json(path, {'protocol_hash': self.key, 'entries': entries})
        self.journal.commit('solutions/initialize', events=[{'type': 'solution_library', 'entries': entries,
            'protocol_hash': self.key, 'reason': 'mature seeds; pending scores are not empirical rankings'}])
        self.materialize()

    def materialize(self):
        for row in self.snapshot():
            target = self.journal.root / 'solution_library' / (fingerprint(row['id']) + '.json')
            body = deepcopy(row)
            if row.get('code'):
                c = row['code']
                source = Path(c['path'])
                if source.is_file() and file_sha256(source) == c['source_sha256']:
                    dest = target.with_name(c['source_sha256'] + '.py')
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(source.read_bytes())
                    body['reusable_source'] = {'path': str(dest), 'sha256': c['source_sha256']}
            atomic_json(target, body)

    def assets(self, sequence=None):
        assets = {}
        for row in self.snapshot(sequence):
            source = row.get('source_snapshot', {})
            if source and source['path'] not in assets and Path(source['path']).is_file() and file_sha256(source['path']) == source['sha256']:
                assets[source['path']] = source['sha256']
            target = self.journal.root / 'solution_library' / (fingerprint(row['id']) + '.json')
            if target.is_file():
                assets[str(target)] = file_sha256(target)
                source = read_json(target).get('reusable_source')
                if source:
                    assets[source['path']] = source['sha256']
        return assets

    def catalog(self, sequence=None):
        return [{**{k: v for k, v in row.items() if k not in ('configuration', 'source_snapshot', 'execution', 'artifacts', 'actual_quantization')},
                 'configuration_asset': str(self.journal.root / 'solution_library' / (fingerprint(row['id']) + '.json'))}
                for row in self.snapshot(sequence)]

    def consider_round(self, number, evidence, experiment, selection, *, validator):
        """Use the existing evidence validator; no extra LLM or accuracy threshold."""
        event_key = 'solutions/round/' + str(number)
        if any(e.get('round') == number and e['type'] == 'solution_considered' for e in self.journal.events):
            return
        p = evidence['payload']
        try:
            validator(experiment, evidence, code_artifact=p["provenance"].get("code_artifact"))
            if (p['measurement_source'] != 'real' or p['status'] != 'completed' or experiment['payload']['level'] != 'L3'
                    or self.brief['evaluation_protocol'].get('formal_incumbent_allowed') is False):
                raise ContractError('not a valid real final evaluation')
            ev = p['observations'].get('evaluations', {})
            metrics = {}
            for task, metric in (('classification', 'top1'), ('segmentation', 'mIoU')):
                row = ev.get(task, {})
                value = row.get('metrics', {}).get(metric)
                if row.get('full_dataset') is not True or type(value) not in (int, float) or not math.isfinite(value):
                    raise ContractError('dual full-task metrics pending')
                metrics[metric] = value
            if not p['provenance'].get('model_ref'):
                raise ContractError('missing frozen model identity')
            if self.brief.get('quantization_contract', {}).get('protocol'):
                if not p['observations'].get('static_w4a4', {}).get('frozen'):
                    raise ContractError('final static W4A4 receipt pending')
            candidate = {'id': p['provenance']['model_ref'], 'title': selection['payload']['selected_slot'] + ' round ' + str(number),
                'status': 'evaluated', 'metrics': metrics, 'protocol_hash': self.key,
                'configuration': deepcopy(self.brief), 'execution': deepcopy(experiment['payload']['execution']),
                'actual_quantization': deepcopy(p['observations'].get('coverage', {})),
                'evidence_ref': evidence['id'], 'experiment_ref': experiment['id'], 'run_dir': str(self.journal.root),
                'selection_ref': selection['id'], 'hypothesis_ref': selection['payload']['selected_hypothesis_ref'],
                **self.source_record(self.journal.cards.get(experiment['id'], {}).get('sequence')),
                'artifacts': p['artifacts']}
            # Preserve exact approved code as part of the complete reusable configuration.
            for task in self.journal.events:
                if task['type'] == 'task_snapshot' and evidence['id'] in task['task'].get('outputs', []):
                    approval = task['task']['inputs'].get('approval')
                    if approval:
                        candidate['code'] = deepcopy(approval['candidate'])
            with file_lock(self.root / 'library.lock', timeout=10):
                path = self.root / 'top3.json'
                current = read_json(path)['entries'] if path.exists() else self.snapshot()
                if candidate['id'] not in {r['id'] for r in current}:
                    current = rank([*current, candidate])
                atomic_json(self.root / 'archive' / (fingerprint(candidate['id']) + '.json'), candidate)
                atomic_json(path, {'protocol_hash': self.key, 'entries': current})
            events = [{'type': 'solution_library', 'entries': current, 'protocol_hash': self.key},
                      {'type': 'solution_considered', 'round': number, 'evidence_ref': evidence['id'],
                       'selected': candidate['id'] in {r['id'] for r in current}}]
        except ContractError as exc:
            events = [{'type': 'solution_considered', 'round': number, 'evidence_ref': evidence['id'],
                       'selected': False, 'reason': str(exc)}]
        self.journal.commit(event_key, events=events)
        self.materialize()
