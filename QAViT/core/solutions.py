"""Reusable complete solutions and independent retention fallback B.

Journal snapshots are the run's read view. Shared protocol buckets are atomic
and locked; source configurations remain available even after replacement.
"""
from copy import deepcopy
from pathlib import Path
import math
from core.cards import ContractError, fingerprint, strict_loads
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
                **({'source_archive_sha256': saved['code_identity']['archive_sha256']}
                   if saved['code_identity'].get('archive_sha256') else {}),
                'source_snapshot': {'path': str(path), 'sha256': file_sha256(path), 'field': field},
                'source_read': {'asset': str(path), 'snapshot_field': field}}

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
            archived = [read_json(p) for p in sorted((self.root / 'archive').glob('*.json'))]
            archived = [r for r in archived if r.get('protocol_hash') == self.key]
            path = self.root / 'top3.json'
            if path.exists():
                entries = read_json(path)['entries']
            else:
                atomic_json(path, {'protocol_hash': self.key, 'entries': entries})
        self.journal.commit('solutions/initialize', events=[{'type': 'solution_library', 'entries': entries,
            'protocol_hash': self.key, 'reason': 'mature seeds; pending scores are not empirical rankings'},
            {'type': 'solution_archive', 'entries': archived, 'protocol_hash': self.key}])
        self.materialize()

    def seeds(self):
        """Seeds remain discoverable even when measured entries fill Top-3."""
        if not hasattr(self, '_seeds'):
            initial = read_json(self.journal.root / 'research_run.json')
            raw = initial.get('source_snapshot', {}).get('docs/solutions/seeds.json')
            # Seed definitions belong to the run's initial snapshot, not whichever
            # engineering intervention happened most recently. Old incomplete
            # metadata does not justify inventing a source binding.
            source = self.source_record(sequence=0) if raw and initial.get('code_identity') else {}
            definitions = strict_loads(raw)['solutions'] if raw else read_json(ROOT / 'QAViT/docs/solutions/seeds.json')['solutions']
            self._seeds = [{**s, 'protocol_hash': self.key, 'configuration': deepcopy(self.brief), **source,
                            'status': 'pending_evaluation', 'metrics': {'top1': None, 'mIoU': None}}
                           for s in definitions]
        return deepcopy(self._seeds)

    def available(self, sequence=None):
        rows = self.snapshot(sequence)
        # Only committed run facts are visible. Never scan the mutable shared
        # bucket during a role read: that would leak future/other-run results.
        historical = {}
        for event in self.journal.events:
            if sequence is not None and event['commit_sequence'] > sequence:
                continue
            if event['type'] in ('solution_library', 'solution_archive'):
                for row in event['entries']:
                    historical[row['id']] = row
            elif event['type'] == 'solution_considered' and event.get('entry'):
                historical[event['entry']['id']] = event['entry']
        known = {r['id'] for r in rows}
        rows += [deepcopy(r) for k, r in historical.items() if k not in known]
        known = {r['id'] for r in rows}
        return rows + [r for r in self.seeds() if r['id'] not in known]

    def fallback(self, incumbent=None, sequence=None):
        """B retains measured outcomes; it never competes for an exploration slot."""
        rows = [r for r in self.snapshot(sequence) if r.get('status') == 'evaluated']
        candidates = [{ 'solution_id': r['id'], 'model_ref': r['id'], 'metrics': r['metrics'],
                        'evidence_refs': [r['evidence_ref']], 'run_dir': r.get('run_dir'),
                        'configuration_asset': str(self.journal.root / 'solution_library' / (fingerprint(r['id']) + '.json')),
                        'measurement_origin': 'current_run' if r.get('run_dir') == str(self.journal.root) else 'historical',
                        'new_measurement': False} for r in rows]
        if incumbent:
            candidates.append({**deepcopy(incumbent), 'metrics': {'top1': incumbent['top1'], 'mIoU': incumbent['mIoU']},
                               'measurement_origin': 'current_run', 'new_measurement': False})
        candidates = [r for r in candidates if all(type(r['metrics'].get(k)) in (int, float)
                      and math.isfinite(r['metrics'][k]) for k in ('top1', 'mIoU'))]
        best = max(candidates, key=lambda r: (r['metrics']['mIoU'], r['metrics']['top1'])) if candidates else None
        if best:
            best.pop('previous', None)
        return {'role': 'retained_verified_result', 'status': 'available' if best else 'pending_no_verified_result',
                'candidate': best, 'competes_in_preselection': False,
                'policy': 'Retain B while exploring H1/H2/H3; historical scores are not current measurements.'}

    def materialize(self):
        for row in self.available():
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
        for row in self.available(sequence):
            target = self.journal.root / 'solution_library' / (fingerprint(row['id']) + '.json')
            if target.is_file():
                assets[str(target)] = file_sha256(target)
                source = read_json(target).get('reusable_source')
                if source:
                    assets[source['path']] = source['sha256']
        return assets

    def catalog(self, sequence=None):
        active = {r['id'] for r in self.snapshot(sequence)}
        return [{**{k: v for k, v in row.items() if k in ('id', 'title', 'status', 'metrics', 'evidence_ref', 'hypothesis_ref')},
                 'retention': 'top3' if row['id'] in active else ('archive' if row.get('status') == 'evaluated' else 'seed'),
                 'configuration_asset': str(self.journal.root / 'solution_library' / (fingerprint(row['id']) + '.json')),
                 'measurement_origin': ('current_run' if row.get('run_dir') == str(self.journal.root) else 'historical') if row.get('status') == 'evaluated' else 'unmeasured_seed',
                 'new_measurement': False}
                for row in self.available(sequence)]

    def consider_round(self, number, evidence, experiment, selection, *, validator):
        """Use the existing evidence validator; no extra LLM or accuracy threshold."""
        event_key = 'solutions/evidence/' + fingerprint(evidence['id'])
        if any(e.get('evidence_ref') == evidence['id'] and e['type'] == 'solution_considered' for e in self.journal.events):
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
            binding = experiment['payload'].get('candidate_binding', {})
            hypothesis_ref = binding.get('hypothesis_ref', selection['payload'].get('selected_hypothesis_ref'))
            candidates = selection['payload'].get('candidate_refs', {})
            slot = next((slot for slot, ref in candidates.items() if ref == hypothesis_ref), None)
            historical_unbound = (not binding and selection['payload'].get('selection_stage') == 'gonly_preselection'
                                  and slot == selection['payload'].get('selected_slot')
                                  and hypothesis_ref == selection['payload'].get('selected_hypothesis_ref'))
            if candidates and not historical_unbound and (slot is None or binding.get('selection_ref') != selection['id']):
                raise ContractError('L3 evidence does not bind an exact synthesized candidate')
            slot = slot or selection['payload'].get('selected_slot')
            if not slot or not hypothesis_ref:
                raise ContractError('L3 candidate identity missing')
            candidate = {'id': p['provenance']['model_ref'], 'title': slot + ' round ' + str(number),
                'status': 'evaluated', 'metrics': metrics, 'protocol_hash': self.key,
                'configuration': deepcopy(self.brief), 'execution': deepcopy(experiment['payload']['execution']),
                'actual_quantization': deepcopy(p['observations'].get('coverage', {})),
                'evidence_ref': evidence['id'], 'experiment_ref': experiment['id'], 'run_dir': str(self.journal.root),
                'selection_ref': selection['id'], 'hypothesis_ref': hypothesis_ref, 'slot': slot,
                **{k:v for k,v in (self.journal.card(hypothesis_ref)['payload'] if hypothesis_ref in self.journal.cards else {}).items()
                   if k in ('topic_ref','method_id','inspiration_id')},
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
                atomic_json(self.root / 'archive' / (fingerprint(evidence['id']) + '.json'), candidate)
                atomic_json(path, {'protocol_hash': self.key, 'entries': current})
            events = [{'type': 'solution_library', 'entries': current, 'protocol_hash': self.key},
                      {'type': 'solution_considered', 'round': number, 'evidence_ref': evidence['id'],
                       'selected': candidate['id'] in {r['id'] for r in current}, 'entry': candidate}]
        except ContractError as exc:
            events = [{'type': 'solution_considered', 'round': number, 'evidence_ref': evidence['id'],
                       'selected': False, 'reason': str(exc)}]
        self.journal.commit(event_key, events=events)
        self.materialize()
