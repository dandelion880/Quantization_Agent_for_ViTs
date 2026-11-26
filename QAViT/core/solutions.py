"""Reusable complete solutions and independent retention fallback B.

Journal snapshots are the run's read view. Shared protocol buckets are atomic
and locked; source configurations remain available even after replacement.
"""
from copy import deepcopy
from pathlib import Path
import math
from core.ptq_protocol import current, protocol_of, reusable_execution, reusable_manifest
from core.cards import ContractError, fingerprint, strict_loads
from core.journal import atomic_json, read_json, file_lock, file_sha256
from runtime.paths import ROOT


def protocol_key(brief):
    return fingerprint({k: brief.get(k) for k in ('model', 'evaluation_protocol', 'quantization_contract', 'hard_constraints')})


CALIBRATION_BINDING_FIELDS = ('calibration_samples', 'data_path', 'data_sha256')


def prepared_compatibility(source, target, old_backend, new_backend):
    """Verify the narrow opt-in migration for a measured, frozen native baseline."""
    from tools.experiments import bind_backend
    from ptq_agent.backends.image_data import ImageManifest

    def comparable(brief, backend):
        model = deepcopy(brief['model'])
        binding = model.pop('binding')
        adapter = model['adapter']
        bound = bind_backend(adapter, segmentation=binding['segmentation'])
        if fingerprint(bound) != binding['sha256']:
            raise ContractError('prepared baseline comparison has a changed model binding')
        for key in CALIBRATION_BINDING_FIELDS:
            adapter.get('backend_overrides', {}).pop(key, None)
        if not adapter.get('backend_overrides'):
            adapter.pop('backend_overrides', None)
        model['fixed_assets'] = [a for a in bound['checkpoint_assets'] if a['path'] != backend['data_path']]
        model['segmentation_assets'] = bound.get('segmentation_assets')
        evaluation = deepcopy(brief['evaluation_protocol'])
        if evaluation.pop('calibration_samples') != backend['calibration_samples'] or (not current(protocol_of(brief)) and evaluation.get('final_training') is not False):
            raise ContractError('prepared baseline requires declared calibration counts and training-free final evaluation')
        hard = deepcopy(brief['hard_constraints'])
        hard.pop('training_allowed', None)
        return model, evaluation, hard, brief.get('quantization_contract')

    if comparable(source, old_backend) != comparable(target, new_backend):
        raise ContractError('prepared baseline model, quantization or final evaluation changed')
    ignored = {*CALIBRATION_BINDING_FIELDS, 'device', 'research_protocol_sha256'}
    if {k: v for k, v in old_backend.items() if k not in ignored} != {k: v for k, v in new_backend.items() if k not in ignored}:
        raise ContractError('prepared baseline backend changed outside calibration or device placement')
    old = ImageManifest(Path(old_backend['data_path']), old_backend['data_sha256'])
    new = ImageManifest(Path(new_backend['data_path']), new_backend['data_sha256'])
    count = old_backend['calibration_samples']
    if (not old.full_validation or not new.full_validation or old.class_to_idx != new.class_to_idx
            or old.PREPROCESS != new.PREPROCESS or old.split_sources != new.split_sources
            or any(old.splits[s] != new.splits[s] for s in ('search', 'final'))
            or len(old.splits['calibration']) != count or len(new.splits['calibration']) != new_backend['calibration_samples']
            or new.splits['calibration'][:count] != old.splits['calibration']):
        raise ContractError('prepared baseline requires an extension preserving calibration prefix and exact search/final views')
    return {'kind': 'prepared_calibration_extension', 'source_protocol_hash': protocol_key(source),
            'target_protocol_hash': protocol_key(target), 'source_calibration_samples': count,
            'target_calibration_samples': new_backend['calibration_samples'],
            'source_data_sha256': old_backend['data_sha256'], 'target_data_sha256': new_backend['data_sha256']}


def rank(rows, priority=("mIoU", "top1")):
    def key(row):
        m = row['metrics']
        return (row.get('status') == 'evaluated', *(m[k] if m.get(k) is not None else -math.inf for k in priority))
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

    def _bound_entries(self, path):
        saved = read_json(path)
        if saved.get('protocol_hash') != self.key or any(r.get('protocol_hash') != self.key for r in saved['entries']):
            raise ContractError('Top-3 与当前冻结模型或评测协议不同')
        return saved['entries']

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
            view = self.brief.get('execution_guidance', {}).get('development', {}).get('manifest', {})
            view_directory = self.root / 'development' / fingerprint(view.get('sha256'))
            prior_views = [read_json(path) for path in sorted(view_directory.glob('*.json'))] if view else []
            path = self.root / 'top3.json'
            if path.exists():
                entries = self._bound_entries(path)
            else:
                atomic_json(path, {'protocol_hash': self.key, 'entries': entries})
        self.journal.commit('solutions/initialize', events=[{'type': 'solution_library', 'entries': entries,
            'protocol_hash': self.key, 'reason': 'mature seeds; pending scores are not empirical rankings'},
            {'type': 'solution_archive', 'entries': archived, 'protocol_hash': self.key},
            *[{'type':'solution_development', 'entry':row} for row in prior_views
              if row.get('protocol_hash') == self.key and row.get('view_sha256') == view.get('sha256')]])
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
        development = {}
        for event in self.journal.events:
            if event['type'] == 'solution_development' and (sequence is None or event['commit_sequence'] <= sequence):
                entry = event['entry']
                development.setdefault(entry['id'], {})[entry['view_sha256']] = entry
        for ref, views in development.items():
            existing = next((row for row in rows if row['id'] == ref), None)
            if existing is None:
                existing = deepcopy(list(views.values())[-1])
                rows.append(existing)
            existing['development'] = {key: {k: row[k] for k in ('metrics','reference_metrics','evidence_ref','experiment_ref')}
                                       for key, row in views.items()}
        known = {r['id'] for r in rows}
        return rows + [r for r in self.seeds() if r['id'] not in known]

    def fallback(self, incumbent=None, sequence=None):
        """B retains measured outcomes; it never competes for an exploration slot."""
        rows = [r for r in self.snapshot(sequence) if r.get('status') == 'evaluated']
        candidates = [{ 'solution_id': r['id'], 'model_ref': r['id'], 'metrics': r['metrics'],
                        'evidence_refs': [r['evidence_ref']], 'run_dir': r.get('run_dir'),
                        'configuration_asset': str(self.journal.root / 'solution_library' / (fingerprint(r['id']) + '.json')),
                        'measurement_origin': r.get('baseline_origin') or ('current_run' if r.get('run_dir') == str(self.journal.root) else 'historical'),
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
        for event in self.journal.events:
            if event['type'] != 'solution_development':
                continue
            row = event['entry']
            if row.get('run_dir') == str(self.journal.root):
                target = self.root / 'development' / fingerprint(row['view_sha256']) / (fingerprint(row['evidence_ref']) + '.json')
                if not target.exists():
                    atomic_json(target, row)
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
        return [{**{k: v for k, v in row.items() if k in ('id', 'title', 'status', 'metrics', 'reference_metrics', 'development', 'evidence_ref', 'hypothesis_ref','comparison_key')},
                 'retention': 'top3' if row['id'] in active else ('archive' if row.get('status') in ('evaluated','development_evaluated') else 'seed'),
                 'configuration_asset': str(self.journal.root / 'solution_library' / (fingerprint(row['id']) + '.json')),
                 'measurement_origin': (row.get('baseline_origin') or ('current_run' if row.get('run_dir') == str(self.journal.root) else 'historical')) if row.get('status') in ('evaluated','development_evaluated') else 'unmeasured_seed',
                 'new_measurement': False}
                for row in self.available(sequence)]

    def _complete_metrics(self, p):
        ev = p['observations'].get('evaluations', {})
        metrics = {}
        tasks=(('classification','top1'),('segmentation','mIoU'))
        evaluation=self.brief.get("evaluation_protocol",{})
        if current(protocol_of(self.brief)):
            priority=evaluation.get("metric_priority",["mIoU","top1"])
            declared=evaluation.get("task_metrics",dict(tasks))
            if not isinstance(priority,(list,tuple)) or not priority or len(set(priority))!=len(priority) or not set(priority)<=set(declared.values()):
                raise ContractError("metric priority must name declared task measurements")
            tasks=tuple(declared.items())
        for task, metric in tasks:
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
        return metrics

    def comparison_conditions(self, payload, execution):
        """Use observed coverage and immutable task views for each scientific comparison."""
        if not current(protocol_of(self.brief)):
            return {}
        observations=payload["observations"]
        receipt=observations.get("static_w4a4", {})
        protocol=self.brief["evaluation_protocol"]
        tasks={}
        for name,value in observations.get("evaluations",{}).items():
            tasks[name]={k:value[k] for k in ("split","sample_count","full_dataset","development_sha256","head_sha256","view_sha256","evaluation_mode") if k in value}
        scope={k:receipt.get(k) for k in ("protocol","extension_dimensions","required_bits","unquantized_edges","task_modules","activation_calibration")}
        scope["additional_states"]={k:{field:v[field] for field in ("bits","high_precision_path","unit") if field in v}
                                    for k,v in receipt.get("boundaries",{}).items() if v.get("high_precision_path") or v.get("unit")}
        deployed=payload.get("provenance",{}).get("execution_facts",{}).get("deployed_stages",execution.get("plan",{}))
        scope["postfit"]=[{k:s[k] for k in ("kind","module","reduce_axes") if k in s} for s in deployed.get("postfit",())
                          if s["kind"] in ("channel_affine","qwt","bias_compensation")]
        scope["structure"]=deployed.get("structure",[])
        conditions={"objective":protocol.get("comparison_intent","matched_quantization_conditions"),"tasks":tasks,
                    "model_and_heads":self.brief["model"].get("binding",{}),"precision":scope,
                    "calibration_view":execution.get("calibration"),"calibration_tasks":execution.get("calibration_tasks"),"calibration_source":execution.get("calibration_generator",{}).get("variant","bound_data"),
                    "data_permissions":protocol.get("data_permissions"),"resource_budget":protocol.get("resource_budget"),
                    "precision_budget":(execution.get("precision_search") or execution.get("recipe",{}).get("allocation",{})).get("budgets")}
        return {"comparison_conditions":conditions,"comparison_key":fingerprint(conditions)}

    def _retain(self, candidate):
        with file_lock(self.root / 'library.lock', timeout=10):
            path = self.root / 'top3.json'
            current = self._bound_entries(path) if path.exists() else self.snapshot()
            group=candidate.get("comparison_key")
            if group:
                group_path=self.root / 'comparisons' / (group+'.json')
                peers=self._bound_entries(group_path) if group_path.exists() else []
                peers=[entry for entry in peers if entry["id"]!=candidate["id"]]
                priority=self.brief["evaluation_protocol"].get("metric_priority",("mIoU","top1"))
                atomic_json(group_path,{"protocol_hash":self.key,"entries":rank([*peers,candidate],priority)})
            compatible=all(r.get("comparison_key")==group for r in current if r.get("status")=="evaluated")
            if compatible and candidate['id'] not in {r['id'] for r in current}:
                current = rank([*current, candidate], self.brief.get("evaluation_protocol", {}).get("metric_priority", ("mIoU", "top1")))
            atomic_json(self.root / 'archive' / (fingerprint(candidate['evidence_ref']) + '.json'), candidate)
            atomic_json(path, {'protocol_hash': self.key, 'entries': current})
        return current

    def import_prepared(self, reference):
        """Accept a measured native baseline from an explicit, round-free preparation run."""
        from core.workflow import Workflow
        if (not isinstance(reference, dict) or not {'run_dir', 'evidence_ref'} <= set(reference)
                or set(reference) - {'run_dir', 'evidence_ref', 'compatibility'}
                or ('compatibility' in reference and reference['compatibility'] != 'prepared_calibration_extension')):
            raise ContractError('prepared_baseline requires run_dir and evidence_ref')
        key = 'solutions/prepared/' + fingerprint(reference)
        if any(e['type'] == 'prepared_baseline_imported' and e['reference'] == reference for e in self.journal.events):
            return
        origin = Workflow((ROOT / reference['run_dir']).resolve(), read_only=True)
        if (origin.config['purpose'] != 'baseline_measurement'
                or (origin.solutions.key != self.key and not reference.get('compatibility'))
                or self.brief['evaluation_protocol'].get('formal_incumbent_allowed') is False):
            raise ContractError('baseline preparation source or frozen protocol mismatch')
        evidence = origin.journal.card(reference['evidence_ref'])
        p = evidence['payload']
        experiment = origin.journal.card(p['experiment_ref'])
        execution = experiment['payload']['execution']
        if (experiment['payload']['level'] != 'L3' or experiment['author'] != 'program'
                or experiment['payload'].get('candidate_binding') or p['measurement_source'] != 'real'
                or execution['service'] != 'quantize_evaluate' or execution.get('requires_code') or not (execution.get('plan') or execution.get('recipe'))
                or not reusable_execution(execution)
                or execution.get('research_protocol_sha256') != origin.solutions.key):
            raise ContractError('prepared baseline needs a complete independently rebuildable native measurement')
        origin.broker.validate_evidence_for(experiment, evidence)
        reload = p['observations'].get('reload', {})
        if not (reload.get('passed') and reload.get('host_verified') and reload.get('evaluated_instance') == 'reloaded'):
            raise ContractError('prepared baseline lacks an independent reload')
        package = {Path(a['path']).name: a['sha256'] for a in p['artifacts'] if Path(a['path']).parent.name == 'model'}
        if set(package) != {'manifest.json', 'weights.pt'} or fingerprint(package) != p['provenance']['model_ref']:
            raise ContractError('prepared baseline evidence does not bind a complete native package')
        manifest = read_json(next(a['path'] for a in p['artifacts'] if Path(a['path']).name == 'manifest.json'
                                  and Path(a['path']).parent.name == 'model'))
        if (not reusable_manifest(manifest, protocol_of(execution))
                or p['provenance'].get('training_declared') != manifest.get('preparation_training')
                or p['provenance'].get('construction') != manifest.get('construction')):
            raise ContractError('prepared baseline has no verified training-free reconstruction')
        compatibility = None
        if origin.solutions.key != self.key:
            from tools.experiment_worker import backend_configuration
            target = backend_configuration({'backend': self.brief['model']['adapter']})['backend']
            compatibility = prepared_compatibility(origin.brief['payload'], self.brief,
                                                    manifest['identity']['config'], target)
            if len(manifest['calibration_ids']) != compatibility['source_calibration_samples']:
                raise ContractError('prepared baseline bundle calibration count disagrees with its origin')
        results = p['observations']['evaluations']
        entry = {'id': p['provenance']['model_ref'], 'title': origin.config['name'], 'status': 'evaluated',
            'baseline_origin': 'preparation', 'protocol_hash': self.key, 'metrics': self._complete_metrics(p),
            'reference_metrics': {k[9:]: v for row in results.values() for k, v in row['metrics'].items()
                                  if k in ('baseline_top1','baseline_top5','baseline_mIoU','baseline_mAcc','baseline_aAcc')},
            'configuration': deepcopy(origin.brief['payload']), 'execution': execution,
            'actual_quantization': p['observations']['coverage'], 'artifacts': p['artifacts'],
            'evidence_ref': evidence['id'], 'experiment_ref': experiment['id'], 'run_dir': str(origin.run_dir),
            **({'protocol_compatibility': compatibility} if compatibility else {}),
            **origin.solutions.source_record(origin.journal.cards[experiment['id']]['sequence'])}
        entry.update(self.comparison_conditions(p,execution))
        current = self._retain(entry)
        self.journal.commit(key, events=[{'type': 'prepared_baseline_imported', 'reference': reference},
            {'type': 'solution_archive', 'entries': [entry], 'protocol_hash': self.key},
            {'type': 'solution_library', 'entries': current, 'protocol_hash': self.key}])
        self.materialize()

    def consider_round(self, number, evidence, experiment, selection, *, validator):
        """Use the existing evidence validator; no extra LLM or accuracy threshold."""
        event_key = 'solutions/evidence/' + fingerprint(evidence['id'])
        if any(e.get('evidence_ref') == evidence['id'] and e['type'] == 'solution_considered' for e in self.journal.events):
            return
        p = evidence['payload']
        try:
            validator(experiment, evidence, code_artifact=p["provenance"].get("code_artifact"))
            dev = experiment['payload']['execution'].get('result_protocol', {}).get('purpose') == 'development'
            if (p['measurement_source'] != 'real' or p['status'] != 'completed' or experiment['payload']['level'] != 'L3'
                    or (not dev and self.brief['evaluation_protocol'].get('formal_incumbent_allowed') is False)):
                raise ContractError('not a valid real final evaluation')
            if dev:
                evaluations = p['observations']['evaluations']
                metrics = {key: evaluations[name]['metrics'][key] for name,key in
                           (('classification','top1'), ('segmentation','mIoU'))}
            else:
                metrics = self._complete_metrics(p)
            binding = experiment['payload'].get('candidate_binding', {})
            if selection is None and not dev:
                raise ContractError('official solution requires a frozen selection')
            selection_payload = selection['payload'] if selection else {}
            hypothesis_ref = binding.get('hypothesis_ref', selection_payload.get('selected_hypothesis_ref'))
            candidates = selection_payload.get('candidate_refs', {})
            slot = next((slot for slot, ref in candidates.items() if ref == hypothesis_ref), None)
            historical_unbound = (not binding and selection_payload.get('selection_stage') == 'gonly_preselection'
                                  and slot == selection['payload'].get('selected_slot')
                                  and hypothesis_ref == selection['payload'].get('selected_hypothesis_ref'))
            if candidates and not historical_unbound and (slot is None or binding.get('selection_ref') != selection['id']):
                raise ContractError('L3 evidence does not bind an exact synthesized candidate')
            slot = slot or selection_payload.get('selected_slot') or binding.get('slot')
            if not dev and (not slot or not hypothesis_ref):
                raise ContractError('L3 candidate identity missing')
            candidate = {'id': p['provenance']['model_ref'], 'title': (slot or 'Development') + (' round ' + str(number) if number is not None else ''),
                'status': 'evaluated', 'metrics': metrics, 'protocol_hash': self.key,
                'configuration': deepcopy(self.brief), 'execution': deepcopy(experiment['payload']['execution']),
                'actual_quantization': deepcopy(p['observations'].get('coverage', {})),
                'evidence_ref': evidence['id'], 'experiment_ref': experiment['id'], 'run_dir': str(self.journal.root),
                'selection_ref': selection['id'] if selection else binding.get('selection_ref'), 'hypothesis_ref': hypothesis_ref, 'slot': slot,
                **{k:v for k,v in (self.journal.card(hypothesis_ref)['payload'] if hypothesis_ref in self.journal.cards else {}).items()
                   if k in ('topic_ref','method_id','inspiration_id')},
                **self.source_record(self.journal.cards.get(experiment['id'], {}).get('sequence')),
                'artifacts': p['artifacts'], **self.comparison_conditions(p,experiment['payload']['execution'])}
            # Preserve exact approved code as part of the complete reusable configuration.
            for task in self.journal.events:
                if task['type'] == 'task_snapshot' and evidence['id'] in task['task'].get('outputs', []):
                    approval = task['task']['inputs'].get('approval')
                    if approval:
                        candidate['code'] = deepcopy(approval['candidate'])
            if dev:
                candidate.update(status='development_evaluated', view_sha256=evaluations['classification']['development_sha256'],
                    reference_metrics={key: evaluations[name]['metrics']['baseline_' + key] for name,key in
                                       (('classification','top1'), ('segmentation','mIoU'))})
                self.journal.commit(event_key, events=[{'type':'solution_development','entry':candidate},
                    {'type':'solution_considered','round':number,'selected':False,'evidence_ref':evidence['id'],
                     'purpose':'development'}])
                self.materialize()
                return
            current = self._retain(candidate)
            events = [{'type': 'solution_library', 'entries': current, 'protocol_hash': self.key},
                      {'type': 'solution_considered', 'round': number, 'evidence_ref': evidence['id'],
                       'selected': candidate['id'] in {r['id'] for r in current}, 'entry': candidate}]
        except ContractError as exc:
            events = [{'type': 'solution_considered', 'round': number, 'evidence_ref': evidence['id'],
                       'selected': False, 'reason': str(exc)}]
        self.journal.commit(event_key, events=events)
        self.materialize()
