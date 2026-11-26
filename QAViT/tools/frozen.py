"""Evidence-bound frozen package loading. No path guesses and no re-calibration."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import importlib
import ast
import hashlib
import sys
from core.ptq_protocol import current, protocol_of, reusable, reusable_execution, reusable_manifest
from core.cards import ContractError, fingerprint, validate_card
from core.journal import file_sha256, read_json, Journal


class FrozenData:
    """Expose immutable data metadata to loaders, never calibration/evaluation reads."""
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)

    def batches(self, split):
        raise ContractError('frozen load cannot read data or recalibrate')

    source_rows = batches


def completed_package(payload, experiment):
    """A diagnostic export alone cannot confer full-model W4A4 status."""
    obs = payload['observations']
    if (experiment['payload']['level'] != 'L3' or payload['measurement_source'] != 'real'
            or not reusable_execution(experiment['payload']['execution'])
            or not obs.get('static_w4a4', {}).get('frozen') or not obs.get('reload', {}).get('host_verified')):
        return {}
    if current(protocol_of(experiment['payload']['execution'])):
        ref=next((a for a in payload["artifacts"] if Path(a["path"]).name=="frozen_evaluation.json"),None)
        ev=obs.get("evaluations",{})
        if not ref or not ev or file_sha256(ref["path"])!=ref["sha256"]:return {}
        facts=read_json(ref["path"])
        if set(facts["tasks"])!=set(ev) or facts["protocol"]!=protocol_of(experiment["payload"]["execution"]):return {}
        return {"verified_evaluation":ref,"frozen_audit":obs["static_w4a4"]}
    reference = next((a for a in payload['artifacts'] if Path(a['path']).name == 'segmentation.json'), None)
    if not reference:
        return {}
    return {'verified_segmentation': reference, 'frozen_audit': obs['static_w4a4']}


def reload_complete(binding, backend, cfg, data):
    from ptq_agent.execution.evaluation import model_digest
    ref = binding.get('verified_evaluation') or binding.get('verified_segmentation')
    if not ref or file_sha256(ref['path']) != ref['sha256']:
        raise ContractError('missing verified complete candidate measurement')
    measured=read_json(ref["path"])
    if binding.get('verified_evaluation') and measured["package_manifest_sha256"]!=file_sha256(Path(binding["model_dir"])/"manifest.json"):
        raise ContractError("frozen task receipt and package bytes differ")
    expected = measured['backbone_hashes']
    model, receipt = load_frozen(binding, backend, cfg, data)
    if [model_digest(backend.model), model_digest(model)] != expected:
        raise ContractError('reloaded package differs from the original evaluated state')
    return model, receipt['manifest']


def original_package(payload, execution, journal):
    """An evaluation has a new receipt but retains the exact originally prepared loader."""
    binding = execution.get('evaluation_package')
    if execution.get('service') != 'evaluate_artifact' or not binding:
        return None
    if payload['provenance'].get('model_ref') != binding.get('model_ref') or not (binding.get('verified_segmentation') or binding.get('verified_evaluation')):
        raise ContractError('evaluation evidence differs from its bound complete frozen package')
    result = deepcopy(binding)
    if not result.get('model_identity') and result.get('candidate'):
        result['model_identity'] = package_identity(result, read_json(Path(result['model_dir']) / 'manifest.json'))
    result['source_cache'] = str(journal.root / 'solution_library' / 'source' / result['source_version'])
    _materialize_source(result)
    return result


def bind_sources(journal, refs, *, training_free_only=False, protocol=None):
    bindings = []
    for ref in refs:
        evidence = journal.card(ref)
        p = evidence['payload']
        if evidence['kind'] != 'EvidenceCard' or evidence['author'] != 'program' or p['status'] != 'completed':
            continue
        if training_free_only and not reusable(p['provenance'].get('training_declared'), p['provenance'].get('construction'), protocol):
            continue  # The metrics remain citable; the trained state cannot supply final parameters.
        verified = p['observations'].get('reload', {}).get('host_verified')
        experiment = journal.card(p['experiment_ref'])
        origin = original_package(p, experiment['payload']['execution'], journal) if verified else None
        if origin:
            bindings.append({**origin, 'evidence_ref':ref, 'measurement_origin':'frozen_package_reevaluation'})
            continue
        manifest = next((a for a in p['artifacts'] if Path(a['path']).name == 'manifest.json' and Path(a['path']).parent.name == 'model'), None)
        if not manifest:
            continue
        directory = Path(manifest['path']).parent
        assets = [a for a in p['artifacts'] if directory in Path(a['path']).parents]
        approval = next((e['task']['inputs'].get('approval') for e in reversed(journal.events)
            if e['type'] == 'task_snapshot' and e['task']['kind'] == 'experiment'
            and e['task']['inputs'].get('experiment_ref') == p['experiment_ref']
            and ref in e['task']['outputs']), None)
        from core.solutions import SolutionLibrary
        experiment = journal.card(p['experiment_ref'])  # Verify committed bytes before using its index.
        source = SolutionLibrary(journal, {}, shared=False).source_record(journal.cards[p['experiment_ref']]['sequence'])
        if not verified:
            # A real custom L1/L2 measurement can export a native diagnostic
            # package. Bind only its exactly identified main package, and let
            # the worker's existing native loader verify it before any use.
            if not native_diagnostic_package(p, experiment, approval, source['source_version'], directory, assets):
                continue
        # Evidence and solution IDs bind the same source era. A resumed host may
        # have changed even though this is still the same scientific run.
        binding = {'evidence_ref': ref, 'experiment_ref': p['experiment_ref'],
            'model_ref': p['provenance']['model_ref'], **({'model_identity': deepcopy(p['provenance']['model_identity'])} if p['provenance'].get('model_identity') else {}),
            'model_dir': str(directory), 'assets': assets,
            'candidate': deepcopy(approval['candidate']) if approval and verified else None,
            **({'source_reload': 'native_loader_required'} if not verified else {}),
            **completed_package(p, experiment),
            **({'verified_increment': p['observations']['refinement']['child_state_sha256']}
               if verified and p['measurement_source'] == 'real' and experiment['payload']['execution'].get('parent_ref')
               and reusable_execution(experiment['payload']['execution']) and
               p['observations'].get('refinement', {}).get('initialization') == 'verified_parent' else {}),
            **deepcopy(source),
            'source_cache': str(journal.root / 'solution_library' / 'source' / source['source_version'])}
        _materialize_source(binding)
        bindings.append(binding)
    return bindings


def native_diagnostic_package(evidence, experiment, approval, origin_core, directory, assets):
    payload = experiment['payload']
    provenance = evidence['provenance']
    if (payload['level'] not in ('L1', 'L2') or payload['execution']['service'] != 'custom_experiment'
            or evidence['measurement_source'] != 'real' or not reusable_execution(payload['execution'])
            or not approval or approval.get('status') != 'approved'):
        return False
    from tools.safeguards import verify_artifact
    if (approval.get('host_implementation_sha256') != origin_core
            or approval.get('experiment_ref') != experiment['id']
            or approval.get('experiment_sha256') != experiment['content_sha256']
            or approval.get('approval_sha256') != fingerprint({k: v for k, v in approval.items() if k != 'approval_sha256'})):
        raise ContractError('native diagnostic package does not match its historical execution approval')
    verify_artifact(approval['candidate'], experiment)
    files = {Path(row['path']).name: row['sha256'] for row in assets}
    if (len(assets) != 2 or set(files) != {'manifest.json', 'weights.pt'}
            or fingerprint(files) != provenance.get('model_ref')):
        return False
    if any(file_sha256(directory / name) != digest for name, digest in files.items()):
        raise ContractError('native diagnostic package changed after its Evidence was committed')
    manifest = read_json(directory / 'manifest.json')
    return (manifest.get('schema_version') == 2 and manifest.get('plan', {}).get('schema_version') == 3
            and manifest.get('identity') == provenance.get('model_identity')
            and (manifest.get('preparation_training') is None and provenance.get('training_declared') is False
                 or reusable_manifest(manifest, protocol_of(payload['execution'])))
            and manifest.get('weights_sha256') == files['weights.pt']
            and manifest.get('content_sha256') == fingerprint({k: v for k, v in manifest.items() if k != 'content_sha256'}))


def bind_solutions(journal, refs, brief):
    """Resolve only the run's committed protocol catalog, never an agent-supplied path."""
    from core.solutions import SolutionLibrary, prepared_compatibility, protocol_key
    rows = {r['id']: r for r in SolutionLibrary(journal, brief).available()}
    bindings = []
    if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
        raise ContractError('solution_refs must be a list of catalog solution IDs')
    for ref in dict.fromkeys(refs):
        row = rows.get(ref)
        if not row or row.get('status') not in ('evaluated','development_evaluated') or row.get('protocol_hash') != protocol_key(brief):
            raise ContractError('frozen solution must be an evaluated entry in this protocol catalog')
        origin = Path(row['run_dir'])
        evidence_path = origin / 'cards' / (hashlib.sha256(row['evidence_ref'].encode()).hexdigest() + '.json')
        evidence = read_json(evidence_path)
        validate_card(evidence)
        p = evidence['payload']
        compatibility = row.get('protocol_compatibility')
        source_brief = row['configuration'] if compatibility else brief
        if (evidence['id'] != row['evidence_ref'] or evidence['author'] != 'program'
                or p['status'] != 'completed' or p['measurement_source'] != 'real'
                or p['provenance'].get('model_ref') != ref
                or p['experiment_ref'] != row['experiment_ref']
                or p['provenance'].get('protocol_hash') != (row['execution']['protocol_hash'] if row.get('status') == 'development_evaluated' else fingerprint(source_brief['evaluation_protocol']))
                or not p['observations'].get('reload', {}).get('host_verified')
                or not p['observations'].get('reload', {}).get('passed')):
            raise ContractError('historical solution evidence identity or reload receipt mismatch')
        for task, metric in (('classification', 'top1'), ('segmentation', 'mIoU')):
            result = p['observations'].get('evaluations', {}).get(task, {})
            complete = (result.get('complete_view') and result.get('development_sha256') == row.get('view_sha256')
                        if row.get('status') == 'development_evaluated' else result.get('full_dataset'))
            if complete is not True or result.get('metrics', {}).get(metric) != row['metrics'].get(metric):
                raise ContractError('historical complete-task scores do not match source evidence')
        inherited = original_package(p, row['execution'], journal)
        if inherited:
            bindings.append({**inherited, 'evidence_ref': evidence['id'], 'solution_id': ref})
            continue
        manifest = next((a for a in p['artifacts'] if Path(a['path']).name == 'manifest.json'
                         and Path(a['path']).parent.name == 'model'), None)
        if not manifest:
            raise ContractError('historical solution has no exported model package')
        if compatibility:
            from tools.experiment_worker import backend_configuration
            if row.get('baseline_origin') != 'preparation' or file_sha256(manifest['path']) != manifest['sha256']:
                raise ContractError('prepared baseline compatibility lacks its original manifest')
            old_backend = read_json(manifest['path'])['identity']['config']
            new_backend = backend_configuration({'backend': brief['model']['adapter']})['backend']
            if compatibility != prepared_compatibility(source_brief, brief, old_backend, new_backend):
                raise ContractError('prepared baseline compatibility receipt does not match the current task')
        directory = Path(manifest['path']).parent
        source_record = historical_source(row)
        source = source_record['source_snapshot']
        binding = {'solution_id': ref, 'evidence_ref': evidence['id'], 'experiment_ref': p['experiment_ref'],
                   'model_ref': ref, **({'model_identity': deepcopy(p['provenance']['model_identity'])} if p['provenance'].get('model_identity') else {}), 'model_dir': str(directory),
                   'assets': [a for a in p['artifacts'] if directory in Path(a['path']).parents],
                   'candidate': deepcopy(row.get('code')), 'source_snapshot': deepcopy(source),
                   'source_version': source_record['source_version'],
                   **{k: source_record[k] for k in ('source_archive_sha256',) if k in source_record},
                   'measurement_origin': row.get('baseline_origin', 'historical'),
                   **({'protocol_compatibility': compatibility} if compatibility else {}),
                   'source_cache': str(journal.root / 'solution_library' / 'source' / source_record['source_version'])}
        source_experiment = read_json(origin / 'cards' / (hashlib.sha256(p['experiment_ref'].encode()).hexdigest() + '.json'))
        validate_card(source_experiment)
        binding.update(completed_package(p, source_experiment))
        # Validate and freeze source before accepting a request; failures do not recalibrate.
        _materialize_source(binding)
        bindings.append(binding)
    return bindings


def historical_source(row):
    """Backfill only from the original committed experiment's implementation era."""
    if row.get('source_snapshot') and row.get('source_version'):
        return {key: deepcopy(row[key]) for key in ('source_snapshot', 'source_version', 'source_archive_sha256') if key in row}
    from core.solutions import SolutionLibrary
    origin = Path(row['run_dir'])
    metadata = read_json(origin / 'research_run.json')
    original = Journal(origin, metadata['run_id'])  # verifies the existing chain, no commits/writes
    experiment = original.cards.get(row['experiment_ref'])
    if not experiment or experiment['kind'] != 'ExperimentCard':
        raise ContractError('historical source backfill needs the original committed experiment')
    record = SolutionLibrary(original, row['configuration'], shared=False).source_record(experiment['sequence'])
    if row.get('source_version') and row['source_version'] != record['source_version']:
        raise ContractError('historical declared source version disagrees with committed implementation era')
    return record


def _source_files(binding):
    record = binding['source_snapshot']
    path = Path(record['path'])
    if not path.is_file() or file_sha256(path) != record['sha256']:
        raise ContractError('historical source snapshot changed or missing')
    body = read_json(path)
    for key in record['field'].split('.'):
        body = body[int(key)] if isinstance(body, list) else body[key]
    if not isinstance(body, dict) or fingerprint({k: hashlib.sha256(v.encode()).hexdigest()
            for k, v in body.items()}) != binding.get('source_archive_sha256', binding['source_version']):
        raise ContractError('historical source version mismatch')
    return body


def _materialize_source(binding, *, create=True):
    """Host creates the cache before Landlock; workers only verify/read it."""
    source = _source_files(binding)
    root = Path(binding['source_cache'])
    files = {k[len('tools/ptq_agent/'):]: v for k, v in source.items() if k.startswith('tools/ptq_agent/')}
    if not files:
        raise ContractError('historical numerical source is missing')
    for name, content in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ContractError('invalid historical source path')
        target = root / 'ptq_agent' / relative
        expected = hashlib.sha256(content.encode()).hexdigest()
        if target.exists() and file_sha256(target) != expected:
            raise ContractError('materialized historical source changed')
        if not target.exists():
            if not create:
                raise ContractError('historical source cache was not prepared by host')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
    actual = {str(p.relative_to(root / 'ptq_agent')) for p in (root / 'ptq_agent').rglob('*.py')}
    if actual != {n for n in files if n.endswith('.py')}:
        raise ContractError('historical numerical source has unexpected files')
    loader = root / 'vision_models.py'
    loader_content = source.get('tools/vision_models.py')
    if loader_content is None:
        raise ContractError('historical model loader source is missing')
    if loader.exists() and file_sha256(loader) != hashlib.sha256(loader_content.encode()).hexdigest():
        raise ContractError('historical model loader source changed')
    if not loader.exists():
        if not create:
            raise ContractError('historical loader cache was not prepared by host')
        loader.write_text(loader_content, encoding='utf-8')
    return root


def _historical_runtime(binding, backend, cfg, identity):
    """Load matching immutable numerical source in a private package namespace."""
    root = _materialize_source(binding, create=False)
    namespace = 'qavit_history_' + binding['source_version']
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(namespace, root / 'ptq_agent' / '__init__.py',
                            submodule_search_locations=[str(root / 'ptq_agent')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
    manifest = read_json(Path(binding['model_dir']) / 'manifest.json')
    # Rebuild the old backend from the declared model/assets, not a fabricated identity.
    expected = deepcopy(identity)
    host = deepcopy(backend.identity())
    old_config = deepcopy(expected['config'])
    old_config['device'] = host['config']['device']
    expected['config']['device'] = host['config']['device']
    # Only the numerical implementation differs: all model, data and protocol identity
    # fields must still match the current task before opening the historical runtime.
    # Loader and data adapter are implementation bytes too. Verify them against
    # the precise archived files before comparing scientific identity across eras.
    historical = importlib.import_module(namespace + '.backends.server').ServerBackend(old_config)
    if historical.identity() != expected:
        raise ContractError('historical loader did not reconstruct the original model/data identity')
    for identity in (expected, host):
        identity.pop('agent_code_sha256', None)
        identity.pop('loader_sha256', None)
        identity.get('data', {}).pop('adapter_sha256', None)
    compatibility = binding.get('protocol_compatibility')
    if compatibility:
        from core.solutions import CALIBRATION_BINDING_FIELDS
        for identity, side in ((expected, 'source'), (host, 'target')):
            config = identity['config']
            if (config.pop('research_protocol_sha256', None) != compatibility[side + '_protocol_hash']
                    or config['calibration_samples'] != compatibility[side + '_calibration_samples']
                    or identity['data'].pop('manifest_sha256') != compatibility[side + '_data_sha256']):
                raise ContractError('prepared baseline reload does not match the bound source and target protocols')
            for key in CALIBRATION_BINDING_FIELDS:
                config.pop(key)
    if expected != host or historical.original_state != backend.original_state:
        raise ContractError('historical package model/data identity does not match current task')
    config = deepcopy(manifest.get('runtime_config', cfg))
    config['backend'] = old_config
    factory = importlib.import_module(namespace + '.execution.model').RTNExecutor
    return namespace, historical, config, factory


def _candidate_module(candidate, namespace=None):
    name = 'qavit_frozen_' + candidate['source_sha256'] + ('_' + namespace if namespace else '')
    spec = importlib.util.spec_from_file_location(name, candidate['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if namespace:
        # Redirect imports only; original approved code bytes and package bytes remain
        # hash-checked. Relative numerical imports keep the private package identity.
        tree = ast.parse(Path(candidate['path']).read_text(), filename=candidate['path'])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (node.module == 'ptq_agent' or node.module.startswith('ptq_agent.')):
                node.module = namespace + node.module[len('ptq_agent'):]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'ptq_agent' or alias.name.startswith('ptq_agent.'):
                        if not alias.asname and alias.name != 'ptq_agent':
                            raise ContractError('historical dotted imports need an explicit alias')
                        alias.asname = alias.asname or 'ptq_agent'
                        alias.name = namespace + alias.name[len('ptq_agent'):]
        exec(compile(tree, candidate['path'], 'exec'), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def package_identity(binding, manifest):
    """The host records the backend identity. The approved exporter owns its manifest schema."""
    if binding.get('model_identity'):
        return binding['model_identity']
    if isinstance(manifest.get('identity'), dict) and 'config' in manifest['identity']:
        return manifest['identity']
    # Legacy custom bundles have only an Evidence reference. Require the original
    # host-produced package and matching artifact receipts before taking its identity.
    directory = Path(binding['model_dir'])
    if not binding.get('candidate') or directory.parent.parent.name != 'jobs':
        raise ContractError('frozen package lacks its bound backend identity')
    origin = directory.parent.parent.parent
    ref = binding['evidence_ref']
    card = validate_card(read_json(origin / 'cards' / (hashlib.sha256(ref.encode()).hexdigest() + '.json')))
    payload = card['payload']
    if (card['kind'] != 'EvidenceCard' or card['author'] != 'program' or card['id'] != ref or payload['status'] != 'completed'
            or payload['provenance'].get('model_ref') != binding['model_ref']
            or payload['experiment_ref'] != binding['experiment_ref']
            or any(asset not in payload['artifacts'] for asset in binding['assets'])):
        raise ContractError('legacy frozen package differs from its source Evidence')
    return payload['provenance']['model_identity']


def load_frozen(binding, backend, cfg, data):
    def verify():
        directory = Path(binding['model_dir'])
        expected = {str(Path(a['path']).resolve()): a['sha256'] for a in binding['assets']}
        actual = {str(p.resolve()): file_sha256(p) for p in directory.rglob('*') if p.is_file()}
        if actual != expected:
            raise ContractError('frozen Evidence package changed or incomplete')
        c = binding.get('candidate')
        if c and file_sha256(c['path']) != c['source_sha256']:
            raise ContractError('frozen loader source changed')
    verify()
    from ptq_agent.execution.model import RTNExecutor
    factory = RTNExecutor
    candidate = binding.get('candidate')
    namespace, loading_backend, loading_cfg = None, backend, cfg
    if binding.get('source_snapshot'):
        _source_files(binding)  # Always verify the declared original source, even on same-version loads.
        manifest_identity = package_identity(binding, read_json(Path(binding['model_dir']) / 'manifest.json'))
        current_identity = backend.identity()
        numerical_hash = manifest_identity.get('agent_code_sha256')
        if binding.get('protocol_compatibility') or not numerical_hash or numerical_hash != current_identity.get('agent_code_sha256'):
            namespace, loading_backend, loading_cfg, factory = _historical_runtime(binding, backend, cfg, manifest_identity)
        else:
            expected = deepcopy(manifest_identity)
            expected['config']['device'] = current_identity['config']['device']
            if expected != current_identity:
                raise ContractError('frozen source model/data identity differs from current backend')
    if candidate:
        module = _candidate_module(candidate, namespace)
        factory = getattr(module, 'make_executor', factory)

    if binding.get('protocol_compatibility'):
        from ptq_agent.execution.evaluation import VerifiedManifestData
        data = VerifiedManifestData(loading_backend, loading_cfg)
    executor = factory(loading_backend, deepcopy(loading_cfg), FrozenData(data))
    model, manifest = executor.load(Path(binding['model_dir']))
    verify()
    loading_backend.reset()
    backend.reset()
    return model, {'evidence_ref': binding['evidence_ref'], 'model_ref': binding['model_ref'],
                   'loaded_without_prepare': True, 'manifest': manifest,
                   'solution_id': binding.get('solution_id'), 'source_version': binding.get('source_version'),
                   **({'source_reload': binding['source_reload']} if 'source_reload' in binding else {}),
                   **({'protocol_compatibility': binding['protocol_compatibility']} if binding.get('protocol_compatibility') else {}),
                   'measurement_origin': binding.get('measurement_origin', 'current_run'), 'new_measurement': False}


def bind_parent_initialization(journal, payload):
    """Only a host-scored calibration receipt can be accepted by the built-in path."""
    from tools.runtime_interface import select_frozen_source
    execution = payload['execution']
    execution.pop('accepted_receipt', None)
    execution.pop('scored_calibration_ids', None)
    if 'parent_ref' not in execution:
        if 'accept_trial' in execution:
            raise ContractError('accept_trial requires an explicit parent_ref')
        return
    parent = select_frozen_source(execution.get('frozen_sources', []), execution['parent_ref'])
    if not (parent.get('verified_segmentation') or parent.get('verified_evaluation') or parent.get('verified_increment')):
        raise ContractError('parent initialization requires an evaluated complete or verified incremental package')
    if 'accept_trial' not in execution:
        return
    choice = execution['accept_trial']
    ref = choice['evidence_ref']
    if ref not in payload.get('evidence_refs', []):
        raise ContractError('cite the scored Evidence for the accepted candidate')
    card = journal.card(ref)
    p = card['payload']
    source = journal.card(p['experiment_ref'])['payload']['execution']
    report = p['observations'].get('result', {})
    if (card['kind'] != 'EvidenceCard' or card['author'] != 'program' or p['status'] != 'completed'
            or p['measurement_source'] != 'real' or p['provenance'].get('training_declared') is not False
            or source.get('service') != 'frozen_helper' or source.get('tool') != 'score_frozen_candidates'
            or report.get('api') != 'frozen_candidate_scores/1' or report['split'] != 'calibration'
            or report['parent_model_ref'] != parent['model_ref']):
        raise ContractError('accepted candidate lacks paired training-free calibration Evidence on this parent')
    row = next((r for r in report['candidates'] if r['id'] == choice['candidate_id']), None)
    if not row:
        raise ContractError('candidate_id was not measured in the cited pool')
    if row['receipt'].get('origin') == 'complete_frozen_package':
        raise ContractError('inherit the scored complete candidate using its source_ref as parent_ref')
    execution['accepted_receipt'] = deepcopy(row['receipt'])
    execution['scored_calibration_ids'] = list(report['sample_ids'])


def load_refinement_parent(binding, backend, cfg, data, *, protocol=None):
    from ptq_agent.execution.frozen_trials import state_identity
    if binding.get('verified_segmentation') or binding.get('verified_evaluation'):
        model, manifest = reload_complete(binding, backend, cfg, data)
    else:
        model, receipt = load_frozen(binding, backend, cfg, data)
        manifest = receipt['manifest']
        if not binding.get('verified_increment') or state_identity(model) != binding['verified_increment']:
            raise ContractError('cold incremental parent differs from its host-verified receipt')
    if not reusable_manifest(manifest, protocol):
        raise ContractError('parent refinement requires verified preparation compatible with the target protocol')
    return model, manifest


def native_parent(backend, cfg, data, model, manifest):
    """Rebind audited v3 native classes across source namespaces, without refitting any grid."""
    from ptq_agent.execution.calibration_search import _quantizers, state_identity
    from ptq_agent.execution.pipeline import QuantizationExecutor, build_model, quantizers, bind_operators, PLUGIN_VERSION
    plan = manifest['plan']
    if manifest.get('implementation') != PLUGIN_VERSION or plan['schema_version'] != 3:
        raise ContractError('custom parent format needs its approved exporter and prepare_from_parent')
    executor = QuantizationExecutor(backend, cfg, data)
    executor.input_shapes = deepcopy(manifest['input_shapes'])
    if not quantizers(model):
        old = _quantizers(model)
        native = build_model(backend, plan,
            {p['module']: {'scale': old[p['module']].input_scale, 'shift': old[p['module']].input_shift}
             for p in plan['preprocess']},
            {q.layer['module']: q.weight_record for q in old.values() if q.layer['weight']},
            {k:q.params() for k,q in old.items() if q.layer['activation']})
        if plan.get("structure"):
            from ptq_agent.execution.structure import configure
            from ptq_agent.execution.window_routes import activate
            # Verify the historical hooks with their original module identities.
            prefix=type(next(iter(old.values()))).__module__.rsplit(".execution.",1)[0]
            historical_structure=importlib.import_module(prefix+".execution.structure")
            configure(native,plan["structure"],historical_structure.fitted_state(model,plan["structure"]))
            activate(native,plan["structure"])
        qmap = quantizers(native)
        for sid, q in qmap.items():
            q.enabled = old[sid].enabled
            if getattr(old[sid],'grid_edit_from_packed_state',False):q.grid_edit_from_packed_state=True
            if hasattr(old[sid],"fitted_input_bias"):
                q.register_buffer("fitted_input_bias",old[sid].fitted_input_bias.detach().clone())
            if hasattr(q, 'weight_enabled'):
                q.weight_enabled = old[sid].weight_enabled
        bind_operators(native, plan, qmap)
        from ptq_agent.execution.output_correction import install
        for name, module in model.named_modules():
            correction = getattr(module, "ptq_output_correction", None)
            if correction is not None:
                install(native,name,correction.kind,dict(correction.named_buffers()))
        if state_identity(native) != state_identity(model):
            raise ContractError('native exporter cannot represent this exact parent state')
        model = native
    executor.audit(model, plan)
    return executor, model
