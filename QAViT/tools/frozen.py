"""Evidence-bound frozen package loading. No path guesses and no re-calibration."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import importlib
import ast
import hashlib
import sys
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


def bind_sources(journal, refs):
    bindings = []
    for ref in refs:
        evidence = journal.card(ref)
        p = evidence['payload']
        if evidence['kind'] != 'EvidenceCard' or evidence['author'] != 'program' or p['status'] != 'completed':
            continue
        if not p['observations'].get('reload', {}).get('host_verified'):
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
        journal.card(p['experiment_ref'])  # Verify the committed card bytes before using its index.
        # Evidence and solution IDs bind the same source era. A resumed host may
        # have changed even though this is still the same scientific run.
        source = SolutionLibrary(journal, {}, shared=False).source_record(journal.cards[p['experiment_ref']]['sequence'])
        binding = {'evidence_ref': ref, 'experiment_ref': p['experiment_ref'],
            'model_ref': p['provenance']['model_ref'], 'model_dir': str(directory), 'assets': assets,
            'candidate': deepcopy(approval['candidate']) if approval else None,
            **deepcopy(source),
            'source_cache': str(journal.root / 'solution_library' / 'source' / source['source_version'])}
        _materialize_source(binding)
        bindings.append(binding)
    return bindings


def bind_solutions(journal, refs, brief):
    """Resolve only the run's committed protocol catalog, never an agent-supplied path."""
    from core.solutions import SolutionLibrary, protocol_key
    rows = {r['id']: r for r in SolutionLibrary(journal, brief).available()}
    bindings = []
    if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
        raise ContractError('solution_refs must be a list of catalog solution IDs')
    for ref in dict.fromkeys(refs):
        row = rows.get(ref)
        if not row or row.get('status') != 'evaluated' or row.get('protocol_hash') != protocol_key(brief):
            raise ContractError('frozen solution must be an evaluated entry in this protocol catalog')
        origin = Path(row['run_dir'])
        evidence_path = origin / 'cards' / (hashlib.sha256(row['evidence_ref'].encode()).hexdigest() + '.json')
        evidence = read_json(evidence_path)
        validate_card(evidence)
        p = evidence['payload']
        if (evidence['id'] != row['evidence_ref'] or evidence['author'] != 'program'
                or p['status'] != 'completed' or p['measurement_source'] != 'real'
                or p['provenance'].get('model_ref') != ref
                or p['experiment_ref'] != row['experiment_ref']
                or p['provenance'].get('protocol_hash') != fingerprint(brief['evaluation_protocol'])
                or not p['observations'].get('reload', {}).get('host_verified')
                or not p['observations'].get('reload', {}).get('passed')):
            raise ContractError('historical solution evidence identity or reload receipt mismatch')
        for task, metric in (('classification', 'top1'), ('segmentation', 'mIoU')):
            result = p['observations'].get('evaluations', {}).get(task, {})
            if result.get('full_dataset') is not True or result.get('metrics', {}).get(metric) != row['metrics'].get(metric):
                raise ContractError('historical complete-task scores do not match source evidence')
        manifest = next((a for a in p['artifacts'] if Path(a['path']).name == 'manifest.json'
                         and Path(a['path']).parent.name == 'model'), None)
        if not manifest:
            raise ContractError('historical solution has no exported model package')
        directory = Path(manifest['path']).parent
        source_record = historical_source(row)
        source = source_record['source_snapshot']
        binding = {'solution_id': ref, 'evidence_ref': evidence['id'], 'experiment_ref': p['experiment_ref'],
                   'model_ref': ref, 'model_dir': str(directory),
                   'assets': [a for a in p['artifacts'] if directory in Path(a['path']).parents],
                   'candidate': deepcopy(row.get('code')), 'source_snapshot': deepcopy(source),
                   'source_version': source_record['source_version'], 'measurement_origin': 'historical',
                   'source_cache': str(journal.root / 'solution_library' / 'source' / source_record['source_version'])}
        # Validate and freeze source before accepting a request; failures do not recalibrate.
        _materialize_source(binding)
        bindings.append(binding)
    return bindings


def historical_source(row):
    """Backfill only from the original committed experiment's implementation era."""
    if row.get('source_snapshot') and row.get('source_version'):
        return {key: deepcopy(row[key]) for key in ('source_snapshot', 'source_version')}
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
            for k, v in body.items()}) != binding['source_version']:
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


def _historical_runtime(binding, backend, cfg):
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
    expected = deepcopy(manifest['identity'])
    host = deepcopy(backend.identity())
    old_config = deepcopy(expected['config'])
    old_config['device'] = host['config']['device']
    expected['config']['device'] = host['config']['device']
    # Only the numerical implementation differs: all model, data and protocol identity
    # fields must still match the current task before opening the historical runtime.
    for identity in (expected, host):
        identity.pop('agent_code_sha256', None)
    if expected != host:
        raise ContractError('historical package model/data identity does not match current task')
    historical = importlib.import_module(namespace + '.backends.server').ServerBackend(old_config)
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
        manifest_identity = read_json(Path(binding['model_dir']) / 'manifest.json')['identity']
        current_identity = backend.identity()
        numerical_hash = manifest_identity.get('agent_code_sha256')
        if not numerical_hash or numerical_hash != current_identity.get('agent_code_sha256'):
            namespace, loading_backend, loading_cfg, factory = _historical_runtime(binding, backend, cfg)
    if candidate:
        module = _candidate_module(candidate, namespace)
        factory = getattr(module, 'make_executor', factory)

    executor = factory(loading_backend, deepcopy(loading_cfg), FrozenData(data))
    model, manifest = executor.load(Path(binding['model_dir']))
    verify()
    loading_backend.reset()
    backend.reset()
    return model, {'evidence_ref': binding['evidence_ref'], 'model_ref': binding['model_ref'],
                   'loaded_without_prepare': True, 'manifest': manifest,
                   'solution_id': binding.get('solution_id'), 'source_version': binding.get('source_version'),
                   'measurement_origin': binding.get('measurement_origin', 'current_run'), 'new_measurement': False}
