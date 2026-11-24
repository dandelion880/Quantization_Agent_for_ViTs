"""Evaluator-owned training views. Construct only after frozen package reload."""
from pathlib import Path
from core.cards import ContractError
from core.journal import file_sha256


def read_view(binding):
    path = Path(binding['path'])
    raw = path.read_bytes()
    import hashlib
    import json
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ContractError('development manifest identity changed')
    view = json.loads(raw)
    if view.get('schema') != 'qavit.development/1':
        raise ContractError('unknown development view schema')
    return view


class ClassificationDev:
    def __init__(self, backend, manifest, digest):
        adapter = backend.data_adapter
        rows = manifest['classification']
        if (adapter.class_to_idx != manifest['class_to_idx'] or adapter.PREPROCESS != manifest['preprocessing']):
            raise ContractError('development classifier mapping/preprocessing mismatch')
        excluded = {r['file_sha256'] for r in adapter.records()}
        self.labels = {}
        for row in rows:
            path = Path(row['path']).resolve()
            if (path.parent.parent.name != 'train' or row['dataset_split'] != 'train'
                    or row['sample_id'] != '/'.join(path.parts[-3:])
                    or adapter.class_to_idx.get(path.parent.name) != row['label']
                    or row['sample_id'] in self.labels or row['file_sha256'] in excluded):
                raise ContractError('invalid, duplicate or overlapping classification development row')
            excluded.add(row['file_sha256'])
            self.labels[row['sample_id']] = row['label']
        self.backend, self.rows = backend, {'dev': rows}
        self.identity = {'full_validation': False, 'development_sha256': digest,
                         'split_ids': {'dev': list(self.labels)}, 'counts': {'dev': len(rows)}}

    def ids(self, split):
        return [r['sample_id'] for r in self.rows[split]]


def segmentation_files(root, rows):
    annotation_dir = root / 'annotations/training'
    files = []
    for row in rows:
        name = row['name']
        path = root / 'images/training' / name
        if (Path(name).name != name or path.resolve().parent != (root/'images/training').resolve()
                or file_sha256(path) != row['file_sha256']
                or file_sha256(annotation_dir / (path.stem + '.png')) != row['annotation_sha256']):
            raise ContractError('invalid ADE training pair or digest')
        files.append(path)
    if not files or len(set(files)) != len(files):
        raise ContractError('empty or duplicated ADE development view')
    return files, annotation_dir


def stage_protocol(brief, execution, level):
    """Use the bound dev protocol only for a complete, frozen dual-task request."""
    protocol = brief['evaluation_protocol']
    rows = execution.get('evaluations', [])
    if not isinstance(rows, list) or any(not isinstance(r,dict) for r in rows):
        raise ContractError('evaluations must be a list of task specifications')
    if not any(r.get('split') in ('dev', 'ade_dev') for r in rows):
        if (level == 'L3' and brief.get('execution_guidance', {}).get('development') and rows
                and execution.get('service') != 'evaluate_artifact'):
            raise ContractError('official stage uses a previously frozen package via evaluate_artifact')
        return protocol
    config = brief.get('execution_guidance', {}).get('development')
    if (level != 'L3' or not config or execution.get('service') not in ('quantize_evaluate','evaluate_artifact')
            or len(rows) != 2 or {(r['kind'], r.get('split')) for r in rows} !=
            {('classification','dev'), ('ade_frozen','ade_dev')}):
        raise ContractError('development requires both bound L3 tasks on a frozen package')
    return {**protocol, 'full_dataset': False, 'final_training': False, 'reload_before_evaluation': True,
            'purpose': 'development', 'evaluations': config['evaluations'],
            'development_view': config['manifest']}


def prepare_with_calibration(run, backend, cfg, data, experiment, directory, approval):
    """Expose only the selected calibration files during independent-dev construction."""
    import sys
    from copy import copy, deepcopy
    from runtime.guards import with_preparation_reads
    from runtime.paths import QAVIT
    from tools.experiment_worker import data_view

    payload = deepcopy(experiment['payload'])
    spec = payload['execution']
    spec['result_protocol'].pop('development_view', None)
    spec.pop('ade_fp_reference', None)
    cfg = {k:v for k,v in cfg.items() if k not in ('_ade_fp_reference', '_development_view')}
    selection = spec.get('calibration')
    ids = set(data_view(data, selection, 'calibration').ids('calibration') if selection else [])
    rows = [r for r in data.rows['calibration'] if r['sample_id'] in ids]
    # Keep parameter identity intact while withholding the evaluator-owned splits.
    local_backend, local_data = copy(backend), copy(data)
    local_backend.rows = rows
    local_backend.data_adapter = copy(backend.data_adapter)
    if hasattr(local_backend.data_adapter, 'splits'):
        local_backend.data_adapter.splits = {'calibration':rows,'search':[],'final':[]}
    local_data.backend = local_backend
    local_data.rows = {'calibration':rows}
    local_data.labels = {sid:value for sid,value in data.labels.items() if sid in ids}
    local_data.identity = {**data.identity, 'split_ids':{'calibration':local_data.ids('calibration')},
                           'counts':{'calibration':len(rows)}, 'full_validation':False}
    # Selection already applied (including a nonzero offset). The public request
    # still documents the original selection for build_plan and its approval.
    # Exporters create model/ themselves and reject pre-existing destinations.
    # The job contains its frozen request and outputs, not evaluator input files.
    paths = [QAVIT, directory, Path(sys.base_prefix)/'lib']
    paths.extend(p for p in (Path('/usr/lib'), Path('/etc/ld.so.cache'),
                              QAVIT/'.runtime_pkgs/common', QAVIT/'.runtime_pkgs/classification') if p.exists())
    paths.extend(Path(r['path']) for r in rows if r.get('path'))
    # Permit the same frozen head/config used by the evaluator, never its image or label directories.
    paths.extend(Path(a['path']) for a in spec.get('backend', {}).get('segmentation_assets', []))
    if spec.get('backend', {}).get('segmentation_assets'):
        paths.append(QAVIT/'.runtime_pkgs/segmentation')
    if approval:
        paths.append(Path(approval['candidate']['path']))
    for binding in spec.get('frozen_sources', []):
        paths.extend(Path(a['path']) for a in binding.get('assets', []))
        if binding.get('candidate'):
            paths.append(Path(binding['candidate']['path']))
    return with_preparation_reads(lambda:run(local_backend, cfg, local_data, payload, True), paths)
