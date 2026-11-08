"""Evidence-bound frozen package loading. No path guesses and no re-calibration."""
from copy import deepcopy
from pathlib import Path
import importlib.util
from core.cards import ContractError
from core.journal import file_sha256


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
        bindings.append({'evidence_ref': ref, 'experiment_ref': p['experiment_ref'],
            'model_ref': p['provenance']['model_ref'], 'model_dir': str(directory), 'assets': assets,
            'candidate': deepcopy(approval['candidate']) if approval else None})
    return bindings


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
    if candidate:
        module_spec = importlib.util.spec_from_file_location('qavit_frozen_' + candidate['source_sha256'], candidate['path'])
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        factory = getattr(module, 'make_executor', RTNExecutor)

    class FrozenData:
        def __getattr__(self, name):
            return getattr(data, name)
        def batches(self, split):
            raise ContractError('frozen load cannot read data or recalibrate')
    executor = factory(backend, deepcopy(cfg), FrozenData())
    model, manifest = executor.load(Path(binding['model_dir']))
    verify()
    backend.reset()
    return model, {'evidence_ref': binding['evidence_ref'], 'model_ref': binding['model_ref'],
                   'loaded_without_prepare': True, 'manifest': manifest}
