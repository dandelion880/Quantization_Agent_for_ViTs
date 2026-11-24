"""Rebuild the exact reviewed role candidate from a Journal, or from its reviewed copy."""
from copy import deepcopy
import argparse
import importlib.util
import subprocess
import tempfile
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--archive-runtime', type=Path)
archived, _ = bootstrap.parse_known_args()
sys.path.insert(0, str(archived.archive_runtime or HERE.parents[2] / 'QAViT'))
from runtime.paths import QAVIT
from runtime.identity import code_identity
from core.solutions import SolutionLibrary
from tools.frozen import _source_files
from core.execution_contract import validate_plan_schema
from core.journal import Journal, read_json, atomic_json, file_sha256
from core.cards import fingerprint
from ptq_agent.execution.model import RTNExecutor, forward, tensor_digest
from ptq_agent.execution.evaluation import model_digest
from tools.experiment_worker import backend_and_data, data_view, module_from_approval
from tools.runtime_interface import ObservedData
from tools.vision_models import setup_segmentation


def rebuild(reviewed, payload, expected_plan_hash, package=None):
    spec = payload['execution']
    setup_segmentation(spec['seed'], 3)
    backend, cfg, data = backend_and_data(spec)
    plan = validate_plan_schema(reviewed.build_plan(
        backend, deepcopy(cfg), deepcopy(payload), deepcopy(spec.get('parameters', {}))))
    assert fingerprint(plan) == expected_plan_hash
    view = ObservedData(data_view(data, spec['calibration'], 'calibration'))
    executor = getattr(reviewed, 'make_executor', RTNExecutor)(backend, deepcopy(cfg), view)
    model, _, _, _ = executor.prepare(plan, {})
    state = model_digest(model)
    restored = None
    if package:
        restored, _ = executor.load(package)
        assert model_digest(restored) == state
    outputs = []
    with torch.inference_mode():
        for x, sid in data_view(data, spec['evaluations'][0]).batches('search'):
            logits = forward(backend, model, x)
            if restored is not None:
                assert torch.equal(logits, forward(backend, restored, x))
            outputs.append({'sample': sid, 'logits_sha256': tensor_digest(logits)})
    return {'plan_sha256': fingerprint(plan), 'layers': plan['layers'], 'actual_model_digest': state,
            'calibration_ids': list(dict.fromkeys(view.reads['calibration'])), 'search_outputs_bitwise_equal': outputs}


def from_journal(run):
    journal = Journal(run, read_json(run/'research_run.json')['run_id'])
    evidence = next(c for c in journal.find_cards('EvidenceCard') if c['payload']['status'] == 'completed'
                    and c['payload']['provenance'].get('model_ref'))
    p = evidence['payload']
    experiment = journal.card(p['experiment_ref'])
    approval = next(e['task']['inputs']['approval'] for e in reversed(journal.events)
                    if e['type'] == 'task_snapshot' and e['task']['kind'] == 'experiment'
                    and evidence['id'] in e['task']['outputs'])
    if approval['host_implementation_sha256'] != code_identity()['core_sha256']:
        if archived.archive_runtime:
            raise ValueError('archived host differs from the reviewed host')
        sequence = journal.cards[experiment['id']]['sequence']
        source = SolutionLibrary(journal, {}, shared=False).source_record(sequence)
        # Only the originally approved host may replay this historical operation.
        with tempfile.TemporaryDirectory(dir=HERE) as work:
            tree = Path(work)/'QAViT'
            for name, content in _source_files(source).items():
                path = (tree/name).resolve()
                if tree not in path.parents:
                    raise ValueError('invalid archive member')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            (tree/'.runtime_pkgs').symlink_to(QAVIT/'.runtime_pkgs', target_is_directory=True)
            subprocess.run([sys.executable, '-B', __file__, str(run), '--archive-runtime', str(tree)], check=True)
        return
    reviewed = module_from_approval(approval, experiment)
    package = next(Path(a['path']).parent for a in p['artifacts'] if a['path'].endswith('/model/manifest.json'))
    manifest = read_json(package/'manifest.json')
    record = rebuild(reviewed, experiment['payload'], fingerprint(manifest['plan']), package)
    assert record['calibration_ids'] == manifest['calibration_ids']
    atomic_json(run/'candidate_rebuild.json', {**record, 'experiment': experiment['id'], 'evidence': evidence['id'],
        'approved_source': approval['candidate'], 'model_ref': p['provenance']['model_ref'], 'package_path': str(package),
        'state_matches_export': True, 'manifest_sha256': file_sha256(package/'manifest.json')})


def from_reviewed_copy():
    saved = read_json(HERE/'RESULTS.json')['role_component']
    source = HERE/saved['approved_file']
    assert source.parent == HERE and file_sha256(source) == saved['source_sha256']
    entry = importlib.util.spec_from_file_location('verified_archived_candidate', source)
    reviewed = importlib.util.module_from_spec(entry)
    entry.loader.exec_module(reviewed)
    record = rebuild(reviewed, {'execution': saved['rebuild_execution']}, saved['plan_sha256'])
    assert all(record[key] == saved[key] for key in record)
    print('Reviewed source, canonical plan, calibration IDs, model state and both search logits match.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(parents=[bootstrap])
    ap.add_argument('run', nargs='?', help='existing validation directory, omit to rebuild the tracked approved source')
    args = ap.parse_args()
    if args.run:
        from_journal(HERE/args.run)
    else:
        from_reviewed_copy()
