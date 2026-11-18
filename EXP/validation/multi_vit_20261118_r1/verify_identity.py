"""Read-only real historical reload and cross-binding checks. Run after verify_numeric."""
from copy import deepcopy
from pathlib import Path
import sys
import hashlib
import importlib.util
from itertools import islice
import tempfile
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'QAViT'))
from runtime.paths import ROOT
from core.journal import Journal, read_json, atomic_json
from core.cards import fingerprint
from core.solutions import SolutionLibrary
from tools.experiment_worker import backend_and_data
from tools.frozen import load_frozen, _materialize_source
from ptq_agent.execution.model import RTNExecutor, forward, tensor_digest
from ptq_agent.schemas import PTQError


def verify_rejection(backend, cfg, package):
    try:
        RTNExecutor(backend, cfg, None).load(package)
    except PTQError as exc:
        if exc.code != 'identity_mismatch':
            raise
        return exc.to_dict()
    raise AssertionError('foreign package loaded')


def main(role_name, dino_dir):
    torch.set_num_threads(3)
    origin = ROOT / 'EXP/runs/deitb_static_w4a4_gonly_20261115_r1'
    journal = Journal(origin, read_json(origin / 'research_run.json')['run_id'])
    evidence = journal.card('EvidenceCard-15a015a9c39044759caf9a9edd7a0c22@1')
    p = evidence['payload']
    experiment = journal.card(p['experiment_ref'])
    package = next(Path(a['path']).parent for a in p['artifacts'] if a['path'].endswith('/model/manifest.json'))
    approval = next(e['task']['inputs'].get('approval') for e in reversed(journal.events)
        if e['type'] == 'task_snapshot' and e['task']['kind'] == 'experiment' and evidence['id'] in e['task']['outputs'])
    source = SolutionLibrary(journal, {}, shared=False).source_record(journal.cards[experiment['id']]['sequence'])
    binding = {**source, 'evidence_ref': evidence['id'], 'experiment_ref': experiment['id'],
               'model_ref': p['provenance']['model_ref'], 'model_dir': str(package),
               'assets': [a for a in p['artifacts'] if package in Path(a['path']).parents],
               'candidate': approval['candidate'] if approval else None, 'source_cache': str(HERE / 'historical_source')}
    _materialize_source(binding)
    backend, cfg, data = backend_and_data(experiment['payload']['execution'])
    model, receipt = load_frozen(binding, backend, cfg, data)
    x, sample = next(data.batches('search'))
    with torch.inference_mode():
        logits = forward(backend, model, x)
    assert torch.isfinite(logits).all()
    out = {'historical_deit': {'run': str(origin), 'experiment': experiment['id'], 'evidence': evidence['id'],
        'model_dir': str(package), 'model_ref': receipt['model_ref'], 'loaded_without_prepare': receipt['loaded_without_prepare'],
        'source_version': source['source_version'], 'sample': sample, 'logits_sha256': tensor_digest(logits)}}
    del model
    entry = importlib.util.spec_from_file_location('verified_historical_deit_loader',
                                                  Path(binding['source_cache'])/'vision_models.py')
    old_loader = importlib.util.module_from_spec(entry)
    entry.loader.exec_module(old_loader)
    fp, load_receipt = old_loader.load_model(backend.spec)
    fp = fp.to(backend.device).eval()
    outputs = []
    with torch.inference_mode():
        for x, sid in islice(data.batches('search'), 2):
            logits = forward(backend, backend.model, x)
            assert torch.equal(fp(x), logits)
            outputs.append({'sample':sid,'logits_sha256':tensor_digest(logits)})
    out['historical_deit']['current_vs_original_fp_bitwise_equal'] = outputs
    out['historical_deit']['original_strict_load'] = load_receipt['strict']
    del fp
    dino = dino_dir/'model'
    out['deit_loading_dino'] = verify_rejection(backend,cfg,dino)
    raw = read_json(ROOT / 'EXP/inputs/dinov3b_static_w4a4/input.json')['research_brief']
    original = read_json(dino/'manifest.json')['identity']
    formal = read_json(HERE / 'context_checks.json')['dino']['protocol_hash']
    other, cfg, _ = backend_and_data({'backend': raw['model']['adapter'], 'research_protocol_sha256': formal})
    except_protocol = deepcopy(other.identity())
    except_protocol['config'].pop('research_protocol_sha256')
    assert except_protocol == original
    out['same_dino_other_protocol'] = verify_rejection(other,cfg,dino)
    local = Path(read_json(HERE / role_name / 'candidate_rebuild.json')['package_path'])
    role_identity = read_json(local/'manifest.json')['identity']
    local_protocol = role_identity['config'].pop('research_protocol_sha256')
    assert role_identity == except_protocol and formal != local_protocol
    out['local_L2_to_formal_protocol'] = {'source_protocol':local_protocol,'formal_protocol':formal,
        'all_other_identity_fields_equal':True,'loader':verify_rejection(other,cfg,local)}
    from tools.vision_models import segmentation_assets
    from tools.segmentation import segmentation_dependencies
    _, settings, _, head = segmentation_assets(cfg['backend'])
    with segmentation_dependencies():
        from mmengine import Config
        raw = torch.load(head['checkpoint'], map_location='cpu')['meta']['cfg']
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=HERE) as temp:
            temp.write(raw)
            temp.flush()
            trained = Config.fromfile(temp.name, import_custom_modules=False).to_dict()
    original = {'model': trained['model'], 'pipeline': trained['test_dataloader']['dataset']['pipeline']}
    assert fingerprint(original) == fingerprint({k: settings[k] for k in original}) == head['task_definition_sha256']
    out['original_head_config'] = {'checkpoint_cfg_sha256':hashlib.sha256(raw.encode()).hexdigest(),
        'model_and_test_pipeline_sha256':fingerprint(original),'matches_bound_config':True}
    atomic_json(HERE / 'identity_checks.json',out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('role_run')
    parser.add_argument('--dino-dir', type=Path, default=HERE/'dino_verified')
    args = parser.parse_args()
    main(args.role_run, args.dino_dir)
