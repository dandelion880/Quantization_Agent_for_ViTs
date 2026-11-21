"""Exercise the evaluator/prepare boundary on real inputs without changing any model."""
import os
from pathlib import Path
import torch
from core.journal import read_json, atomic_json
from runtime.paths import ROOT
from runtime.guards import protect_files
from runtime.gpu_resources import acquire, available_inventory
from tools.experiment_worker import backend_and_data
from tools.development import prepare_with_calibration, read_view
from runtime.identity import code_identity

out = Path(__file__).resolve().parent
brief = read_json(out/'input.json')['research_brief']
binding = brief['execution_guidance']['development']['manifest']
view = read_view(binding)
spec = {'backend': brief['model']['adapter'], 'result_protocol':{'development_view': binding},
        'calibration': {'split':'calibration','count':1,'offset':1}}
pool = available_inventory({})
leased, fds = acquire([g['index'] for g in pool if g['index'] != 0], bindings=pool)
gpu = next(g for g in pool if g['index'] == leased[0])
os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],QAVIT_RESOURCE_ASSIGNED='1')
backend, cfg, data = backend_and_data(spec)
torch.set_num_threads(2)
probe_dir = out/'preparation_probe'
probe_dir.mkdir(exist_ok=True)
protect_files(probe_dir, cuda_device_nodes=[gpu['device_node']])
private = {'development_manifest':binding['path'], 'development_image':view['classification'][0]['path'],
           'search_image':data.rows['search'][0]['path'], 'final_image':data.rows['final'][0]['path'],
           'other_calibration':data.rows['calibration'][0]['path']}
def probe(local, cfg, restricted, payload, selected):
    assert len(local.rows) == 1 and len(restricted.ids('calibration')) == 1
    assert set(restricted.rows) == {'calibration'}
    assert 'development_view' not in payload['execution']['result_protocol']
    denied = {}
    for k,path in private.items():
        try:
            with open(path, 'rb') as stream:
                stream.read(1)
        except PermissionError:
            denied[k] = True
    assert set(denied) == set(private)
    x, sid = next(restricted.batches('calibration'))
    with torch.inference_mode():
        local.forward_model(local.model, x)
    return {'sample_id':sid,'device':str(x.device),'denied_read':denied,'source':code_identity()['core_sha256']}
receipt = prepare_with_calibration(probe, backend, cfg, data, {'payload':{'execution':spec}},probe_dir,None)
receipt.update(evaluator_can_read_after=all(Path(p).open('rb').read(1) for p in private.values()),
               gpu=gpu['uuid'], scope='engineering probe, no scientific metric')
atomic_json(probe_dir/'receipt.json',receipt)
