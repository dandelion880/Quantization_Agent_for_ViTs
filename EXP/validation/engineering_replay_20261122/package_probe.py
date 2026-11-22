"""Reopen one actual nested L3 package then measure a single ordered search sample."""
import argparse
from pathlib import Path
import time
import torch
from runtime.paths import ROOT
from runtime.guards import protect_files
from core.journal import read_json, atomic_json, file_sha256
from core.cards import fingerprint
from tools.experiment_worker import backend_and_data
from tools.runtime_interface import ObservedData
from tools.frozen import reload_complete
from ptq_agent.execution.evaluation import model_digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--data-only',action='store_true',help='decode the complete declared local view on CPU, without model forward')
    args = parser.parse_args()
    directory = args.output.resolve()
    if directory.parent != Path(__file__).resolve().parent:
        parser.error('use a fresh directory beside this script')
    directory.mkdir(exist_ok=False)
    run = ROOT/'EXP/runs/dinov3b_static_w4a4_independent_20261121_r1'
    request = run/'jobs/task-7353e9d0a927a030725092f8-a0001/request.json'
    diagnostic = run/'jobs/task-3a5b25ba50042105441f119e-a0001/request.json'
    spec = read_json(request)['experiment']['payload']['execution']
    selected_request = read_json(diagnostic)
    selected = selected_request['experiment']['payload']['execution']['parameters']['diagnostic_view']
    if args.data_only:
        from runtime.identity import code_identity
        assert not torch.cuda.is_available(), 'set CUDA_VISIBLE_DEVICES="" for the CPU decode diagnostic'
        started = time.monotonic()
        execution = selected_request['experiment']['payload']['execution']
        # Device placement belongs to this decode probe. Keep the exact original manifest and source order.
        execution['backend'].setdefault('backend_overrides', {})['device'] = 'cpu'
        torch.set_num_threads(2)
        backend, _, data = backend_and_data(execution)
        protection = protect_files(directory)
        decoded = []
        load = backend.data_adapter.load_image
        def observed_load(row):
            decoded.append(row['sample_id'])
            return load(row)
        backend.data_adapter.load_image = observed_load
        view = ObservedData(data, selections=[('diagnostic_view',selected)],withhold_final=True)
        pool = data.ids('search')
        ids = [sid for _,sid in view.batches('search')]
        metadata = view.identity
        assert len(pool) == len(set(pool)) == 1000 and len(ids) == 64
        assert metadata['batch_size'] == 1 and metadata['ordering_seeds']['search'] == selected['ordering_seed']
        assert metadata['diagnostic_view']['ordered_pool_ids'] == pool
        assert metadata['split_ids']['search'] == pool[512:576] == decoded == ids == view.reads['search']
        assert metadata['counts'] == {'search':64}
        atomic_json(directory/'receipt.json',{'scope':'all 64 ordered real CPU decodes; no forward or task score',
            'core_sha256':code_identity()['core_sha256'],'diagnostic_request_sha256':file_sha256(diagnostic),
            'selection':selected,'split_ids':ids,'count':len(decoded),'ordering_sha256':fingerprint(ids),
            'decoder_seen_sha256':fingerprint(decoded),'first_id':ids[0],'last_id':ids[-1],
            'status':'completed','protection':protection,'elapsed_seconds':time.monotonic()-started})
        return
    package = next(b for b in spec['frozen_sources'] if b['model_dir'].endswith('task-34987abf61fad915c5e72787-a0001/model'))
    started = time.monotonic()
    torch.set_num_threads(2)
    backend,cfg,data = backend_and_data(spec)
    protection = protect_files(directory,cuda_device_nodes=['/dev/nvidia1'])
    receipt = {'input_sha256':file_sha256(request),'diagnostic_request_sha256':file_sha256(diagnostic),
               'source':package['evidence_ref'],'model_ref':package['model_ref'],'protection':protection}
    try:
        with torch.inference_mode():
            model, manifest = reload_complete(package,backend,cfg,data)
            view = ObservedData(data, selections=[('diagnostic_view',selected)],withhold_final=True)
            ids = view.ids('search')
            assert len(ids)==64 and ids==data.ids('search')[512:576]
            image,sid = next(view.batches('search'))
            y = backend.forward_model(model,image)
            assert view.reads == {'search':[ids[0]]} and torch.isfinite(y).all()
            receipt.update(status='completed',state_sha256=model_digest(model),logit_shape=list(y.shape),
                           sample_id=sid,logit_norm=float(y.float().norm()),actual_reads=view.reads,
                           bound_ids_sha256=fingerprint(ids),scope='one-image loader diagnostic; no task score')
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc))
        raise
    finally:
        atomic_json(directory/'receipt.json',{**receipt,'elapsed_seconds':time.monotonic()-started})
if __name__ == '__main__':
    main()
