"""Read-only historical DeiT package: direct/helper numerical equality on two images.
CUDA_VISIBLE_DEVICES=0 python EXP/validation/capabilities_multi_l3_20261117_r1/validate_frozen_helpers_gpu.py
No next-round workflow or final/full task evaluation is started.
"""
from copy import deepcopy
from pathlib import Path
import gc
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'QAViT'))
from runtime.paths import QAVIT
from core.journal import Journal, read_json, atomic_json, file_sha256
from core.solutions import SolutionLibrary
from tools.frozen import load_frozen, _materialize_source
from tools.experiments import preflight_experiment
from tools.experiment_worker import backend_and_data, data_view
from tools.frozen_helpers import frozen_helper_service
from ptq_agent.execution.propagation import profile_frozen_propagation
from ptq_agent.execution.calibration_search import search_frozen_scales
import torch

def main():
    started = time.time()
    torch.set_num_threads(4)
    output = Path(__file__).resolve().parent / 'frozen_helpers_gpu'
    output.mkdir(exist_ok=True)
    origin = ROOT / 'EXP/runs/deitb_static_w4a4_gonly_20261115_r1'
    metadata = read_json(origin / 'research_run.json')
    journal = Journal(origin, metadata['run_id'])
    evidence = journal.card('EvidenceCard-15a015a9c39044759caf9a9edd7a0c22@1')
    p = evidence['payload']
    source_exp = journal.card(p['experiment_ref'])
    manifest_asset = next(a for a in p['artifacts'] if a['path'].endswith('/model/manifest.json'))
    package = Path(manifest_asset['path']).parent
    approval = next(e['task']['inputs'].get('approval') for e in reversed(journal.events)
        if e['type'] == 'task_snapshot' and e['task']['kind'] == 'experiment'
        and e['task']['inputs'].get('experiment_ref') == p['experiment_ref']
        and evidence['id'] in e['task']['outputs'])
    source = SolutionLibrary(journal, {}, shared=False).source_record(journal.cards[p['experiment_ref']]['sequence'])
    binding = {'evidence_ref': evidence['id'], 'experiment_ref': p['experiment_ref'],
        'model_ref': p['provenance']['model_ref'], 'model_dir': str(package),
        'assets': [a for a in p['artifacts'] if package in Path(a['path']).parents],
        'candidate': deepcopy(approval['candidate']) if approval else None,
        **source, 'source_cache': str(output / 'historical_source')}
    _materialize_source(binding)  # cache only under this validation directory
    brief = next(journal.card(ref)['payload'] for ref, row in journal.cards.items() if row['kind'] == 'ResearchBrief')
    backend, cfg, data = backend_and_data(source_exp['payload']['execution'])
    print('loaded real DeiT/ImageNet reference', flush=True)
    results = []
    cases = [('profile_frozen_propagation', {'blocks':['blocks.0','blocks.1'], 'split':'search', 'count':2,
               'token_layout':{'axis':1,'cls_index':0,'patch_start':1},
               'interventions':[{'id':'fp0','kind':'fp_block','block':'blocks.0'}]}),
             ('search_frozen_scales', {'count':2,'metric':'mse','block':'blocks.0',
               'candidates':[{'id':str(f),'scales':{'blocks.0.attn.qkv':{'activation':f}}} for f in (.8,1.,1.2)]})]
    for tool,args in cases:
        payload = preflight_experiment({'level':'L1','question':'GPU thin-adapter equivalence smoke',
            'measurements':['result'], 'evidence_refs':[evidence['id']],
            'execution':{'service':'frozen_helper','tool':tool,'arguments':deepcopy(args)}}, brief)
        payload['execution']['frozen_sources']=[binding]
        args_direct = deepcopy(args)
        split = args_direct.pop('split','calibration')
        args_direct.pop('count')
        view = data_view(data, payload['execution']['calibration' if split == 'calibration' else 'evaluation'])
        model, receipt = load_frozen(binding,backend,cfg,data)
        if tool == 'profile_frozen_propagation':
            expected = profile_frozen_propagation(backend,model,view,split=split,
                source={k:v for k,v in receipt.items() if k != 'manifest'},**args_direct)
        else:
            expected = search_frozen_scales(backend,model,view,**args_direct)
            expected['source']={k:v for k,v in receipt.items() if k != 'manifest'}
        del model
        gc.collect()
        actual = frozen_helper_service({'payload':payload},backend,cfg,data,output)
        atomic_json(output / (tool+'.json'), actual)
        assert actual['observations']['result'] == expected, tool + ': adapter differs from direct numerical call'
        results.append({'tool':tool,'direct_adapter_exact_equal':True,'sample_ids':actual['sample_selection'],
            'frozen_loads':actual['execution_facts']['frozen_loads']})
        print(tool + ': exact equality',flush=True)
    unchanged = all(file_sha256(a['path']) == a['sha256'] for a in binding['assets'])
    assert unchanged
    atomic_json(output / 'summary.json', {'status':'passed','validation_kind':'real DeiT/ImageNet two-sample GPU smoke; not L3',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__, 'seconds':time.time()-started,
        'source_evidence':evidence['id'],'source_model':binding['model_ref'],'source_version':binding['source_version'],
        'original_assets_unchanged':unchanged,'results':results})
    print('passed; source assets unchanged',flush=True)

if __name__ == '__main__': main()
