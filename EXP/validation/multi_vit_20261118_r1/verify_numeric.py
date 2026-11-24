"""Paired integration checks on bound real assets. All counts are subset counts."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'QAViT'))
from runtime.paths import ROOT
from core.journal import atomic_json, file_sha256, read_json
from core.cards import fingerprint
from core.static_w4a4 import check_plan
from ptq_agent.execution.baselines import build_baseline
from ptq_agent.execution.evaluation import model_digest
from ptq_agent.execution.model import forward, RTNExecutor
from ptq_agent.execution.propagation import profile_frozen_propagation
from ptq_agent.execution.calibration_search import search_frozen_scales
from ptq_agent.execution.sites import site_catalog
from ptq_agent.profiling.adapters import boundary_tensor
from ptq_agent.profiling.torch_profiler import TorchProfiler
import numpy as np
from tools.experiment_worker import backend_and_data, data_view, evaluate_services, tensor_probe
from tools.vision_models import setup_segmentation
import torch


def verify(config, output):
    setup_segmentation(42, 3)
    adapter = {'kind': 'server', 'config_path': config}
    backend, cfg, data = backend_and_data({'backend': adapter})
    catalog = site_catalog(backend, cfg)
    summary = {'model_id': backend.spec['model_id'], 'checkpoint_sha256': backend.checkpoint_hash,
               'load': backend.load_report, 'config': config, 'config_sha256': file_sha256(ROOT/config),
               'device': torch.cuda.get_device_name(), 'seed': 42, 'scope': 'integration subset; no L3'}
    first = backend.spec['block_paths'][0]
    last = backend.spec['block_paths'][-1]
    profile = TorchProfiler(backend, cfg['constraints'])
    captured, identifiers, shape = profile.capture([first], 1, 42, 0)
    activation, same_ids = backend.tensor(first, 'activation', 1, 42, 0)
    assert identifiers == same_ids and np.array_equal(captured[first],activation)
    summary['raw_profile'] = {'first_block':first, 'input_shape':shape, 'sample_ids':identifiers,
                              'captured':list(activation.shape), 'same_tensor_both_tools':True}
    x, sid = next(data.batches('calibration'))
    handles, captures = [], {}
    def capture(name):
        def hook(m, args, kwargs, out):
            captures[name] = (args, kwargs, boundary_tensor(out))
        return hook
    for name in (first, last):
        handles.append(backend.model.get_submodule(name).register_forward_hook(capture(name), with_kwargs=True))
    with torch.inference_mode():
        logits = forward(backend, backend.model, x)
    for h in handles:
        h.remove()
    with torch.inference_mode():
        for name, (args, kwargs, value) in captures.items():
            assert torch.equal(boundary_tensor(backend.model.get_submodule(name)(*args, **kwargs)), value)
        summary['feature_checks'] = {'sample_id': sid, 'block_replay': 'bitwise equal', 'logits': list(logits.shape)}
        if backend.task_head:
            features = backend.model.forward_features(x)
            prefix, dim = backend.spec['prefix_token_count'], backend.spec['hidden_dim']
            assert torch.equal(logits, backend.model.head(features['x_norm_clstoken']))
            maps = backend.model.get_intermediate_layers(x, n=[0, len(backend.model.blocks)-1], norm=True, reshape=True)
            for tokens, spatial in zip([captures[first][2], captures[last][2]], maps):
                assert torch.equal(backend.model.norm(tokens)[:, prefix:], spatial.flatten(2).transpose(1, 2))
            assert torch.equal(features['x_norm_patchtokens'], maps[-1].flatten(2).transpose(1, 2))
            assert torch.equal(features['x_storage_tokens'], backend.model.norm(captures[last][2])[:,1:prefix])
            summary['feature_checks'].update(native_intermediates_bitwise_equal=True,
                normalized_cls_logits_bitwise_equal=True, features={k:list(v.shape) for k,v in features.items() if torch.is_tensor(v)})
    summary['sites'] = dict(Counter(v['kind'] for v in catalog.values()))
    plan = build_baseline(backend, 'repq')
    plan['preprocess'] = []
    for layer in plan['layers']:
        layer['options'] = {}
        for target in ('weight','activation'):
            if layer[target]:
                layer[target + '_method'] = 'uniform_minmax'
                layer[target + '_calibration'] = {'strategy':'minmax'}
    required = check_plan(plan,catalog)
    linears = [n for n,m in backend.model.named_modules() if isinstance(m,torch.nn.Linear) and n != 'head']
    for name in linears:
        assert name + '/weight' in required['required_boundaries']
    attns = [n for n,m in backend.model.named_modules() if hasattr(m,'qkv') and hasattr(m,'num_heads')]
    for name in attns:
        for kind in ('attention_query','attention_key','attention_value','post_softmax'):
            assert name+'::'+kind+'/activation' in required['required_boundaries']
    summary['required_count'] = len(required['required_boundaries'])
    experiment = {'id':'integration-subset', 'payload':{'level':'L2','resources':{'cpu_threads':3},
        'execution':{'service':'quantize_evaluate','plan':plan,'resolved_choices':{'training':False},
                     'quantization_contract':{'protocol':'static_w4a4/1'},
                     'calibration':{'split':'calibration','count':2},
                     'evaluations':[{'kind':'classification','split':'search','count':2},{'kind':'ade_frozen','count':1}]}}}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'input.json',experiment)
    result = evaluate_services(experiment,backend,cfg,data,output,None)
    atomic_json(output/'result.json',result)
    summary['subset'] = {k:{f:r[f] for f in ('sample_count','sample_ids','metrics','full_dataset')}
                         for k,r in result['observations']['evaluations'].items()}
    summary['reload'] = result['observations']['reload']
    summary['model_ref'] = result['model_ref']
    manifest = read_json(output/'model/manifest.json')
    summary['calibration_ids'] = manifest['calibration_ids']
    summary['weights_sha256'] = manifest['weights_sha256']
    model, _ = RTNExecutor(backend,cfg,None).load(output/'model')
    summary['model_digest'] = model_digest(model)
    view = data_view(data,{'split':'calibration','count':1})
    report = profile_frozen_propagation(backend,model,view,blocks=[first,last])
    atomic_json(output/'propagation.json',report)
    summary['diagnostic_groups'] = report['decomposition_cross_term'][first]
    site = next(k for k,v in catalog.items() if v['kind']=='linear' and k.startswith(first+'.'))
    candidates = [{'id':str(f),'scales':{site:{'activation':f}}} for f in (1.0,0.9)]
    scopes = ['patch'] + (['special'] if backend.spec['special_token_count'] else [])
    for scope in scopes:
        report = search_frozen_scales(backend,deepcopy(model),view,candidates,block=first,
                                      reference='local_fp',token_scope=scope)
        atomic_json(output/('search_'+scope+'.json'),report)
    assert model_digest(model) == summary['model_digest']
    summary['tensor_probes'] = {}
    attn = attns[0]
    for probe_site in (attn + '.qkv', attn + '::attention_key'):
        work = output / ('probe_' + probe_site.replace('::','_'))
        work.mkdir()
        probe = {'payload':{'level':'L2','execution':{'site':probe_site,'method':'uniform_minmax',
            'parameters':{'bits':4,'signed':True,'granularity':'per_tensor'},
            'resolved_choices':{'activation_bits':4},'calibration':{'split':'calibration','count':1},
            'evaluation':{'split':'search','count':1}}}}
        result = tensor_probe(probe,backend,cfg,data,work,None)
        atomic_json(work/'result.json',result)
        summary['tensor_probes'][probe_site] = {
            'calibration_ids': result['sample_selection']['calibration_ids'],
            'elements': result['observations']['elements'], 'local_output':result['observations']['operator_output_error']}
    summary['segmentation_parity'] = read_json(output/'segmentation.json')['fp_adapter_parity']
    backend.reset()
    atomic_json(output/'summary.json',summary)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.config,args.output)
