"""Original H3 mechanism functions against a real newly exported tiny frozen ViT."""
from pathlib import Path
import sys, time, importlib.util, traceback
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'QAViT'))
import runtime.paths
import torch
from core.journal import atomic_json, file_sha256
from tools.runtime_interface import interface_fixture, execute_custom
from tools.experiments import normalize_source_refs
from ptq_agent.execution.baselines import build_baseline
from ptq_agent.execution.model import RTNExecutor
from ptq_agent.execution.propagation import profile_frozen_propagation
from tools.evaluating import DataView
OUT=Path(__file__).resolve().parent
start=time.time(); torch.set_num_threads(2)
result={'scope':'synthetic_computation; real PyTorch tiny ViT, original H3 functions; not DeiT mechanism or task evidence'}
try:
    backend,cfg,data=interface_fixture({'backend':{'config_path':'QAViT/configs/deitb_backend.json'}}, {
        'shape':[1,3,32,32], 'sites':['patch_embed.proj','blocks.0.attn.qkv','blocks.1.mlp.fc2'],
        'seed':42,'counts':{'calibration':2,'search':1,'final':1}})
    backend.identity=lambda:{'fixture':'h3_tiny_replay','state':backend.original_state}
    embed=backend.model.patch_embed
    embed.grid_size=tuple(i//p for i,p in zip(embed.img_size,embed.patch_size))
    plan=build_baseline(backend,'adalog')
    for layer in plan['layers']:
        if layer.get('activation_method')=='adalog':
            layer['options']['log_base_candidates']=[19]
            layer['activation_calibration']['factors']=[1.0]
    executor=RTNExecutor(backend,cfg,data)
    folder=OUT/'tiny_model'
    if not folder.exists():
        model,weights,activations,audit=executor.prepare(plan,{})
        executor.export(model,plan,weights,activations,folder,audit)
    else:
        model,_=executor.load(folder)  # Read-only repeat of the exact existing package.

    # Whole package verification and the production RTN loader run inside load_frozen.
    binding={'evidence_ref':'engineering-h3-tiny-evidence','model_ref':'engineering-h3-tiny-model',
        'model_dir':str(folder),'candidate':None,
        'assets':[{'path':str(p),'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in folder.iterdir()]}
    original=OUT/'replay_implementation.py'
    source=original.read_text().replace("['blocks.0.attn', 'blocks.5.attn', 'blocks.11.attn']", "['blocks.0.attn', 'blocks.1.attn']")
    source=source.replace('EvidenceCard-46fbe7b34ad44eea8f72d04756d69074@1',binding['evidence_ref']).replace('c1c5ac43b4957e8afca2e8c4caca8f3cd1d3cc75eb72e0c8529bf2d00b6a27db',binding['model_ref'])
    code=OUT/'tiny_implementation.py'; code.write_text(source)
    spec=importlib.util.spec_from_file_location('h3_tiny_replay',code);plugin=importlib.util.module_from_spec(spec);spec.loader.exec_module(plugin)
    payload=normalize_source_refs({'controls':{'evidence_refs':[binding['evidence_ref']]}, 'execution':{
        'service':'custom_experiment','requires_code':True,'resolved_choices':{},'frozen_sources':[binding],
        'calibration':{'split':'calibration','count':2,'offset':0}}})
    measured=execute_custom(plugin,payload,backend,cfg,data,OUT)
    atomic_json(OUT/'h3_tiny_measurement.json',measured)
    # Reuse the same verified frozen model for paired global boundary diagnostics.
    from tools.frozen import load_frozen
    loaded,receipt=load_frozen(binding,backend,cfg,data)
    propagation=profile_frozen_propagation(backend,loaded,data,blocks=['blocks.0','blocks.1'],split='calibration',
        token_layout={'axis':1,'cls_index':0,'patch_start':1},
        interventions=[{'id':'block0_fp','kind':'fp_block','block':'blocks.0'}],
        classification_output=None,source={'model_ref':receipt['model_ref']})
    atomic_json(OUT/'h3_tiny_propagation.json',propagation)
    result.update(status='completed',elapsed_seconds=time.time()-start,execution_facts=measured['execution_facts'],
        paired_comparisons=measured['observations']['search_paired_comparisons'],
        path_checks=measured['observations']['path_checks'],lambda_one_control=measured['observations']['lambda_one_control'],
        original_function_source=str(original),adapted_source_sha256=file_sha256(code),sample_ids=measured['sample_selection'])
except Exception as exc:
    result.update(status='failed',error={'code':type(exc).__name__,'message':str(exc)},elapsed_seconds=time.time()-start)
    traceback.print_exc()
finally:
    atomic_json(OUT/'h3_tiny_result.json',result)
    print({k:v for k,v in result.items() if k in ('status','error','elapsed_seconds')},flush=True)
