"""Small paired DINO measurements on the bound P64. Never read official evaluation labels."""
from pathlib import Path
import argparse
import json
import time
import torch
from runtime.paths import ROOT
from runtime.identity import code_identity
from core.journal import read_json, atomic_json, file_sha256
from tools.experiment_worker import backend_and_data, data_view
from tools.frozen import reload_complete
from tools.runtime_interface import ObservedData
from tools.segmentation import DenseTask
from ptq_agent.execution.frozen_trials import score_frozen_candidates
from ptq_agent.execution.model import tensor_digest

OUT = Path(__file__).resolve().parent
RUN = ROOT/'EXP/runs/dinov3b_static_w4a4_independent_20261121_r1'
REQUEST = RUN/'jobs/task-3a5b25ba50042105441f119e-a0001/request.json'
OBJECTIVES = {'cls_kl':{'cls.teacher_kl':1}, 'cls_mse':{'cls.mse':1}, 'patch_cos':{'later.cosine':1},
              'dense_nmse':{'head.nmse':1}, 'balanced':{'cls.teacher_kl':1,'head.nmse':1}}
OBS = [{'id':'cls','view':'classification','output':'classification_logits','metrics':['teacher_kl','mse']},
       {'id':'later','view':'classification','output':'block','block':'blocks.11','tokens':'patch','metrics':['nmse','cosine']},
       {'id':'head','view':'dense','output':'head_features','metrics':['mse','nmse','cosine']}]
P = 'blocks.8.attn::post_softmax'
V = 'blocks.8.attn::v'
def setup():
    spec = read_json(REQUEST)['experiment']['payload']['execution']
    parent = spec['frozen_sources'][0]
    backend, cfg, data = backend_and_data(spec)
    model, manifest = reload_complete(parent,backend,cfg,data)
    return backend,cfg,data,model,manifest,parent,spec

def grid(vsite):
    a,b = {P:{'activation':.8}}, {vsite:{'activation':.9}}
    return [{'id':k,'scales':s} for k,s in [('parent',{}),('P08',a),('V09',b),('both',{**a,**b}),
            ('P12',{P:{'activation':1.2}}),('V11',{vsite:{'activation':1.1}})]]

def run(dest):
    start=time.monotonic()
    backend,cfg,data,model,manifest,parent,spec = setup()
    init_s = time.monotonic()-start
    vsite = next(m.ptq_site_id for m in model.modules() if getattr(m,'layer',{}).get('kind')=='attention_value' and m.layer['module']=='blocks.8.attn')
    candidates=grid(vsite)
    dense=DenseTask(backend)
    def measure(rows,split,count,offset,obs=OBS):
        view=ObservedData(data,selections=[('diagnostic_view',{'split':split,'count':count,'offset':offset})],withhold_final=True)
        t=time.monotonic()
        result=score_frozen_candidates(backend,model,view,rows,observations_spec=obs,objectives=OBJECTIVES if split=='calibration' else None,
            split=split,source_ref=parent['model_ref'],input_shapes=manifest['input_shapes'],dense_task=dense)
        result.update(actual_reads=view.reads,seconds=time.monotonic()-t)
        return result
    calibration=measure(candidates,'calibration',4,0)
    atomic_json(dest/'calibration.json',calibration)
    selected=set(calibration['winners'].values())|{'parent','both'}
    confirm=measure([c for c in candidates if c['id'] in selected],'search',6,512,OBS+[
        {'id':'seg','view':'dense','output':'segmentation_logits','metrics':['teacher_kl']}])
    atomic_json(dest/'confirmation.json',confirm)
    atomic_json(dest/'run.json',{'request':str(REQUEST),'request_sha256':file_sha256(REQUEST),'parent':parent['model_ref'],
        'source':parent['evidence_ref'],'model_identity':backend.identity(),'core':code_identity()['core_sha256'],
        'seed':backend.config['seed'],'device':torch.cuda.get_device_name(),'startup_seconds':init_s,
        'seconds':time.monotonic()-start,'peak_mib':torch.cuda.max_memory_allocated()/2**20})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name')
    args=parser.parse_args()
    dest=OUT/args.name
    dest.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    run(dest)
