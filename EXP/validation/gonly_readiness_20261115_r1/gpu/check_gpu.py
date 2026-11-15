"""Real DeiT smoke: 1 calibration image; 512 classification + 1 ADE image.
Not scientific evidence or full L3. A fresh package avoids historical identity mutation.
"""
from pathlib import Path
import sys, os, time, json, gc
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'QAViT'))
from runtime.paths import ROOT
from core.journal import atomic_json,read_json,file_sha256
from core.cards import fingerprint
from runtime.gpu_resources import gpu_bindings,acquire
from tools.experiment_worker import backend_and_data,evaluate_services,data_view
from tools.experiments import runtime_identity,measurement_implementation
from ptq_agent.execution.baselines import build_baseline
import torch
OUT=Path(__file__).resolve().parent
if (OUT/'parallel').exists():
 raise SystemExit('Preparation output already exists; use the frozen-package check instead of recalibrating.')
bindings=gpu_bindings([0,1]); gpus,fds=acquire([0,1],2,bindings=bindings)
assert len(gpus)==2,(gpus,bindings)
os.environ['CUDA_VISIBLE_DEVICES']=','.join(r['uuid'] for r in bindings)
os.environ['QAVIT_ASSIGNED_GPU_UUIDS']=os.environ['CUDA_VISIBLE_DEVICES']
os.environ['QAVIT_GPU_LEASE_FDS']=','.join(map(str,fds))
os.environ['QAVIT_RESOURCE_ASSIGNED']='1'
torch.set_num_threads(2)
spec={'service':'quantize_evaluate','backend':{'kind':'server','config_path':'QAViT/configs/deitb_backend.json'},
 'seed':42,'resolved_choices':{'training':False},'calibration':{'split':'calibration','count':1},
 'evaluations':[{'kind':'classification','split':'search','count':512},{'kind':'ade_frozen','count':1}],
 'quantization_contract':{'protocol':'static_w4a4/1'}}
experiment={'id':'gpu_readiness_smoke@1','run_id':'gonly_readiness_gpu','payload':{'level':'L2','execution':spec,'resources':{'cpu_threads':2}}}
try:
 backend,cfg,data=backend_and_data(spec)
 spec['plan']=build_baseline(backend,'repq')
 spec['plan']['preprocess']=[]
 for layer in spec['plan']['layers']:
  layer['activation_calibration']={'strategy':'minmax'}
  if layer['weight']:layer['weight_calibration']={'strategy':'minmax'}
 spec['runtime']=runtime_identity();spec['measurement_implementation']=measurement_implementation()
 directory=OUT/'parallel';directory.mkdir(exist_ok=True)
 atomic_json(directory/'request.json',{'deadline_unix':time.time()+900,'experiment':experiment,'physical_gpu_bindings':bindings})
 start=time.time(); result=evaluate_services(experiment,backend,cfg,data,directory,None,gpus)
 atomic_json(directory/'result.json',result)
 atomic_json(OUT/'parallel_summary.json',{'elapsed_seconds':time.time()-start,
  'receipt':read_json(directory/'dual_task_receipt.json'),
  'evaluations':{k:{a:b for a,b in v.items() if a!='sample_ids'} for k,v in result['observations']['evaluations'].items()},
  'scope':'engineering smoke, fresh uniform W4A4 minmax calibration1, no reparameterization; no performance claim'})
 del result;gc.collect();torch.cuda.empty_cache()
 # Same immutable package, explicit one-card fallback; no prepare/recalibration.
 spec.update(service='evaluate_artifact',model_dir=str(directory/'model'))
 spec.pop('calibration')
 serial=OUT/'serial';serial.mkdir(exist_ok=True)
 start=time.time();result=evaluate_services(experiment,backend,cfg,data,serial,None,gpus[:1])
 atomic_json(serial/'result.json',result)
 atomic_json(OUT/'serial_summary.json',{'elapsed_seconds':time.time()-start,'model_ref':result['model_ref'],
  'evaluations':{k:{a:b for a,b in v.items() if a!='sample_ids'} for k,v in result['observations']['evaluations'].items()}})
finally:
 if 'backend' in globals():backend.reset()
 for fd in fds:os.close(fd)
