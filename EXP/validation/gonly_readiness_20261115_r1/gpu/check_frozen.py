"""Same immutable native package: dual-task/serial equality and real GPU failure cleanup.
A small engineering check, not L3 evidence or a new method verification.
"""
from pathlib import Path
import sys, os, time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'QAViT'))
from runtime.paths import ROOT
from core.journal import atomic_json,read_json
from runtime.gpu_resources import gpu_bindings,acquire
from runtime.identity import measurement_implementation
from runtime.task_queue import process_alive
from tools.experiment_worker import backend_and_data,evaluate_services,plan_constraints
from tools.experiments import runtime_identity
from tools.dual_task_evaluation import evaluate_parallel,package_files
from ptq_agent.execution.model import RTNExecutor
import torch
SOURCE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('QAVIT_GPU_CHECK_OUTPUT', str(SOURCE)))
if (OUT/'GPU_RESULTS.json').exists():
 raise SystemExit('Completed output exists; set QAVIT_GPU_CHECK_OUTPUT to a new directory.')
OUT.mkdir(parents=True, exist_ok=True)
bindings=gpu_bindings([0,1]);gpus,fds=acquire([0,1],2,bindings=bindings)
assert len(gpus)==2
os.environ['CUDA_VISIBLE_DEVICES']=bindings[0]['uuid']
os.environ['QAVIT_ASSIGNED_GPU_UUIDS']=','.join(r['uuid'] for r in bindings)
os.environ['QAVIT_GPU_LEASE_FDS']=','.join(map(str,fds))
os.environ['QAVIT_RESOURCE_ASSIGNED']='1'
torch.set_num_threads(2)
experiment=read_json(SOURCE/'parallel/request.json')['experiment']
experiment['id']='gpu_readiness_frozen@1'
spec=experiment['payload']['execution']
model_dir=SOURCE/'parallel/model'
spec.update(service='evaluate_artifact',model_dir=str(model_dir))
spec.pop('calibration',None)
spec['evaluations'][0]['count']=64
spec['runtime']=runtime_identity();spec['measurement_implementation']=measurement_implementation()
initial=package_files(model_dir)
try:
 backend,cfg,data=backend_and_data(spec)
 results={}
 for mode,devices in [('dual',[0,1]),('serial',[0])]:
  directory=OUT/mode;directory.mkdir(exist_ok=True)
  if (directory/'result.json').exists():
   result=read_json(directory/'result.json');elapsed=None
  else:
   atomic_json(directory/'request.json',{'deadline_unix':time.time()+300,'experiment':experiment,'physical_gpu_bindings':bindings})
   start=time.time();result=evaluate_services(experiment,backend,cfg,data,directory,None,devices)
   elapsed=time.time()-start;atomic_json(directory/'result.json',result)
  results[mode]={'elapsed_seconds':elapsed,'model_ref':result['model_ref'],
    'evaluations':result['observations']['evaluations']}
 assert results['dual']['model_ref']==results['serial']['model_ref']
 equality={}
 for kind in ['classification','segmentation']:
  a,b=[results[mode]['evaluations'][kind] for mode in ['dual','serial']]
  operational={'latency_ms','baseline_latency_ms','cuda_peak_memory_mib','measured_speedup'}
  quality=[{k:v for k,v in r['metrics'].items() if k not in operational} for r in [a,b]]
  raw=[read_json(OUT/m/(kind+'.json')) for m in ['dual','serial']]
  integer_equal=(raw[0]['classification_counts']==raw[1]['classification_counts'] if kind=='classification'
                 else raw[0]['metrics']==raw[1]['metrics'])
  equality[kind]={'same_sample_ids':a['sample_ids']==b['sample_ids'],'exact_quality_metrics':quality[0]==quality[1],
                  'integer_counts_or_confusion_matrices_equal':integer_equal,'sample_count':a['sample_count'],
                  'operational_metrics_by_mode':{m:{k:v for k,v in r['metrics'].items() if k in operational}
                                                for m,r in zip(['dual','serial'],[a,b])}}
  assert all(equality[kind][k] for k in ['same_sample_ids','exact_quality_metrics','integer_counts_or_confusion_matrices_equal']),equality
 # An extra local buffer makes the parent instance disagree with its frozen
 # package. The child must reject it; neither package nor shared FP is touched.
 model,manifest=RTNExecutor(backend,cfg,data).load(model_dir)
 model.register_buffer('readiness_fault_only',torch.tensor([1.],device=backend.device),persistent=False)
 spec['evaluations'][0]['count']=1
 failure=OUT/'identity_failure';failure.mkdir(exist_ok=True)
 atomic_json(failure/'request.json',{'deadline_unix':time.time()+120})
 try:
  coverage=plan_constraints(manifest['plan'],spec['resolved_choices'],backend,cfg,spec['quantization_contract'])
  evaluate_parallel(experiment,backend,cfg,data,model,manifest['plan'],coverage,True,model_dir,failure,gpus,None)
 except Exception as exc:
  error=type(exc).__name__+': '+str(exc)
 else:raise AssertionError('mismatched parent must not be accepted')
 assert 'identity' in error,error
 assert not process_alive(read_json(failure/'processes/dual-segmentation.json'))
 assert initial==package_files(model_dir)
 summary={'scope':'engineering smoke only; uniform static W4A4 with calibration1; no new scientific evidence',
  'gpu_bindings':bindings,'model_assets':initial,'parallel':read_json(OUT/'dual/dual_task_receipt.json'),
  'equality':equality,'failure':{'error':error,'child_alive':False,'package_unchanged':True},
  'runs':{mode:{**r,'evaluations':{k:{a:b for a,b in row.items() if a!='sample_ids'} for k,row in r['evaluations'].items()}}
          for mode,r in results.items()}}
 atomic_json(OUT/'GPU_RESULTS.json',summary)
finally:
 if 'backend' in globals():backend.reset()
 for fd in fds:os.close(fd)
