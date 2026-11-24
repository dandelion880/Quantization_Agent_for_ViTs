import sys, json, os, gc, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'QAViT'),str(ROOT/'QAViT/tools')]
from core.journal import Journal, read_json, atomic_json, file_sha256
from core.solutions import SolutionLibrary
from tools.frozen import bind_solutions,load_frozen
from tools.experiment_worker import backend_and_data
from runtime.gpu_resources import gpu_bindings, acquire
# Acquire the shared UUID lease before constructing a CUDA backend.
bindings = gpu_bindings([4])
selected, lease_fds = acquire([4], pinned=4, bindings=bindings)
os.environ['CUDA_VISIBLE_DEVICES']=bindings[0]['uuid']
import atexit
atexit.register(lambda: [os.close(fd) for fd in lease_fds])
import torch
out=Path(__file__).parent
bucket=next((ROOT/'EXP/solutions').glob('*/top3.json'))
rows=read_json(bucket)['entries']
brief=rows[0]['configuration']
j=Journal(out/'binding_run','history-readiness')
if not j.events:j.commit('catalog',events=[{'type':'solution_library','entries':rows,'protocol_hash':rows[0]['protocol_hash']}])
bound = {row['id']:bind_solutions(j,[row['id']],brief)[0] for row in rows if row['status']=='evaluated'}
if '--guarded' in sys.argv:
 from runtime.guards import protect_files
 out=out/'guarded_job';out.mkdir(exist_ok=True)
 torch.cuda.init()
 protect_files(out,cuda_device_nodes=[b['device_node'] for b in bindings])
results=[]
for row in rows:
 if row['status']!='evaluated':continue
 started=time.monotonic()
 binding=bound[row['id']]
 before={a['path']:file_sha256(a['path']) for a in binding['assets']}
 backend,cfg,data=backend_and_data(row['execution'])
 model,receipt=load_frozen(binding,backend,cfg,data)
 # Historical packages retain wrapper flags; explicit eval does not refit any tensor.
 model.eval()
 image,sid=next(data.batches('search'))
 with torch.inference_mode():
  logits=backend.forward_model(model,image)
  fp=backend.forward_model(backend.model,image)
 torch.cuda.synchronize()
 after={a['path']:file_sha256(a['path']) for a in binding['assets']}
 assert before==after and torch.isfinite(logits).all()
 results.append({'solution_id':row['id'],'historical_scores':row['metrics'],
  'source_version':receipt['source_version'],'loaded_without_prepare':receipt['loaded_without_prepare'],
  'measurement_origin':receipt['measurement_origin'],'new_check':{'sample_id':sid,'samples':1,'split':'search',
   'output_shape':list(logits.shape),'quantized_argmax':logits.argmax(-1).tolist(),'fp_argmax':fp.argmax(-1).tolist()},
  'package_unchanged':before==after,'elapsed_seconds':time.monotonic()-started})
 atomic_json(out/'history_gpu_leased.json',{'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'results':results,'scope':'load and one-image forward; no task accuracy or calibration'})
 print(row['title'], 'loaded and forwarded',flush=True)
 del model,backend,data,logits,fp,image
 gc.collect();torch.cuda.empty_cache()
