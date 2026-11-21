"""Bounded engineering probe of the real evaluate_artifact worker and secondary lease."""
from pathlib import Path
from copy import deepcopy
import os,time
from runtime.paths import ROOT
from runtime.gpu_resources import available_inventory,acquire
from core.cards import fingerprint
from core.journal import atomic_json,read_json
from tools.experiments import asset,preflight_experiment
from tools.frozen import completed_package, _materialize_source
from tools.experiment_worker import run_experiment
d=Path(__file__).resolve().parent
out=d/('custom_dual_checked' if '--check-eligibility' in __import__('sys').argv else 'custom_dual')
out.mkdir(exist_ok=True)
settings=read_json(d/'input.json')
row=read_json(next((ROOT/'EXP/runs').glob('dino*/solution_library/7f70*.json')))
old=next(a for a in row['artifacts'] if Path(a['path']).name=='segmentation.json')
job=Path(old['path']).parent
source=read_json(job/'request.json')['experiment']
evidence=read_json(job/'result.json')['data']
evidence['artifacts']=row['artifacts']
package=job/'model'
binding={k:row[k] for k in ('evidence_ref','source_snapshot','source_version','source_archive_sha256')}
binding.update(model_ref=row['id'],candidate=row['code'],model_dir=str(package),assets=[
 a for a in row['artifacts'] if package in Path(a['path']).parents],source_cache=str(d/'sources'/row['source_version']))
binding.update(completed_package(evidence,source))
_materialize_source(binding)
view=read_json(d/'development.json')
for kind in ('classification','ade_frozen'):
    view[kind]=view[kind][:2]
atomic_json(out/'probe_view.json',view)
experiment=deepcopy(source)
spec=experiment['payload']['execution']
spec.update(service='evaluate_artifact',requires_code=False,model_dir=str(package),model_assets=binding['assets'],
            evaluation_package=binding)
for key in ('calibration','ade_fp_reference','measurement_implementation','runtime','frozen_sources','interface','result_protocol'):
    spec.pop(key,None)
dev=settings['research_brief']['execution_guidance']['development']
dev['manifest']=asset(out/'probe_view.json')
for r in dev['evaluations']: r['count']=2
spec['evaluations']=dev['evaluations']
from tools.development import stage_protocol
spec['result_protocol']=stage_protocol(settings['research_brief'],spec,'L3')
spec['protocol_hash']=fingerprint(spec['result_protocol'])
inventory=available_inventory({})
from runtime.gpu_resources import placement
support=placement(experiment,{'allowed_gpus':[v['index'] for v in inventory]})
assert support['parallelism']['dual_task_supported']
from runtime.gpu_resources import ResourceBusy
until = time.monotonic() + 150
while time.monotonic() < until:
    try:
        gpus,fds=acquire([v['index'] for v in inventory if v['index']!=0],2,bindings=inventory)
    except ResourceBusy:
        gpus,fds=[],[]
    if len(gpus)==2:
        break
    for fd in fds:
        os.close(fd)
    time.sleep(5)
else:
    raise ResourceBusy('two unoccupied secondary-probe cards not available in the bounded window')
leases=[next(v for v in inventory if v['index']==i) for i in gpus]
os.environ.update(QAVIT_RESOURCE_ASSIGNED='1',CUDA_VISIBLE_DEVICES=leases[0]['uuid'],
    QAVIT_ASSIGNED_GPU_UUIDS=','.join(v['uuid'] for v in leases),QAVIT_GPU_LEASE_FDS=','.join(map(str,fds)))
request=dict(experiment=experiment,gpu_bindings=leases,gpus=gpus,deadline_unix=time.time()+300)
result=run_experiment(request,out)
atomic_json(out/'debug.json',{'scope':'engineering_probe','result':result,'gpu_bindings':leases})
