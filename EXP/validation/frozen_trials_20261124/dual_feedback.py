"""Use normal evaluator adapters and exact cold packages on a paired, evaluator-owned dev prefix."""
from pathlib import Path
import argparse
import gc
import time
import torch
from numerical import OUT, RUN, setup
from core.cards import fingerprint
from core.journal import read_json, atomic_json, file_sha256
from tools.frozen import FrozenData, native_parent
from tools.experiment_worker import evaluate_task, evaluation_summary
from ptq_agent.execution.frozen_trials import accepted_copy, state_identity
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.model import forward
from ptq_agent.execution.evaluation import model_digest
from core.static_w4a4 import native_audit, check_plan
from ptq_agent.execution.sites import site_catalog

def main(dest, selected, ncls, nade):
    start=time.monotonic()
    backend,cfg,data,original,manifest,binding,spec=setup()
    native, model=native_parent(backend,cfg,data,original,manifest)
    x,_=next(data.batches('calibration'))
    assert torch.equal(forward(backend,original,x),forward(backend,model,x))
    del original
    plan=manifest['plan']
    freeze=read_json(OUT/'paired/calibration.json')
    brief=read_json(RUN/'research_run.json')['config']['research_brief']
    cfg['_development_view']=brief['execution_guidance']['development']['manifest']
    # The private manifest is accessed only by the independent evaluator below.
    rows={r['id']:r['receipt'] for r in freeze['candidates']}
    results={}
    for name in selected:
        t=time.monotonic()
        job=dest/name
        job.mkdir()
        child,receipt=accepted_copy(model,rows[name],source_ref=binding['model_ref'],input_shapes=manifest['input_shapes'])
        coverage=native.audit(child,plan)
        pkg=job/'model'
        m=native.export(child,plan,{}, {},pkg,coverage)
        m.update(calibration_ids=manifest['calibration_ids'],preparation_training=False,refinement={
            **receipt,'initialization':'verified_parent','scored_calibration_ids':freeze['sample_ids']})
        m.pop('content_sha256')
        m['content_sha256']=fingerprint(m)
        atomic_json(pkg/'manifest.json',m)
        files={f.name:file_sha256(f) for f in pkg.iterdir()}
        rec={'candidate':receipt,'model_ref':fingerprint(files),'package':str(pkg),'files':files,'tasks':{}}
        for kind,split,n in [('classification','dev',ncls),('ade_frozen','ade_dev',nade)]:
            cold,_=QuantizationExecutor(backend,cfg,FrozenData(data)).load(pkg)
            assert state_identity(cold)==receipt['child_state_sha256'] and torch.equal(forward(backend,child,x),forward(backend,cold,x))
            full=native_audit(cold,plan,site_catalog(backend),check_plan(plan,site_catalog(backend)))
            atomic_json(job/'static.json',full)
            req={'kind':kind,'split':split,'count':n}
            val=evaluate_task(backend,cfg,data,cold,plan,req,job,True,coverage)
            key,summary=evaluation_summary(req,val)
            atomic_json(job/(key+'.json'),val)
            rec['tasks'][key]={k:v for k,v in summary.items() if k in ('metrics','sample_ids','sample_count','split','full_dataset')}
            del cold
            gc.collect()
        assert state_identity(model)==receipt['parent_state_sha256'] and files=={f.name:file_sha256(f) for f in pkg.iterdir()}
        rec['seconds']=time.monotonic()-t
        results[name]=rec
        atomic_json(dest/'results.json',{'source':binding['evidence_ref'],'seed':backend.config['seed'],'dev_manifest':cfg['_development_view']['sha256'],
            'coverage':'paired prefix of bound development views; independent of calibration and search; no official evaluation',
            'device':torch.cuda.get_device_name(),'peak_mib':torch.cuda.max_memory_allocated()/2**20,
            'seconds':time.monotonic()-start,'results':results})
        del child
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('arms', nargs='+')
    p.add_argument('--ncls',type=int,default=24)
    p.add_argument('--nade',type=int,default=6)
    a=p.parse_args()
    torch.set_num_threads(2)
    dest=OUT/a.name
    dest.mkdir(exist_ok=False)
    main(dest,a.arms,a.ncls,a.nade)
