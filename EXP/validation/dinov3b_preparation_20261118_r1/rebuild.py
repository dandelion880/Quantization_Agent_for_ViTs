"""Independent regeneration from the bound FP assets, exact plan and calibration manifest."""
import time
from copy import deepcopy
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'QAViT'))
from runtime.paths import QAVIT
from core.cards import fingerprint
from core.journal import read_json, atomic_json
from tools.experiment_worker import backend_and_data, evaluate_services
from tools.frozen import FrozenData
from ptq_agent.execution.model import RTNExecutor
from ptq_agent.execution.evaluation import model_digest
from runtime.guards import install_gradient_guard
import torch
def main():
    out=HERE/'independent_rebuild'
    out.mkdir(exist_ok=False)
    job,=list((HERE/'baseline_partition/jobs').glob('*/request.json'))
    request=read_json(job)
    experiment=deepcopy(request['experiment'])
    spec=experiment['payload']['execution']
    spec['evaluations']=[{'kind':'classification','split':'search','count':2}]
    experiment['payload']['level']='L2'
    start=time.monotonic()
    import random, numpy as np
    seed=spec['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(experiment['payload']['resources']['cpu_threads'])
    install_gradient_guard(True)
    backend,cfg,data=backend_and_data(spec)
    random.seed(seed); np.random.seed(seed%(2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    result=evaluate_services(experiment,backend,cfg,data,out,None)
    verified = {}
    from core.journal import file_sha256
    for name,path in [('independent',out/'model'),('formal',job.parent/'model')]:
        model,manifest=RTNExecutor(backend,cfg,FrozenData(data)).load(path)
        verified[name]={'digest':model_digest(model),'calibration_ids':manifest['calibration_ids'],
            'plan_sha256':fingerprint(manifest['plan']), 'training':manifest.get('preparation_training')}
        del model
    atomic_json(out/'rebuild_check.json',{'states':verified,'equal':verified['formal']==verified['independent'],
        'source_request':str(job),'source_request_sha256':file_sha256(job),'wall_seconds':time.monotonic()-start,
        'reload':result['observations']['reload']})
    assert verified['formal']==verified['independent']
if __name__=='__main__':main()
