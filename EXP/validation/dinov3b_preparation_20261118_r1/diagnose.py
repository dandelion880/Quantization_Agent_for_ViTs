"""Isolate source of a collapsed pilot without any fitting or parameter sweep."""
from pathlib import Path
import sys, time
import torch
from itertools import islice
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT/'QAViT'))
from runtime.paths import QAVIT
from core.journal import atomic_json, read_json
from tools.experiment_worker import backend_and_data
from tools.frozen import FrozenData
from ptq_agent.execution.pipeline import QuantizationExecutor, quantizers
from ptq_agent.execution.model import forward
from tools.vision_models import setup_segmentation
def main():
    setup_segmentation(42,3)
    source = HERE/'pilot_group32_r2'
    cfg = read_json(ROOT/'EXP/inputs/dinov3b_static_w4a4/input.json')
    backend, options, data = backend_and_data({'backend':cfg['research_brief']['model']['adapter']})
    model, manifest=QuantizationExecutor(backend,options,FrozenData(data)).load(source/'model')
    metrics={}
    for w, a in ((False,False),(True,False),(False,True),(True,True)):
        for q in quantizers(model).values():
            if getattr(q,'weight_record',None) is not None: q.weight_enabled=w
            q.enabled=a
        rows=[]
        with torch.inference_mode():
            for x,sid in islice(data.batches('search'),24):
                q=forward(backend,model,x)
                y=forward(backend,backend.model,x)
                rows.append({'id':sid,'label':data.labels[sid],'pred':q.argmax().item(),'fp':y.argmax().item(),
                             'relative_error':float((y-q).square().sum()/y.square().sum())})
        metrics[f'weight_{w}_activation_{a}']=rows
    atomic_json(HERE/'pilot_component_controls.json',{'source':str(source),'data':metrics})
    print({k:sum(r['label']==r['pred'] for r in v) for k,v in metrics.items()})
if __name__=='__main__':main()
