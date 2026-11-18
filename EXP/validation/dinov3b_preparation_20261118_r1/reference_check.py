"""Exact reuse and changed-view miss on the real frozen FP head and actual ADE images."""
from pathlib import Path
import sys, os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(Path(os.environ.get('QAVIT_PILOT_SOURCE', ROOT/'QAViT')).resolve()))
from runtime.paths import QAVIT
from core.journal import atomic_json, read_json, file_sha256
from tools.experiment_worker import backend_and_data
from tools.segmentation import evaluate
from tools.vision_models import setup_segmentation
from copy import deepcopy
def main():
    setup_segmentation(42,4)
    inp=read_json(ROOT/'EXP/inputs/dinov3b_static_w4a4/input.json')
    backend, _, _ = backend_and_data({'backend':inp['research_brief']['model']['adapter']})
    out=HERE/'fp_reference_check'
    out.mkdir(exist_ok=True)
    model=deepcopy(backend.model)
    source=out/'source.json'
    first=evaluate(model,backend,out,1)
    atomic_json(source,first)
    ref={'path':str(source),'sha256':file_sha256(source)}
    reused=evaluate(model,backend,out,1,fp_reference=ref)
    changed=evaluate(model,backend,out,2,fp_reference=ref)
    assert first['metrics']==reused['metrics'] and reused['fp_reference_sha256']==ref['sha256']
    assert changed['fp_identity'] != first['fp_identity'] and changed['fp_reference_sha256'] is None
    atomic_json(out/'receipt.json',{'one':first['fp_identity'],'two':changed['fp_identity'],
            'reused_reference':ref,'same_counts_and_metrics':True,'changed_view_measured':True})
if __name__=='__main__':main()
