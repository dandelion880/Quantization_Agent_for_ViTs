"""Compare crop/feature/logit receipts to the installed task's real inference loop."""
from copy import deepcopy
from unittest.mock import patch
import time
import sys
import torch
from core.journal import atomic_json
from numerical import OUT, setup, OBS, P
from tools.segmentation import DenseTask
from tools.experiment_worker import data_view
from ptq_agent.execution.model import tensor_digest
from ptq_agent.execution.frozen_trials import image_views, score_frozen_candidates
from ptq_agent.execution.propagation import profile_frozen_propagation
from runtime.identity import code_identity

@torch.inference_mode()
def main():
    t=time.monotonic()
    b,cfg,data,model,m,source,_=setup()
    dense=DenseTask(b)
    from mmengine.dataset import pseudo_collate
    from tools.vision_models import segmentation_assets
    from tools.segmentation import task_pipeline
    from PIL import Image
    view=data_view(data,{'split':'calibration','count':2})
    settings=segmentation_assets(b.config)[1]
    native_pipeline=task_pipeline(settings,annotations=False)
    units=[]
    for row,(x,sid) in zip(view.source_rows('calibration'),view.batches('calibration')):
        expected=list(dense.windows(row))
        with Image.open(row['path']) as raw:
            original_hw=[raw.height,raw.width]
        assert row['sample_id']==sid and torch.equal(x.cpu(),b.data_adapter.load_image(row))
        packed=native_pipeline(dict(img_path=row['path'],seg_fields=[]))
        mm=dense.segmentor
        prepared=mm.data_preprocessor(pseudo_collate([packed]),training=False)
        meta=prepared['data_samples'][0].metainfo
        actual=[]
        native_decode=mm.encode_decode
        def measured(window, metas):
            i=len(actual)
            w,ours,g=expected[i]
            assert torch.equal(window,w) and original_hw==g['original_hw']
            assert tuple(metas[0]['img_shape'])==tuple(ours['img_shape'])
            seen={}
            def record(_m,_args,out):
                seen['features']=out
            with mm.backbone.register_forward_hook(record):
                logits=native_decode(window,metas)
            features=seen['features']
            computed=dense.observe(b.model,w,ours,logits=True)
            diff={str(layer):float((computed[str(layer)]-f).abs().max()) for layer,f in zip(dense.indices,features)}
            # The evaluation adapter has a documented parity tolerance vs the modern MMSeg FP backbone.
            nmse={str(layer):float((computed[str(layer)]-f).double().square().sum()/f.double().square().sum()) for layer,f in zip(dense.indices,features)}
            err=float((computed['logits']-logits).double().square().sum()/logits.double().square().sum())
            assert max(nmse.values())<1e-8 and err<1e-8
            actual.append({'sample':sid,'file':row['file_sha256'],'window':i,**g,'input':tensor_digest(window),
                           'task_fp_nmse':nmse,'task_fp_max_abs':diff,'task_logit_nmse':err})
            return logits
        with patch.object(mm,'encode_decode',side_effect=measured):
            mm.inference(prepared['inputs'],[deepcopy(meta)])
        assert len(actual)==len(expected)
        units+=actual
    assert len(units)>len(view.ids('calibration'))
    prop=profile_frozen_propagation(b,model,view,blocks=[b.spec['block_paths'][-1]],split='calibration',
                                     view='dense',output='head_features')
    assert [(v['sample_id'],v['index'],v['input_sha256']) for v in prop['view_units']]==[(v['sample'],v['window'],v['input']) for v in units]
    single=data_view(data,{'split':'calibration','count':1})
    pool=[{'id':'parent','scales':{}},{'id':'P08','scales':{P:{'activation':.8}}}]
    obs=OBS+[{'id':'seg','view':'dense','output':'segmentation_logits','metrics':['teacher_kl','nmse']}]
    def score(pool):
        return score_frozen_candidates(b,model,single,pool,observations_spec=obs,
                   input_shapes=m['input_shapes'],source_ref=source['model_ref'],dense_task=dense)
    left,right=score(pool),score(pool[::-1])
    assert left['candidates']==right['candidates'][::-1] and left['views']==right['views']
    assert all(v['output_geometry'] for v in left['views'])
    atomic_json(OUT/(sys.argv[1] if len(sys.argv) > 1 else 'parity.json'),{'source':source['model_ref'],'units':units,'samples':view.ids('calibration'),
        'task_identity':dense.identity,'classification_and_dense_shapes':left['views'],
        'dense_diagnostic_units':prop['unit_count'],'ordering_scores':{r['id']:r['scores'] for r in left['candidates']},
        'order_invariant':True,'state_sha256':left['parent_state_sha256'],'core':code_identity()['core_sha256'],
        'seconds':time.monotonic()-t,'peak_mib':torch.cuda.max_memory_allocated()/2**20})
if __name__=='__main__':
    torch.set_num_threads(2)
    main()
