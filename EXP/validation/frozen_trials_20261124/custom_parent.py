"""Preserve the archived nonlinear H2 parent and verify reconstruction of an accepted native-grid edit."""
import time
import torch
from numerical import OUT, RUN, setup, P
from core.journal import read_json, atomic_json
from core.cards import ContractError
from ptq_agent.execution.calibration_search import state_identity, _quantizers, apply_grid_changes
from ptq_agent.execution.frozen_trials import score_frozen_candidates, accepted_copy
from ptq_agent.execution.model import tensor_digest, forward
from ptq_agent.schemas import PTQError
from tools.frozen import load_refinement_parent, native_parent
from tools.experiment_worker import data_view
def main():
    t=time.monotonic()
    b,cfg,data,base,_,p64,_=setup()
    recovered=state_identity(base)
    assert recovered==read_json(OUT/'dev/results.json')['results']['parent']['candidate']['child_state_sha256']
    del base
    historical=next(RUN.glob('jobs/task-7353*/request.json'))
    h2=next(s for s in read_json(historical)['experiment']['payload']['execution']['frozen_sources'] if s['model_ref'].startswith('344d1d2c7d'))
    model,m=load_refinement_parent(h2,b,cfg,data)
    q=_quantizers(model)[P]
    assert type(q).__name__=='PowerP4'
    try: native_parent(b,cfg,data,model,m)
    except ContractError as exc: native_guard=str(exc)
    else: raise ValueError('unsupported format silently converted')
    try: apply_grid_changes(model,{P:{'activation':.9}}, {})
    except PTQError as exc: grid_guard=str(exc)
    else: raise ValueError('inactive scale silently edited')
    v='blocks.8.attn::attention_value'
    view=data_view(data,{'split':'calibration','count':2})
    scores=score_frozen_candidates(b,model,view,[{'id':'noop','scales':{}},{'id':'v09','scales':{v:{'activation':.9}}}],
        observations_spec=[{'id':'cls','view':'classification','output':'classification_logits','metrics':['teacher_kl','nmse']}],source_ref=h2['model_ref'])
    row=scores['candidates'][1]
    child,receipt=accepted_copy(model,row['receipt'],source_ref=h2['model_ref'],input_shapes={})
    cold,_=load_refinement_parent(h2,b,cfg,data)
    duplicate,r=accepted_copy(cold,row['receipt'],source_ref=h2['model_ref'],input_shapes={})
    x,_=next(view.batches('calibration'))
    assert r==receipt and state_identity(duplicate)==state_identity(child) and torch.equal(forward(b,child,x),forward(b,duplicate,x))
    assert state_identity(model)==state_identity(cold)==scores['parent_state_sha256']
    assert all(torch.equal(getattr(_quantizers(child)[P],key),getattr(q,key)) for key in ('h2_levels','h2_midpoints'))
    atomic_json(OUT/'custom_parent.json',{'measurement':'exact cold custom parent; child independently reconstructed in memory',
        'reverted_p64_state':recovered,'custom_source':{k:h2[k] for k in ('evidence_ref','model_ref','source_version')},
        'custom_class':type(q).__name__,'custom_code_sha256':h2['candidate']['source_sha256'],'scores':scores,
        'receipt':receipt,'logits_sha256':tensor_digest(forward(b,child,x)),'verified_custom_levels':tensor_digest(q.h2_levels),
        'native_format_guard':native_guard,'unsupported_scale_guard':grid_guard,'seconds':time.monotonic()-t})
if __name__=='__main__':
    torch.set_num_threads(2)
    main()
