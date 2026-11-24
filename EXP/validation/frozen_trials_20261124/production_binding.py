"""Exercise ordinary broker.freeze from committed worker Evidence and cold-load its exact incremental source."""
from copy import deepcopy
from pathlib import Path
import shutil
import time
import torch
from numerical import OUT
from core.cards import ContractError
from core.journal import Journal, read_json, atomic_json
from runtime.budget import BudgetLedger
from tools.experiments import ExperimentBroker
from tools.experiment_worker import backend_and_data
from tools.frozen import load_refinement_parent
from ptq_agent.execution.calibration_search import state_identity

def main():
    start=time.monotonic()
    root=OUT/'rebind'
    shutil.copytree(OUT/'workers/journal',root)
    era=read_json(OUT/'era/binding.json')
    # The local validation journal started after program-scored P08. These are its actual executing source bytes.
    atomic_json(root/'research_run.json',{'code_identity':{'core_sha256':era['source_version'],'archive_sha256':era['source_archive_sha256']},
                                         'source_snapshot':read_json(OUT/'era/source.json')['files']})
    j=Journal(root,'incremental-validation')
    ref=read_json(OUT/'workers/adaptive/evidence.json')['ref']
    scored=read_json(OUT/'workers/score/evidence.json')['ref']
    e=j.card(ref)['payload']
    state=e['observations']['refinement']['child_state_sha256']
    model_ref=e['provenance']['model_ref']
    ledger=BudgetLedger(Journal(OUT/'rebind_budget','verify'),{'wall_seconds':1200})
    broker=ExperimentBroker(j,next(r for r in j.cards if r.startswith('ResearchBrief-')),ledger)
    draft={'level':'L2','resources':{'estimated_seconds':300,'cpu_threads':2},'question':'continue verified child','measurements':['identity'],
        'evidence_refs':[ref],'execution':{'service':'frozen_helper','tool':'score_frozen_candidates','source_ref':model_ref,
        'arguments':{'count':1,'observations_spec':[{'id':'cls','view':'classification','output':'classification_logits','metrics':['nmse']}],
                     'candidates':[{'id':'noop','scales':{}}]}}}
    card=broker.freeze(draft,requester='generation')
    spec=card['payload']['execution'];src=spec['frozen_sources'][0]
    assert src['model_ref']==model_ref and src['verified_increment']==state
    b,cfg,data=backend_and_data(spec)
    model,m=load_refinement_parent(src,b,cfg,data)
    assert state_identity(model)==state
    wrong=deepcopy(draft)
    wrong['evidence_refs'].append(scored)
    wrong['execution']={'service':'quantize_evaluate','training':False,'parent_ref':model_ref,
                        'accept_trial':{'evidence_ref':scored,'candidate_id':'adaptive'}}
    try: broker.freeze(wrong,requester='generation')
    except ContractError as exc:
        assert 'on this parent' in str(exc), str(exc)
        refusal=str(exc)
    else: raise ValueError('accepted a receipt from grandparent')
    atomic_json(OUT/'production_binding.json',{'real_source_evidence':ref,'model_ref':model_ref,'cold_state':state,
            'calibration_ids':m['calibration_ids'],'source_version':src['source_version'],'stale_receipt_response':refusal,
            'next_experiment':card['content_sha256'],'seconds':time.monotonic()-start})
if __name__=='__main__':
    torch.set_num_threads(2)
    main()
