"""Verify real prepared B import, frozen reload and the launch input, without advancing any round."""
import json
from pathlib import Path
import sys
import time
from copy import deepcopy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'QAViT'))
from runtime.paths import QAVIT
from runtime.configuration import config_check
from core.workflow import Workflow
from core.cards import fingerprint
from core.journal import atomic_json,read_json
from tools.experiments import preflight_experiment
from tools.frozen import bind_solutions, load_frozen
from tools.experiment_worker import backend_and_data
from ptq_agent.execution.model import forward
from ptq_agent.execution.evaluation import model_digest
import torch

def main():
    baseline=read_json(HERE/'baseline_partition/baseline_receipt.json')
    cfg=read_json(ROOT/'EXP/inputs/dinov3b_static_w4a4/input.json')
    cfg.update(name='dinov3b_static_w4a4_gonly_next4',prepared_baseline=baseline['reference'])
    cfg['budget'].update(max_rounds=4,wall_seconds=43200)
    # Keep 'auto' unexpanded in the portable production input.
    resolved=config_check(deepcopy(cfg),resolve_resources=True)
    test=deepcopy(cfg)
    test.update(name='verify_prepared_handoff',purpose='validation',enable_l3=False)
    path=HERE/'prepared_handoff'
    wf=Workflow(path) if path.exists() else Workflow(path,test)
    journal_start=wf.journal.sequence
    wf.solutions.import_prepared(cfg['prepared_baseline'])
    assert wf.journal.sequence==journal_start
    from core.cards import ContractError
    altered = deepcopy(test)
    altered.pop('prepared_baseline')
    altered['research_brief']['evaluation_protocol']['calibration_samples'] -= 1
    alternate_path = HERE/'mismatched_handoff'
    alternate = Workflow(alternate_path) if alternate_path.exists() else Workflow(alternate_path,altered)
    rejected = []
    for reason, library, source in [('different protocol',alternate.solutions,cfg['prepared_baseline']),
             ('component run',wf.solutions,{'run_dir':str((HERE/'roles_direct').relative_to(ROOT)),
                                             'evidence_ref':cfg['prepared_baseline']['evidence_ref']})]:
        try:
            library.import_prepared(source)
        except ContractError:
            rejected.append(reason)
    assert len(rejected)==2 and wf.journal.sequence==journal_start
    row=wf.solutions.fallback()
    candidate=row['candidate']
    assert row['status']=='available' and not row['competes_in_preselection']
    assert candidate['solution_id']==baseline['model_ref'] and candidate['measurement_origin']=='preparation'
    brief=wf.brief['payload']
    assert wf.solutions.key==baseline['protocol']
    binding,=bind_solutions(wf.journal,[candidate['solution_id']],brief)
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    pilot=read_json(HERE/'baseline_plan.json')
    protocol=brief['evaluation_protocol']
    definition={'level':'L3','question':'准备复核正式评测合同','measurements':['top1','mIoU'],
        'execution':{'service':'quantize_evaluate','plan':pilot['plan'],'training':False,
        'calibration':{'split':'calibration','count':protocol['calibration_samples']},'evaluations':protocol['evaluations']}}
    formal=preflight_experiment(definition,brief,resolved['runtime'],executable=True)
    import os
    from runtime.task_queue import process_identity
    wf.budget.begin_segment(owner=process_identity(os.getpid()))
    try:
        frozen=wf.broker.freeze(definition,requester='program')
    finally:
        wf.budget.end_segment(processes_stopped=True, reason='planning entry checked; no research tasks dispatched')
    reference=frozen['payload']['execution']['ade_fp_reference']
    measured=read_json(reference['path'])
    assert measured['images']==2000 and measured['fp_identity'] and measured['fp_reference_sha256'] is None
    assert [(r['kind'],r['count']) for r in formal['execution']['evaluations']]==[('classification',50000),('ade_frozen',2000)]
    backend,options,data=backend_and_data(frozen['payload']['execution'])
    model,receipt=load_frozen(binding,backend,options,data)
    assert receipt['loaded_without_prepare']
    with torch.inference_mode():
        image,sid=next(data.batches('search'))
        logits=forward(backend,model,image)
    saved=read_json(HERE/'independent_rebuild/rebuild_check.json')
    assert saved['equal'] and model_digest(model)==saved['states']['formal']['digest']
    assert not any(e['type'] in ('round_started','round_finished') for e in wf.journal.events)
    assert not wf.journal.find_cards('HypothesisCard') and not wf.journal.find_cards('EvidenceCard')
    dest=ROOT/'EXP/inputs/dinov3b_static_w4a4_gonly_next4/input.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
    atomic_json(path/'handoff_check.json',{'B':row,'protocol':wf.solutions.key,'frozen_loader':receipt,
        'sample_id':sid,'independent_logits_top5':logits[0].topk(5).indices.tolist(),'digest':model_digest(model),
        'baseline_plan_sha256':pilot['plan_sha256'],'evaluations':formal['execution']['evaluations'],
        'route':resolved['llm']['agent_models'],'budget':cfg['budget'],'runtime':cfg['runtime'],
        'ade_fp_reference_binding':reference,'ade_fp_identity':measured['fp_identity'],
        'rejected_sources':rejected,'prepared_input':str(dest.relative_to(ROOT)),'input_sha256':fingerprint(cfg),'formal_rounds_launched':0})
if __name__=='__main__': main()
