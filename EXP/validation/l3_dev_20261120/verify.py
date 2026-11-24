"""Verify the retained measurements and required FP pairing without scheduling a run."""
from pathlib import Path
from copy import deepcopy
from core.journal import Journal, atomic_json, file_sha256, read_json
from core.cards import ContractError
from runtime.identity import code_identity
from tools.experiments import ExperimentBroker
from tools.result_contract import check_results
out=Path(__file__).resolve().parent
j=Journal(out/'journal','development_20261120')
brief=j.find_cards('ResearchBrief')[0]
broker=ExperimentBroker(j,brief['id'],None)
checked=[]
for evidence in j.find_cards('EvidenceCard'):
    ep=evidence['payload']
    exp=j.card(ep['experiment_ref'])
    broker.validate_evidence_for(exp,evidence)
    for name,metric in [('classification','baseline_top1'),('segmentation','baseline_mIoU')]:
        broken=deepcopy(ep)
        del broken['observations']['evaluations'][name]['metrics'][metric]
        try:
            check_results(exp['payload'],brief['payload'],broken)
        except ContractError as exc:
            assert '指标' in str(exc),str(exc)
        else:
            raise AssertionError('missing paired FP was accepted')
    checked.append(evidence['id'])
sha=file_sha256(out/'development.json')
assert len(checked)==2 and sha==brief['payload']['execution_guidance']['development']['manifest']['sha256']
for path in ('custom_dual/dual_task_receipt.json','custom_dual_checked/dual_task_receipt.json'):
    receipt=read_json(out/path)
    assert receipt['status']=='completed' and receipt['overlap_seconds']>0
assert len(read_json(out/'binding.json'))==2
assert len(read_json(out/'catalog.json'))==2
assert len(read_json(out/'engineering_matched_result.json')['evidence'])==1
for log in ('final_tests.log','binding_tests.log'):
    assert (out/log).read_text().rstrip().endswith('OK'), log
atomic_json(out/'verification.json', {'core_sha256':code_identity()['core_sha256'],
    'view_sha256':sha,'verified_development_evidence':checked,'missing_fp_rejections':4,
    'official_and_dev_catalog_records':2,'bound_historical_packages':2,
    'successful_engineering_replay':read_json(out/'engineering_matched_result.json')['evidence'],
    'recorder_card':j.find_cards('MetareviewCard')[0]['id'],
    'regression_logs':{f:file_sha256(out/f) for f in ('final_tests.log','binding_tests.log')},
    'actual_gpu_probes':{p:read_json(out/p) for p in ('preparation_probe/receipt.json',
        'custom_dual/dual_task_receipt.json','custom_dual_checked/dual_task_receipt.json')}})
