"""Resolve actual catalog entries through the regular broker, without scheduling evaluation."""
from pathlib import Path
from copy import deepcopy
from runtime.paths import ROOT
from runtime.identity import code_identity,code_snapshot
from core.cards import make_card
from core.journal import Journal,read_json,atomic_json
from core.solutions import protocol_key
from tools.experiments import ExperimentBroker
out=Path(__file__).resolve().parent
root=out/'binding_check'
root.mkdir(exist_ok=True)
j=Journal(root,'binding-check')
if not (root/'research_run.json').exists():
    identity=code_identity()
    atomic_json(root/'research_run.json',{'purpose':'engineering binding check','code_identity':identity,'source_snapshot':code_snapshot(identity)})
cfg=read_json(out/'input.json')
historical=[read_json(next((ROOT/'EXP/runs').glob('dino*/solution_library/'+v+'*.json'))) for v in ('7c18','7f70')]
# Use the original resolved model binding; the dev manifest is evaluator-only.
brief=deepcopy(historical[1]['configuration'])
brief['execution_guidance']['development']=cfg['research_brief']['execution_guidance']['development']
assert protocol_key(brief)==historical[1]['protocol_hash']
if not j.find_cards('ResearchBrief'):
    card=make_card('ResearchBrief',j.run_id,brief,author='user')
    j.commit('binding-context',cards=[card],events=[{'type':'solution_archive','entries':historical}])
b=j.find_cards('ResearchBrief')[0]
from runtime.budget import BudgetLedger
broker=ExperimentBroker(j,b['id'],BudgetLedger(j,cfg['budget']))
from runtime.task_queue import process_identity
import os
import atexit
broker.ledger.begin_segment(owner=process_identity(os.getpid()))
atexit.register(lambda: broker.ledger.end_segment(processes_stopped=True,reason='catalog binding check completed'))
records=[]
for old in historical:
    draft={'level':'L3','question':'Measure frozen package on the two fixed development tasks',
           'decision_impact':'Compare same-view FP retention','scope':{},
           'measurements':['top1','mIoU'],'controls':{'reference':'frozen FP'},'solution_refs':[old['id']],
           'execution':{'service':'evaluate_artifact','resolved_choices':{'training':False},
                        'evaluations':brief['execution_guidance']['development']['evaluations']}}
    frozen=broker.freeze(draft,requester='generation')
    spec=frozen['payload']['execution']
    package=spec['evaluation_package']
    assert package['model_ref']==old['id'] and package['verified_segmentation']
    assert 'calibration' not in spec and spec['result_protocol']['purpose']=='development'
    records.append({'model':old['id'],'evaluation_protocol':spec['protocol_hash'],'source_ref':package['evidence_ref'],
                    'frozen_source':package['source_version'],'reload_source':package['model_dir']})
atomic_json(out/'binding.json',records)
