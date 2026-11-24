"""Continue the measured Coding breakpoint into a new isolated short trajectory."""
from pathlib import Path
import argparse
import hashlib
import os
import subprocess
import sys
import time

from runtime.paths import ROOT, QAVIT
from core.cards import make_card, validate_schema
from core.evidence import evidence_facts
from core.journal import Journal, atomic_json, read_json
from agents.roles import role_spec, submission_schema
from runtime.identity import code_identity
from runtime.model_router import call_once
from tools.experiments import ExperimentBroker
from tools.safeguards import approval_receipt, verify_artifact
from tools.evaluation.breakpoint import start_request
from tools.evaluation.common import confined, digest, EVALUATIONS, output_directory
from tools.evaluation.judge import evaluation_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coding',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    coll = read_json(Path(__file__).with_name('collection.json'))
    root = EVALUATIONS / coll['evaluation']
    trace = read_json(confined(args.coding,EVALUATIONS)/'trace.json')
    case = coll['cases']['code_repair']
    req = start_request(root,case)
    if trace['status'] != 'submitted' or trace['start'] != req['execution_id']:
        raise ValueError('requires a completed Coding revision of this original request')
    candidate = trace['steps'][-1]['artifact']
    confined(candidate['path'],args.coding)
    ctx = req['spec']['context']
    original = ctx['original_request']
    verify_artifact(candidate,original)
    if args.output.resolve().parent != EVALUATIONS:
        parser.error('output must be a new evaluation directory')
    out = output_directory(args.output,create=True)
    branch = out/'branch'
    m = evaluation_manifest(root)
    objects = {r['path']:r for r in m['source_files']}
    def raw(name):
        row = objects[name]
        data = confined(root/row['object'],root/'sources').read_bytes()
        if digest(data) != row['sha256']:
            raise ValueError(name)
        return data
    def copy(name):
        target = branch/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw(name))
    copy('research_run.json')
    for seq in range(1,req['spec']['context']['read_view']['sequence']+1):
        name = f'commits/{seq:09d}.json'
        copy(name)
        for row in read_json(branch/name)['cards']:
            copy('cards/'+hashlib.sha256(row['ref'].encode()).hexdigest()+'.json')
    journal = Journal(branch,original['run_id'])
    from runtime.budget import BudgetLedger
    ledger = BudgetLedger(Journal(out/'budget','local-replay'),{'wall_seconds': original['payload']['resources']['estimated_seconds']+900})
    broker = ExperimentBroker(journal,ctx['research_brief']['id'],ledger,read_json(branch/'research_run.json')['config']['runtime'])
    experiment = broker.bind_execution(journal.card(original['id']),candidate)
    journal.commit('replay/new-experiment',cards=[experiment])
    common = {k:ctx[k] for k in ('model_metadata','research_brief','constraints_source','code_interface','interface_reference','original_request')}
    review_ctx = {**common,'frozen_experiment':experiment,'candidate_source':Path(candidate['path']).read_text(),
                  'task':{'candidate':candidate,'experiment_ref':experiment['id'],'request_ref':original['id']}}
    settings = req['spec']['settings']
    def call(role,purpose,context):
        spec = role_spec(role,purpose,context,settings)
        reply = call_once(spec,out/role,time.time()+300)['response']
        atomic_json(out/role/'response.json',reply)
        return reply
    reply = call('safeguard','safeguard',review_ctx)
    decision = reply.get('decision') or {}
    if decision.get('action') != 'submit':
        atomic_json(out/'result.json',{'status':'review_followup_unmeasured','tools':reply['tool_calls']})
        return
    review = decision['content']
    validate_schema(review,submission_schema('safeguard'),'review')
    feedback = {'review':review}
    if review['approved']:
        binding = {'source_sha256':candidate['source_sha256'],'request_ref':original['id'],
                   'request_sha256':original['content_sha256'],'experiment_ref':experiment['id'],
                   'experiment_sha256':experiment['content_sha256']}
        rc = make_card('ReviewDecisionCard',journal.run_id,
            {'evidence_refs':[],'uncertainties':[],**review,'phase':'safeguard','reviewed_binding':binding},
            author='safeguard',refs=[experiment['id']])
        approval = approval_receipt(experiment,candidate,rc)
        journal.commit('replay/new-review',cards=[rc])
        # Only the original local 12-image diagnostic is in this case's budget.
        spec = experiment['payload']['execution']
        if experiment['payload']['level'] not in ('L1','L2') or spec['service']!='custom_experiment' or spec['calibration']['count']!=12:
            raise ValueError('request exceeds local replay scope')
        job = out/'numeric'
        uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        request = {'kind':'experiment','experiment':experiment,'approval':approval,'task_id':'replay-local',
                   'execution_id':'replay-local-a0001','core_sha256':code_identity()['core_sha256'],
                   'resource_keys':['gpu:'+uuid],'gpu':1,'gpus':[1],'gpu_bindings':[{'index':1,'uuid':uuid,'device_node':'/dev/nvidia1'}],
                   'deadline_unix':time.time()+experiment['payload']['resources']['estimated_seconds']}
        atomic_json(job/'request.json',request)
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':'1','QAVIT_RESOURCE_ASSIGNED':'1'}
        with (job/'worker.log').open('w') as log:
            subprocess.run([sys.executable,'-B',str(QAVIT/'runtime/task_worker.py'),str(job/'request.json')],env=env,stdout=log,stderr=log,check=True)
        receipt = read_json(job/'result.json')
        feedback.update(execution=receipt['status'])
        if receipt['status']=='completed':
            ev = broker.evidence(experiment,receipt,job/'result.json',code_artifact=candidate)
            broker.validate_evidence_for(experiment,ev,code_artifact=candidate)
            journal.commit('replay/measurement',cards=[ev])
            feedback.update(evidence=evidence_facts(ev),observations=ev['payload']['observations'])
        else:
            feedback['error'] = receipt['error']
    # Return only this branch's new state to its historical scientific requester.
    import json
    generator = json.loads(raw('jobs/task-b23ea3009beff0ed0600d200-a0001/request.json'))['spec']
    feedback_ctx = {**generator['context'],'tool_results':[{'name':'request_experiment','original_request':original['payload'],
                                                          'result':feedback}]}
    answer = call('generation','generate',feedback_ctx)
    atomic_json(out/'result.json',{'status':'feedback_delivered','review_approved':review['approved'],
                'execution':feedback.get('execution'),'evidence':feedback.get('evidence',{}).get('source'),
                'generation_action':answer.get('decision'),'new_tool_choices':[c['name'] for c in answer['tool_calls']],
                'new_experiments':'unmeasured'})
if __name__ == '__main__':
    main()
