"""Program-authored dual-task measurement without a research round or candidate."""
from pathlib import Path
import os
import sys
import time
from copy import deepcopy
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'QAViT'))
from runtime.paths import QAVIT
from runtime.task_queue import process_identity
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from core.cards import fingerprint

def main(pilot=None, output=None):
    if pilot:
        numerical_input = read_json(HERE / pilot / 'input.json')
        plan = numerical_input['experiment']['payload']['execution']['plan']
    else:
        numerical_input = read_json(HERE / 'baseline_plan.json')
        plan, pilot = numerical_input['plan'], numerical_input['source_pilot']
    if numerical_input['plan_sha256'] != fingerprint(plan):
        raise ValueError('baseline definition digest mismatch')
    path = Path(output).resolve() if output else HERE / ('baseline_' + Path(pilot).name.removeprefix('pilot_'))
    cfg = read_json(ROOT / 'EXP/inputs/dinov3b_static_w4a4/input.json')
    cfg.update(name='DINOv3-B static W4A4 preparation ' + pilot, purpose='baseline_measurement', enable_l3=True)
    cfg['runtime'].update(allowed_gpus=[0,1], cpu_threads=4)
    wf = Workflow(path) if path.exists() else Workflow(path, cfg)
    timing = time.monotonic()
    wf.budget.begin_segment(owner=process_identity(os.getpid()))
    try:
        previous = wf.queue.get_by_key('baseline/measurement')
        if not previous:
            protocol = wf.brief['payload']['evaluation_protocol']
            experiment = wf.broker.freeze({'level':'L3','question':'以同一冻结骨干测量来源明确的准备基线及配对 FP。','measurements':['top1','top5','mIoU','mAcc','aAcc'],
                'controls':{'source':'engineering preparation','pilot':pilot,'plan_sha256':fingerprint(plan)},
                'scope':{'full_backbone':True},'resources':{'gpu_count':2}, 'execution':{'service':'quantize_evaluate',
                'plan':plan,'training':False,'calibration':{'split':'calibration','count':protocol['calibration_samples']},
                'evaluations':deepcopy(protocol['evaluations'])}}, requester='program')
            wf.journal.commit('baseline/specification', cards=[experiment])
            tid=wf.queue.enqueue('baseline/measurement','experiment',inputs={'experiment_ref':experiment['id'],'approval':None},
                resource_keys=wf.experiment_resources(experiment), max_retries=0)
        else:
            tid = previous['task_id']
            prior = wf.journal.card(previous['inputs']['experiment_ref'])
            if prior['payload']['controls']['plan_sha256'] != fingerprint(plan):
                raise ValueError('existing measurement uses a different baseline definition')
        while True:
            wf.poll()
            wf.update_numeric_wait()
            task = wf.queue.get(tid)
            if task['state'] == 'completed':
                break
            if task['state'] in ('cancelled','failed','needs_clarification'):
                raise RuntimeError(task.get('error'))
            wf.dispatch()
            wf.update_numeric_wait()
            time.sleep(2)
        evidence = wf.journal.card(task['outputs'][0])
        experiment = wf.journal.card(evidence['payload']['experiment_ref'])
        wf.broker.validate_evidence_for(experiment,evidence)
        p = evidence['payload']
        assert p['status']=='completed' and p['measurement_source']=='real'
        assert not any(e['type']=='round_finished' for e in wf.journal.events)
        receipt = {'reference':{'run_dir':str(path.relative_to(ROOT)),'evidence_ref':evidence['id']},
            'model_ref':p['provenance']['model_ref'], 'protocol':wf.solutions.key,
            'plan_sha256':fingerprint(plan),'experiment':experiment['id'],'seconds':time.monotonic()-timing,
            'evaluations':{k:v['metrics'] for k,v in p['observations']['evaluations'].items()}, 'resources':p['resources']}
        atomic_json(path/'baseline_receipt.json',receipt)
        print(receipt, flush=True)
    finally:
        wf.runtime.stop()
        wf.budget.end_segment(processes_stopped=True, reason='preparation measurement closed')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', nargs='?')
    parser.add_argument('--output', help='fresh run directory inside this project EXP')
    args = parser.parse_args()
    main(args.pilot, args.output)
