"""Real three-candidate scheduling/calibration/frozen dual-task subset validation.
No LLM, no research round, no full L3 or promotion to B.
"""
from pathlib import Path
from copy import deepcopy
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'QAViT'))
from core.workflow import Workflow
from core.cards import make_card
from core.journal import atomic_json, read_json
from runtime.identity import code_identity
from tests.support import validation_config, experiment_payload, hypothesis
from tests.test_plan_execution_consistency import numeric_plan

OUT = Path(__file__).resolve().parent
config = validation_config()
config['budget'] = {'wall_seconds': 180, 'clock_mode': 'active_segments',
                    'exclude_numeric_wait': True, 'max_rounds': 1, 'min_free_bytes': 1024 ** 3}
config['runtime'].update(allowed_gpus=[2, 3, 4], experiment_concurrency=3, cpu_threads=2)
brief = config['research_brief']
brief.update(scientific_question='Bounded GPU scheduling validation only, not a scientific experiment round',
             hard_constraints={'training': False}, search_space={},
             evaluation_protocol={'kind': 'engineering_subset', 'formal_incumbent_allowed': False})
brief['model'] = {'name': 'DeiT-B', 'scope': {'model': 'deit_base_patch16_224'}, 'adapter': {
    'kind': 'server', 'config_path': 'QAViT/configs/deitb_backend.json', 'backend_overrides': {'device': 'cuda:0'}}}
brief.pop('experiment_example', None)
name = sys.argv[1] if len(sys.argv) > 1 else 'gpu_candidates_r1'
wf = Workflow(OUT / name, config)
wf.budget.begin_segment()
runtimes = [wf.runtime]
report = {'scope': 'engineering subset; real DeiT/ImageNet/ADE assets; no full L3',
          'real_llm_used': False, 'full_research_loop_started': False,
          'source_identity': code_identity(), 'hardware': wf.config['runtime'], 'tasks': {}}
start = time.time()

def enqueue(key, payload, hypothesis_ref=None):
    card = wf.broker.freeze(payload, requester='user')
    wf.journal.commit('spec/' + key, cards=[card])
    tid = wf.queue.enqueue(key, 'experiment', inputs={'experiment_ref': card['id'],
        **({'round': 1, 'hypothesis_ref': hypothesis_ref} if hypothesis_ref else {})})
    report['tasks'][key] = {'task_id': tid, 'experiment_ref': card['id'], 'hypothesis_ref': hypothesis_ref}
    return tid


def drain(ids, restore=False):
    global wf
    restored = False
    while not all(wf.queue.get(t)['state'] == 'completed' for t in ids):
        if time.time() - start > 900:
            raise TimeoutError('900 second engineering smoke cap')
        wf.poll(); wf.update_numeric_wait()
        for tid in ids:
            task = wf.queue.get(tid)
            if task['state'] in ('failed', 'cancelled', 'retry'):
                raise RuntimeError(str({'task': task['key'], 'error': task.get('error')}))
        wf.dispatch(); wf.update_numeric_wait()
        if restore and not restored and len(wf.queue.snapshot(states={'running'}, kinds={'experiment'})) == 3:
            wf = Workflow(wf.run_dir)
            runtimes.append(wf.runtime)
            restored = True
            report['live_restore'] = True
        time.sleep(.1)
    wf.budget.numeric_wait(False)


def collect(key):
    row = report['tasks'][key]
    task = wf.queue.get(row['task_id'])
    evidence = wf.journal.card(task['outputs'][0])
    assert evidence['payload']['experiment_ref'] == row['experiment_ref']
    assert evidence['payload']['measurement_source'] == 'real'
    wf.broker.validate_evidence_for(wf.journal.card(row['experiment_ref']), evidence)
    request = read_json(task['request_path'])
    directory = Path(task['request_path']).parent
    row.update(evidence_ref=evidence['id'], allocated_gpus=request['gpus'],
        parallelism=request['parallelism'], started=task['started_unix'], finished=task['finished_unix'],
        classification=read_json(directory / 'classification.json'),
        segmentation=read_json(directory / 'segmentation.json'),
        model_dir=str(directory / 'model'))
    return row

try:
    ids = []
    for slot in ('H1', 'H2', 'H3'):
        hyp = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, slot), author='user')
        wf.journal.commit('candidate/' + slot, cards=[hyp])
        p = experiment_payload('L2')
        p.update(question='Bounded identical-method scheduling control ' + slot,
                 scope={'model': 'deit_base_patch16_224'},
                 replication={'requested': True, 'purpose': 'independent scheduling control',
                              'configuration_difference': {'slot': slot}})
        p['execution'] = {'service': 'quantize_evaluate', 'requires_code': False,
            'plan': numeric_plan(3, granularity='per_tensor'), 'seed': 42,
            'resolved_choices': {'training': False}, 'calibration': {'split': 'calibration', 'count': 2},
            'evaluations': [{'kind': 'classification', 'split': 'search', 'count': 3},
                            {'kind': 'ade_frozen', 'count': 1}]}
        p['resources'].update(cpu_threads=2, estimated_seconds=600)
        ids.append(enqueue(slot, p, hyp['id']))
    drain(ids, restore=True)
    rows = [collect(slot) for slot in ('H1', 'H2', 'H3')]
    assert all(len(row['allocated_gpus']) == 1 for row in rows)
    assert len({row['allocated_gpus'][0] for row in rows}) == 3
    report['three_candidate_overlap_seconds'] = min(r['finished'] for r in rows) - max(r['started'] for r in rows)
    assert report['three_candidate_overlap_seconds'] > 0
    # Re-measure precisely H1's frozen model on two GPUs after all candidates finish.
    p['execution'] = {'service': 'evaluate_artifact', 'requires_code': False,
        'model_dir': rows[0]['model_dir'], 'seed': 42, 'resolved_choices': {'training': False},
        'evaluations': [{'kind': 'classification', 'split': 'search', 'count': 3},
                        {'kind': 'ade_frozen', 'count': 1}]}
    p['question'] = 'Same frozen H1 package: serial versus parallel task equivalence'
    p['resources']['gpu_count'] = 2
    p['replication']['configuration_difference'] = {'validation': 'same_frozen_parallel'}
    parallel = enqueue('same_frozen_parallel', p)
    drain([parallel])
    same = collect('same_frozen_parallel')
    assert len(same['allocated_gpus']) == 2
    first = rows[0]
    assert first['classification']['sample_ids'] == same['classification']['sample_ids']
    assert first['classification']['classification_counts'] == same['classification']['classification_counts']
    assert first['segmentation']['sample_ids'] == same['segmentation']['sample_ids']
    assert first['segmentation']['metrics'] == same['segmentation']['metrics']
    receipt = read_json(Path(wf.queue.get(parallel)['request_path']).parent / 'dual_task_receipt.json')
    report['same_frozen_parallel_receipt'] = receipt
    assert receipt['overlap_seconds'] > 0
    report['same_frozen_serial_parallel_metrics_equal'] = True
    for runtime in runtimes: runtime.stop()
    wf.budget.end_segment(processes_stopped=True, reason='bounded validation pause')
    paused = wf.budget.snapshot()['elapsed_seconds']
    wf = Workflow(wf.run_dir)
    assert wf.budget.snapshot()['elapsed_seconds'] == paused
    wf.budget.begin_segment(); wf.budget.end_segment(processes_stopped=True, reason='resume checked; no new experiment')
    report.update(status='completed', budget=wf.budget.snapshot())
except BaseException as exc:
    report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
    raise
finally:
    for runtime in runtimes: runtime.stop()
    if wf.budget.snapshot().get('active_segment'):
        wf.budget.end_segment(processes_stopped=True, reason='validation cleanup')
    report['wall_seconds'] = time.time() - start
    atomic_json(OUT / (name + '_report.json'), report)
    print(json.dumps({'status': report['status'], 'report': str(OUT / (name + '_report.json'))}))
