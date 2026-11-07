"""Bounded real GPU components: same frozen model, 3 ImageNet images, 1 ADE image.

No LLM calls, new method search, full L3, or formal research loop.
"""
from pathlib import Path
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'QAViT'))
from core.workflow import Workflow
from core.journal import read_json, atomic_json
from runtime.identity import code_identity


def main():
    out = Path(__file__).resolve().parent
    source = out.parent / 'parallel_finish_20261106_r1/gpu_smoke_r3'
    config = read_json(source / 'research_run.json')['config']
    config['budget'].update(wall_seconds=25)
    config['runtime'].update(allowed_gpus='auto', experiment_concurrency='auto')
    config['runtime'].pop('gpu_bindings', None)
    model_dir = next(p.parent for p in (source / 'jobs').glob('*/model/weights.pt'))
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'gpu_components_r2'
    wf = Workflow(out / run_name, config)
    wf.budget.begin_segment()
    started = time.time()
    report = {'source_identity': code_identity(), 'source_plan': str(model_dir / 'manifest.json'),
              'scope': 'same pretrained DeiT-B frozen package; W8A8 only blocks.0.mlp.fc1/fc2; '
                       '3 ImageNet search images and 1 ADE validation image, frozen head, not full L3',
              'real_llm_used': False, 'full_research_loop_started': False,
              'hardware': wf.config['runtime'], 'tasks': {}}
    try:
        pool = wf.config['runtime']['allowed_gpus']
        if len(pool) < 3:
            raise RuntimeError('This bounded three-device validation needs three visible authorized GPUs')
        def drain():
            while any(wf.queue.get(r['task_id'])['state'] != 'completed' for r in report['tasks'].values()):
                if time.time() - started > 420:
                    raise TimeoutError('bounded component validation ceiling')
                wf.budget.check(allow_numeric_drain=wf.numeric_running())
                wf.poll()
                wf.update_numeric_wait()
                for row in report['tasks'].values():
                    task = wf.queue.get(row['task_id'])
                    if task['state'] in ('failed', 'cancelled', 'retry'):
                        raise RuntimeError(str(task.get('error')))
                wf.dispatch()
                wf.update_numeric_wait()
                time.sleep(.1)

        for key, gpu, evaluations in (
            ('serial_metrics', pool[-1], [{'kind': 'classification', 'split': 'search', 'count': 3,
                'metrics': ['top1', 'relative_mse', 'latency_ms', 'cuda_peak_memory_mib']}]),
            ('parallel_classification', pool[0], [{'kind': 'classification', 'split': 'search', 'count': 3,
                'metrics': ['top1', 'relative_mse']}]),
            ('ade_uuid', pool[-1], [{'kind': 'ade_frozen', 'count': 1, 'metrics': ['mIoU', 'mAcc', 'aAcc']}])):
            payload = {'level': 'L2', 'question': '有界验证 UUID 设备、必需指标回退与同一冻结模型分片一致性',
                'measurements': ['protocol results'], 'execution': {
                    'service': 'evaluate_artifact', 'requires_code': False, 'model_dir': str(model_dir),
                    'resolved_choices': {'training': False}, 'seed': 42, 'evaluations': evaluations},
                'resources': {'cpu_threads': 2, 'estimated_seconds': 240, 'gpu': gpu, 'gpu_count': 2},
                'replication': {'requested': True, 'purpose': 'bounded engineering validation',
                                'configuration_difference': {'validation': key}}}
            if key == 'serial_metrics':
                payload['execution'].update(service='quantize_evaluate',
                    plan=read_json(model_dir / 'manifest.json')['plan'], calibration={'split': 'calibration', 'count': 2})
                payload['execution'].pop('model_dir')
            card = wf.broker.freeze(payload, requester='user')
            wf.journal.commit('spec/' + key, cards=[card])
            tid = wf.queue.enqueue(key, 'experiment', inputs={'experiment_ref': card['id']})
            report['tasks'][key] = {'task_id': tid, 'experiment_ref': card['id']}
            if key == 'serial_metrics':
                drain()
                wf.budget.numeric_wait(False)
                model_dir = Path(wf.queue.get(tid)['request_path']).parent / 'model'
                report['model_dir'] = str(model_dir)
        drain()
        wf.budget.numeric_wait(False)
        for key, row in report['tasks'].items():
            task = wf.queue.get(row['task_id'])
            evidence = wf.journal.card(task['outputs'][0])
            assert evidence['payload']['experiment_ref'] == row['experiment_ref']
            row.update(evidence_ref=evidence['id'], resources=evidence['payload']['resources'],
                       attempts=task['attempt'], retry_count=task['retry_count'],
                       started=task['started_unix'], finished=task['finished_unix'])
            for name in ('classification.json', 'segmentation.json'):
                path = next((a['path'] for a in evidence['payload']['artifacts'] if Path(a['path']).name == name), None)
                if path:
                    row[name] = read_json(path)
        one = report['tasks']['serial_metrics']['classification.json']
        many = report['tasks']['parallel_classification']['classification.json']
        assert one['sample_ids'] == many['sample_ids']
        assert one['classification_counts'] == many['classification_counts']
        assert one['parallelism']['shards'] == 1 and many['parallelism']['shards'] == 2
        for name in ('latency_ms', 'cuda_peak_memory_mib'):
            assert math.isfinite(one['metrics'][name])
        differences = []
        for name, sums in one['metrics']['raw_error_sums'].items():
            for field, value in sums.items():
                difference = abs(value - many['metrics']['raw_error_sums'][name][field]) / max(1, abs(value))
                assert difference < 1e-12, (name, field, difference)
                differences.append(difference)
        ade = report['tasks']['ade_uuid']['segmentation.json']
        assert ade['images'] == 1 and not ade['full_dataset_evaluation']
        assert ade['head_unchanged'] and ade['backbones_unchanged']
        a, b = (report['tasks'][k] for k in ('ade_uuid', 'parallel_classification'))
        overlap = min(a['finished'], b['finished']) - max(a['started'], b['started'])
        assert overlap > 0
        wf.runtime.stop()
        wf.budget.end_segment(processes_stopped=True, reason='bounded validation pause')
        consumed = wf.budget.snapshot()['elapsed_seconds']
        wf = Workflow(wf.run_dir)
        assert wf.budget.snapshot()['elapsed_seconds'] == consumed
        assert all(t['state'] == 'completed' for t in wf.queue.snapshot().values())
        report.update(status='passed', overlap_seconds=overlap, max_relative_accumulator_difference=max(differences),
                      paused_consumption=consumed)
    except BaseException as exc:
        report.update(status='failed', error={'code': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        wf.runtime.stop()
        wf.poll()
        wf.budget.numeric_wait(False)
        wf.budget.end_segment(processes_stopped=True, reason='bounded validation complete')
        report.update(wall_seconds=time.time() - started, budget=wf.budget.snapshot(),
                      no_owned_processes=not wf.runtime.live_processes(),
                      source_unchanged=code_identity() == report['source_identity'])
        atomic_json(out / (run_name + '_RESULT.json'), report)
        print({k: report.get(k) for k in ('status', 'error', 'wall_seconds', 'no_owned_processes', 'source_unchanged')}, flush=True)


if __name__ == '__main__':
    main()
