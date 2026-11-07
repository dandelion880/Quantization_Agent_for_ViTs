"""Reuse the serial evaluator on disjoint single-image shards of one frozen package."""
from pathlib import Path
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import atomic_json, read_json, file_sha256
from runtime.paths import QAVIT
from runtime.task_queue import process_identity
from tools.evaluating import evaluate, aggregate


def row(value, experiment, model_dir, identity):
    return {'data': {'evaluation': value, 'proposal_id': experiment['id'],
                     'plan_key': file_sha256(model_dir / 'manifest.json'),
                     'model_dir': str(model_dir), 'code_files': {}, 'data_protocol': identity},
            'cpu_seconds': None, 'elapsed_seconds': None, 'peak_rss_kib': None}


def evaluate_parallel(experiment, backend, cfg, data, model, plan, requested, model_dir, directory, gpus):
    from runtime.codex_cli import parent_guard
    from runtime.task_runtime import TaskRuntime
    from ptq_agent.execution.evaluation import model_digest
    split, count = requested['split'], requested.get('count')
    size = len(data.ids(split)) if count is None else count
    gpus = gpus[:size]  # Never emit empty shards or duplicate samples.
    if len(gpus) < 2:
        return evaluate(backend, cfg, data, model, plan, split, count)
    children, rows, received = [], [], set()
    lease_fds = tuple(int(fd) for fd in os.environ.get('QAVIT_GPU_LEASE_FDS', '').split(',') if fd)
    digest, baseline = model_digest(model), model_digest(backend.model)
    assets = {str(p.relative_to(model_dir)): file_sha256(p) for p in model_dir.rglob('*') if p.is_file()}
    try:
        for index, gpu in enumerate(gpus[1:], 1):
            output = directory / 'shards' / str(index)
            output.mkdir(parents=True, exist_ok=True)
            request = {'experiment': experiment, 'model_dir': str(model_dir), 'model_assets': assets,
                       'model_digest': digest, 'baseline_digest': baseline,
                       'split': split, 'count': count, 'shard': [index, len(gpus)]}
            path = output / 'request.json'
            atomic_json(path, request)
            uuids = os.environ.get('QAVIT_ASSIGNED_GPU_UUIDS', '').split(',')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[index] if len(uuids) > index else str(gpu),
                       QAVIT_RESOURCE_ASSIGNED='1')
            with (output / 'worker.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-B', str(QAVIT / 'tools/classification_shards.py'), str(path)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    preexec_fn=parent_guard(os.getpid()), pass_fds=lease_fds)
            children.append((child, output, process_identity(child.pid)))
            atomic_json(directory / 'processes' / ('shard-' + str(index) + '.json'), children[-1][2])
        start = time.time()
        value = evaluate(backend, cfg, data, model, plan, split, count, (0, len(gpus)))
        atomic_json(directory / 'shards/0/result.json', {'evaluation': value, 'started_unix': start,
                    'finished_unix': time.time(), 'model_digest': digest, 'baseline_digest': baseline})
        rows.append(row(value, experiment, model_dir, data.identity))
        for child, output, identity in children:
            # The parent task timer remains the single timeout for all shards.
            while not (output / 'result.json').exists() and child.poll() is None:
                time.sleep(.05)
            if not (output / 'result.json').exists():
                raise ContractError('classification shard failed: ' + str(output))
            result = read_json(output / 'result.json')
            if (result.get('request_sha256') != file_sha256(output / 'request.json')
                    or result['model_digest'] != digest or result['baseline_digest'] != baseline):
                raise ContractError('classification shards did not use the same frozen models')
            rows.append(row(result['evaluation'], experiment, model_dir, data.identity))
            received.add(child.pid)
        value = aggregate(rows)['data']['evaluation']
        value['parallelism'] = {'physical_gpus': gpus, 'shards': len(gpus),
                                'model_digest': digest, 'baseline_digest': baseline,
                                'aggregation': 'integer correct counts and FP64 error sums; original sample order'}
        return value
    finally:
        for child, output, identity in children:
            if child.pid in received:
                # Validated durable data is ready. Job-level supervision retains
                # all leases through teardown; do not spend the numerical timeout
                # waiting for Python/CUDA process destructors.
                child.poll()
                continue
            if child.poll() is None:
                TaskRuntime.stop_identity(identity)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # registered identity remains owned by TaskRuntime


def main():
    from tools.experiment_worker import backend_and_data
    from ptq_agent.execution.model import RTNExecutor
    from ptq_agent.execution.evaluation import model_digest
    from tools.experiments import verify_assets
    import torch
    path = Path(sys.argv[1])
    request = read_json(path)
    experiment = request['experiment']
    verify_assets(experiment['payload']['execution'])
    torch.set_num_threads(experiment['payload']['resources']['cpu_threads'])
    model_dir = Path(request['model_dir'])
    if request['model_assets'] != {str(p.relative_to(model_dir)): file_sha256(p) for p in model_dir.rglob('*') if p.is_file()}:
        raise ContractError('frozen shard model package changed')
    backend, cfg, data = backend_and_data(experiment['payload']['execution'])
    model, manifest = RTNExecutor(backend, cfg, data).load(model_dir)
    digest, baseline = model_digest(model), model_digest(backend.model)
    if digest != request['model_digest'] or baseline != request['baseline_digest']:
        raise ContractError('loaded shard model identity differs')
    start = time.time()
    value = evaluate(backend, cfg, data, model, manifest['plan'], request['split'], request['count'], request['shard'])
    atomic_json(path.parent / 'result.json', {'evaluation': value, 'model_digest': digest,
                'baseline_digest': baseline, 'request_sha256': file_sha256(path),
                'started_unix': start, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
