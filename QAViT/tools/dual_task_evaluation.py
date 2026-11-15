"""Two single-GPU tasks, one already frozen package; no calibration or sharding.

The experiment remains one budget/timeout unit. The existing TaskRuntime owns
both inherited leases and the registered child identity, including after resume.
"""
from pathlib import Path
import gc
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, read_json, file_sha256
from runtime.paths import QAVIT
from runtime.task_queue import process_identity
from runtime.failures import RemoteWorkerError, failure_details


def package_files(directory):
    return {str(p.relative_to(directory)): file_sha256(p)
            for p in directory.rglob('*') if p.is_file()}


def check_identity(result, request, request_path):
    for key in ('model_digest', 'baseline_digest', 'plan_sha256', 'model_assets'):
        if result.get(key) != request[key]:
            raise ContractError('dual-task frozen identity mismatch: ' + key)
    if result.get('request_sha256') != file_sha256(request_path):
        raise ContractError('dual-task request identity mismatch')
    if result.get('status') != 'completed':
        raise RemoteWorkerError(result.get('error'), worker='ade_frozen',
                                receipt=request_path.with_name('result.json'),
                                worker_log=request_path.with_name('worker.log'))


def check_loaded_identity(result, request, path):
    """A custom loader's returned identity is its implementation contract."""
    try:
        check_identity(result, request, path)
    except ContractError as exc:
        raise ContractError(str(exc)) from exc


def evaluate_parallel(experiment, backend, cfg, data, model, plan, coverage, native,
                      model_dir, directory, gpus, approval, *, deadline_unix=None):
    from tools.experiment_worker import evaluate_task, evaluation_summary
    from runtime.codex_cli import parent_guard
    from runtime.task_runtime import TaskRuntime
    from ptq_agent.execution.evaluation import model_digest
    requested = {r['kind']: r for r in experiment['payload']['execution']['evaluations']}
    output = directory / 'dual_tasks' / 'segmentation'
    output.mkdir(parents=True, exist_ok=True)
    deadline = deadline_unix
    request = {'experiment': experiment, 'approval': approval, 'native': native,
        'model_dir': str(model_dir), 'model_assets': package_files(model_dir),
        'model_digest': model_digest(model), 'baseline_digest': model_digest(backend.model),
        'plan_sha256': fingerprint(plan), 'plan': plan, 'coverage': coverage,
        'requested': requested['ade_frozen'], 'deadline_unix': deadline, 'start_barrier': True}
    path = output / 'request.json'
    atomic_json(path, request)
    uuids = os.environ.get('QAVIT_ASSIGNED_GPU_UUIDS', '').split(',')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[1] if len(uuids) > 1 else str(gpus[1]),
               QAVIT_RESOURCE_ASSIGNED='1')
    lease_fds = tuple(int(fd) for fd in os.environ.get('QAVIT_GPU_LEASE_FDS', '').split(',') if fd)
    child = None
    try:
        with (output / 'worker.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), str(path)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                preexec_fn=parent_guard(os.getpid()), pass_fds=lease_fds)
        identity = process_identity(child.pid)
        atomic_json(directory / 'processes' / 'dual-segmentation.json', identity)
        # Validate the independent loader before starting either measurement.
        # In particular small subsets must overlap in evaluation, not merely in
        # process lifetime while the secondary GPU is still loading the model.
        while not (output / 'ready.json').exists():
            if (output / 'result.json').exists():
                check_identity(read_json(output / 'result.json'), request, path)
                raise ContractError('dual-task child completed without ready handshake')
            if child.poll() is not None:
                raise RemoteWorkerError({'code': 'ChildProcessError', 'origin': 'host',
                    'message': 'ADE worker exited before ready', 'returncode': child.returncode},
                    worker='ade_frozen', receipt=output / 'result.json', worker_log=output / 'worker.log')
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError('dual-task deadline exceeded while loading')
            time.sleep(.05)
        check_identity(read_json(output / 'ready.json'), request, path)
        atomic_json(output / 'start.json', {'request_sha256': file_sha256(path)})
        start = time.time()
        value = evaluate_task(backend, cfg, data, model, plan, requested['classification'], directory, native, coverage)
        finished = time.time()
        value['parallelism'] = {'mode': 'dual_task', 'physical_gpu': gpus[0], 'shards': 1,
                                'started_unix': start, 'finished_unix': finished}
        atomic_json(directory / 'classification.json', value)  # durable partial result on sibling failure
        name, summary = evaluation_summary(requested['classification'], value)
        while not (output / 'result.json').exists() and child.poll() is None:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError('dual-task experiment deadline exceeded')
            time.sleep(.05)
        if not (output / 'result.json').exists():
            raise RemoteWorkerError({'code': 'ChildProcessError', 'origin': 'host',
                'message': 'ADE worker exited without result', 'returncode': child.poll()},
                worker='ade_frozen', receipt=output / 'result.json', worker_log=output / 'worker.log')
        receipt = read_json(output / 'result.json')
        check_identity(receipt, request, path)
        if package_files(model_dir) != request['model_assets']:
            raise ContractError('dual-task evaluation changed frozen package')
        segmentation = receipt['evaluation']
        segmentation['parallelism'] = {'mode': 'dual_task', 'physical_gpu': gpus[1], 'shards': 1,
            'started_unix': receipt['started_unix'], 'finished_unix': receipt['finished_unix']}
        atomic_json(directory / 'segmentation.json', segmentation)
        second, second_summary = evaluation_summary(requested['ade_frozen'], segmentation)
        atomic_json(directory / 'dual_task_receipt.json', {
            'status': 'completed', 'physical_gpus': list(gpus[:2]),
            'gpu_uuids': [u for u in uuids[:2] if u], 'model_digest': request['model_digest'],
            'baseline_digest': request['baseline_digest'], 'model_assets': request['model_assets'],
            'classification_interval': [start, finished],
            'segmentation_interval': [receipt['started_unix'], receipt['finished_unix']],
            'overlap_seconds': max(0, min(finished, receipt['finished_unix']) - max(start, receipt['started_unix'])),
            'budget': 'one experiment wall interval; child time is not added to active/numeric wait',
            'deadline_unix': deadline})
        return {name: summary, second: second_summary}
    except BaseException as exc:
        partial = ['classification.json'] if (directory / 'classification.json').exists() else []
        child_result = output / 'result.json'
        if child_result.exists() and read_json(child_result).get('status') == 'completed':
            partial.append('dual_tasks/segmentation/result.json')
        atomic_json(directory / 'dual_task_failure.json', {
            'status': 'failed', 'error': failure_details(exc, {'approval': approval}),
            'partial_results': partial,
            'child_result_ref': str(child_result) if child_result.exists() else None,
            'model_assets': request['model_assets'], 'finished_unix': time.time()})
        raise
    finally:
        if child is not None:
            # Always reap, including after successful durable output. On parent
            # termination TaskRuntime also owns this registered process group.
            if child.poll() is None:
                TaskRuntime.stop_identity(identity)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def run(request, path):
    import torch
    from copy import deepcopy
    from tools.experiment_worker import backend_and_data, module_from_approval, evaluate_task, plan_constraints
    from tools.experiments import verify_assets
    from ptq_agent.execution.model import RTNExecutor
    from ptq_agent.execution.evaluation import model_digest
    from core.execution_contract import check_fp_state
    from runtime.guards import install_gradient_guard
    experiment = request['experiment']
    spec = experiment['payload']['execution']
    verify_assets(spec)
    torch.set_num_threads(experiment['payload']['resources']['cpu_threads'])
    seed = spec['seed']
    torch.manual_seed(seed)
    guard = install_gradient_guard(True)
    model_dir = Path(request['model_dir'])
    if package_files(model_dir) != request['model_assets']:
        raise ContractError('dual-task frozen model package changed before load')
    backend, cfg, data = backend_and_data(spec)
    # Match the parent's post-construction RNG reset; constructors may seed.
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        factory = RTNExecutor
        if not request['native']:
            plugin = module_from_approval(request['approval'], experiment)
            factory = plugin.make_executor
        from tools.frozen import FrozenData
        executor = factory(backend, deepcopy(cfg), FrozenData(data))
        model, manifest = executor.load(model_dir)  # deliberately never prepare/fit/export
        plan = manifest['plan'] if request['native'] else request['plan']
        coverage = plan_constraints(plan, spec['resolved_choices'], backend, cfg,
                                    spec.get('quantization_contract'), native=request['native'])
        check_fp_state(model, backend, spec.get('quantization_contract', {}))
        if spec.get('quantization_contract', {}).get('protocol'):
            from core.static_w4a4 import check_audit, native_audit
            from ptq_agent.execution.sites import site_catalog
            if request['native']:
                native_audit(model, plan, site_catalog(backend), coverage['static_w4a4'])
            else:
                check_audit(executor.audit(model, plan).get('static_w4a4'), coverage['static_w4a4'])
        result = {'model_digest': model_digest(model), 'baseline_digest': model_digest(backend.model),
                  'plan_sha256': fingerprint(plan), 'model_assets': package_files(model_dir),
                  'request_sha256': file_sha256(path), 'status': 'completed'}
        check_loaded_identity(result, request, path)
        if request.get('start_barrier'):
            atomic_json(path.parent / 'ready.json', result)
            while not (path.parent / 'start.json').exists():
                if request.get('deadline_unix') is not None and time.time() >= request['deadline_unix']:
                    raise TimeoutError('dual-task deadline exceeded waiting for start')
                time.sleep(.05)
            if read_json(path.parent / 'start.json').get('request_sha256') != file_sha256(path):
                raise ContractError('dual-task start handshake identity mismatch')
        start = time.time()
        if request.get('deadline_unix') is not None and start >= request['deadline_unix']:
            raise TimeoutError('dual-task deadline exceeded before evaluation')
        value = evaluate_task(backend, cfg, data, model, plan, request['requested'], path.parent,
                              request['native'], coverage)
        check_loaded_identity({**result, 'model_digest': model_digest(model),
            'baseline_digest': model_digest(backend.model), 'model_assets': package_files(model_dir)}, request, path)
        return {**result, 'evaluation': value, 'started_unix': start, 'finished_unix': time.time()}
    finally:
        guard.remove()
        backend.reset()
        del backend
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    path = Path(sys.argv[1])
    request = read_json(path)
    try:
        result = run(request, path)
    except BaseException as exc:
        traceback.print_exc()
        result = {k: request[k] for k in ('model_digest', 'baseline_digest', 'plan_sha256', 'model_assets')}
        result.update(status='failed', error=failure_details(exc, request),
                      request_sha256=file_sha256(path), finished_unix=time.time())
    atomic_json(path.parent / 'result.json', result)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
