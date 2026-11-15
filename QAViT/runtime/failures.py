"""Preserve numerical failure provenance across the worker process boundary."""
from pathlib import Path
import traceback

from core.cards import ContractError


class RemoteWorkerError(ContractError):
    """An identity-checked child failure, with its original exception code."""
    def __init__(self, error, *, worker, receipt, worker_log):
        if (not isinstance(error, dict) or not isinstance(error.get('code'), str)
                or not isinstance(error.get('message'), str)
                or error.get('origin') not in ('host', 'implementation')):
            raise ContractError('child failure receipt has no valid structured provenance')
        self.error = {**error, 'subprocess': {'worker': worker, 'receipt': str(receipt),
                                             'worker_log': str(worker_log)}}
        super().__init__(error['message'])

    def to_dict(self):
        return dict(self.error)


def failure_origin(exc, request):
    """Use actual failing frames, not an exception's name, to route code repair."""
    if isinstance(exc, RemoteWorkerError):
        return exc.error['origin']
    candidate = (request.get('approval') or {}).get('candidate') or {}
    source = candidate.get('path')
    if not source:
        return 'host'
    frames = traceback.extract_tb(exc.__traceback__)
    if any(Path(frame.filename).resolve() == Path(source).resolve() for frame in frames):
        return 'implementation'
    # Contracts on returned state run after the approved plugin frame exits.
    validators = {'tools/result_contract.py', 'core/execution_contract.py', 'core/static_w4a4.py'}
    if isinstance(exc, ContractError) and frames:
        last = frames[-1]
        if (any(last.filename.endswith('/' + name) for name in validators)
                or last.name in ('execute_custom', 'tensor_probe', 'evaluate_services', 'plan_constraints', 'check_loaded_identity')
                or (last.name == 'run_experiment' and any(term in str(exc) for term in
                    ('缺少供宿主重载', '导出包必须位于', '改变了冻结基准模型')))):
            return 'implementation'
    return 'host'


def failure_details(exc, request):
    error = exc.to_dict() if hasattr(exc, 'to_dict') else {'code': type(exc).__name__, 'message': str(exc)}
    return {**error, 'origin': failure_origin(exc, request)}
