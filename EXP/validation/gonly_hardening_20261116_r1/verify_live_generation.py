"""Read-only verification of real local Generation/torch receipts; never reruns work."""
from pathlib import Path
import importlib.util
import json

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('live_generation_validation', BASE / 'live_generation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from core.journal import atomic_json, file_sha256, read_json
from runtime.task_queue import process_alive


def verify(name='live_generation_r1'):
    wf = module.Workflow(BASE / name, read_only=True)
    head = (wf.journal.sequence, wf.journal.commits[-1]['sha256'])
    result = module.summary(wf, 'validation_completed', None)
    verified = []
    for card in wf.journal.find_cards('EvidenceCard'):
        payload = card['payload']
        execution_id = payload['execution_id']
        directory = wf.run_dir / 'jobs' / execution_id
        request_path, receipt_path = directory / 'request.json', directory / 'result.json'
        request, receipt = read_json(request_path), read_json(receipt_path)
        assert request['kind'] == 'experiment'
        assert receipt['execution_id'] == execution_id and receipt['task_id'] == request['task_id']
        assert receipt['request_sha256'] == file_sha256(request_path)
        assert receipt['experiment_ref'] == payload['experiment_ref']
        assert receipt['experiment_sha256'] == wf.journal.card(payload['experiment_ref'])['content_sha256']
        assert receipt['status'] == 'completed'
        assert receipt['data']['measurement_source'] == payload['measurement_source'] == 'synthetic_computation'
        assert receipt['core_sha256'] == request['core_sha256'] == result['core_identity']['core_sha256']
        provenance = receipt['data']['provenance']
        assert provenance['model_identity']['source'] == 'synthetic_torch_vit_validation_only'
        assert provenance['model_unchanged'] and provenance['isolation']['mechanism'] == 'Linux Landlock'
        assert provenance['gradient_audit']['optimizer_steps'] == 0
        for artifact in payload['artifacts']:
            assert file_sha256(artifact['path']) == artifact['sha256']
        task = wf.queue.get(receipt['task_id'])
        assert task['kind'] == 'experiment' and task['state'] == 'completed' and card['id'] in task['outputs']
        spawned = [e['task']['process'] for e in wf.journal.events if e['type'] == 'task_snapshot'
                   and e['task']['task_id'] == task['task_id'] and e['task'].get('process')]
        assert spawned and not any(process_alive(p) for p in spawned)
        verified.append({'evidence_ref': card['id'], 'execution_id': execution_id,
                         'request_path': str(request_path), 'request_sha256': file_sha256(request_path),
                         'receipt_path': str(receipt_path), 'receipt_sha256': file_sha256(receipt_path),
                         'core_sha256': receipt['core_sha256'], 'measurement_source': payload['measurement_source'],
                         'spawned_worker_processes': spawned, 'all_exited': True,
                         'worker_resources': receipt['resources'], 'verified': True})
    assert all(c['error'] is None and c['requested'] == {'model': 'gpt-6-astra', 'reasoning_effort': 'high'}
               for c in result['calls'])
    result['receipt_verification'] = verified
    result['verification_note'] = (
        'Initial harness incorrectly required measurement_source=real for the synthetic tiny backend. '
        'Initial result is preserved. Corrected only the EXP verifier and re-read original requests, '
        'receipts, process snapshots and Evidence, without extra model or numeric calls. '
        'Core source, permissions and numerical protocol were unchanged.')
    result['checks']['bound_real_worker_receipts_and_hashes'] = bool(verified) and all(x['verified'] for x in verified)
    result['checks']['registered_workers_all_exited'] = all(x['all_exited'] for x in verified)
    result['validation_passed'] = result['status'] == 'validation_completed' and all(result['checks'].values())
    assert head == (wf.journal.sequence, wf.journal.commits[-1]['sha256'])
    result['read_only_verification_head'] = head
    atomic_json(BASE / (name + '_result.json'), result)
    return result


if __name__ == '__main__':
    result = verify()
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'validation_passed')}, ensure_ascii=False))
