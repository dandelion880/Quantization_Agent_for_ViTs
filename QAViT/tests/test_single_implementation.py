"""Single-implementation queue regression; scripted roles, no empirical claim."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, fingerprint, make_card
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tools.safeguards import approval_receipt, validate_approval

SOURCE = 'def fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n'


class SingleImplementation(TempCase):
    begin = delivery.DeliveryRegression.begin

    def start(self):
        wf = self.workflow(1)
        author = wf.enqueue_role('r001/generate/H1', 'generate', refs=[wf.brief['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        researcher, _ = self.begin(wf, wf.queue.get(author))
        draft = {'question': 'Compare local activation error', 'measurements': ['input_relative_mse'],
                 'execution': {'service': 'tensor_probe', 'requires_code': True}}
        wf.receive_tools(researcher, {'tool_calls': [{'id': 'experiment', 'name': 'request_experiment',
            'arguments': {'experiment': draft}}]}, 'SCRIPTED')
        wf.queue.ready()
        request = wf.queue.claim(wf.queue.get(author)['waiting_for'][0])
        wf.progress_request(request)
        return wf, researcher, wf.queue.get(request['task_id'])

    def progress(self, wf, request):
        wf.queue.ready()
        current = wf.queue.get(request['task_id'])
        if current['state'] != 'running': current = wf.queue.claim(current['task_id'])
        with patch.object(wf.broker, 'cached', return_value=None): wf.progress_request(current)
        return wf.queue.get(request['task_id'])

    def coding(self, wf, request, source=SOURCE):
        revision = request.get('code_revision', 0)
        suffix = '/revision' + str(revision) if revision else ''
        key = request['key'] + '/code' + suffix
        task, context = self.begin(wf, wf.tasks()[key])
        self.assertIn('original_request', context)
        choices = {'site': 'blocks.0.mlp.fc2', 'parameters': {'bits': 4, 'signed': True, 'granularity': 'per_tensor'},
                   'calibration': {'split': 'calibration', 'count': 2},
                   'evaluation': {'split': 'search', 'count': 1},
                   'resolved_choices': {'training': False, 'activation_calibration': 'static'}}
        wf.submit(task, {'source': source, 'execution_choices': choices}, 'SCRIPTED')
        return self.progress(wf, request)

    def approve(self, wf, request):
        reviews = [t for t in wf.tasks().values() if t['inputs'].get('purpose') == 'safeguard' and t['state'] != 'completed']
        task, context = self.begin(wf, reviews[-1])
        self.assertIn('candidate_source', context)
        self.assertIn('original_request', context)
        self.assertNotIn('independent_reference_source', context)
        wf.submit(task, {'approved': True, 'rationale': 'Scripted review for queue regression only'}, 'SCRIPTED')
        return self.progress(wf, request)

    def test_minimal_request_defers_execution_binding_and_single_source_approval(self):
        wf, researcher, request = self.start()
        self.assertNotIn('experiment_ref', request)
        original = wf.journal.card(request['request_ref'])['payload']['original_request']
        self.assertEqual(set(original), {'question', 'measurements', 'execution'})
        self.assertNotIn('resources', original)
        self.assertEqual([t['inputs'].get('purpose') for t in wf.tasks().values() if t['inputs'].get('parent_task') == request['task_id']], ['code'])
        request = self.coding(wf, request)
        self.assertIn('experiment_ref', request)
        request = self.approve(wf, request)
        trial = wf.tasks()[request['key'] + '/execute']
        approval = trial['inputs']['approval']; execution = wf.journal.card(request['experiment_ref'])
        self.assertNotIn('reference', approval)
        self.assertNotIn('verification_sha256', approval)
        validate_approval(approval, execution)
        altered = deepcopy(execution); altered['payload']['execution']['evaluation']['count'] = 2
        altered['content_sha256'] = fingerprint({k: v for k, v in altered.items() if k != 'content_sha256'})
        with self.assertRaisesRegex(ContractError, '执行条件'): validate_approval(approval, altered)
        path = Path(approval['candidate']['path']); path.write_text(path.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ContractError, '源码哈希'): validate_approval(approval, execution)
        self.assertFalse(any(t['kind'] == 'verification' or t['inputs'].get('purpose') == 'reference' for t in wf.tasks().values()))

    def test_execution_error_revises_same_request_and_returns_evidence_to_researcher(self):
        wf, researcher, request = self.start()
        request = self.approve(wf, self.coding(wf, request))
        trial = wf.tasks()[request['key'] + '/execute']; wf.queue.ready(); wf.queue.claim(trial['task_id'])
        wf.queue.fail(trial['task_id'], {'code': 'ValueError', 'message': 'invalid implementation reshape'})
        request = self.progress(wf, request)
        self.assertEqual(request['code_revision'], 1)
        self.assertEqual(request['code_revision_feedback']['error']['code'], 'ValueError')
        original = request['request_ref']
        request = self.progress(wf, request)
        request = self.approve(wf, self.coding(wf, request, SOURCE + '\n# fixed implementation\n'))
        self.assertEqual(request['request_ref'], original)
        self.assertEqual(wf.queue.get(trial['task_id'])['state'], 'failed')
        trial = wf.tasks()[request['key'] + '/execute/revision1']; wf.queue.ready(); wf.queue.claim(trial['task_id'])
        evidence = make_card('EvidenceCard', wf.journal.run_id, {'experiment_ref': request['experiment_ref'],
            'execution_fingerprint': 'fixture', 'status': 'completed', 'observations': {'mse': 0.1},
            'provenance': {}, 'resources': {}, 'artifacts': [], 'measurement_source': 'test_double',
            'execution_id': 'scripted'}, refs=[request['experiment_ref']])
        wf.queue.finish(trial['task_id'], cards=[evidence], outputs=[evidence['id']])
        request = self.progress(wf, request)
        self.assertEqual(request['result']['evidence_ref'], evidence['id'])
        researcher, _ = self.begin(wf, wf.queue.get(researcher['task_id']))
        self.assertIn(evidence['id'], wf.required_refs(researcher))

    def test_actual_cpu_worker_failure_and_revision_recovers(self):
        # Roles are scripted; both failed and successful numerical workers really run.
        import time
        wf, researcher, request = self.start()
        bad = SOURCE.replace('return parameters', "raise ValueError('injected fit implementation failure')")
        request = self.approve(wf, self.coding(wf, request, bad))
        def execute(current, suffix=''):
            tid = wf.tasks()[current['key'] + '/execute' + suffix]['task_id']
            wf.dispatch()
            deadline = time.monotonic() + 40
            while wf.queue.get(tid)['state'] == 'running' and time.monotonic() < deadline:
                time.sleep(.1)
                wf.poll()
            self.assertNotEqual(wf.queue.get(tid)['state'], 'running')
            return wf.queue.get(tid)
        try:
            failed = execute(request)
            self.assertEqual(failed['state'], 'failed')
            self.assertEqual(failed['error']['origin'], 'implementation')
            request = self.progress(wf, request)
            self.assertEqual(request['code_revision'], 1)
            request = self.progress(wf, request)
            request = self.approve(wf, self.coding(wf, request, SOURCE))
            completed = execute(request, '/revision1')
            self.assertEqual(completed['state'], 'completed', completed.get('error'))
            evidence = wf.journal.card(completed['outputs'][0])
            self.assertEqual(evidence['payload']['measurement_source'], 'synthetic_computation')
            self.assertEqual(evidence['payload']['observations']['input_error']['relative_mse'], 0)
            request = self.progress(wf, request)
            self.assertEqual(request['result']['evidence_ref'], evidence['id'])
            resumed, _ = self.begin(wf, wf.queue.get(researcher['task_id']))
            self.assertIn(evidence['id'], wf.required_refs(resumed))
        finally:
            wf.runtime.stop()

    def test_unchanged_failed_source_is_never_executed_again(self):
        wf, _, request = self.start()
        request = self.approve(wf, self.coding(wf, request))
        trial = wf.tasks()[request['key'] + '/execute']; wf.queue.ready(); wf.queue.claim(trial['task_id'])
        wf.queue.fail(trial['task_id'], {'code': 'KeyError', 'message': 'wrong field'})
        request = self.progress(wf, request); request = self.progress(wf, request)
        request = self.coding(wf, request)
        self.assertEqual(request['result']['reason'], 'implementation_unchanged_after_failure')
        self.assertNotIn(request['key'] + '/execute/revision1', wf.tasks())
