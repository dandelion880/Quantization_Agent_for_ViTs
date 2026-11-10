"""CPU regression for code-manifest identity versus producer permissions."""
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.support import experiment_payload
from tools.safeguards import write_artifact


class CodeAssetOwnershipRegression(TempCase):
    begin = delivery.DeliveryRegression.begin
    read = delivery.DeliveryRegression.read
    debate_task = delivery.DeliveryRegression.debate_task

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU only'))
        worker.start(); self.addCleanup(worker.stop)

    def public_spec(self, wf):
        payload = experiment_payload()
        payload['execution'].update(requires_code=True, interface='quantizer_v1')
        spec = make_card('ExperimentCard', wf.journal.run_id, payload)
        wf.journal.commit('public-spec', cards=[spec])
        return spec

    def code_source(self, marker):
        return '# ' + marker + '\ndef fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n'

    def produced_code(self, wf, spec, position):
        rid = wf.queue.enqueue('private-request', 'experiment_request', inputs={
            'parent_task': position['task_id'], 'requester': 'judger',
            'refs': [wf.brief['id']], 'draft': spec['payload']})
        wf.queue.ready(); wf.queue.claim(rid)
        wf.queue.change(rid, 'running', patch={'experiment_ref': spec['id']})
        wf.progress_request(wf.queue.get(rid))
        code = wf.tasks()['private-request/code']; wf.queue.ready(); code = wf.queue.claim(code['task_id'])
        wf.submit(code, {'interface': 'quantizer_v1', 'source': self.code_source('PRIVATE_CODE_OWNER'),
                         'explanation': 'CPU fixture'}, 'CPU_FIXTURE')
        code = wf.queue.get(code['task_id'])
        return rid, code, code['result']['artifact']

    def test_private_code_on_public_spec_not_granted_by_draft_path(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        own, other = self.debate_task(wf), self.debate_task(wf, side='B')
        rid, code, artifact = self.produced_code(wf, spec, other)
        draft = experiment_payload(); draft['scope']['path'] = artifact['path']
        wf.broker.preflight(draft)
        tid = wf.enqueue_role('interpret-descendant', 'interpret', refs=[wf.brief['id']],
            context={'experiment_draft': draft, 'requester': 'generation'}, parent_task=own['task_id'])
        task, _ = self.begin(wf, wf.queue.get(tid))
        self.assertTrue(wf.can_read_card(task, spec))
        self.assertFalse(wf.can_read_card(task, wf.journal.card(code['task_card'])))
        with self.assertRaises(ContractError): self.read(wf, task, 'read_asset', asset=artifact['path'])

    def test_bound_safeguard_can_read_same_branch_produced_code(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        own = self.debate_task(wf)
        rid, code, artifact = self.produced_code(wf, spec, own)
        tid = wf.enqueue_role('bound-safeguard', 'safeguard', refs=[wf.brief['id'], spec['id']],
            context={'experiment_ref': spec['id'], 'candidate': artifact}, parent_task=rid)
        task, _ = self.begin(wf, wf.queue.get(tid))
        self.assertTrue(wf.can_read_card(task, wf.journal.card(code['task_card'])))
        self.assertIn('PRIVATE_CODE_OWNER', self.read(wf, task, 'read_asset', asset=artifact['path'])['text'])

    def test_unregistered_manifest_requires_exact_host_handoff(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        artifact = write_artifact(wf.run_dir, spec, {'interface': 'quantizer_v1',
            'source': self.code_source('UNREGISTERED_FIXTURE')}, 'coding')
        draft = experiment_payload(); draft['scope']['path'] = artifact['path']
        tid = wf.enqueue_role('unbound-interpret', 'interpret', refs=[wf.brief['id']],
            context={'experiment_draft': draft, 'requester': 'generation'})
        task, _ = self.begin(wf, wf.queue.get(tid))
        with self.assertRaises(ContractError): self.read(wf, task, 'read_asset', asset=artifact['path'])
        wf.queue.finish(tid)
        tid = wf.enqueue_role('fixture-safeguard', 'safeguard', refs=[wf.brief['id'], spec['id']],
            context={'experiment_ref': spec['id'], 'candidate': artifact})
        task, _ = self.begin(wf, wf.queue.get(tid))
        self.assertIn('UNREGISTERED_FIXTURE', self.read(wf, task, 'read_asset', asset=artifact['path'])['text'])

    def test_private_owner_cannot_be_overridden_by_manifest_fallback(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        own, other = self.debate_task(wf), self.debate_task(wf, side='B')
        rid, code, artifact = self.produced_code(wf, spec, other)
        tid = wf.enqueue_role('opposite-bound-safeguard', 'safeguard', refs=[wf.brief['id'], spec['id']],
            context={'experiment_ref': spec['id'], 'candidate': artifact}, parent_task=own['task_id'])
        task = wf.queue.get(tid)
        self.assertFalse(wf.can_read_card(task, wf.journal.card(code['task_card'])))
        self.assertNotIn(artifact['path'], wf.authorized_assets(task))
        with self.assertRaises(ContractError): wf.request_body(task)

    def test_requester_can_read_completed_code_from_own_experiment(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        own = self.debate_task(wf)
        rid, code, artifact = self.produced_code(wf, spec, own)
        own, _ = self.begin(wf, wf.queue.get(own['task_id']))
        self.assertTrue(wf.can_read_card(own, wf.journal.card(code['task_card'])))
        self.assertIn('PRIVATE_CODE_OWNER', self.read(wf, own, 'read_asset', asset=artifact['path'])['text'])

    def test_source_handoff_is_not_limited_to_judger_role(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        tid = wf.enqueue_role('own-generation', 'generate', refs=[wf.brief['id']],
            context={'round': 1, 'slot': 'H1', 'hypothesis_type': 'existing_method'})
        own = wf.queue.get(tid)
        rid, code, artifact = self.produced_code(wf, spec, own)
        own, _ = self.begin(wf, wf.queue.get(tid))
        self.assertEqual(own['role'], 'generation')
        self.assertIn('PRIVATE_CODE_OWNER', self.read(wf, own, 'read_asset', asset=artifact['path'])['text'])

    def test_own_future_code_remains_hidden_until_next_call(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        own, _ = self.begin(wf, self.debate_task(wf))
        rid, code, artifact = self.produced_code(wf, spec, own)
        self.assertFalse(wf.can_read_card(own, wf.journal.card(code['task_card'])))
        with self.assertRaises(ContractError): self.read(wf, own, 'read_asset', asset=artifact['path'])

    def test_own_code_read_preserves_frozen_file_hash_check(self):
        wf = self.workflow(1); spec = self.public_spec(wf)
        own = self.debate_task(wf)
        rid, code, artifact = self.produced_code(wf, spec, own)
        own, _ = self.begin(wf, wf.queue.get(own['task_id']))
        self.assertTrue(wf.can_read_card(own, wf.journal.card(code['task_card'])))
        source = Path(artifact['path'])
        source.write_text(source.read_text() + '# changed after reader snapshot\n')
        with self.assertRaises(ContractError): self.read(wf, own, 'read_asset', asset=artifact['path'])
