"""Host revision transactions, evidence handoff and exact-version scheduling."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests import test_gonly as gonly
from tests import test_memory_delivery as delivery
from tests import test_generation_loop as loop
from tests.support import hypothesis


class CandidateRevision(TempCase):
    workflow_case = gonly.GOnlyRegression.workflow_case
    drive = gonly.GOnlyRegression.drive
    begin = delivery.DeliveryRegression.begin

    def revision(self, wf, task, parent, evidence=None):
        body = hypothesis(1, 'H2')
        body.update(title='Narrow intervention after contrary evidence', evidence_refs=evidence or [])
        args = {'parent_ref': parent, 'change_type': 'scientific',
                'reason': 'Total error contradicts local objective; revised scope remains unvalidated',
                'hypothesis': body}
        wf.receive_tools(task, {'tool_calls': [{'id': 'revise', 'name': 'revise_candidate', 'arguments': args}]}, 'fixture')
        wf.queue.ready()
        child = wf.queue.claim(wf.queue.get(task['task_id'])['waiting_for'][0])
        result = wf.host_tool(child)
        return child, result

    def test_experiment_revision_resume_reselect_and_exact_l3_binding(self):
        wf, task = self.drive(self.workflow_case(), stop=lambda t: t['inputs']['purpose'] == 'select')
        from core.journal import atomic_json
        body = wf.request_body(task)
        path = wf.run_dir / 'select_request.json'
        atomic_json(path, body)
        wf.queue.change(task['task_id'], 'running', patch={'request_path': str(path)})
        task = wf.queue.get(task['task_id'])
        old = deepcopy(wf.round_hypotheses(1)['H2'])
        request = loop.GenerationLoop.ask(self, wf, task)
        evidence = loop.GenerationLoop.measured(self, wf, request, 1)
        task, _ = self.begin(wf, wf.queue.get(task['task_id']))
        child, result = self.revision(wf, task, old['id'], [evidence['id']])
        self.assertEqual(result['validation_status'], 'unvalidated_revision')
        new = wf.journal.card(result['candidate_ref'])
        self.assertEqual(new['previous'], old['id'])
        self.assertEqual(new['version'], old['version'] + 1)
        self.assertEqual(wf.journal.card(old['id']), old)
        self.assertEqual(new['payload']['evidence_refs'], [evidence['id']])
        # Crash after publication, before writing host-tool completion.
        wf = Workflow(wf.run_dir)
        self.assertEqual(wf.host_tool(child), result)
        self.assertEqual(len([e for e in wf.journal.events if e['type'] == 'candidate_revised']), 1)
        wf.queue.finish(child['task_id'], result=result)
        with self.assertRaisesRegex(ContractError, '过期'):
            wf.submit(task, {'selected_slot': 'H2', 'rationale': 'stale'}, None)
        wf.advance()
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'cancelled')
        select = wf.queue.get_by_key(wf.selection_key(1))
        self.assertNotEqual(select['task_id'], task['task_id'])
        self.assertEqual(select['inputs']['context']['candidate_refs']['H2'], new['id'])
        self.assertIn(new['id'], wf.required_refs(select))
        revision = select['inputs']['context']['revision_context'][0]
        self.assertEqual(revision['parent_ref'], old['id'])
        self.assertEqual(revision['basis_evidence_refs'], [evidence['id']])
        self.assertEqual(revision['validation_status'], 'unvalidated_revision')
        count = len(wf.tasks())
        wf.advance(); wf.advance()
        self.assertEqual(len(wf.tasks()), count)
        wf.queue.ready(); select = wf.queue.claim(select['task_id'])
        wf.submit(select, {'selected_slot': 'H2', 'rationale': 'next measurement of revised candidate',
                           'evidence_refs': [evidence['id']]}, None)
        wf.advance()
        plan = wf.tasks()['r001/plan_l3']
        self.assertEqual(plan['inputs']['context']['hypothesis_ref'], new['id'])
        self.assertFalse(wf.summary()['formal_validation_success'])

    def test_engineering_repair_and_mixed_batch_do_not_create_revision(self):
        wf, task = self.drive(self.workflow_case(), stop=lambda t: t['inputs']['purpose'] == 'select')
        args = {'parent_ref': wf.round_hypotheses(1)['H2']['id'], 'change_type': 'engineering',
                'reason': 'fix importer only', 'hypothesis': hypothesis(1, 'H2')}
        with self.assertRaises(ContractError):
            wf.receive_tools(task, {'tool_calls': [{'id': 'revision', 'name': 'revise_candidate', 'arguments': args}]}, None)
        args['change_type'] = 'scientific'
        with self.assertRaisesRegex(ContractError, '单独'):
            wf.receive_tools(task, {'tool_calls': [{'id': x, 'name': 'revise_candidate', 'arguments': args} for x in ('a', 'b')]}, None)
        self.assertFalse([e for e in wf.journal.events if e['type'] == 'candidate_revised'])
