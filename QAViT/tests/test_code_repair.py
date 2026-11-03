"""独立 CPU 子流程夹具；不进入真实研究 Memory，不调用 LLM/GPU。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import make_card
from core.workflow import Workflow
from tests.support import experiment_payload, review
from tests.test_infrastructure import TempCase


class CodeRepairRegression(TempCase):
    def setup_request(self):
        wf = self.workflow(1)
        draft = experiment_payload()
        draft['execution'].update(requires_code=True, interface='quantizer_v1')
        draft['validation'] = {'mathematics': 'fixture only', 'atol': 1e-6, 'rtol': 1e-6,
                               'cases': [{'seed': 1, 'values': [0., 1.]}]}
        tid = wf.queue.enqueue('fixture/request', 'experiment_request', inputs={
            'requester': 'judger', 'draft': draft, 'refs': [wf.brief['id']]})
        self.progress(wf, tid)
        return wf, tid

    def progress(self, wf, tid):
        wf.queue.ready()
        task = wf.queue.get(tid)
        self.assertIn(task['state'], ('ready', 'running'), 'request must be woken by the real queue before progressing')
        if task['state'] == 'ready': task = wf.queue.claim(tid)
        wf.progress_request(task)

    def complete(self, wf, key, *, result=None, card=None):
        wf.queue.ready()
        task = wf.queue.claim(wf.tasks()[key]['task_id'])
        wf.queue.finish(task['task_id'], result=result or {}, outputs=[card['id']] if card else [], cards=[card] if card else [])

    def review_attempt(self, wf, tid, *, approve=False, numerical_result=None):
        task = wf.queue.get(tid)
        revision = task.get('code_revision', 0)
        code = 'fixture/request/code' + (f'/revision{revision}' if revision else '')
        self.complete(wf, code, result={'artifact': {'source_sha256': f'fixture-candidate-{revision}', 'interface': 'quantizer_v1', 'path': '/fixture/not-executed'}})
        ref = 'fixture/request/reference' + (f'/revision{revision}' if revision else '')
        if wf.tasks()[ref]['state'] != 'completed':
            self.complete(wf, ref, result={'artifact': {'source_sha256': 'fixture-reference', 'interface': 'quantizer_v1', 'path': '/fixture/not-executed'}})
        self.progress(wf, tid)
        numerical = next(t for t in wf.tasks().values() if t['kind'] == 'verification' and t['state'] not in ('completed','failed','cancelled'))
        self.complete(wf, numerical['key'], result=numerical_result or {'status': 'passed', 'cases': [{'fixture': True}]})
        self.progress(wf, tid)
        safeguard = next(t for t in wf.tasks().values() if t['inputs'].get('purpose') == 'safeguard' and t['state'] not in ('completed','failed','cancelled'))
        card = make_card('ReviewDecisionCard', wf.journal.run_id, review('safeguard', approved=approve,
            fidelity='fixture rejection or acceptance', numerical_correctness='fixture only', experimental_validity='fixture only'))
        self.complete(wf, safeguard['key'], card=card)
        with patch('core.experiment_requests.approval_receipt', return_value={'candidate': {'source_sha256': f'fixture-candidate-{revision}', 'interface': 'quantizer_v1'}}):
            self.progress(wf, tid)
        return card

    def test_rejection_returns_to_coding_and_resume_does_not_duplicate(self):
        wf, tid = self.setup_request()
        rejected = self.review_attempt(wf, tid)
        self.assertNotEqual(wf.queue.get(tid)['state'], 'completed')
        self.assertEqual(wf.queue.get(tid)['code_revision'], 1)
        self.progress(wf, tid)
        revised = wf.tasks()['fixture/request/code/revision1']
        self.assertEqual(revised['inputs']['context']['revision_feedback']['review_ref'], rejected['id'])
        self.assertNotIn('revision_feedback', wf.tasks()['fixture/request/reference']['inputs']['context'])
        self.assertNotIn('revision_feedback', wf.tasks()['fixture/request/reference/revision1']['inputs']['context'])
        self.assertEqual(wf.tasks()['fixture/request/reference/revision1']['inputs']['refs'], wf.tasks()['fixture/request/reference']['inputs']['refs'])
        before = set(wf.tasks())
        restored = Workflow(wf.run_dir)
        restored.queue.ready()
        self.assertEqual(restored.queue.get(tid)['state'], 'waiting')
        self.assertEqual(set(restored.tasks()), before)
        self.assertEqual(len(restored.journal.find_cards('ExperimentCard')), 1)
        self.review_attempt(restored, tid, approve=True)
        self.assertIn('fixture/request/execute/revision1', restored.tasks())
        verifications = [t for t in restored.tasks().values() if t['kind'] == 'verification']
        self.assertEqual(len(verifications), 2)
        self.assertNotEqual(verifications[0]['inputs']['candidate'], verifications[1]['inputs']['candidate'])

    def test_repeated_rejections_are_bounded_and_never_execute(self):
        wf, tid = self.setup_request()
        for _ in range(3):
            self.review_attempt(wf, tid)
            if wf.queue.get(tid)['state'] != 'completed': self.progress(wf, tid)
        self.assertEqual(wf.queue.get(tid)['result']['status'], 'rejected')
        self.assertEqual(wf.queue.get(tid)['code_revision'], 2)
        self.assertNotIn('fixture/request/execute', wf.tasks())
        self.assertEqual(len(wf.journal.find_cards('ReviewDecisionCard')), 3)
        self.assertEqual(len(wf.journal.find_cards('EvidenceCard')), 0)

    def test_reference_revision_receives_only_its_own_execution_failure(self):
        self.check_reference_failure_delivery('reference')

    def test_candidate_failure_does_not_reach_independent_reference(self):
        self.check_reference_failure_delivery('candidate')

    def test_string_comparison_failure_is_not_a_reference_execution_error(self):
        wf,tid=self.setup_request()
        self.review_attempt(wf,tid,numerical_result={'status':'rejected',
            'reference_sha256':'fixture-reference',
            'failure':'structure mismatch at $.observations.coverage; COMPARISON_SECRET'})
        self.progress(wf,tid)
        context=wf.tasks()['fixture/request/reference/revision1']['inputs']['context']
        self.assertEqual(set(context),{'experiment_ref'})
        self.assertNotIn('COMPARISON_SECRET',str(context))
        self.assertEqual(wf.queue.get(tid)['state'],'waiting')

    def check_reference_failure_delivery(self, side):
        wf,tid=self.setup_request()
        self.review_attempt(wf,tid,numerical_result={'status':'rejected',
            'reference_sha256':'fixture-reference', 'candidate_sha256':'CANDIDATE_SECRET',
            'cases':[{'candidate_output':'COMPARISON_SECRET'}],
            'failure':{'side':side,'error':{'type':'ValueError','message':'own runtime error',
                                           'extra':'DO_NOT_FORWARD'}}})
        self.progress(wf,tid)
        initial=wf.tasks()['fixture/request/reference']
        revised=wf.tasks()['fixture/request/reference/revision1']
        context=revised['inputs']['context']
        self.assertEqual(revised['inputs']['refs'],initial['inputs']['refs'])
        self.assertNotIn('revision_feedback',context)
        if side=='reference':
            self.assertEqual(set(context),{'experiment_ref','self_execution_failure'})
            failure=context['self_execution_failure']
            self.assertEqual(failure['source_sha256'],'fixture-reference')
            self.assertEqual(failure['error'],{'type':'ValueError','message':'own runtime error'})
        else:self.assertEqual(set(context),{'experiment_ref'})
        self.assertNotIn('SECRET',str(context));self.assertNotIn('DO_NOT_FORWARD',str(context))
        self.assertEqual(len(wf.journal.find_cards('EvidenceCard')),0)

    def test_runtime_interface_failure_revises_and_revalidates_without_replaying_failed_code(self):
        wf, tid = self.setup_request()
        for revision in range(3):
            self.review_attempt(wf, tid, approve=True)
            key = 'fixture/request/execute' + (f'/revision{revision}' if revision else '')
            wf.queue.ready()
            child = wf.queue.claim(wf.tasks()[key]['task_id'])
            wf.queue.fail(child['task_id'], {'code': 'TypeError', 'message': 'unhashable type: dict'})
            with patch('core.experiment_requests.approval_receipt', return_value={'candidate': {'source_sha256': f'fixture-candidate-{revision}', 'interface': 'quantizer_v1'}}):
                self.progress(wf, tid)
            if revision < 2:
                self.assertEqual(wf.queue.get(tid)['code_revision'], revision + 1)
                self.progress(wf, tid)
                revised = wf.tasks()[f'fixture/request/code/revision{revision+1}']
                self.assertEqual(revised['inputs']['context']['revision_feedback']['error']['code'], 'TypeError')
                self.assertEqual(wf.queue.get(child['task_id'])['state'], 'failed')
                restored = Workflow(wf.run_dir)
                restored.queue.ready()
                self.assertEqual(restored.queue.get(tid)['state'], 'waiting')
                self.assertEqual(set(wf.tasks()), set(restored.tasks()))
                wf = restored
        self.assertEqual(wf.queue.get(tid)['state'], 'failed')
        self.assertEqual(len([t for t in wf.tasks().values() if t['kind'] == 'verification']), 3)
        self.assertEqual(len([t for t in wf.tasks().values() if t['kind'] == 'experiment']), 3)
        self.assertEqual(len(wf.journal.find_cards('ExperimentCard')), 1)
        self.assertEqual(len(wf.journal.find_cards('EvidenceCard')), 0)

    def test_specification_clarification_returns_to_original_requester_after_restore(self):
        wf, tid = self.setup_request()
        rejected = self.review_attempt(wf, tid)
        self.progress(wf, tid)
        parent = wf.queue.enqueue('fixture/requester', 'llm', role='judger', inputs={'purpose': 'arbitrate'})
        wf.queue.ready(); wf.queue.claim(parent)
        wf.wait_for(wf.queue.get(parent), [tid])
        request = wf.queue.get(tid)
        wf.queue.change(tid, 'waiting', patch={'inputs': {**request['inputs'], 'parent_task': parent}})
        child = wf.tasks()['fixture/request/code/revision1']
        wf.queue.ready(); wf.queue.claim(child['task_id'])
        question = {'question': 'Frozen validation cases miss required boundary; revise specification?', 'reason': 'Changing implementation cannot repair fixed cases'}
        wf.queue.change(child['task_id'], 'needs_clarification', patch={'clarification': question})
        restored = Workflow(wf.run_dir)
        restored.advance(); restored.queue.ready()
        self.assertEqual(restored.queue.get(tid)['result']['clarification'], question)
        self.assertEqual(restored.queue.get(tid)['result']['review_ref'], rejected['id'])
        self.assertEqual(restored.queue.get(child['task_id'])['state'], 'cancelled')
        self.assertEqual(restored.tasks()['fixture/request/reference/revision1']['state'], 'cancelled')
        self.assertEqual(restored.queue.get(parent)['state'], 'ready')
        self.assertEqual(restored.queue.get(parent)['tool_results'][-1]['result']['reason'], 'frozen_specification_clarification')
        before = restored.journal.sequence
        restored.return_code_clarifications()
        self.assertEqual(restored.journal.sequence, before)
        self.assertEqual(len(restored.journal.find_cards('ExperimentCard')), 1)
        self.assertEqual(len(restored.journal.find_cards('EvidenceCard')), 0)


class RecorderRepairRegression(TempCase):
    def test_invalid_analysis_preserves_original_draft_and_repairs_same_task(self):
        from copy import deepcopy
        from tests.support import meta
        from tests.test_experiment_preflight import DraftRepairRegression
        wf = self.workflow(1)
        judgment = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation'))
        wf.journal.commit('fixture/judgment', cards=[judgment])
        tid = wf.enqueue_role('r001/record', 'record', refs=[judgment['id']], context={
            'round': 1, 'selection': None, 'selection_ref': None})
        wf.queue.ready(); task = wf.queue.claim(tid)
        draft = meta(1); draft['scientific_analysis'] = ''
        response = {'decision': {'action': 'submit', 'content': draft}, 'tool_calls': []}
        rejected = DraftRepairRegression.deliver(self, wf, task, response)
        self.assertEqual(wf.queue.get(tid)['state'], 'retry')
        self.assertEqual(wf.queue.get(tid)['repair_feedback']['rejected_response'], rejected)
        self.assertEqual(wf.journal.find_cards('MetareviewCard'), [])
        restored = Workflow(wf.run_dir)
        self.assertEqual(restored.context(restored.queue.get(tid))['repair_feedback']['rejected_response'], rejected)
        restored.queue.ready(); task = restored.queue.claim(tid)
        repaired = deepcopy(rejected)
        repaired['decision']['content']['judgment_audit'] = meta(1, [judgment['id']])['judgment_audit']
        repaired['decision']['content']['scientific_analysis'] = meta(1)['scientific_analysis']
        DraftRepairRegression.deliver(self, restored, task, repaired)
        self.assertEqual(restored.queue.get(tid)['state'], 'completed')
        self.assertEqual(len(restored.journal.find_cards('MetareviewCard')), 1)
        self.assertEqual(restored.memory.snapshot()['L3'], restored.queue.get(tid)['outputs'])


if __name__ == '__main__': unittest.main()
