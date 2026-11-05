"""贯通边界回归；脚本数据只验证上下文、审计和预算控制，不代表科研测量。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from core.journal import atomic_json
from core.workflow import Workflow
from runtime.budget import BudgetLedger, BudgetExhausted
from tests.test_infrastructure import TempCase
from tests.support import hypothesis, experiment_payload, meta, review


class ReadinessRegression(TempCase):
    def evidence(self, wf):
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'fixture', 'status': 'completed',
            'observations': {'mse': .1}, 'provenance': {}, 'resources': {}, 'artifacts': [],
            'measurement_source': 'synthetic_computation', 'execution_id': 'fixture'}, refs=[spec['id']])
        old = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/old', cards=[spec, evidence, old])
        return spec, evidence, old

    def test_stateless_continuation_keeps_explicitly_read_old_cards_and_definitions(self):
        from tests.test_memory_delivery import DeliveryRegression
        wf = self.workflow(1)
        spec, evidence, old = self.evidence(wf)
        tid = wf.enqueue_role('fixture/generate', 'generate', refs=[wf.brief['id']],
                              context={'round': 1, 'slot': 'H2', 'hypothesis_type': 'engineering'})
        helper = DeliveryRegression()
        task, _ = helper.begin(wf, wf.queue.get(tid))
        task, _ = helper.tools(wf, task, [('read_card', {'ref': c['id']}) for c in (spec, evidence, old)])
        for i in range(10):
            new = make_card('HypothesisCard', wf.journal.run_id, hypothesis(2+i, 'H3'))
            wf.journal.commit('fixture/new/' + str(i), cards=[new])
        wf = Workflow(wf.run_dir)
        got = {r['result']['id'] for r in wf.context(task)['tool_results'] if 'id' in r['result']}
        self.assertTrue({spec['id'], evidence['id'], old['id']} <= got)

    def test_judger_gets_feedback_reminders_and_can_read_full_metareview_on_demand(self):
        wf = self.workflow(1)
        card = make_card('MetareviewCard', wf.journal.run_id, meta(1), author='recorder')
        wf.journal.commit('fixture/meta', cards=[card], events=[wf.memory.apply_metareview(card)])
        value = wf.retriever.retrieve('judger', query={'model': 'unmatched'}, after_version=wf.journal.memory_version)
        self.assertEqual(value['cards'], [])
        self.assertEqual(value['latest_metareview_ref'], card['id'])
        tid = wf.enqueue_role('fixture/screen', 'screen', context={})
        context = wf.context(wf.queue.get(tid))
        self.assertEqual(context['memory']['reminders']['read_card']['ref'], card['id'])
        self.assertFalse(context['memory']['reminders']['required'])
        self.assertFalse(context['memory']['reminders']['fields'])
        self.assertNotIn('scientific_analysis', context['memory']['reminders'])
        self.assertEqual(wf.host_tool({'inputs': {'parent_task': tid, 'name': 'read_card', 'arguments': {'ref': card['id']}}}), card)

    def test_code_gets_request_and_positions_keep_isolation(self):
        wf = self.workflow(1)
        spec, evidence, old = self.evidence(wf)
        tid = wf.enqueue_role('fixture/code', 'code', refs=[wf.brief['id'], spec['id']],
                              context={'experiment_ref': spec['id']})
        task = wf.queue.get(tid)
        value = wf.context(task)
        self.assertEqual(value['research_brief']['id'], wf.brief['id'])
        self.assertEqual(value['original_request']['id'], spec['id'])
        self.assertIn('directory', value['memory'])
        other = make_card('ReviewDecisionCard', wf.journal.run_id, review('position',
                          round=1, pair=['H1','H2'], exchange=1, side='B'))
        wf.journal.commit('fixture/position', cards=[other])
        tid = wf.enqueue_role('fixture/position', 'position', refs=[other['id']],
                              context={'round':1, 'pair':['H1','H2'], 'exchange':1, 'side':'A'})
        with self.assertRaises(ContractError):
            wf.context(wf.queue.get(tid))  # 显式误绑定对方对象也不能绕过隔离

    def test_recorder_selective_untested_analysis_and_empty_knowledge(self):
        wf = self.workflow(1)
        _, _, old = self.evidence(wf)
        tid = wf.enqueue_role('fixture/record', 'record', refs=[old['id']], context={
            'round':1, 'selection':None, 'selection_ref':None})
        wf.queue.ready(); task = wf.queue.claim(tid)
        payload = meta(1)
        payload['judgment_audit'] = [{'judgment_ref':old['id'], 'assessment':'untested',
                                     'evidence_refs':[], 'reasoning':'未做对应实验，不能判定预测。'}]
        wf.submit(task, payload, 'fixture')
        self.assertEqual(wf.queue.get(tid)['state'], 'completed')
        self.assertEqual(wf.memory.snapshot()['L2'], {})

    def test_single_request_refusal_is_not_global_exhaustion(self):
        budget = BudgetLedger(self.journal(), {'wall_seconds':100, 'closeout_reserve_seconds':20,
                              'shutdown_reserve_seconds':5}, clock=lambda:100)
        with self.assertRaises(ContractError) as raised:
            budget.reserve('oversize', 'experiment', estimated_seconds=80)
        self.assertNotIsInstance(raised.exception, BudgetExhausted)
        budget.reserve('record', 'llm', estimated_seconds=90)
        self.assertEqual(budget.check()['remaining_seconds'], 100)

    def test_dispatch_refusal_returns_to_parent(self):
        wf = self.workflow(1)
        tid = wf.queue.enqueue('fixture/literature', 'literature', inputs={'action':'search','query':'fixture'})
        # 请求超出可用时间，但全局预算仍在，不能终止driver。
        with patch.object(wf.budget, 'clock', return_value=wf.budget.snapshot()['deadline_unix']-10):
            wf.dispatch()
        task = wf.queue.get(tid)
        self.assertEqual(task['state'], 'failed')
        self.assertEqual(task['error']['code'], 'RequestBudgetInsufficient')
        self.assertEqual(wf.budget.snapshot()['reservations'], {})

    def test_closeout_records_partial_round_without_fabricating_completion(self):
        from tests.support import validation_config
        import time
        cfg = validation_config()
        cfg['budget'].update(wall_seconds=100, closeout_reserve_seconds=30,
                             shutdown_reserve_seconds=5, min_free_bytes=0)
        wf = Workflow(self.root / 'closeout', cfg)
        wf.advance()  # 已排队H1/H2/H3，但尚未执行。
        deadline = wf.budget.snapshot()['deadline_unix']
        def submit_recorder():
            count = 0
            for candidate in wf.queue.ready():
                task = wf.queue.claim(candidate['task_id'])
                self.assertEqual(task['inputs']['purpose'], 'record')
                self.assertTrue(task['inputs']['context']['incomplete_round'])
                wf.submit(task, meta(1, task['inputs']['context']['recording_batch']['judgment_refs']), 'fixture')
                count += 1
            return count
        with patch.object(wf.budget, 'clock', return_value=deadline-25), \
             patch.object(wf, 'dispatch', side_effect=submit_recorder):
            result = wf.run()
        self.assertEqual(result['status'], 'budget_stopped')
        self.assertFalse(result['workflow_completed'])
        self.assertTrue(result['state_saved'])
        self.assertFalse(any(e['type']=='round_finished' for e in wf.journal.events))
        self.assertEqual(len(wf.memory.snapshot()['L3']), 0)  # No science was produced before closeout.
        self.assertTrue(result['scientific_review_complete'])
        self.assertFalse(any(t['inputs'].get('purpose') == 'record' for t in wf.tasks().values()))
        self.assertEqual(wf.budget.snapshot()['deadline_unix'], deadline)
        self.assertEqual(sum(t['state']=='cancelled' for t in wf.tasks().values()), 3)

    def test_global_exhaustion_still_stops(self):
        clock = [100]
        budget = BudgetLedger(self.journal(), {'wall_seconds':100}, clock=lambda:clock[0])
        clock[0] = 201
        with self.assertRaises(BudgetExhausted):
            budget.reserve('late', 'llm', estimated_seconds=1)


if __name__ == '__main__':
    unittest.main()
