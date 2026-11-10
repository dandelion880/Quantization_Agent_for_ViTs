"""Numerical task deadlines stay separate from LLM timeouts and active budget."""
import time
import unittest
from copy import deepcopy

from core.cards import ContractError
from core.journal import read_json
from runtime.budget import BudgetLedger
from runtime.configuration import config_check
from runtime.task_queue import GlobalTaskQueue
from runtime.task_runtime import TaskRuntime
from tests.support import validation_config
from tests.test_infrastructure import TempCase
from tools.experiments import prepare_request


class NumericalTimeoutDefaults(unittest.TestCase):
    def test_draft_defaults_by_level_independent_of_llm_timeout(self):
        brief = validation_config()['research_brief']
        for level, expected in [('L1', 3600), ('L2', 10800), ('L3', 86400)]:
            draft = {'level': level, 'question': 'Timeout routing', 'measurements': ['elapsed']}
            original = deepcopy(draft)
            self.assertEqual(prepare_request(draft, brief, {'task_timeout_seconds': 3})[
                'resources']['estimated_seconds'], expected)
            self.assertEqual(draft, original)

    def test_config_and_explicit_task_overrides(self):
        cfg = validation_config()
        cfg['runtime']['numeric_task_timeout_seconds'] = {'L2': 12000}
        runtime = config_check(cfg)['runtime']
        self.assertEqual(runtime['numeric_task_timeout_seconds'],
                         {'L1': 3600, 'L2': 12000, 'L3': 86400})
        draft = {'level': 'L2', 'question': 'Timeout routing', 'measurements': ['elapsed']}
        self.assertEqual(prepare_request(draft, cfg['research_brief'], runtime)[
            'resources']['estimated_seconds'], 12000)
        draft['resources'] = {'estimated_seconds': 17}
        self.assertEqual(prepare_request(draft, cfg['research_brief'], runtime)[
            'resources']['estimated_seconds'], 17)

    def test_invalid_numeric_timeout_configuration_is_rejected(self):
        for invalid in [None, 3, {'L0': 10}, {'L1': 0}, {'L2': -1},
                        {'L3': float('inf')}, {'L1': True}, {'L2': '3600'}]:
            with self.subTest(value=invalid):
                cfg = validation_config()
                cfg['runtime']['numeric_task_timeout_seconds'] = invalid
                with self.assertRaises(ContractError):
                    config_check(cfg)


class NumericalWorkerDeadlines(TempCase):
    def test_default_l3_deadline_exceeds_active_budget_and_llm_timeout(self):
        journal = self.journal()
        budget = BudgetLedger(journal, {'wall_seconds': 10, 'clock_mode': 'active_segments',
                                        'exclude_numeric_wait': True})
        budget.begin_segment()
        queue = GlobalTaskQueue(journal)
        tid = queue.enqueue('numeric', 'experiment')
        queue.promote()
        runtime = TaskRuntime(journal.root, queue, budget, {'task_timeout_seconds': .1})
        try:
            runtime.launch(queue.get(tid), {'kind': 'test_control', 'value': 3,
                           'experiment': {'id': 'numeric-timeout-fixture', 'content_sha256': 'fixture',
                                          'payload': {'level': 'L3'}}}, input_versions={})
            request = read_json(queue.get(tid)['request_path'])
            self.assertGreater(request['deadline_unix'] - time.time(), 86390)
            reservation = budget.snapshot()['reservations'][request['execution_id']]
            self.assertEqual(reservation['estimated_seconds'], 86400)
            self.assertLess(budget.snapshot()['remaining_seconds'], 10)
            for _ in range(300):
                results = runtime.poll()
                if results:
                    break
                time.sleep(.01)
            self.assertEqual(results[0][1]['status'], 'completed')
            runtime.settle(results[0][1])
            queue.finish(tid)
        finally:
            runtime.stop()

