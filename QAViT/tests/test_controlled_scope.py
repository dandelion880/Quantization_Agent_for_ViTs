"""CPU/control-flow fixtures only; no real model calls or DeiT measurements."""
from pathlib import Path
import json
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import Journal, atomic_json, read_json
from core.workflow import Workflow
from runtime.budget import BudgetLedger, BudgetExhausted, RequestBudgetInsufficient
from runtime.run_segments import run_managed
from runtime.task_runtime import TaskRuntime
from runtime.task_queue import GlobalTaskQueue
from tests.support import validation_config, hypothesis
from tests.test_infrastructure import TempCase, WorkflowSimulation


class ScopeBoundary(TempCase):
    def setup_scope(self, budget=None):
        cfg = validation_config()
        cfg.update(purpose="infrastructure_test", scope_target={"round": 2, "stage": "generation"})
        cfg["budget"]["min_free_bytes"] = 0
        cfg["budget"].update(budget or {})
        wf = Workflow(self.root / "scope", cfg)
        if cfg['budget'].get('clock_mode') == 'active_segments':
            wf.budget.begin_segment()
        script = WorkflowSimulation()
        for _ in range(100):
            self.assertFalse(wf.advance())
            if "r002/generate/H1" in wf.tasks():
                return wf
            for candidate in wf.queue.ready():
                task = wf.queue.claim(candidate["task_id"])
                wf.submit(task, script.scripted(wf, task), "SCRIPTED_CONTROL_FLOW_ONLY")
        self.fail("round one fixture did not finish")

    def submit_h(self, wf, slot):
        wf.queue.ready()
        task = wf.queue.claim(wf.tasks()["r002/generate/" + slot]["task_id"])
        wf.submit(task, WorkflowSimulation().scripted(wf, task), "SCRIPTED_CONTROL_FLOW_ONLY")

    def test_partial_generation_children_last_receipt_and_repeat_resume(self):
        wf = self.setup_scope()
        self.assertEqual(len(wf.memory.snapshot()["L3"]), 1)
        for slot in ("H1", "H2"):
            self.submit_h(wf, slot)
            wf = Workflow(wf.run_dir)
            self.assertFalse(wf.advance())
            self.assertFalse(wf.advance())
            self.assertFalse(any("r002/screen" in k for k in wf.tasks()))
        # A required child must settle even if its parent's durable H is present.
        parent = wf.tasks()["r002/generate/H3"]
        child = wf.queue.enqueue("r002/generate/H3/control_child", "host_tool",
                                 inputs={"parent_task": parent["task_id"]})
        self.submit_h(wf, "H3")
        self.assertFalse(wf.advance())
        wf.queue.ready(); wf.queue.claim(child); wf.queue.finish(child, result={"source": "test_double"})
        self.assertTrue(wf.advance())
        seq = wf.journal.sequence
        self.assertTrue(wf.advance())
        self.assertEqual(seq, wf.journal.sequence)
        for _ in range(2):
            wf = Workflow(wf.run_dir)
            with patch.object(wf, "dispatch", side_effect=AssertionError("target must never dispatch")):
                result = wf.run()
            self.assertEqual(result["status"], "scope_target_reached")
            self.assertTrue(result["scope_target_reached"])
            self.assertFalse(result["workflow_completed"])
            self.assertEqual(result["completed_rounds"], [1])
            self.assertFalse(any(k.startswith("r002/") and "/generate/" not in k for k in wf.tasks()))
        self.assertEqual(len(wf.memory.snapshot()["L3"]), 1)
        self.assertTrue(wf.memory.recording_batch()["review_required"])

    def test_budget_closeout_does_not_add_round_two_recorder(self):
        wf = self.setup_scope()
        self.submit_h(wf, "H1")
        wf.event("budget_closeout_started", "budget/closeout", round=2)
        with patch.object(wf, "dispatch", side_effect=AssertionError("no second round recorder")):
            report = wf.run()
        self.assertEqual(report["status"], "budget_stopped")
        self.assertFalse(any(t["inputs"].get("purpose") == "record" and
            t["inputs"]["context"].get("round") == 2 for t in wf.tasks().values()))

    def test_last_durable_generation_receipt_during_shutdown_marks_only_scope(self):
        wf = self.setup_scope()
        for slot in ("H1", "H2"):
            self.submit_h(wf, slot)
        wf.queue.ready()
        task = wf.queue.claim(wf.tasks()["r002/generate/H3"]["task_id"])
        execution = "last-receipt-control-fixture"
        wf.budget.reserve(execution, "llm", estimated_seconds=1)
        receipt = {"status": "completed", "execution_id": execution, "data": {"response": {
            "source": "test_double", "decision": {"action": "submit",
            "content": WorkflowSimulation().scripted(wf, task)}}}}
        with patch.object(wf.runtime, "poll", side_effect=[[], KeyboardInterrupt(),
                [(task, receipt, self.root / "SCRIPTED_ONLY.json")]]), \
             patch.object(wf, "dispatch", side_effect=AssertionError("shutdown cannot dispatch")):
            report = wf.run()
        self.assertEqual(report["status"], "scope_target_reached")
        self.assertEqual(report["completed_rounds"], [1])
        self.assertFalse(any("r002/screen" in k for k in wf.tasks()))

    def test_no_unused_recorder_reserve_blocks_partial_round_two(self):
        wf = self.setup_scope({'clock_mode': 'active_segments', 'wall_seconds': 10800,
                               'closeout_reserve_seconds': 900, 'shutdown_reserve_seconds': 60})
        self.submit_h(wf, 'H1')
        self.submit_h(wf, 'H2')
        state = wf.budget.snapshot()
        deadline = state['deadline_unix']
        self.assertEqual(state['configured_closeout_reserve_seconds'], 900)
        self.assertEqual(state['effective_closeout_reserve_seconds'], 0)
        release = state['closeout_reserve_releases'][0]
        self.assertEqual(release['recorder_state'], 'completed')
        self.assertTrue(release['recorder_outputs'])
        for remaining in (959, 100):
            wf = Workflow(wf.run_dir)
            clock = lambda: deadline - remaining
            wf.budget.clock = clock
            self.assertFalse(wf.release_scope_closeout_reserve())
            self.assertEqual(len(wf.budget.snapshot()['closeout_reserve_releases']), 1)
            self.assertEqual(wf.budget.snapshot()['deadline_unix'], deadline)
            self.assertEqual(wf.budget.input_receipt()['research_remaining_seconds'], remaining - 60)
            wf.budget.check(estimated_seconds=remaining - 60)
            with self.assertRaises(RequestBudgetInsufficient):
                wf.budget.check(estimated_seconds=remaining - 59)
            with patch('runtime.driver.time.time', clock), \
                 patch.object(wf, 'dispatch', side_effect=KeyboardInterrupt()) as dispatch:
                report = wf.run()
            dispatch.assert_called_once()
            self.assertEqual(report['status'], 'interrupted')
            self.assertNotIn('cancelled', report['tasks'])
            self.assertFalse(any(e['type'] == 'budget_closeout_started' for e in wf.journal.events))
        wf.budget.clock = lambda: deadline - 60
        with patch('runtime.driver.time.time', wf.budget.clock), patch.object(wf, 'dispatch') as dispatch:
            report = wf.run()
        dispatch.assert_not_called()
        self.assertEqual(report['status'], 'budget_exhausted')
        self.assertEqual(report['budget']['worker_deadline_unix'], deadline - 60)
        self.assertFalse(any('r002/screen' in k or 'r002/record' in k for k in wf.tasks()))

    def test_release_after_poll_before_closeout_and_preserve_unfinished_first_round(self):
        cfg = validation_config()
        cfg.update(scope_target={'round': 2, 'stage': 'generation'})
        cfg['budget'].update(clock_mode='active_segments', wall_seconds=10800,
                             closeout_reserve_seconds=900, shutdown_reserve_seconds=60, min_free_bytes=0)
        wf = Workflow(self.root / 'release_after_poll', cfg)
        wf.budget.begin_segment()
        wf.advance(); wf.queue.ready()
        generator = wf.queue.claim(wf.tasks()['r001/generate/H1']['task_id'])
        wf.submit(generator, WorkflowSimulation().scripted(wf, generator), 'SCRIPTED_HYPOTHESIS_ONLY')
        task = wf.enqueue_recorder(1)
        self.assertIsNotNone(task)
        self.assertFalse(wf.release_scope_closeout_reserve())
        self.assertEqual(wf.budget.snapshot()['effective_closeout_reserve_seconds'], 900)
        wf.queue.ready(); claimed = wf.queue.claim(task)
        script = WorkflowSimulation().scripted(wf, claimed)
        original_poll = wf.poll
        accepted = [False]
        def poll():
            if not accepted[0]:
                wf.submit(claimed, script, 'SCRIPTED_RECORDER_CONTROL_FLOW_ONLY')
                accepted[0] = True
            return original_poll()
        deadline = wf.budget.snapshot()['deadline_unix']
        wf.budget.clock = lambda: deadline - 959
        with patch('runtime.driver.time.time', wf.budget.clock), patch.object(wf, 'poll', poll), \
             patch.object(wf, 'advance', return_value=False), \
             patch.object(wf, 'dispatch', side_effect=KeyboardInterrupt()) as dispatch:
            report = wf.run()
        dispatch.assert_called_once()
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['budget']['effective_closeout_reserve_seconds'], 0)
        self.assertFalse(any(e['type'] == 'round_finished' for e in wf.journal.events))

    def test_deferred_first_round_release_does_not_claim_recorder_success_or_change_legacy(self):
        for mode in ('active_segments', 'continuous'):
            cfg = validation_config()
            cfg.update(scope_target={'round': 2, 'stage': 'generation'})
            cfg['budget'].update(clock_mode=mode, closeout_reserve_seconds=900,
                                 shutdown_reserve_seconds=60, min_free_bytes=0)
            wf = Workflow(self.root / mode, cfg)
            if mode == 'active_segments':
                wf.budget.begin_segment()
            wf.advance(); wf.queue.ready()
            generator = wf.queue.claim(wf.tasks()['r001/generate/H1']['task_id'])
            wf.submit(generator, WorkflowSimulation().scripted(wf, generator), 'SCRIPTED_HYPOTHESIS_ONLY')
            task = wf.enqueue_recorder(1)
            self.assertIsNotNone(task)
            wf.queue.ready(); wf.queue.claim(task)
            wf.queue.fail(task, {'code': 'scripted_recorder_failure'})
            self.assertFalse(wf.release_scope_closeout_reserve())
            wf.event('round_finished', 'SCRIPTED_DEFERRED_ROUND_ONLY', round=1, scientific_review='pending')
            self.assertEqual(wf.release_scope_closeout_reserve(), mode == 'active_segments')
            self.assertEqual(wf.queue.get(task)['state'], 'failed')
            self.assertFalse(wf.journal.find_cards('MetareviewCard'))
            self.assertEqual(wf.budget.snapshot()['effective_closeout_reserve_seconds'],
                             0 if mode == 'active_segments' else 900)


class ActiveSegments(TempCase):
    def test_new_explicit_total_authorization_keeps_spending_segments_and_fixed_deadline(self):
        clock = [100.0]
        ledger = self.ledger(clock)
        path = self.root / 'USER_AUTHORIZATION_TEST_FIXTURE.json'
        atomic_json(path, {'author': 'user', 'total_wall_seconds': 150,
                           'all_other_budgets_unchanged': True})
        ledger.begin_segment()
        with self.assertRaises(ContractError):
            ledger.authorize_total_wall_seconds(150, reason='explicit test authorization', authorization_path=path)
        clock[0] = 130
        ledger.end_segment(processes_stopped=True, reason='test fully stopped')
        old_segments = ledger.snapshot()['segments']
        ledger.authorize_total_wall_seconds(150, reason='explicit test authorization', authorization_path=path)
        state = ledger.snapshot()
        self.assertEqual(state['elapsed_seconds'], 30)
        self.assertEqual(state['remaining_seconds'], 120)
        self.assertEqual(state['segments'], old_segments)
        self.assertEqual(state['config']['wall_seconds'], 100)
        self.assertEqual(state['total_wall_seconds'], 150)
        self.assertEqual(state['wall_budget_authorizations'][0]['additional_seconds'], 50)
        ledger.authorize_total_wall_seconds(150, reason='same authorization replay', authorization_path=path)
        self.assertEqual(len(ledger.snapshot()['wall_budget_authorizations']), 1)
        clock[0] = 1000
        ledger = BudgetLedger(Journal(ledger.journal.root, 'run'), clock=lambda: clock[0])
        self.assertEqual(ledger.snapshot()['remaining_seconds'], 120)
        ledger.begin_segment()
        self.assertEqual(ledger.snapshot()['deadline_unix'], 1120)
        clock[0] = 1030
        self.assertEqual(ledger.snapshot()['deadline_unix'], 1120)
        self.assertEqual(ledger.snapshot()['worker_deadline_unix'], 1115)
        self.assertEqual(ledger.snapshot()['elapsed_seconds'], 60)
        self.assertEqual(ledger.input_receipt()['total_wall_seconds'], 150)
        with self.assertRaises(ContractError):
            ledger.extend(50, 'old generic extend remains forbidden')
        ledger.end_segment(processes_stopped=True, reason='test fully stopped')
        atomic_json(path, {'author': 'user', 'total_wall_seconds': 200,
                           'all_other_budgets_unchanged': False})
        with self.assertRaises(ContractError):
            ledger.authorize_total_wall_seconds(200, reason='invalid fixture', authorization_path=path)
        self.assertEqual(ledger.snapshot()['total_wall_seconds'], 150)

    def test_rebuilt_summary_preserves_original_stop_error(self):
        wf = self.workflow(1)
        error = {'code': 'original_failure', 'message': 'explicit test fixture'}
        wf.event('run_status', 'fixture/stop', status='blocked', error=error)
        self.assertEqual(wf.summary()['error'], error)
        self.assertIsNone(wf.summary('running')['error'])

    def ledger(self, clock):
        return BudgetLedger(self.journal(), {"wall_seconds": 100, "clock_mode": "active_segments",
            "shutdown_reserve_seconds": 5}, clock=lambda: clock[0])

    def test_pause_resume_fixed_deadline_and_legacy_semantics(self):
        clock = [100.0]
        ledger = self.ledger(clock)
        ledger.begin_segment(started_unix=98)
        self.assertEqual(ledger.snapshot()["deadline_unix"], 198)
        clock[0] = 120
        self.assertEqual(ledger.snapshot()["deadline_unix"], 198)
        ledger.end_segment(processes_stopped=True, reason="test confirmation")
        ledger.end_segment(processes_stopped=True, reason="idempotent")
        clock[0] = 1000
        self.assertEqual(ledger.snapshot()["elapsed_seconds"], 22)
        self.assertIsNone(ledger.snapshot()["worker_deadline_unix"])
        ledger = BudgetLedger(Journal(ledger.journal.root, "run"), clock=lambda: clock[0])
        ledger.begin_segment(started_unix=998)
        self.assertEqual(ledger.snapshot()["deadline_unix"], 1076)
        self.assertEqual(ledger.snapshot()["worker_deadline_unix"], 1071)
        clock[0] = 1040
        self.assertEqual(ledger.snapshot()["remaining_seconds"], 36)
        self.assertEqual(ledger.snapshot()["elapsed_seconds"], 64)
        with self.assertRaises(ContractError):
            ledger.extend(100, "not authorized")
        clock[0] = 1077
        with self.assertRaises(BudgetExhausted):
            ledger.reserve("late", "llm", estimated_seconds=1)
        ledger.end_segment(processes_stopped=True, reason="deadline")
        self.assertEqual(ledger.snapshot()["elapsed_seconds"], 101)

    def test_missing_stop_counts_unknown_interval_and_requires_confirmation(self):
        clock = [100.0]
        ledger = self.ledger(clock)
        ledger.begin_segment()
        clock[0] = 145
        recovered = BudgetLedger(Journal(ledger.journal.root, "run"), clock=lambda: clock[0])
        with self.assertRaises(ContractError):
            recovered.begin_segment()
        with self.assertRaises(ContractError):
            recovered.end_segment(processes_stopped=False, reason="still running")
        recovered.end_segment(processes_stopped=True, reason="recovered", recovered=True)
        clock[0] = 170
        recovered.begin_segment()
        self.assertEqual(recovered.snapshot()["remaining_seconds"], 55)
        self.assertEqual(recovered.snapshot()["deadline_unix"], 225)
        self.assertTrue(recovered.snapshot()["segments"][0]["recovered_without_stop"])

    def test_real_cpu_worker_stale_intent_and_unsettled_attempt(self):
        journal = self.journal()
        ledger = BudgetLedger(journal, {"wall_seconds": 60, "clock_mode": "active_segments",
                                       "shutdown_reserve_seconds": 5})
        queue = GlobalTaskQueue(journal)
        runtime = TaskRuntime(journal.root, queue, ledger, {"cpu_threads": 1})
        self.addCleanup(runtime.stop)
        ledger.begin_segment()
        task_id = queue.enqueue("cpu", "test_control")
        queue.ready()
        execution = task_id + "-a0001"
        ledger.reserve(execution, "test_control", estimated_seconds=1)
        path = journal.root / "jobs" / execution / "request.json"
        atomic_json(path, {"deadline_unix": time.time() - 10, "task_id": task_id,
                          "execution_id": execution, "budget_segment_id": "segment-1"})
        runtime.settle_abandoned_intents()
        ledger.end_segment(processes_stopped=True, reason="CPU fixture paused")
        ledger.begin_segment()
        runtime.launch(queue.get(task_id), {"kind": "test_control", "value": 5, "estimated_seconds": 10}, input_versions={})
        current = queue.get(task_id)
        request = read_json(Path(current["request_path"]))
        self.assertEqual(request["budget_segment_id"], "segment-2")
        self.assertEqual(current["attempt"], 2)
        self.assertLessEqual(request["deadline_unix"], ledger.snapshot()["worker_deadline_unix"])
        self.assertLess(read_json(path)["deadline_unix"], time.time())
        rows = []
        until = time.time() + 15
        while not rows and time.time() < until:
            rows = runtime.poll(); time.sleep(.05)
        self.assertEqual(rows[0][1]["data"], {"source": "test_double", "value": 5})
        runtime.settle(rows[0][1]); queue.finish(task_id)
        runtime.stop()
        ledger.end_segment(processes_stopped=True, reason="CPU fixture stopped")
        self.assertEqual(ledger.snapshot()["settlements"][execution]["reported_usage"], None)
        self.assertFalse(runtime.live_processes())

    def test_monitor_confirms_supervisor_exit_without_any_science(self):
        cfg = validation_config()
        cfg["budget"].update(clock_mode="active_segments", min_free_bytes=0)
        cfg.update(scope_target={"round": 2, "stage": "generation"})
        wf = Workflow(self.root / "managed", cfg)
        wf.event("scope_target_reached", "scope/target_reached", target=cfg["scope_target"],
                 hypothesis_refs=[], task_ids=[], completed_rounds=[])
        # Explicit terminal control fixture: no fabricated H or numerical Evidence.
        with patch("runtime.run_segments.subprocess.Popen", side_effect=AssertionError("must not spawn")):
            result = run_managed(wf, started_unix=time.time() - .1)
        self.assertEqual(result["status"], "scope_target_reached")
        self.assertIsNone(result["budget"]["active_segment"])
        self.assertGreaterEqual(result["budget"]["elapsed_seconds"], .1)

    def test_monitor_enforces_deadline_on_real_cpu_sleep_process(self):
        import subprocess
        cfg = validation_config()
        cfg["budget"].update(clock_mode="active_segments", wall_seconds=2,
                             shutdown_reserve_seconds=1, min_free_bytes=0)
        wf = Workflow(self.root / "timed", cfg)
        spawn = subprocess.Popen
        children = []
        def cpu_only(*args, **kwargs):
            child = spawn([sys.executable, "-c", "import time; time.sleep(20)"], **kwargs)
            children.append(child)
            return child
        with patch("runtime.run_segments.subprocess.Popen", side_effect=cpu_only):
            report = run_managed(wf, started_unix=time.time())
        self.assertIsNotNone(children[0].poll())
        self.assertLess(report["budget"]["elapsed_seconds"], 2)
        self.assertEqual(report["status"], "budget_exhausted")
        self.assertIsNone(report["budget"]["active_segment"])
        self.assertFalse(wf.runtime.live_processes())
        self.assertFalse(wf.journal.find_cards("EvidenceCard"))


class PreflightRecovery(TempCase):
    def test_error_nature_controls_retries_before_and_after_model_call(self):
        for before_call in (True, False):
            for code, temporary in (("llm_transport_error", True), ("llm_timeout", True),
                    ("llm_authentication", False), ("llm_quota_exhausted", False),
                    ("llm_quota_unknown", False), ("llm_credit_policy", False),
                    ("llm_tool_isolation", False), ("llm_model_mismatch", False)):
                wf = self.workflow(1) if not (self.root / "research").exists() else Workflow(self.root / "research")
                identifier = wf.enqueue_role(f"test/{before_call}/{code}", "generate", context={})
                path = wf.run_dir / "test_audits" / f"{before_call}_{code}" / "result.json"
                atomic_json(path.with_name("llm_audit.json"), {"model_call_started": not before_call})
                tries = 3 if temporary else 1
                for i in range(tries):
                    wf.queue.change(identifier, "ready", patch={"retry_after": 0})
                    task = wf.queue.claim(identifier)
                    execution = identifier + str(i)
                    wf.budget.reserve(execution, "llm", estimated_seconds=1)
                    wf.accept(task, {"status": "failed", "execution_id": execution,
                        "error": {"code": code, "message": "explicit injected preflight failure"}}, path)
                    current = wf.queue.get(identifier)
                    self.assertEqual(current["state"], "retry" if temporary and i < 2 else "failed")
                    self.assertEqual(bool(wf.transport_blocked), not temporary)
                    if temporary and i < 2:
                        self.assertGreater(current["retry_after"], time.time())
                wf.transport_blocked = None

    def test_model_catalog_and_account_rpc_preserve_temporary_category(self):
        from runtime import codex_cli
        from tests.test_cli_versions import CLIVersionGates
        from ptq_agent.schemas import PTQError
        self.assertEqual(codex_cli.account_rpc_error("account/read", {"message": "HTTP 503 connection reset"}).code,
                         "llm_transport_error")
        self.assertEqual(codex_cli.account_rpc_error("account/read", {"message": "HTTP 401 unauthorized"}).code,
                         "llm_authentication")
        for status in (408, 500, 502, 503, 504):
            self.assertEqual(codex_cli.account_rpc_error("account/read", {"message": f"HTTP {status}"}).code,
                             "llm_transport_error")
        with patch("runtime.subscription.cli.run", side_effect=[(0, "codex-cli 0.166.0", ""),
                (1, "", "HTTP 503 temporarily unavailable")]), \
             patch("runtime.subscription.cli.account_status") as account:
            with self.assertRaises(PTQError) as error:
                CLIVersionGates().client().preflight()
        self.assertEqual(error.exception.code, "llm_transport_error")
        account.assert_not_called()

    def test_real_preflight_is_repeated_and_no_call_before_success(self):
        from runtime.model_router import call_once
        from runtime import codex_cli
        from agents.roles import role_spec
        from tests.support import validation_config
        spec = role_spec("generation", "generate", {}, validation_config()["llm"])
        with patch("runtime.subscription.SubscriptionTransport.preflight", side_effect=[
                codex_cli.fail("timeout", "temporary fixture"), {"source": "test_double"}]) as preflight, \
             patch("runtime.model_router.cli.run", side_effect=codex_cli.fail("authentication", "post-preflight fixture")) as model:
            for i in range(2):
                with self.assertRaises(Exception):
                    call_once(spec, self.root / str(i), time.time() + 10)
                self.assertEqual(model.call_count, i)
        self.assertEqual(preflight.call_count, 2)
        self.assertFalse(read_json(self.root / "0/llm_audit.json")["model_call_started"])
        self.assertTrue(read_json(self.root / "1/llm_audit.json")["model_call_started"])
