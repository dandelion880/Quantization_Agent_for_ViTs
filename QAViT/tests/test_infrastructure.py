"""可重复单元/集成测试；脚本化 Agent 明确只用于调度验证。"""
from copy import deepcopy
from pathlib import Path
import json
import os
import signal
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.paths import ROOT
from core.cards import ContractError, canonical, fingerprint, make_card, strict_loads, check_constraints
from core.journal import Journal, atomic_json, read_json, ConcurrentCommit
from runtime.task_queue import GlobalTaskQueue, process_identity, process_alive
from runtime.task_runtime import TaskRuntime
from runtime.budget import BudgetLedger, BudgetExhausted, RequestBudgetInsufficient
from core.memory import ResearchMemory, MemoryRetriever
from core.debate import INITIAL_PAIRS, ranking_table, validate_verdict
from core.workflow import Workflow
from tools.experiments import execution_fingerprint
from tests.support import validation_config, experiment_payload, hypothesis, review, meta


class TempCase(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "EXP/cache/tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="v2-test-", dir=parent)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def journal(self, name="run"):
        return Journal(self.root / name, name)

    def workflow(self, rounds=2):
        config = validation_config()
        config["purpose"] = "infrastructure_test"
        config["budget"].update(max_rounds=rounds, min_free_bytes=0)
        return Workflow(self.root / "research", config)


class CardsAndJournal(TempCase):
    def test_published_fact_survives_checkpoint_enospc(self):
        from unittest.mock import patch
        journal = self.journal()
        card = make_card("HypothesisCard", "run", hypothesis(1, "H1"))
        save = atomic_json
        def failing_index(path, value):
            if Path(path).name == "checkpoint.json":
                raise OSError(28, "explicit ENOSPC fault injection")
            return save(path, value)
        with patch("core.journal.atomic_json", side_effect=failing_index):
            journal.commit("published_despite_index_failure", cards=[card])
        self.assertEqual(journal.index_error["errno"], 28)
        recovered = Journal(journal.root, "run")
        self.assertEqual(recovered.card(card["id"]), card)
        self.assertEqual(recovered.sequence, 1)

    def test_strict_json_and_open_scientific_content(self):
        for value in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises((ValueError, ContractError)):
                strict_loads(value)
        payload = hypothesis(1, "H1")
        payload["mechanism"] = "数学推导 $E[(Q(x)-x)^2]$ " * 15000
        payload["extended_theory"] = {"anything": [1, None, {"derivation": "complete"}]}
        card = make_card("HypothesisCard", "run", payload, author="generation")
        self.assertGreater(len(canonical(card)), 200000)
        with self.assertRaises(ContractError):
            make_card("HypothesisCard", "run", hypothesis(1, "H4"), author="evolution")

    def test_journal_fault_boundaries_and_immutable_versions(self):
        journal = self.journal()
        card = make_card("HypothesisCard", "run", hypothesis(1, "H1"))
        def die(stage):
            if stage == "cards_written":
                raise RuntimeError("simulated crash before commit")
        with self.assertRaises(RuntimeError):
            journal.commit("first", cards=[card], fault=die)
        self.assertEqual(Journal(journal.root, "run").cards, {})
        journal.commit("first", cards=[card])
        self.assertEqual(journal.commit("first", cards=[card])["sequence"], 1)
        with self.assertRaises(ContractError):
            journal.commit("illegal-recommit", cards=[card])
        version = make_card("HypothesisCard", "run", {**card["payload"], "title": "new version"}, previous=card)
        journal.commit("revision", cards=[version])
        self.assertEqual(journal.card(card["id"])["payload"]["title"], card["payload"]["title"])
        with self.assertRaises(ConcurrentCommit):
            journal.commit("stale-writer", events=[], expected_sequence=0)
        raw = read_json(journal.card_path(card["id"]))
        raw["payload"]["title"] = "tampered"
        atomic_json(journal.card_path(card["id"]), raw)
        with self.assertRaises(ContractError):
            Journal(journal.root, "run")

    def test_no_inherited_bitwidth_training_or_model(self):
        check_constraints({"hard_constraints": {}}, {"training_allowed": True, "activation_bits": 8, "model": "anotherViT"})
        with self.assertRaises(ContractError):
            check_constraints({"hard_constraints": {"activation_bits": [4, 8]}}, {"activation_bits": 3})
        check_constraints({"hard_constraints": {"calibration_samples": {"min": 16, "max": 128}}}, {"calibration_samples": 48})


class QueueAndBudget(TempCase):
    def test_disk_scan_tolerates_atomic_temporary_file_disappearance(self):
        from unittest.mock import patch
        journal = self.journal()
        budget = BudgetLedger(journal, {"wall_seconds": 100, "max_artifact_bytes": 1 << 20}, clock=lambda: 100)
        temporary = journal.root / '.pending-worker'
        temporary.write_text('in-flight atomic write')
        original = Path.is_file
        def disappear(path):
            value = original(path)
            if path == temporary and value:
                path.unlink()
            return value
        with patch.object(Path, 'is_file', disappear):
            budget.check()
        (journal.root / 'durable.bin').write_bytes(b'x' * (1 << 20))
        with self.assertRaises(BudgetExhausted):
            budget.check()

    def test_disk_scan_preserves_real_io_failures(self):
        from unittest.mock import patch
        journal = self.journal()
        budget = BudgetLedger(journal, {"wall_seconds": 100, "max_artifact_bytes": 1 << 20}, clock=lambda: 100)
        target = journal.root / 'unreadable.bin'; target.write_text('durable')
        original = Path.is_file
        def unreadable(path):
            if path == target:
                raise PermissionError('explicit storage fault injection')
            return original(path)
        with patch.object(Path, 'is_file', unreadable), self.assertRaises(PermissionError):
            budget.check()

    def test_llm_timeout_is_clamped_by_deadline_not_a_completion_estimate(self):
        journal, clock = self.journal(), [100.0]
        budget = BudgetLedger(journal, {"wall_seconds": 30}, clock=lambda: clock[0])
        timeout = budget.snapshot()["remaining_seconds"]
        clock[0] += .1
        budget.reserve("llm", "llm", estimated_seconds=timeout)
        with self.assertRaises(RequestBudgetInsufficient):
            budget.reserve("experiment", "experiment", estimated_seconds=timeout)
        clock[0] = 131
        with self.assertRaises(BudgetExhausted):
            budget.reserve("late", "llm", estimated_seconds=1)

    def test_parent_deadline_cannot_be_silently_extended(self):
        journal = self.journal()
        budget = BudgetLedger(journal, {"wall_seconds": 200, "parent_deadline_unix": 150}, clock=lambda: 100)
        with self.assertRaises(ContractError):
            budget.extend(1, "child authorization alone does not change parent budget")
        self.assertEqual(budget.snapshot()["deadline_unix"], 150)

    def test_storage_failure_returns_honest_unsaved_state(self):
        from unittest.mock import patch
        workflow = self.workflow(1)
        with patch.object(workflow, "event", side_effect=OSError(28, "explicit ENOSPC fault injection")):
            result = workflow.run()
        self.assertEqual(result["status"], "storage_blocked")
        self.assertFalse(result["state_saved"])
        self.assertFalse(result["stop_event_saved"])
        self.assertEqual(len(workflow.queue.snapshot()), 0)

    def test_dependencies_idempotence_retries_resource_locks(self):
        journal, clock = self.journal(), [100.0]
        queue = GlobalTaskQueue(journal)
        a = queue.enqueue("a", "test_control", resource_keys=["gpu:0"])
        self.assertEqual(a, queue.enqueue("a", "test_control"))
        b = queue.enqueue("b", "test_control", depends_on=[a])
        c = queue.enqueue("c", "test_control", resource_keys=["gpu:0"])
        self.assertNotIn(b, [t["task_id"] for t in queue.ready()])
        queue.claim(a)
        with self.assertRaises(ContractError):
            queue.claim(c)
        queue.fail(a, {"code": "temporary"}, retryable=True)
        queue.change(a, "retry", patch={"retry_after": 0})
        queue.promote(); queue.claim(a); queue.finish(a, result={"ok": True})
        self.assertEqual(queue.get(a)["attempt"], 2)
        self.assertIn(b, [t["task_id"] for t in queue.ready()])
        self.assertEqual(GlobalTaskQueue(Journal(journal.root, "run")).get(a)["state"], "completed")

    def test_wall_clock_includes_downtime_and_explicit_extension(self):
        journal, clock = self.journal(), [100.0]
        ledger = BudgetLedger(journal, {"max_rounds": 10, "wall_seconds": 100}, clock=lambda: clock[0])
        ledger.start_round(1)
        for i in range(20):
            ledger.reserve(str(i), "experiment", estimated_seconds=1)
        clock[0] = 210
        resumed = BudgetLedger(Journal(journal.root, "run"), clock=lambda: clock[0])
        with self.assertRaises(BudgetExhausted):
            resumed.check()
        self.assertEqual(resumed.snapshot()["l1_l2_call_limit"], None)
        self.assertEqual(resumed.snapshot()["llm_call_limit"], None)
        resumed.extend(100, "explicit test authorization")
        self.assertEqual(resumed.check()["remaining_seconds"], 90)

    def test_real_process_receipt_and_lost_process_recovery(self):
        journal = self.journal()
        queue = GlobalTaskQueue(journal)
        budget = BudgetLedger(journal, {"wall_seconds": 60})
        runtime = TaskRuntime(journal.root, queue, budget, {"cpu_threads": 1})
        self.addCleanup(runtime.stop)
        task_id = queue.enqueue("physical", "test_control")
        queue.promote()
        runtime.launch(queue.get(task_id), {"kind": "test_control", "value": 17, "estimated_seconds": 20}, input_versions={})
        end = time.monotonic() + 25
        while time.monotonic() < end:
            rows = runtime.poll()
            if rows:
                break
            time.sleep(.1)
        self.assertEqual(len(rows), 1)
        task, result, path = rows[0]
        self.assertEqual(result["data"], {"source": "test_double", "value": 17})
        runtime.settle(result)
        queue.finish(task_id, result=result["data"])
        # 模拟已领取但在 fork 前崩溃的持久 launch intent。
        lost = queue.enqueue("lost", "test_control")
        queue.promote()
        directory = journal.root / "jobs/lost"; directory.mkdir()
        request = {"task_id": lost, "execution_id": "lost-1", "core_sha256": "fixture", "deadline_unix": time.time() + 30}
        atomic_json(directory / "request.json", request)
        budget.reserve("lost-1", "test_control", estimated_seconds=1)
        queue.claim(lost, request_path=str(directory / "request.json"))
        queue.change(lost, "running", patch={"started_unix": time.time() - 10})
        recovered = TaskRuntime(journal.root, GlobalTaskQueue(Journal(journal.root, "run")), budget, {})
        rows = recovered.poll()
        self.assertEqual(rows[0][1]["status"], "interrupted")
        self.assertEqual(rows[0][1]["resources"]["wall_seconds"], None)

    def test_numeric_deadline_does_not_repeat_work_but_interruptions_still_retry(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from unittest.mock import patch
        from runtime import task_worker
        from runtime.identity import code_identity

        workflow = self.workflow(1)
        for index, (kind, signum, expired) in enumerate([
                                ('experiment', signal.SIGALRM, True),
                ('experiment', signal.SIGTERM, False),
                ('llm', signal.SIGALRM, False)]):
            with self.subTest(kind=kind, signum=signum):
                identifier = workflow.queue.enqueue('signal/' + str(index), kind)
                workflow.queue.promote()
                execution = identifier + '-signal'
                path = workflow.run_dir / 'jobs' / execution / 'request.json'
                atomic_json(path, {'kind': kind, 'task_id': identifier, 'execution_id': execution,
                    'deadline_unix': time.time() + .05, 'core_sha256': code_identity()['core_sha256']})
                workflow.budget.reserve(execution, kind, estimated_seconds=1)
                workflow.queue.claim(identifier, request_path=str(path))
                saved = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
                def wait_or_interrupt(*args):
                    if signum == signal.SIGTERM:
                        signal.raise_signal(signal.SIGTERM)
                    time.sleep(1)  # The real worker timer must stop numerical work.
                try:
                    with patch.object(sys, 'argv', ['task_worker.py', str(path)]), \
                         patch('runtime.task_worker.work', side_effect=wait_or_interrupt), \
                         redirect_stderr(StringIO()):
                        task_worker.main()
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    for sig, handler in saved.items():
                        signal.signal(sig, handler)
                receipt = read_json(path.with_name('result.json'))
                self.assertEqual(workflow.poll(), 1)
                self.assertEqual(receipt['status'], 'failed' if expired else 'interrupted')
                self.assertEqual(receipt['error']['code'], 'TimeoutError' if expired else 'InterruptedError')
                self.assertNotIn('data', receipt)
                task = workflow.queue.get(identifier)
                self.assertEqual(task['state'], 'failed' if expired else 'retry')
                self.assertEqual(task['retry_count'], 0 if expired else 1)
                self.assertFalse(workflow.journal.find_cards('EvidenceCard'))


class DebateAndMemory(TempCase):
    def test_direct_results_cycle_and_five_exchange_bound(self):
        matches = {"-".join(pair): {"outcome": "A"} for pair in INITIAL_PAIRS}
        self.assertEqual(ranking_table(matches, ["H1", "H2", "H3"])["groups_by_direct_results"][0], ["H1"])
        matches["H1-H3"]["outcome"] = "B"
        self.assertTrue(ranking_table(matches, ["H1", "H2", "H3"])["cycles"])
        with self.assertRaises(ContractError):
            validate_verdict({"outcome": "continue"}, 5)
        validate_verdict({"outcome": "A", "decisive_reason": "clear evidence"}, 1)

    def test_memory_has_no_legacy_state_and_cache_is_not_replication(self):
        journal = self.journal()
        memory = ResearchMemory(journal)
        self.assertIsNone(memory.snapshot()["L1"]["incumbent"])
        self.assertEqual(memory.snapshot()["L2"], {})
        experiment = make_card("ExperimentCard", "run", experiment_payload(), author="judger")
        journal.commit("experiment", cards=[experiment])
        evidence = make_card("EvidenceCard", "run", {"experiment_ref": experiment["id"], "execution_fingerprint": "x", "status": "completed",
            "observations": {"mse": .1}, "provenance": {}, "resources": {}, "artifacts": [],
            "measurement_source": "synthetic_computation", "execution_id": "actual-1", "scope": {"schema_version": "qavit.evidence_scope/1", "model": "tiny_vit", "measurement_kind": "fixture_only"}}, refs=[experiment["id"]])
        journal.commit("evidence", cards=[evidence])
        payload = meta(1)
        payload["evidence_refs"] = [evidence["id"]]
        payload["knowledge_updates"] = [{"claim_id": "k1", "text": "scoped numeric observation", "status": "replicated",
            "scope": {"model": "tiny_vit", "measurement_kind": "fixture_only"}, "evidence_refs": [evidence["id"], evidence["id"]], "rationale": "test",
            "replication_analysis": "incorrectly duplicates same execution"}]
        with self.assertRaises(ContractError):
            memory.validate_metareview(make_card("MetareviewCard", "run", payload, author="recorder"))
        payload["knowledge_updates"][0].update(status="observed_once", evidence_refs=[evidence["id"]])
        card = make_card("MetareviewCard", "run", payload, author="recorder", refs=[evidence["id"]])
        event = memory.apply_metareview(card)
        journal.commit("meta", cards=[card], events=[event])
        result = MemoryRetriever(journal).retrieve("generation", after_version=1, query={"model": "tiny_vit"})
        self.assertEqual(result["latest_metareview_ref"], card["id"])
        self.assertNotIn("L2", result)
        knowledge = MemoryRetriever(journal).retrieve("generation", query={"layer": "L2", "model": "tiny_vit"})
        self.assertEqual(knowledge["entries"][0]["status"], "observed_once")
        self.assertEqual(knowledge["entries"][0]["ref"], card["id"])
        self.assertEqual(result["cards"], [])

    def test_fingerprint_isolates_configuration_seed_and_reuses_question(self):
        wf = self.workflow(1)
        a = wf.broker.freeze(experiment_payload(), requester="judger")
        b_payload = experiment_payload(); b_payload["question"] = "different scientific interpretation"
        b = wf.broker.freeze(b_payload, requester="judger")
        self.assertEqual(execution_fingerprint(a), execution_fingerprint(b))
        b_payload["execution"]["seed"] += 1
        c = wf.broker.freeze(b_payload, requester="judger")
        self.assertNotEqual(execution_fingerprint(a), execution_fingerprint(c))
        l2 = deepcopy(a)
        l2["payload"]["level"] = "L2"
        self.assertNotEqual(execution_fingerprint(a), execution_fingerprint(l2))
        explanation = deepcopy(a)
        explanation["payload"]["controls"]["description"] = "another wording for the same frozen execution"
        self.assertEqual(execution_fingerprint(a), execution_fingerprint(explanation))


class WorkflowSimulation(TempCase):
    def scripted(self, wf, task, *, ineligible=False, five=False):
        c, purpose = task["inputs"]["context"], task["inputs"]["purpose"]
        if purpose in ("generate", "evolve"):
            value = hypothesis(c["round"], c["slot"])
            if wf.memory.snapshot()["L3"]:
                value["feedback_used"] = [{"metareview_ref": wf.memory.snapshot()["L3"][-1], "change": "changed test prediction in response to previous open question"}]
            if purpose == "evolve":
                value.update(parent_refs=[c["parent_ref"]], improvement_objective="test goal")
            return value
        if purpose == "screen":
            rejected = ineligible and c["slot"] == "H3"
            return review("screening", eligible=not rejected, hard_rejection_reasons=["explicit fixture contradiction"] if rejected else [],
                          similarity_analysis="different fixture", evidence_gaps=[])
        if purpose == "position":
            return review("position", side=c["side"], arguments=["test"], objections=["test"], responses_to_prior=[], key_predictions=["p1"])
        if purpose == "arbitrate":
            more = five and c["pair"] == ["H1", "H2"] and c["exchange"] < 5
            return review("verdict", outcome="continue" if more else "A", decisive_reason="test control flow", changed_judgments=[], next_questions=[])
        if purpose == "rank":
            table = ranking_table(wf.selection_matches(c), c["eligible"])
            return review("ranking", selected_slot=table["groups_by_direct_results"][0][0] if c["eligible"] else None,
                          provisional=table["requires_provisional_basis"], selection_basis="stage-specific direct wins in fixture")
        if purpose == "record":
            return meta(c['round'], c['recording_batch']['judgment_refs'])
        self.fail("unexpected purpose " + purpose)

    def drive(self, *, ineligible=False, five=False, rounds=2):
        wf = self.workflow(rounds)
        reloaded = False
        for tick in range(350):
            if wf.advance():
                break
            ready = wf.queue.ready()
            self.assertTrue(ready, "scripted workflow stalled")
            # 真实调用是串行的；模拟也逐条领取，防止绕过资源锁。
            for candidate in ready:
                task = wf.queue.claim(candidate["task_id"])
                self.assertEqual(task["kind"], "llm")
                if task["inputs"]["purpose"] == "position":
                    context = wf.context(task)
                    for card in context["memory"]["cards"]:
                        self.assertTrue(wf.can_read_card(task, card))
                wf.submit(task, self.scripted(wf, task, ineligible=ineligible, five=five), self.root / "SCRIPTED_TEST_ONLY.json")
            if tick == 4:
                wf = Workflow(wf.run_dir)
                reloaded = True
        else:
            self.fail("scripted workflow did not terminate")
        self.assertTrue(reloaded)
        self.assertEqual(len(wf.memory.snapshot()["L3"]), rounds)
        self.assertIsNone(wf.memory.snapshot()["L1"]["incumbent"])
        self.assertEqual(len(wf.journal.find_cards("EvidenceCard")), 0)
        for number in range(1, rounds + 1):
            self.assertEqual(len(wf.matches(number)), 4)
            self.assertEqual(len(wf.round_hypotheses(number)), 4)
        return wf

    def test_two_round_four_matches_feedback_and_resume(self):
        wf = self.drive()
        self.assertTrue(all(m["exchanges"] == 1 for m in wf.matches(1).values()))
        self.assertTrue(all(h["payload"]["feedback_used"] for h in wf.round_hypotheses(2).values()))

    def test_ineligible_pairs_are_records_not_fake_wins(self):
        wf = self.drive(ineligible=True, rounds=1)
        self.assertEqual(sum(m["outcome"] == "ineligible" for m in wf.matches(1).values()), 2)
        self.assertTrue(all(m["exchanges"] == 0 for m in wf.matches(1).values() if m["outcome"] == "ineligible"))

    def test_fifth_exchange_is_a_hard_ceiling(self):
        wf = self.drive(five=True, rounds=1)
        self.assertEqual(wf.matches(1)["H1-H2"]["exchanges"], 5)
        self.assertFalse(any("/e6/" in key for key in wf.tasks()))


class PureSemantics(unittest.TestCase):
    """不依赖磁盘写入的科研语义门禁，不能代替持久化或真实Agent验证。"""
    def test_subscription_transport_preserves_open_experiment_and_scientific_content(self):
        from agents.roles import role_spec, tools_for
        from ptq_agent.schemas import validate
        schema = next(t["parameters"] for t in tools_for("arbitrate") if t["name"] == "request_experiment")
        validate({"experiment": experiment_payload()}, schema)
        specification = role_spec("generation", "generate", {}, {})
        validate({"action": "submit", "content": hypothesis(1, "H1")}, specification["response_schema"])

    def test_actual_plan_rejects_wrong_bits_scope_and_dtype(self):
        from tools.experiment_worker import plan_constraints
        from tests.test_execution_contracts import plan_fixture
        plan = plan_fixture()
        plan["layers"] = plan["layers"][:1]
        plan["layers"][0]["activation"].update(bits=4, format="UINT4")
        with self.assertRaises(ContractError):
            plan_constraints(plan, {"weight_bits": 4})
        with self.assertRaises(ContractError):
            plan_constraints(plan, {"quantization_scope": {"sites": ["missing_site"]}})
        with self.assertRaises(ContractError):
            plan_constraints(plan, {"compute_dtype": "FP16"})
        result = plan_constraints(plan, {"weight_bits": 8, "activation_bits": 4, "compute_dtype": "FP32"})
        self.assertEqual(result["weight_sites"], 1)

    def test_retired_dual_implementation_worker_is_unavailable(self):
        from runtime.task_worker import work
        from agents.tasks import TASKS
        self.assertNotIn('reference', TASKS)
        with self.assertRaisesRegex(ContractError, '未知工作进程任务类型'):
            work({'kind': 'verification'}, Path('.'))

    def test_cycle_uncertainty_and_early_stop(self):
        matches = {"-".join(p): {"outcome": "A"} for p in INITIAL_PAIRS}
        matches["H1-H3"]["outcome"] = "B"
        table = ranking_table(matches, ["H1", "H2", "H3"])
        self.assertTrue(table["requires_provisional_basis"])
        self.assertTrue(table["cycles"])
        validate_verdict({"outcome": "B", "decisive_reason": "explicit decisive evidence"}, 1)
        with self.assertRaises(ContractError):
            validate_verdict({"outcome": "continue"}, 5)

    def test_unconstrained_training_and_precision_are_open(self):
        for training in (True, False):
            for bits in (2, 4, 8):
                check_constraints({"hard_constraints": {}}, {"training": training, "weight_bits": bits})
        with self.assertRaises(ContractError):
            check_constraints({"hard_constraints": {"training": False}}, {"training": True})


class ImplementationApproval(TempCase):
    SOURCE = "def fit(x, parameters):\n    return {}\ndef apply(x, state, parameters):\n    return x\n"

    def approved(self, approved=True):
        from tools.safeguards import write_artifact, approval_receipt
        payload = experiment_payload()
        payload['execution'].update(requires_code=True, interface='quantizer_v1')
        card = make_card('ExperimentCard', 'approval_fixture', payload)
        candidate = write_artifact(self.root, card, {'interface': 'quantizer_v1', 'source': self.SOURCE}, 'coding')
        binding = {'experiment_ref': card['id'], 'experiment_sha256': card['content_sha256'],
                   'request_ref': card['id'], 'request_sha256': card['content_sha256'],
                   'source_sha256': candidate['source_sha256']}
        verdict = make_card('ReviewDecisionCard', 'approval_fixture',
            review('safeguard', approved=approved, reviewed_binding=binding))
        return card, candidate, approval_receipt(card, candidate, verdict)

    def test_worker_distinguishes_host_interface_failure_from_implementation(self):
        from runtime.task_worker import failure_origin
        source = self.root / 'implementation.py'
        source.write_text("def execute():\n    raise TypeError('implementation argument')\n")
        namespace = {}
        exec(compile(source.read_text(), str(source), 'exec'), namespace)
        request = {'approval': {'candidate': {'path': str(source)}}}
        try:
            namespace['execute']()
        except TypeError as exc:
            self.assertEqual(failure_origin(exc, request), 'implementation')
        try:
            raise TypeError('host interface failed before invoking plugin')
        except TypeError as exc:
            self.assertEqual(failure_origin(exc, request), 'host')

    def test_single_source_approval_needs_no_numerical_or_reference_record(self):
        from tools.safeguards import validate_approval
        card, _, approval = self.approved()
        validate_approval(approval, card)
        self.assertNotIn('reference', approval)
        self.assertNotIn('numerical', approval)

    def test_rejection_and_changed_code_or_execution_invalidate_approval(self):
        from tools.safeguards import validate_approval
        with self.assertRaisesRegex(ContractError, '未批准'):
            self.approved(False)
        card, artifact, approval = self.approved()
        changed = deepcopy(card['payload'])
        changed['execution']['seed'] += 1
        altered = make_card('ExperimentCard', card['run_id'], changed)
        with self.assertRaisesRegex(ContractError, '执行条件'):
            validate_approval(approval, altered)
        Path(artifact['path']).write_text(self.SOURCE + '# revision\n')
        with self.assertRaisesRegex(ContractError, '源码哈希'):
            validate_approval(approval, card)


if __name__ == "__main__":
    unittest.main(verbosity=2)
