"""本轮三项局部回归：脚本化工作流与一个随机 CPU 小模型，不调用 LLM/GPU。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from core.debate import INITIAL_PAIRS, ranking_table
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from tests.support import validation_config, experiment_payload, review, meta
from tests import test_infrastructure as infrastructure
from tests.test_infrastructure import TempCase


class RankingRegression(TempCase):
    def rank_task(self, wf, outcomes, eligible=None):
        matches = {"-".join(pair): {"outcome": outcome} for pair, outcome in zip(INITIAL_PAIRS, outcomes)}
        cards = [make_card("HypothesisCard", wf.journal.run_id, infrastructure.hypothesis(1, slot), author="generation")
                 for slot in ("H1", "H2", "H3")]
        wf.journal.commit("fixture/initial", cards=cards)
        identifier = wf.enqueue_role("fixture/rank", "rank", context={
            "round": 1, "selection_stage": "parent", "eligible": eligible or ["H1", "H2", "H3"],
            "candidate_refs": {c["payload"]["slot"]: c["id"] for c in cards}})
        wf.queue.ready()
        task = wf.queue.claim(identifier)
        return task, matches

    def test_unanimous_winner_survives_lower_tie_or_undetermined(self):
        for lower in ("tie", "undetermined"):
            with self.subTest(lower=lower):
                wf = Workflow(self.root / lower, validation_config())
                task, matches = self.rank_task(wf, ["A", "A", lower])
                self.assertTrue(ranking_table(matches, task["inputs"]["context"]["eligible"])["requires_provisional_basis"])
                with patch.object(wf, "matches", return_value=matches):
                    with self.assertRaisesRegex(ContractError, "全胜"):
                        wf.submit(task, review("ranking", selected_slot="H2", provisional=True,
                                               selection_basis="lower pairs are uncertain"), "fixture")
                    wf.submit(task, review("ranking", selected_slot="H1", provisional=True,
                                           selection_basis="H1 directly beats every eligible peer"), "fixture")

    def test_true_cycle_keeps_existing_provisional_rule(self):
        wf = self.workflow(1)
        task, matches = self.rank_task(wf, ["A", "B", "A"])
        with patch.object(wf, "matches", return_value=matches):
            with self.assertRaisesRegex(ContractError, "暂定|全序"):
                wf.submit(task, review("ranking", selected_slot="H2", provisional=False,
                                       selection_basis="cycle fixture"), "fixture")
            wf.submit(task, review("ranking", selected_slot="H2", provisional=True,
                                   selection_basis="same permitted provisional choice"), "fixture")
        self.assertEqual(wf.result_card(wf.queue.get(task["task_id"]))["payload"]["selected_slot"], "H2")

    def test_nonempty_revision_rejected_and_original_winner_preserved(self):
        wf = self.workflow(1)
        task, matches = self.rank_task(wf, ["A", "A", "tie"])
        old = make_card("ReviewDecisionCard", wf.journal.run_id, review("verdict", outcome="A"), author="judger")
        experiment = make_card("ExperimentCard", wf.journal.run_id, experiment_payload())
        evidence = make_card("EvidenceCard", wf.journal.run_id, {
            "experiment_ref": experiment["id"], "execution_fingerprint": "fixture", "status": "completed",
            "observations": {"fixture": True}, "provenance": {}, "resources": {}, "artifacts": [],
            "measurement_source": "test_double", "execution_id": "revision-fixture"}, refs=[experiment["id"]])
        wf.journal.commit("fixture/revision", cards=[old, experiment, evidence])
        matches["H1-H3"].update(round=1, pair="H1-H3", exchanges=1, verdict_ref=old["id"])
        revised = review("ranking", selected_slot="H3", provisional=True, selection_basis="revision fixture",
                         verdict_revisions=[review("revision", pair="H1-H3", outcome="B",
                                                   decisive_reason="new fixture observation", evidence_refs=[evidence["id"]])])
        with patch.object(wf, "matches", side_effect=lambda _: deepcopy(matches)):
            sequence = wf.journal.sequence
            with self.assertRaisesRegex(ContractError, "已结案裁决不可改写"):
                wf.submit(task, deepcopy(revised), "fixture")
            self.assertEqual(wf.journal.sequence, sequence)
            self.assertEqual(wf.matches(1)["H1-H3"]["outcome"], "A")
            revised.pop("verdict_revisions")
            with self.assertRaisesRegex(ContractError, "全胜"):
                wf.submit(task, deepcopy(revised), "fixture")
            revised["selected_slot"] = "H1"
            wf.submit(task, revised, "fixture")
        self.assertFalse(any(e["type"] == "match_closed" for e in wf.journal.events))
        self.assertEqual(wf.result_card(wf.queue.get(task["task_id"]))["payload"]["selected_slot"], "H1")

    def test_winner_only_needs_to_beat_eligible_peers(self):
        wf = self.workflow(1)
        task, matches = self.rank_task(wf, ["A", "ineligible", "ineligible"], eligible=["H1", "H2"])
        with patch.object(wf, "matches", return_value=matches):
            with self.assertRaisesRegex(ContractError, "全胜"):
                wf.submit(task, review("ranking", selected_slot="H2", provisional=True,
                                       selection_basis="eligible subset fixture"), "fixture")
            wf.submit(task, review("ranking", selected_slot="H1", provisional=False,
                                   selection_basis="direct win over the only eligible peer"), "fixture")


class L3Regression(TempCase):
    def planned(self, name, plan_state=None):
        config = validation_config()
        config.update(purpose="infrastructure_test", enable_l3=True)
        config["budget"].update(max_rounds=1, min_free_bytes=0)
        wf = Workflow(self.root / name, config)
        for _ in range(50):
            self.assertFalse(wf.advance())
            for candidate in wf.queue.ready():
                task = wf.queue.claim(candidate["task_id"])
                if task["inputs"]["purpose"] == "plan_l3":
                    if plan_state:
                        wf.queue.change(task["task_id"], plan_state, patch={"error": {"message": "fixture plan failure"}})
                        return wf, task
                    draft = experiment_payload("L3")
                    draft["execution"].update(service="evaluate_artifact", parameters={}, evaluations=[{
                        "kind": "classification", "split": "final", "count": 2}])
                    wf.submit(task, draft, "fixture")
                    wf.advance()
                    wf.queue.ready()
                    trial = wf.queue.claim(wf.tasks()["r001/L3"]["task_id"])
                    return wf, trial
                wf.submit(task, infrastructure.WorkflowSimulation.scripted(self, wf, task), "fixture")
        self.fail("did not reach L3")

    def recorder(self, wf):
        self.assertFalse(wf.advance())
        task = wf.tasks()["r001/record"]
        self.assertIn(task["task_id"], {t["task_id"] for t in wf.queue.ready()})
        task = wf.queue.claim(task["task_id"])
        return task, wf.request_body(task)["spec"]["context"]["task"]

    def test_failure_cancel_and_rejection_reach_recorder_with_reason(self):
        for state in ("failed", "cancelled", "rejected"):
            with self.subTest(state=state):
                wf, trial = self.planned(state)
                reason = "fixture L3 failure detail: " + state
                if state == "failed":
                    child = wf.queue.enqueue("r001/L3/execute", "experiment")
                    wf.queue.ready()
                    wf.queue.claim(child)
                    wf.queue.fail(child, {"code": "fixture_execution_error", "message": reason})
                    wf.wait_children_or_fail(trial, [child])
                elif state == "cancelled":
                    wf.queue.change(trial["task_id"], "cancelled", patch={"error": {"message": reason}})
                else:
                    wf.queue.finish(trial["task_id"], result={"status": "rejected", "reason": reason})
                task, context = self.recorder(wf)
                self.assertIn("l3_outcome", context)
                outcome = context["l3_outcome"]
                self.assertEqual(outcome["status"], state)
                self.assertIn(reason, str(outcome))
                self.assertIn("execution_state", outcome)
                self.assertTrue(outcome["valid_evidence_missing"])
                self.assertFalse(outcome["formal_validation_success"])
                wf.submit(task, meta(1, task['inputs']['context']['recording_batch']['judgment_refs']), "fixture")
                report = wf.run()  # 已提交 Recorder；此处不会派发任何真实工作进程。
                self.assertEqual(report["status"], "completed")
                self.assertTrue(report["workflow_completed"])
                self.assertFalse(report["formal_validation_success"])

    def test_completed_without_evidence_is_not_formal_success(self):
        wf, trial = self.planned("missing")
        wf.queue.finish(trial["task_id"], result={"status": "completed"})
        task, context = self.recorder(wf)
        self.assertIn("l3_outcome", context)
        self.assertTrue(context["l3_outcome"]["valid_evidence_missing"])
        self.assertFalse(context["l3_outcome"]["formal_validation_success"])
        wf.submit(task, meta(1, task['inputs']['context']['recording_batch']['judgment_refs']), "fixture")
        report = wf.run()
        self.assertTrue(report["workflow_completed"])
        self.assertFalse(report["formal_validation_success"])

    def evidence(self, wf, trial, *, source_level="L3"):
        experiment = wf.broker.freeze(trial["inputs"]["draft"], requester="judger")
        wf.queue.change(trial["task_id"], "running", patch={"experiment_ref": experiment["id"]}, cards=[experiment])
        source = experiment
        if source_level != "L3":
            payload = deepcopy(experiment["payload"])
            payload["level"] = source_level
            source = make_card("ExperimentCard", wf.journal.run_id, payload)
            wf.journal.commit("fixture/other-level", cards=[source])
        data = {"observations": {"evaluations": {"classification": {
                    "metrics": {"fixture_metric": 0.5}, "sample_count": 2, "sample_ids": ["fixture-0", "fixture-1"],
                    "split": "final", "full_dataset": False}}},
                "provenance": {"protocol_hash": source["payload"]["execution"]["protocol_hash"],
                               "execution_facts": {"measurement_kind": "contract_fixture_only"}},
                "measurement_source": "synthetic_computation", "artifacts": []}
        receipt = {"status": "completed", "experiment_ref": source["id"], "experiment_sha256": source["content_sha256"],
                   "data": data, "resources": {}, "execution_id": "scripted-report-fixture"}
        data["provenance"]["execution_facts"]["data"] = deepcopy(data["observations"]["evaluations"])
        path = wf.run_dir / "jobs/fixture/result.json"
        atomic_json(path, receipt)
        evidence = wf.broker.evidence(source, receipt, path)
        child = wf.queue.enqueue("r001/L3/execute", "experiment", inputs={"experiment_ref": source["id"]})
        wf.queue.ready()
        wf.queue.claim(child)
        wf.queue.finish(child, outputs=[evidence["id"]], cards=[evidence])
        return experiment, evidence

    def test_formal_success_requires_evidence_matching_specification(self):
        wf, trial = self.planned("valid")
        experiment, evidence = self.evidence(wf, trial)
        wf.broker.validate_evidence_for(experiment, evidence)
        equivalent = deepcopy(experiment["payload"])
        equivalent["question"] = "another question using the same frozen computation"
        wf.broker.validate_evidence_for(make_card("ExperimentCard", wf.journal.run_id, equivalent), evidence)
        for mismatch in ("test_double", "protocol", "count", "empty", "fingerprint"):
            with self.subTest(mismatch=mismatch), self.assertRaises(ContractError):
                invalid = deepcopy(evidence)
                body = invalid["payload"]
                if mismatch == "test_double":
                    body["measurement_source"] = "test_double"
                elif mismatch == "protocol":
                    body["provenance"]["protocol_hash"] = "other protocol"
                elif mismatch == "count":
                    body["observations"]["evaluations"]["classification"]["sample_count"] = 1
                elif mismatch == "empty":
                    body["observations"] = {}
                else:
                    body["execution_fingerprint"] = "different specification"
                wf.broker.validate_evidence_for(experiment, invalid)
        wf.queue.finish(trial["task_id"], outputs=[evidence["id"]], result={"status": "completed", "evidence_ref": evidence["id"]})
        task, context = self.recorder(wf)
        self.assertTrue(context["l3_outcome"]["formal_validation_success"])
        self.assertFalse(context["l3_outcome"]["valid_evidence_missing"])
        wf.submit(task, meta(1, task['inputs']['context']['recording_batch']['judgment_refs']), "fixture")
        report = wf.run()
        self.assertTrue(report["workflow_completed"])
        self.assertTrue(report["formal_validation_success"])  # 仅测试报告判定，不是实测科研成功。
        Path(evidence["payload"]["artifacts"][0]["path"]).write_text("tampered fixture")
        self.assertFalse(wf.summary()["formal_validation_success"])

    def test_l2_evidence_cannot_validate_l3(self):
        wf, trial = self.planned("wrong-level")
        _, evidence = self.evidence(wf, trial, source_level="L2")
        wf.queue.finish(trial["task_id"], outputs=[evidence["id"]], result={"status": "completed"})
        _, context = self.recorder(wf)
        self.assertTrue(context["l3_outcome"]["valid_evidence_missing"])
        self.assertTrue(context["l3_outcome"]["evidence_issues"])
        self.assertFalse(wf.summary("completed")["formal_validation_success"])

    def test_interpretation_failure_preserves_available_measurement(self):
        wf, trial = self.planned("interpretation-failure")
        _, evidence = self.evidence(wf, trial)
        wf.queue.fail(trial["task_id"], {"message": "fixture interpretation failed after execution"})
        _, context = self.recorder(wf)
        outcome = context["l3_outcome"]
        self.assertEqual(outcome["execution_state"], "completed")
        self.assertEqual(outcome["valid_evidence_refs"], [evidence["id"]])
        self.assertFalse(outcome["valid_evidence_missing"])
        self.assertFalse(outcome["formal_validation_success"])

    def test_cached_evidence_survives_interpretation_failure(self):
        wf, trial = self.planned("cached-interpretation-failure")
        experiment, evidence = self.evidence(wf, trial)
        # 后续请求复用已提交证据；它自己没有 execute 子任务。
        cached_id = wf.queue.enqueue("r002/L3", "experiment_request", inputs=trial["inputs"])
        wf.queue.ready()
        wf.queue.claim(cached_id)
        wf.queue.change(cached_id, "running", patch={"experiment_ref": experiment["id"]})
        interpretation = wf.enqueue_role("r002/L3/interpret", "interpret", refs=[experiment["id"], evidence["id"]],
                                         context={"cache_reused": True})
        wf.queue.ready()
        wf.queue.claim(interpretation)
        wf.queue.fail(interpretation, {"message": "fixture cached interpretation failed"})
        wf.wait_children_or_fail(wf.queue.get(cached_id), [interpretation])
        outcome = wf.l3_outcome(2)
        self.assertEqual(outcome["execution_state"], "cache_reused")
        self.assertEqual(outcome["valid_evidence_refs"], [evidence["id"]])
        self.assertFalse(outcome["valid_evidence_missing"])
        self.assertIn("fixture cached interpretation failed", str(outcome))
        self.assertFalse(outcome["formal_validation_success"])

    def test_planning_failure_still_allows_recorder(self):
        wf, _ = self.planned("plan-failure", plan_state="failed")
        _, context = self.recorder(wf)
        self.assertIn("fixture plan failure", str(context["l3_outcome"]))
        self.assertEqual(context["l3_outcome"]["execution_state"], "not_started")


class ProfileRegression(TempCase):
    def measured(self):
        from tools.experiment_worker import backend_and_data, profile_service
        wf = self.workflow(1)
        payload = experiment_payload()
        payload["execution"].update(service="profile", tool="profile_structure",
                                      arguments={"modules": ["blocks.0.mlp.fc2"], "count": 1, "seed": 0, "offset": 0})
        experiment = wf.broker.freeze(payload, requester="judger")
        wf.journal.commit("fixture/experiment", cards=[experiment])
        backend, cfg, data = backend_and_data(experiment["payload"]["execution"])
        directory = wf.run_dir / "jobs/measure"
        result = profile_service(experiment, backend, cfg, data, directory)
        result.update(measurement_source="synthetic_computation", provenance={
            "model_identity": backend.identity(), "protocol_hash": experiment["payload"]["execution"]["protocol_hash"],
            "execution_facts": result.pop("execution_facts")})
        receipt = {"status": "completed", "experiment_ref": experiment["id"],
                   "experiment_sha256": experiment["content_sha256"], "data": result,
                   "execution_id": "fixture-measure", "resources": {}}
        path = directory / "result.json"
        atomic_json(path, receipt)
        evidence = wf.broker.evidence(experiment, receipt, path)
        wf.journal.commit("fixture/evidence", cards=[evidence])
        return wf, payload, backend, cfg, data, directory, result

    def test_successful_measurement_is_registered(self):
        from ptq_agent.profiling.evidence import read_returned_evidence
        _, _, _, _, _, directory, result = self.measured()
        host = SimpleNamespace(run_dir=directory / "profile", state=read_json(directory / "profile/state.json"))
        reference = result["observations"]["result"]["evidence"]
        payload = read_returned_evidence(host, reference)
        self.assertEqual(payload["result"]["modules"][0]["name"], "blocks.0.mlp.fc2")

    def test_next_job_reads_committed_measurement_without_forward(self):
        from tools.experiment_worker import profile_service
        wf, payload, backend, cfg, data, _, result = self.measured()
        reference = result["observations"]["result"]["evidence"]
        payload["execution"].update(tool="read_profile_evidence", arguments={
            "evidence": reference, "section": "modules", "offset": 0, "limit": 1})
        # 重建宿主工作流，再冻结下一作业；不依赖上个 Python 会话中的 Host。
        wf = Workflow(wf.run_dir)
        experiment = wf.broker.freeze(payload, requester="judger")
        with patch("ptq_agent.profiling.programs.ProfilePrograms.structure", side_effect=AssertionError("must not remeasure")):
            read = profile_service(experiment, backend, cfg, data, wf.run_dir / "jobs/read")
        self.assertEqual(read["observations"]["result"]["rows"][0]["name"], "blocks.0.mlp.fc2")
        self.assertEqual(read_json(wf.run_dir / "jobs/read/profile/state.json")["budget"]["measurements"], 0)

    def test_unauthorized_path_is_rejected(self):
        from tools.experiment_worker import profile_service
        wf, payload, backend, cfg, data, directory, result = self.measured()
        reference = result["observations"]["result"]["evidence"]
        for index, unauthorized in enumerate(("evidence/unregistered.json", "../measure/profile/" + reference,
                                              str(directory / "profile" / reference))):
            with self.subTest(reference=unauthorized), self.assertRaises(ContractError):
                payload["execution"].update(tool="read_profile_evidence", arguments={
                    "evidence": unauthorized, "section": "modules", "offset": 0, "limit": 1})
                experiment = wf.broker.freeze(payload, requester="judger")
                profile_service(experiment, backend, cfg, data, wf.run_dir / ("jobs/denied-" + str(index)))

    def test_cross_run_binding_and_uncommitted_file_are_rejected(self):
        wf, payload, _, _, _, directory, result = self.measured()
        reference = result["observations"]["result"]["evidence"]
        payload["execution"].update(tool="read_profile_evidence", arguments={
            "evidence": reference, "section": "modules", "offset": 0, "limit": 1})
        authorized = wf.broker.freeze(payload, requester="judger")
        other = Workflow(self.root / "other-run", validation_config())
        payload["execution"]["profile_evidence_bindings"] = authorized["payload"]["execution"]["profile_evidence_bindings"]
        orphan = other.run_dir / "jobs/orphan/profile" / reference
        atomic_json(orphan, read_json(directory / "profile" / reference))
        with self.assertRaisesRegex(ContractError, "本次运行已提交"):
            other.broker.freeze(payload, requester="judger")

    def test_restore_rejects_tampering_and_identity_mismatch(self):
        from tools.experiment_worker import profile_service
        from ptq_agent.schemas import PTQError
        wf, payload, backend, cfg, data, directory, result = self.measured()
        reference = result["observations"]["result"]["evidence"]
        payload["execution"].update(tool="read_profile_evidence", arguments={
            "evidence": reference, "section": "modules", "offset": 0, "limit": 1})
        experiment = wf.broker.freeze(payload, requester="judger")
        identity = {**backend.identity(), "model_sha256": "different-model"}
        with patch.object(backend, "identity", return_value=identity), self.assertRaisesRegex(PTQError, "身份不匹配"):
            profile_service(experiment, backend, cfg, data, wf.run_dir / "jobs/identity-mismatch")
        (directory / "profile" / reference).write_text("tampered fixture")
        with self.assertRaisesRegex(ContractError, "资产已变化"):
            profile_service(experiment, backend, cfg, data, wf.run_dir / "jobs/tampered")


if __name__ == "__main__":
    unittest.main()
