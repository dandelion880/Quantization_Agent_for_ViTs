"""实验输入的 CPU 合同回归；所有回执夹具仅检查接口，不代表科研测量。"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, fingerprint, make_card
from core.journal import atomic_json, file_sha256, read_json
from core.workflow import Workflow
from runtime.budget import BudgetLedger, BudgetExhausted, RequestBudgetInsufficient
from runtime.paths import ROOT, QAVIT
from tests.support import validation_config, experiment_payload, review
from tests.test_infrastructure import TempCase
from tools.data_receipt import data_receipt
from tools.experiments import preflight_experiment


class InputCase(TempCase):
    def setUp(self):
        super().setUp()
        blocker = patch("runtime.task_runtime.TaskRuntime.launch", side_effect=AssertionError("CPU input tests cannot launch workers"))
        blocker.start()
        self.addCleanup(blocker.stop)


class DataReceiptRegression(InputCase):
    def manifest_brief(self):
        folder = self.root / "assets"
        folder.mkdir()
        cfg = read_json(QAVIT / "configs/deitb_backend.json")
        path = folder / "backend.json"
        atomic_json(path, cfg)
        counts = {"calibration": 2, "search": 3, "final": 5}
        # 只用于元数据投影的微型清单，不声称通过 ImageManifest 或真实图像校验。
        manifest = {"schema": "imagefolder_manifest_v1", "preprocessing": {"fixture": True},
                    "class_to_idx": {"fixture": 0}, "splits": {s: [
                        {"sample_id": f"fixture-{s}-{i}", "dataset_split": "train" if s == "calibration" else "val"}
                        for i in range(n)] for s, n in counts.items()}}
        data_path = folder / "manifest.json"
        atomic_json(data_path, manifest)
        brief = validation_config()["research_brief"]
        brief["model"]["adapter"] = {"kind": "server", "config_path": str(path), "backend_overrides": {
            "data_path": "manifest.json", "data_protocol": manifest["schema"], "data_sha256": file_sha256(data_path),
            **{s + "_samples": n for s, n in counts.items()}}}
        return brief, data_path

    def test_existing_configuration_manifest_and_hash_are_reported_without_loading_model(self):
        brief = validation_config()["research_brief"]
        brief["model"]["adapter"] = {"kind": "server", "config_path": "QAViT/configs/deitb_backend.json"}
        original = deepcopy(brief)
        with patch("tools.experiments.asset", side_effect=AssertionError("no checkpoint freezing")), \
             patch("tools.experiment_worker.backend_and_data", side_effect=AssertionError("no model or image loading")):
            receipt = data_receipt(brief)
        cfg = read_json(QAVIT / "configs/deitb_backend.json")["backend"]
        self.assertEqual({r["split"]: r["available_samples"] for r in receipt["available_data"]},
                         {s: cfg[s + "_samples"] for s in ("calibration", "search", "final")})
        self.assertEqual(receipt["manifest_source"]["sha256"], cfg["data_sha256"])
        self.assertEqual(receipt["ordering_seeds"], cfg["data_seeds"])
        self.assertFalse(receipt["unknowns"])
        self.assertFalse(receipt["checks"]["image_contents_rechecked"])
        self.assertIsNone(receipt["selected_data"]["calibration"])
        self.assertEqual(brief, original)
        self.assertLess(len(json.dumps(receipt)), 7000)

    def test_relative_overrides_and_selected_views_are_distinct_and_not_mutated(self):
        brief, path = self.manifest_brief()
        execution = {"calibration": {"split": "calibration", "count": 1, "offset": 1},
                     "evaluations": [{"kind": "classification", "split": "search", "count": 2}]}
        saved = deepcopy(execution)
        receipt = data_receipt(brief, execution)
        self.assertEqual(receipt["manifest_source"]["path"], str(path))
        self.assertEqual(receipt["available_data"][0]["index_range"], [0, 2])
        self.assertEqual(receipt["selected_data"]["calibration"], saved["calibration"])
        self.assertEqual(receipt["selected_data"]["evaluations"], saved["evaluations"])
        self.assertIsNone(receipt["selected_data"]["evaluation"])
        self.assertEqual(execution, saved)
        full_view = {"split": "calibration", "count": None, "offset": 1}
        receipt = data_receipt(brief, {"calibration": full_view})
        self.assertEqual(receipt["selected_data"]["calibration"], full_view)
        self.assertIn("显式 count=null", receipt["meaning"])

    def test_bad_hash_count_missing_assets_and_factory_are_explicitly_unknown(self):
        brief, path = self.manifest_brief()
        changed = deepcopy(brief)
        changed["model"]["adapter"]["backend_overrides"]["search_samples"] = 99
        receipt = data_receipt(changed)
        self.assertIsNone(receipt["available_data"][1]["available_samples"])
        self.assertIn("backend.search_samples=99", " ".join(receipt["unknowns"]))
        changed["model"]["adapter"]["backend_overrides"]["data_sha256"] = "wrong hash"
        receipt = data_receipt(changed)
        self.assertTrue(all(row["available_samples"] is None for row in receipt["available_data"]))
        path.unlink()  # 本测试生成的临时清单。
        self.assertIn("读取或校验失败", " ".join(data_receipt(brief)["unknowns"]))
        receipt = data_receipt(validation_config()["research_brief"])
        self.assertFalse(receipt["available_data"])
        self.assertIn("提供器未执行", " ".join(receipt["unknowns"]))
        self.assertIsNone(receipt["prior_validation"])

    def test_prior_validation_requires_matching_manifest_counts_order_and_real_source(self):
        brief, path = self.manifest_brief()
        receipt = data_receipt(brief)
        identity = {"counts": receipt["configured_pool_counts"], "ordering_seeds": receipt["ordering_seeds"], "full_validation": False}
        spec = make_card("ExperimentCard", "fixture", experiment_payload())
        body = {"experiment_ref": spec["id"], "execution_fingerprint": "fixture", "status": "completed",
                "observations": {"fixture_only": True}, "provenance": {"data_identity": identity,
                    "model_identity": {"data": {"manifest_sha256": file_sha256(path)}}},
                "resources": {}, "artifacts": [], "measurement_source": "real", "execution_id": "contract-fixture"}
        card = make_card("EvidenceCard", "fixture", body)
        self.assertEqual(data_receipt(brief, evidence=[card])["prior_validation"]["evidence_ref"], card["id"])
        for key in ("manifest", "counts", "seeds", "source"):
            bad = deepcopy(card)
            if key == "manifest": bad["payload"]["provenance"]["model_identity"]["data"]["manifest_sha256"] = "wrong"
            elif key == "counts": bad["payload"]["provenance"]["data_identity"]["counts"]["search"] = 99
            elif key == "seeds": bad["payload"]["provenance"]["data_identity"]["ordering_seeds"]["search"] = 99
            else: bad["payload"]["measurement_source"] = "test_double"
            with self.subTest(key=key):
                self.assertIsNone(data_receipt(brief, evidence=[bad])["prior_validation"])

    def test_changed_frozen_config_is_not_presented_as_verified_availability(self):
        brief, _ = self.manifest_brief()
        adapter = deepcopy(brief["model"]["adapter"])
        path = Path(adapter["config_path"])
        adapter["config_asset"] = {"path": str(path), "sha256": file_sha256(path)}
        cfg = read_json(path)
        cfg["backend"]["search_samples"] = 7
        atomic_json(path, cfg)
        value = data_receipt(brief, {"backend": adapter})
        self.assertFalse(value["checks"]["frozen_config_matches"])
        self.assertTrue(all(row["available_samples"] is None for row in value["available_data"]))
        self.assertIn("config_asset.sha256", " ".join(value["unknowns"]))

    def test_unstructured_asset_inputs_report_unknown_without_blocking_task_context(self):
        brief = validation_config()["research_brief"]
        brief["model"]["adapter"] = "adapter still to be specified"
        self.assertIn("不是配置对象", " ".join(data_receipt(brief)["unknowns"]))
        brief["model"]["adapter"] = {}
        for evaluations in (None, "evaluation scope still to be specified", ["classification"]):
            brief["evaluation_protocol"] = {"evaluations": evaluations}
            with self.subTest(evaluations=evaluations):
                self.assertIn("不是结构化评测列表", " ".join(data_receipt(brief)["unknowns"]))

    def test_ade_receipt_uses_existing_settings_and_marks_content_validation_unknown(self):
        from tools import segmentation
        root = self.root / "ade"
        for kind, suffix in (("images", ".jpg"), ("annotations", ".png")):
            folder = root / kind / "validation"
            folder.mkdir(parents=True)
            for i in range(3): (folder / (str(i) + suffix)).write_text("directory fixture; not an image")
        settings = self.root / "ade.json"
        atomic_json(settings, {"data_root": str(root), "pipeline": [{"fixture": True}]})
        brief = validation_config()["research_brief"]
        brief["evaluation_protocol"] = {"evaluations": [{"kind": "ade_frozen", "count": None}]}
        with patch.object(segmentation, "SETTINGS", settings):
            receipt = data_receipt(brief)
            row, = receipt["available_data"]
            self.assertEqual(row["available_samples"], 3)
            self.assertEqual(row["source"]["path"], str(settings))
            self.assertIn("未读取内容", row["checks"])
            (root / "annotations/validation/2.png").unlink()
            self.assertIsNone(data_receipt(brief)["available_data"][0]["available_samples"])

    def test_task_inputs_keep_role_boundaries_and_do_not_copy_debate_memory(self):
        wf = self.workflow(1)
        cards = [make_card("ReviewDecisionCard", wf.journal.run_id, review("interpretation", rationale=f"fixture {i}")) for i in range(12)]
        spec = make_card("ExperimentCard", wf.journal.run_id, experiment_payload())
        wf.journal.commit("fixture/science", cards=[*cards, spec])
        for purpose in ("generate", "plan_l3", "code", "record"):
            context = {"round": 1}
            refs = [wf.brief["id"]]
            if purpose == "code": context["experiment_ref"] = spec["id"]; refs.append(spec["id"])
            task = wf.queue.get(wf.enqueue_role("fixture/" + purpose, purpose, refs=refs, context=context))
            value = wf.context(task)
            with self.subTest(purpose=purpose):
                self.assertEqual("data_receipt" in value, purpose in ("generate", "plan_l3"))
                self.assertIn("budget_receipt", value)
                self.assertNotIn("scientific_index", value["memory"])
                self.assertEqual({c["id"] for c in cards} <= {c["id"] for c in value["memory"]["cards"]}, purpose == "record")
                if purpose == "plan_l3": self.assertIsNone(value["data_receipt"]["selected_data"]["calibration"])


class BudgetInputRegression(InputCase):
    def ledger(self):
        self.now = [100.]
        return BudgetLedger(self.journal(), {"wall_seconds": 1000, "max_rounds": 2, "closeout_reserve_seconds": 100,
                            "shutdown_reserve_seconds": 20, "llm_call_limit": 2}, clock=lambda: self.now[0])

    def test_receipt_and_check_share_balances_without_deducting_estimates_twice(self):
        ledger = self.ledger()
        ledger.reserve("experiment", "experiment", estimated_seconds=350)
        ledger.reserve("llm", "llm", estimated_seconds=900)
        sequence = ledger.journal.sequence
        value = ledger.input_receipt()
        self.assertEqual(value["remaining_seconds"], 1000)
        self.assertEqual(value["research_remaining_seconds"], 880)
        self.assertEqual(value["unsettled_estimated_seconds_by_kind"], {"experiment": 350, "llm": 900})
        self.assertEqual(ledger.check(estimated_seconds=880, check_disk=False)["research_remaining_seconds"], 880)
        with self.assertRaises(RequestBudgetInsufficient): ledger.check(estimated_seconds=881, check_disk=False)
        self.assertEqual(ledger.journal.sequence, sequence)

    def test_settlement_elapsed_time_resume_and_call_cap_match_actual_enforcement(self):
        ledger = self.ledger()
        ledger.reserve("one", "llm", estimated_seconds=900)
        ledger.settle("one", status="completed", resources={"wall_seconds": 80})
        self.now[0] += 200
        ledger = BudgetLedger(ledger.journal, clock=lambda: self.now[0])
        value = ledger.input_receipt()
        self.assertEqual((value["elapsed_seconds"], value["remaining_seconds"], value["research_remaining_seconds"]), (200, 800, 680))
        self.assertEqual((value["unsettled_reservations"], value["settled_reservations"]), (0, 1))
        self.assertEqual(value["llm_calls"], {"limit": 2, "reserved": 1, "remaining": 1})
        ledger.reserve("two", "llm", estimated_seconds=800)
        self.assertEqual(ledger.input_receipt()["llm_calls"]["remaining"], 0)
        with self.assertRaises(BudgetExhausted): ledger.reserve("three", "llm", estimated_seconds=1)
        ledger.extend(50, "explicit test-only extension")
        self.assertEqual(ledger.input_receipt()["research_remaining_seconds"], 730)
        self.now[0] = 1150
        self.assertEqual(ledger.input_receipt()["remaining_seconds"], 0)
        with self.assertRaises(BudgetExhausted): ledger.check(check_disk=False)

    def test_context_uses_one_snapshot_and_distinguishes_closeout_balance(self):
        wf = self.workflow(1)
        deadline = wf.budget.snapshot()["deadline_unix"]
        task = wf.queue.get(wf.enqueue_role("fixture/plan", "plan_l3"))
        with patch.object(wf.budget, "clock", return_value=deadline - 12):
            value = wf.context(task)
            self.assertEqual(value["available_seconds"], value["budget_receipt"]["remaining_seconds"])
            self.assertEqual(value["budget_receipt"]["research_remaining_seconds"], wf.budget.check(check_disk=False)["research_remaining_seconds"])
        ledger = self.ledger()
        self.now[0] = 1000
        self.assertEqual(ledger.input_receipt()["remaining_seconds"], 100)
        self.assertEqual(ledger.input_receipt()["research_remaining_seconds"], 0)
        with self.assertRaises(RequestBudgetInsufficient): ledger.check(estimated_seconds=1, check_disk=False)
        ledger.reserve("closeout", "llm", estimated_seconds=100)


class BriefEvaluationRegression(InputCase):
    def configuration(self, protocol):
        cfg = validation_config()
        cfg["purpose"] = "infrastructure_test"
        cfg["budget"]["min_free_bytes"] = 0
        cfg["research_brief"]["evaluation_protocol"] = deepcopy(protocol)
        return cfg

    def draft(self, evaluations):
        value = experiment_payload("L3")
        value["execution"].update(service="evaluate_artifact", parameters={}, evaluations=deepcopy(evaluations))
        return value

    def test_plan_submission_accepts_classification_subset_single_full_and_dual_protocols(self):
        cases = [({"kind": "classification", "split": "search", "count": 2}, False),
                 ({"kind": "classification", "split": "final", "count": None}, True)]
        protocols = [{"evaluations": [row], "full_dataset": full} for row, full in cases]
        protocols.append({"evaluations": [cases[1][0], {"kind": "ade_frozen", "count": None}], "full_dataset": True})
        for index, protocol in enumerate(protocols):
            wf = Workflow(self.root / str(index), self.configuration(protocol))
            task = wf.queue.get(wf.enqueue_role("fixture/plan", "plan_l3"))
            wf.queue.ready(); task = wf.queue.claim(task["task_id"])
            draft = self.draft(list(reversed(protocol["evaluations"])))
            original = deepcopy(draft)
            with patch.object(wf.broker, "freeze", side_effect=AssertionError("preflight cannot freeze")):
                wf.submit(task, draft, "CPU_CONTRACT_FIXTURE")
            self.assertEqual(wf.queue.get(task["task_id"])["state"], "completed")
            self.assertEqual(wf.queue.get(task["task_id"])["result"]["draft"]["execution"]["evaluations"], original["execution"]["evaluations"])
            self.assertEqual(draft, original)

    def test_explicit_evaluation_conflicts_are_rejected_and_unspecified_fields_are_not_guessed(self):
        brief = self.configuration({"evaluations": [{"kind": "classification", "split": "search", "count": 2}]})["research_brief"]
        for evaluations in ([], [{"kind": "classification", "split": "final", "count": 2}],
                [{"kind": "classification", "split": "search", "count": None}],
                [{"kind": "classification", "split": "search"}],
                [*brief["evaluation_protocol"]["evaluations"], {"kind": "ade_frozen", "count": None}]):
            with self.subTest(evaluations=evaluations), self.assertRaisesRegex(ContractError, r"execution.evaluations.*ResearchBrief.evaluation_protocol.evaluations"):
                preflight_experiment(self.draft(evaluations), brief)
        brief["evaluation_protocol"] = {"evaluations": [{"kind": "classification", "split": "search"}]}
        value = preflight_experiment(self.draft([{"kind": "classification", "split": "search", "count": 1}]), brief)
        self.assertEqual(value["execution"]["evaluations"][0]["count"], 1)
        brief["evaluation_protocol"] = {"kind": "custom statistics; no classification/segmentation requirement"}
        draft = self.draft([])
        draft["execution"].update(service="custom_experiment", requires_code=True, interface="experiment_v1")
        self.assertEqual(preflight_experiment(draft, brief)['stage'], 'request')
        with self.assertRaisesRegex(ContractError, "L3 缺少可核验"):
            preflight_experiment(draft, brief, executable=True)
        brief["evaluation_protocol"]["required_results"] = [{"path": "statistic", "schema": {"type": "number"}}]
        draft["validation"]["interface_cases"] = [{"seed": 1, "shape": [1, 4, 8], "sites": ["blocks.0.mlp.fc2"],
            "counts": {"calibration": 2, "search": 4, "final": 2}}]
        self.assertEqual(preflight_experiment(draft, brief)["execution"]["evaluations"], [])

    def evidence(self, wf, *, full=False, commit=True):
        rows = wf.brief["payload"]["evaluation_protocol"]["evaluations"]
        spec = wf.broker.freeze(self.draft(rows), requester="judger")
        wf.journal.commit("fixture/spec", cards=[spec])
        evaluations = {}
        for row in rows:
            name = "segmentation" if row["kind"] == "ade_frozen" else "classification"
            count = row.get("count") or 2
            evaluations[name] = {"sample_count": count, "sample_ids": [f"fixture-{name}-{i}" for i in range(count)],
                "full_dataset": full, "split": "ade_validation" if name == "segmentation" else row["split"],
                "metrics": {"top1": 50., "mIoU": 40., "mAcc": 45., "aAcc": 46.}}
        receipt = {"status": "completed", "experiment_ref": spec["id"], "experiment_sha256": spec["content_sha256"],
            "execution_id": "CONTRACT_FIXTURE_NO_EXECUTION", "resources": {}, "data": {
                "observations": {"evaluations": evaluations, "fixture_only": True}, "measurement_source": "synthetic_computation",
                "provenance": {"protocol_hash": spec["payload"]["execution"]["protocol_hash"], "model_ref": "fixture-model",
                               "execution_facts": {"measurement_kind": "contract_fixture_only", "data": deepcopy(evaluations)}}}}
        path = wf.run_dir / "fixture_receipt.json"
        atomic_json(path, receipt)
        card = wf.broker.evidence(spec, receipt, path)
        if commit: wf.journal.commit("fixture/evidence", cards=[card])
        return spec, card, receipt, path

    def test_evidence_completion_respects_protocol_and_preserves_incumbent_qualification(self):
        protocols = [
            {"evaluations": [{"kind": "classification", "split": "search", "count": 2}], "full_dataset": False},
            {"evaluations": [{"kind": "classification", "split": "final", "count": 2}], "full_dataset": True},
            {"evaluations": [{"kind": "classification", "split": "final", "count": None}, {"kind": "ade_frozen", "count": None}], "full_dataset": True}]
        for index, protocol in enumerate(protocols):
            with self.subTest(protocol=protocol):
                wf = Workflow(self.root / str(index), self.configuration(protocol))
                spec, evidence, _, _ = self.evidence(wf, full=protocol["full_dataset"])
                wf.broker.validate_evidence_for(spec, evidence)
                candidate = {"model_ref": "fixture-model", "protocol_hash": spec["payload"]["execution"]["protocol_hash"], "evidence_refs": [evidence["id"]]}
                if index < 2:
                    with self.assertRaisesRegex(ContractError, "双任务完整评测"):
                        wf.memory.consider_incumbent(candidate, candidate["protocol_hash"])
                else:
                    self.assertIsNotNone(wf.memory.consider_incumbent(candidate, candidate["protocol_hash"]))
                if protocol["full_dataset"]:
                    bad = deepcopy(evidence)
                    bad["payload"]["observations"]["evaluations"]["classification"]["full_dataset"] = False
                    with self.assertRaises(ContractError): wf.broker.validate_evidence_for(spec, bad)
                wrong = deepcopy(spec)
                wrong["payload"]["execution"]["evaluations"][0]["split"] = "other_split"
                with self.assertRaisesRegex(ContractError, "evaluation_protocol"):
                    wf.broker.validate_evidence_for(wrong, evidence)

    def test_brief_prohibits_incumbent_even_if_full_evidence_is_returned(self):
        protocol = {"evaluations": [{"kind": "classification", "split": "final", "count": None},
                    {"kind": "ade_frozen", "count": None}], "full_dataset": True, "formal_incumbent_allowed": False}
        wf = Workflow(self.root / "no_incumbent", self.configuration(protocol))
        spec, _, receipt, path = self.evidence(wf, full=True, commit=False)
        task_id = wf.queue.enqueue("fixture/execute", "experiment", inputs={"experiment_ref": spec["id"]})
        wf.queue.ready(); task = wf.queue.claim(task_id)
        with patch.object(wf.runtime, "settle"), patch.object(wf.memory, "consider_incumbent", side_effect=AssertionError("Brief forbids promotion")):
            wf.accept(task, receipt, path)
        self.assertEqual(wf.queue.get(task_id)["state"], "completed")
        self.assertIsNone(wf.memory.snapshot()["L1"]["incumbent"])
        self.assertTrue(any(e["type"] == "incumbent_ineligible" and "formal_incumbent_allowed" in e["reason"] for e in wf.journal.events))


if __name__ == "__main__":
    unittest.main()
