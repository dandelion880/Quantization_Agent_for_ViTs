"""两阶段选择的 CPU 控制流回归；全部角色响应为脚本夹具，不调用 LLM/GPU。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from core.debate import INITIAL_PAIRS, PAIRS, ranking_table
from core.workflow import Workflow
from tests.support import validation_config, hypothesis, review, meta, experiment_payload
from tests.test_infrastructure import TempCase


class ParentEvolutionRegression(TempCase):
    def setUp(self):
        super().setUp()
        launch = patch("runtime.task_runtime.TaskRuntime.launch", side_effect=AssertionError("CPU control-flow tests must not launch workers"))
        launch.start()
        self.addCleanup(launch.stop)

    def workflow_case(self, name="research", rounds=1, l3=True):
        config = validation_config()
        config.update(purpose="infrastructure_test", enable_l3=l3)
        config["budget"].update(max_rounds=rounds, min_free_bytes=0)
        return Workflow(self.root / name, config)

    def drive(self, wf, *, parent="H1", rejected=(), initial=None, final="A", final_choice="H4",
              stop=None, reload=False, five=False):
        for _ in range(150):
            if wf.advance():
                return wf, None
            ready = wf.queue.ready()
            self.assertTrue(ready, "scripted workflow stalled")
            for candidate in ready:
                task = wf.queue.claim(candidate["task_id"])
                c, purpose = task["inputs"]["context"], task["inputs"]["purpose"]
                if stop and stop(task):
                    return wf, task
                if purpose in ("generate", "evolve"):
                    value = hypothesis(c["round"], c["slot"])
                    if purpose == "evolve":
                        value.update(parent_refs=[c["parent_ref"]], improvement_objective="address the selected parent's critique")
                    latest = wf.memory.snapshot()["L3"]
                    if latest:
                        value["feedback_used"] = [{"metareview_ref": latest[-1], "change": "revise fixture prediction from final-selection feedback"}]
                elif purpose == "screen":
                    value = review("screening", eligible=c["slot"] not in rejected,
                                   hard_rejection_reasons=["fixture constraint violation"] if c["slot"] in rejected else [],
                                   similarity_analysis="fixture", evidence_gaps=[])
                elif purpose == "position":
                    value = review("position", side=c["side"], arguments=["fixture argument"],
                                   objections=["critique for " + c["position_for"]], responses_to_prior=[], key_predictions=["p1"])
                elif purpose == "arbitrate":
                    pair = c["pair"]
                    outcome = final if pair[0] == "H4" else (initial if initial else "B" if pair[1] == parent else "A")
                    if five and pair[0] == "H4" and c["exchange"] < 5:
                        outcome = "continue"
                    value = review("verdict", outcome=outcome, decisive_reason="fixture mechanism comparison: " + "-".join(pair))
                elif purpose == "rank":
                    # 明确裁决由程序复用；脚本 Judger 仅处理未决阶段。
                    value = review("ranking", selection_stage=c["selection_stage"],
                                   selected_slot=parent if c["selection_stage"] == "parent" else final_choice,
                                   provisional=True, selection_basis="fixture: explicitly prefer this eligible candidate under unresolved evidence")
                elif purpose == "plan_l3":
                    wf.queue.fail(task["task_id"], {"code": "scripted_no_experiment", "message": "test only inspects final handoff"})
                    continue
                elif purpose == "record":
                    value = meta(c['round'], c['recording_batch']['judgment_refs'])
                else:
                    self.fail("unexpected task " + purpose)
                wf.submit(task, value, "SCRIPTED_CPU_ONLY")
            if reload:
                wf = Workflow(wf.run_dir)
        self.fail("scripted workflow did not finish")

    def selected(self, wf, stage="final", number=1):
        key = f"r{number:03d}/" + ("select_parent" if stage == "parent" else "rank")
        return wf.result_card(wf.tasks().get(key))

    def test_each_parent_and_both_final_outcomes_reach_l3_and_recorder(self):
        for parent in ("H1", "H2", "H3"):
            for final in ("A", "B"):
                with self.subTest(parent=parent, final=final):
                    wf = self.workflow_case(parent + final)
                    wf, plan = self.drive(wf, parent=parent, final=final,
                                          stop=lambda t: t["inputs"]["purpose"] == "plan_l3")
                    first, last = self.selected(wf, "parent"), self.selected(wf)
                    winner = "H4" if final == "A" else parent
                    self.assertEqual(first["payload"]["selected_slot"], parent)
                    self.assertEqual(last["payload"]["selected_slot"], winner)
                    self.assertNotEqual(first["id"], last["id"])
                    self.assertEqual((first["author"], last["author"]), ("program", "program"))
                    self.assertNotIn("verdict_revisions", first["payload"])
                    self.assertNotIn("verdict_revisions", last["payload"])
                    self.assertEqual(last["payload"]["parent_selection_ref"], first["id"])
                    self.assertEqual(plan["inputs"]["context"]["selection_ref"], last["id"])
                    self.assertEqual(plan["inputs"]["context"]["hypothesis_ref"], last["payload"]["selected_hypothesis_ref"])
                    self.assertIn(last["payload"]["selected_hypothesis_ref"], plan["inputs"]["refs"])
                    self.assertEqual(set(wf.matches(1)), {"-".join(p) for p in INITIAL_PAIRS} | {"H4-" + parent})
                    self.assertEqual(wf.round_hypotheses(1)["H4"]["parents"], [first["payload"]["selected_hypothesis_ref"]])
                    self.assertEqual(wf.memory.snapshot()["L1"]["round_selection"], {"slot": winner, "review_ref": last["id"]})
                    wf.queue.fail(plan["task_id"], {"code": "scripted_no_experiment"})
                    wf, _ = self.drive(wf, parent=parent, final=final)
                    recorder = wf.tasks()["r001/record"]
                    self.assertEqual(recorder["inputs"]["context"]["selection"], winner)
                    self.assertEqual(recorder["inputs"]["context"]["selection_ref"], last["id"])
                    self.assertTrue({first["id"], last["id"]} <= set(recorder["inputs"]["context"]["recording_batch"]["judgment_refs"]))

    def test_evolution_gets_pinned_parent_rationale_critiques_and_evidence(self):
        wf = self.workflow_case()
        wf, initial = self.drive(wf, initial="tie", stop=lambda t: t["inputs"]["purpose"] == "rank")
        experiment = make_card("ExperimentCard", wf.journal.run_id, experiment_payload())
        evidence = make_card("EvidenceCard", wf.journal.run_id, {
            "experiment_ref": experiment["id"], "execution_fingerprint": "fixture", "status": "completed",
            "observations": {"fixture": True}, "provenance": {}, "resources": {}, "artifacts": [],
            "measurement_source": "test_double", "execution_id": "explicit-control-flow-fixture"}, refs=[experiment["id"]])
        wf.journal.commit("fixture/evidence", cards=[experiment, evidence])
        wf.submit(initial, review("ranking", selected_slot="H2", provisional=True,
                                  selection_basis="H2 has the clearest testable unresolved prediction", evidence_refs=[evidence["id"]]), "fixture")
        self.assertIsNone(wf.memory.snapshot()["L1"]["round_selection"])
        wf, task = self.drive(wf, parent="H2", initial="tie", stop=lambda t: t["inputs"]["purpose"] == "evolve")
        c = wf.context(task)["task"]
        first = self.selected(wf, "parent")
        self.assertEqual(c["parent_ref"], first["payload"]["selected_hypothesis_ref"])
        self.assertEqual(c["parent_hypothesis"], wf.journal.card(c["parent_ref"]))
        self.assertEqual(c["selection_basis"], first["payload"]["selection_basis"])
        self.assertTrue(c["parent_provisional"])
        self.assertTrue(c["critique_refs"])
        self.assertIn(evidence["id"], c["evidence_refs"])
        visible = {card["id"] for card in wf.context(task)["memory"]["cards"]}
        self.assertTrue({first["id"], *c["critique_refs"], evidence["id"]} <= visible
                        and c["parent_ref"] not in visible)

    def test_unresolved_stages_keep_reasons_and_never_merge_rankings(self):
        wf = self.workflow_case()
        wf, final = self.drive(wf, parent="H3", initial="undetermined", final="tie",
                               stop=lambda t: t["inputs"]["purpose"] == "rank" and t["inputs"]["context"]["selection_stage"] == "final")
        self.assertTrue(self.selected(wf, "parent")["payload"]["provisional"])
        context = wf.context(final)["task"]
        self.assertEqual(set(context["matches"]), {"H4-H3"})
        self.assertEqual(set(context["candidate_refs"]), {"H4", "H3"})
        for change in ({"selected_slot": "H1"}, {"provisional": False}, {"selection_basis": " "}, {"selection_stage": "parent"}):
            value = review("ranking", selected_slot="H3", provisional=True, selection_basis="retain parent pending evidence")
            value.update(change)
            with self.subTest(change=change), self.assertRaises(ContractError):
                wf.submit(final, value, "fixture")
        wf.submit(final, review("ranking", selected_slot="H3", provisional=True,
                                selection_basis="retain parent pending evidence"), "fixture")
        self.assertEqual(self.selected(wf)["payload"]["selected_slot"], "H3")
        self.assertTrue(self.selected(wf)["payload"]["provisional"])
        with self.assertRaisesRegex(ContractError, "全局排名"):
            ranking_table(wf.matches(1), ["H1", "H2", "H3", "H4"])
        with self.assertRaisesRegex(ContractError, "全局排名"):
            ranking_table({"-".join(p): {"outcome": "A"} for p in PAIRS}, ["H1", "H2", "H3", "H4"])

    def test_ineligible_h4_keeps_parent_and_all_initial_ineligible_closes_normally(self):
        for rejected in (("H4",), ("H1", "H2"), ("H1", "H2", "H3")):
            with self.subTest(rejected=rejected):
                wf, _ = self.drive(self.workflow_case("reject_" + "_".join(rejected)), parent="H3", rejected=rejected)
                if len(rejected) == 3:
                    self.assertIsNone(self.selected(wf)["payload"]["selected_slot"])
                    self.assertNotIn("r001/evolve/H4", wf.tasks())
                    self.assertNotIn("r001/plan_l3", wf.tasks())
                    self.assertEqual(len(wf.matches(1)), 3)
                elif rejected == ("H4",):
                    self.assertEqual(self.selected(wf)["payload"]["selected_slot"], "H3")
                    self.assertEqual(wf.matches(1)["H4-H3"]["outcome"], "ineligible")
                    self.assertFalse(any("/match/H4-H3/e" in key for key in wf.tasks()))
                else:
                    self.assertEqual(self.selected(wf, "parent")["payload"]["selected_slot"], "H3")
                self.assertEqual(wf.tasks()["r001/record"]["state"], "completed")
                self.assertTrue(any(e["type"] == "round_finished" for e in wf.journal.events))

    def test_restore_never_changes_parent_id_or_repeats_tasks_and_matches(self):
        wf, evolution = self.drive(self.workflow_case(), parent="H2", stop=lambda t: t["inputs"]["purpose"] == "evolve")
        first = self.selected(wf, "parent")
        original = wf.journal.card(first["payload"]["selected_hypothesis_ref"])
        replacement = make_card("HypothesisCard", wf.journal.run_id, {**original["payload"], "title": "later fixture revision"},
                                author="generation", previous=original)
        wf.journal.commit("fixture/later-version", cards=[replacement])
        wf = Workflow(wf.run_dir)
        keys, sequence = set(wf.tasks()), wf.journal.sequence
        for _ in range(3):
            self.assertFalse(wf.advance())
        self.assertEqual(set(wf.tasks()), keys)
        self.assertEqual(wf.journal.sequence, sequence)
        self.assertEqual(self.selected(wf, "parent")["id"], first["id"])
        draft = hypothesis(1, "H4")
        draft["improvement_objective"] = "fixture"
        for parents in ([replacement["id"]], ["H2"], [original["logical_id"]], [original["id"], original["id"]], []):
            with self.subTest(parents=parents), self.assertRaisesRegex(ContractError, "parent_refs|引用"):
                wf.submit(evolution, {**draft, "parent_refs": parents}, "fixture")
        wf.submit(evolution, {**draft, "parent_refs": [original["id"]]}, "fixture")
        with self.assertRaisesRegex(ContractError, "持久化|固定身份"):
            wf.submit(evolution, {**draft, "parent_refs": [replacement["id"]]}, "fixture")
        wf, plan = self.drive(wf, parent="H2", final="B", reload=True, stop=lambda t: t["inputs"]["purpose"] == "plan_l3")
        self.assertEqual(plan["inputs"]["context"]["hypothesis_ref"], original["id"])
        events = [e for e in wf.journal.events if e["type"] == "match_closed"]
        self.assertEqual(len(events), 4)
        keys, sequence = set(wf.tasks()), wf.journal.sequence
        wf = Workflow(wf.run_dir)
        for _ in range(3):
            self.assertFalse(wf.advance())
        self.assertEqual(set(wf.tasks()), keys)
        self.assertEqual(wf.journal.sequence, sequence)
        other = "H1"
        with self.assertRaisesRegex(ContractError, "H4 只能"):
            wf.progress_match(1, ("H4", other), wf.round_hypotheses(1), {})

    def test_final_selection_cannot_revise_any_closed_match_with_new_evidence(self):
        wf, task = self.drive(self.workflow_case(), final="tie",
                              stop=lambda t: t["inputs"]["purpose"] == "rank")
        first = self.selected(wf, "parent")
        closed = deepcopy(wf.matches(1))
        experiment = make_card("ExperimentCard", wf.journal.run_id, experiment_payload())
        evidence = make_card("EvidenceCard", wf.journal.run_id, {
            "experiment_ref": experiment["id"], "execution_fingerprint": "fixture", "status": "completed",
            "observations": {"fixture": True}, "provenance": {}, "resources": {}, "artifacts": [],
            "measurement_source": "test_double", "execution_id": "fixture-revision"}, refs=[experiment["id"]])
        wf.journal.commit("fixture/revision", cards=[experiment, evidence])
        value = review("ranking", selected_slot="H1", provisional=True, selection_basis="new fixture evidence provisionally supports parent",
                       evidence_refs=[evidence["id"]], verdict_revisions=[review("revision", pair="H1-H2", outcome="B",
                           decisive_reason="fixture evidence", evidence_refs=[evidence["id"]])])
        sequence = wf.journal.sequence
        with self.assertRaisesRegex(ContractError, "已结案裁决不可改写"):
            wf.submit(task, value, "fixture")
        value["verdict_revisions"][0]["pair"] = "H4-H1"
        with self.assertRaisesRegex(ContractError, "已结案裁决不可改写"):
            wf.submit(task, value, "fixture")
        self.assertEqual(wf.journal.sequence, sequence)
        value["verdict_revisions"] = []  # 兼容空字段但不写入新的选择卡。
        wf.submit(task, value, "fixture")
        self.assertEqual(wf.matches(1), closed)
        self.assertEqual(self.selected(wf)["payload"]["selected_slot"], "H1")
        self.assertTrue(self.selected(wf)["payload"]["provisional"])
        self.assertNotIn("verdict_revisions", self.selected(wf)["payload"])
        self.assertEqual(self.selected(wf, "parent"), first)

    def test_next_round_uses_final_selection_feedback_and_five_exchange_limit(self):
        wf, _ = self.drive(self.workflow_case(rounds=2, l3=False), parent="H3", final="B", five=True, reload=True)
        for number in (1, 2):
            self.assertEqual(wf.matches(number)["H4-H3"]["exchanges"], 5)
            self.assertEqual(len(wf.matches(number)), 4)
            self.assertEqual(self.selected(wf, number=number)["payload"]["selected_slot"], "H3")
        self.assertFalse(any("/e6/" in key for key in wf.tasks()))
        previous = wf.result_card(wf.tasks()["r001/record"])
        for slot in ("H1", "H2", "H3"):
            task = wf.tasks()["r002/generate/" + slot]
            self.assertIn(previous["id"], task["inputs"]["refs"])
            self.assertEqual(wf.round_hypotheses(2)[slot]["payload"]["feedback_used"][0]["metareview_ref"], previous["id"])
        self.assertEqual(wf.memory.snapshot()["L1"]["round_selection"]["review_ref"], self.selected(wf, number=2)["id"])


if __name__ == "__main__":
    unittest.main()
