"""按场推进与科研上下文的 CPU 回归；脚本卡片均为夹具，禁止启动 worker。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card, fingerprint
from core.debate import INITIAL_PAIRS
from core.journal import atomic_json, file_sha256
from core.workflow import Workflow
from tests.support import experiment_payload, hypothesis, review
from tests import test_infrastructure as infrastructure, test_parent_evolution as evolution


class SerialDebateMemoryRegression(infrastructure.TempCase):
    drive = evolution.ParentEvolutionRegression.drive
    workflow_case = evolution.ParentEvolutionRegression.workflow_case

    def setUp(self):
        super().setUp()
        launch = patch("runtime.task_runtime.TaskRuntime.launch", side_effect=AssertionError("CPU fixtures must not launch workers"))
        launch.start()
        self.addCleanup(launch.stop)

    def science(self, wf, label, *, count=1, number=1, large=False, commit=True):
        cards = []
        for i in range(count):
            spec = make_card("ExperimentCard", wf.journal.run_id,
                             {**experiment_payload(), "question": f"FIXTURE {label} statistics {i}"})
            observations = {"mean": i / 100, "sample_count": 2, "explanation": f"FIXTURE observation {label}/{i}"}
            artifacts = []
            if large and i == count - 1:
                observations["raw"] = list(range(25000))
                asset = wf.run_dir / f"{label}_raw.json"
                atomic_json(asset, observations)
                artifacts = [{"path": str(asset), "sha256": file_sha256(asset), "bytes": asset.stat().st_size}]
            evidence = make_card("EvidenceCard", wf.journal.run_id, {
                "experiment_ref": spec["id"], "execution_fingerprint": f"fixture-{label}-{i}", "status": "completed",
                "observations": observations, "provenance": {}, "resources": {}, "artifacts": artifacts,
                "measurement_source": "test_double", "execution_id": f"fixture-{label}-{i}"}, refs=[spec["id"]])
            theory = make_card("HypothesisCard", wf.journal.run_id, {
                **hypothesis(number, "H3"), "mechanism": f"FIXTURE complete theoretical argument {label}/{i}"})
            interpretation = make_card("ReviewDecisionCard", wf.journal.run_id, review("interpretation",
                rationale=f"FIXTURE complete statistical interpretation {label}/{i}", evidence_refs=[evidence["id"]]),
                refs=[spec["id"], evidence["id"], theory["id"]])
            cards.extend([spec, evidence, theory, interpretation])
        if commit:
            wf.journal.commit("fixture/science/" + label, cards=cards)
        return cards

    def debate_task(self, wf, purpose="position", *, number=1, exchange=1, side="A", suffix=""):
        tid = wf.enqueue_role(f"fixture/{purpose}/{side}/{exchange}/{suffix}", purpose, refs=[wf.brief["id"]], context={
            "round": number, "pair": ["H1", "H2"], "exchange": exchange, "side": side,
            "position_for": "H1" if side == "A" else "H2", "max_exchanges": 5})
        return wf.queue.get(tid)

    def read(self, wf, task, name, **arguments):
        return wf.host_tool({"inputs": {"name": name, "arguments": arguments, "parent_task": task["task_id"]}})

    def test_matches_advance_independently_and_join_only_after_last_close_and_restore(self):
        wf, arb = self.drive(self.workflow_case(l3=False), stop=lambda t: t["inputs"]["purpose"] == "arbitrate")
        for pair in INITIAL_PAIRS:
            for side in ("A", "B"):
                self.assertIn("r001/match/" + "-".join(pair) + "/e1/" + side, wf.tasks())
        self.assertFalse(wf.matches(1))
        spec, evidence = self.science(wf, "own_match", commit=False)[:2]
        wf.journal.commit("fixture/own_match_evidence", cards=[spec, evidence])
        wf.submit(arb, review("verdict", outcome="continue", decisive_reason="fixture: one unresolved question",
                              evidence_refs=[evidence["id"]]), "fixture")
        wf = Workflow(wf.run_dir)
        wf.advance()
        self.assertIn("r001/match/H1-H2/e2/A", wf.tasks())
        self.assertNotIn("r001/select_parent", wf.tasks())
        # 其它两场可以先结案；即使已知 H3 两胜，也必须等待第三场正式结案。
        for pair in ("H2-H3", "H1-H3"):
            task = wf.tasks()["r001/match/" + pair + "/e1/arbitrate"]
            wf.queue.ready()
            task = wf.queue.claim(task["task_id"])
            wf.submit(task, review("verdict", outcome="B", decisive_reason="fixture: H3 wins"), "fixture")
            wf = Workflow(wf.run_dir)
            wf.advance()
            self.assertNotIn("r001/select_parent", wf.tasks())
            self.assertNotIn("r001/evolve/H4", wf.tasks())
        wf, arb = self.drive(wf, parent="H3", stop=lambda t: t["inputs"]["purpose"] == "arbitrate")
        self.assertEqual(arb["inputs"]["context"]["pair"], ["H1", "H2"])
        self.assertEqual(arb["inputs"]["context"]["exchange"], 2)
        own_prior = wf.result_card(wf.tasks()["r001/match/H1-H2/e1/arbitrate"])
        self.assertIn(own_prior["id"], arb["inputs"]["refs"])
        inherited = {card["id"] for card in wf.context(arb)["memory"]["cards"]}
        self.assertTrue({own_prior["id"], spec["id"], evidence["id"]} <= inherited)
        foreign = {row["verdict_ref"] for row in wf.matches(1).values()}
        self.assertFalse(foreign.intersection(arb["inputs"]["refs"]))
        wf.submit(arb, review("verdict", outcome="tie", decisive_reason="fixture: no decisive advantage"), "fixture")
        # 裁决卡已提交但结案事件尚未发布时，不提前汇合。
        self.assertNotIn("r001/select_parent", wf.tasks())
        wf = Workflow(wf.run_dir)
        wf.advance()
        self.assertEqual(wf.matches(1)["H1-H2"]["outcome"], "tie")
        parent = wf.result_card(wf.tasks()["r001/select_parent"])
        self.assertEqual(parent["payload"]["selected_slot"], "H3")
        self.assertEqual(parent["author"], "program")
        sequence, keys = wf.journal.sequence, set(wf.tasks())
        for _ in range(3):
            wf = Workflow(wf.run_dir)
            wf.advance()
        self.assertEqual((wf.journal.sequence, set(wf.tasks())), (sequence, keys))
        wf, _ = self.drive(wf, parent="H3", reload=True)
        closures = [e for e in wf.journal.events if e["type"] == "match_closed"]
        self.assertEqual([e["pair"] for e in closures], ["H2-H3", "H1-H3", "H1-H2", "H4-H3"])
        publications = [e["commit_sequence"] for e in wf.journal.events if e["type"] == "task_snapshot"
                        and e["task"]["state"] == "pending" and "/match/H4-H3/" in e["task"]["key"]]
        self.assertGreater(min(publications), max(e["commit_sequence"] for e in closures[:3]))
        sequence = wf.journal.sequence
        for _ in range(3):
            wf = Workflow(wf.run_dir)
            self.assertTrue(wf.advance())
        self.assertEqual(wf.journal.sequence, sequence)

    def test_initial_closure_permutations_preserve_selection_and_no_early_join(self):
        from itertools import permutations
        for order in permutations(("H1-H2", "H1-H3", "H2-H3")):
            with self.subTest(order=order):
                wf, first = self.drive(self.workflow_case("order_" + "_".join(order), l3=False),
                                       stop=lambda t: t["inputs"]["purpose"] == "arbitrate")
                arbs = {pair: wf.tasks()["r001/match/" + pair + "/e1/arbitrate"] for pair in order}
                for index, pair in enumerate(order):
                    task = arbs[pair]
                    if task["task_id"] != first["task_id"]:
                        wf.queue.ready()
                        task = wf.queue.claim(task["task_id"])
                    wf.submit(task, review("verdict", outcome="B", decisive_reason="fixture: right candidate wins"), "fixture")
                    wf = Workflow(wf.run_dir)
                    wf.advance()
                    self.assertEqual(len(wf.matches(1)), index + 1)
                    if index < 2:
                        self.assertNotIn("r001/select_parent", wf.tasks())
                        self.assertNotIn("r001/evolve/H4", wf.tasks())
                    else:
                        selected = wf.result_card(wf.tasks()["r001/select_parent"])
                        self.assertEqual(selected["payload"]["selected_slot"], "H3")
                        self.assertEqual(set(selected["payload"]["reused_verdict_refs"]),
                                         {v["verdict_ref"] for v in wf.matches(1).values()})

    def index(self, wf, task):
        offset, rows = 0, []
        while offset is not None:
            page = self.read(wf, task, "retrieve_memory", query={"browse": True}, offset=offset, limit=12)
            rows.extend(page["entries"])
            offset = page["next_offset"]
        return {r["ref"]: r for r in rows}

    def full_card(self, wf, task, ref):
        result = self.read(wf, task, "read_card", ref=ref)
        if "id" in result:
            return result
        parts = [result["text"]]
        while result["next_offset"] is not None:
            result = self.read(wf, task, "read_card", ref=ref, offset=result["next_offset"])
            parts.append(result["text"])
        return json.loads("".join(parts))

    def test_all_current_round_science_remains_discoverable_and_readable_beyond_first_page(self):
        wf = self.workflow(1)
        wf.budget.start_round(1)
        cards = self.science(wf, "current", count=13)
        refs = {c["id"] for c in cards}
        for purpose in ("position", "arbitrate"):
            with self.subTest(purpose=purpose):
                task = self.debate_task(wf, purpose)
                memory = wf.request_body(task)["spec"]["context"]["memory"]
                self.assertFalse(refs.intersection(c["id"] for c in memory["cards"]))
                self.assertIsNotNone(memory["directory"]["next_offset"])
                index = self.index(wf, task)
                self.assertTrue(refs <= index.keys())
                for card in cards:
                    self.assertEqual(self.full_card(wf, task, card["id"]), card)
                self.assertNotIn("TaskCard", {r["kind"] for r in index.values()})

    def test_older_science_has_complete_paged_index_and_raw_reads(self):
        wf = self.workflow(2)
        wf.budget.start_round(1)
        cards = self.science(wf, "older", count=13, large=True)
        wf.budget.start_round(2)
        task = self.debate_task(wf, number=2)
        index = self.index(wf, task)
        self.assertTrue({c["id"] for c in cards} <= index.keys())
        for card in cards:
            self.assertEqual(index[card["id"]]["content_sha256"], card["content_sha256"])
            self.assertEqual(self.full_card(wf, task, card["id"]), card)
        raw = next(c for c in cards if c["kind"] == "EvidenceCard" and c["payload"]["artifacts"])
        self.assertEqual(index[raw["id"]]["experiment_ref"], raw["payload"]["experiment_ref"])
        for path, expected in ((str(wf.journal.card_path(raw["id"])), raw),
                               (raw["payload"]["artifacts"][0]["path"], raw["payload"]["observations"])):
            offset, parts = 0, []
            while offset is not None:
                page = self.read(wf, task, "read_asset", asset=path, offset=offset, limit=12000)
                parts.append(page["text"]); offset = page["next_offset"]
            self.assertGreater(len(parts), 1)
            self.assertEqual(json.loads("".join(parts)), expected)

    def test_same_exchange_new_experiment_reaches_resumed_position_and_arbitrate(self):
        for purpose in ("position", "arbitrate"):
            with self.subTest(purpose=purpose):
                wf = self.workflow_case(purpose, l3=False)
                wf.budget.start_round(1)
                task = self.debate_task(wf, purpose)
                opening = wf.run_dir / "jobs/opening/request.json"
                atomic_json(opening, wf.request_body(task))
                wf.queue.ready()
                task = wf.queue.claim(task["task_id"], request_path=str(opening))
                with patch.object(wf.broker, "freeze", side_effect=AssertionError("no actual freeze in context fixture")):
                    wf.receive_tools(task, {"tool_calls": [{"id": "measure", "name": "request_experiment",
                        "arguments": {"experiment": experiment_payload()}}]}, "SCRIPTED_CPU_ONLY")
                waiting = wf.queue.get(task["task_id"])
                self.assertEqual(waiting["state"], "waiting")
                child_id, = waiting["waiting_for"]
                wf.queue.ready()
                wf.queue.claim(child_id)
                cards = self.science(wf, "returned", commit=False, large=True)
                wf.queue.finish(child_id, cards=cards, outputs=[cards[1]["id"], cards[3]["id"]],
                                result={"fixture": "scripted experiment response, no execution"})
                # 另有许多较新的公开材料，返回结果不能仅靠词法 top 8 才可见。
                self.science(wf, "later", count=12)
                wf = Workflow(wf.run_dir)
                wf.queue.ready()
                task = wf.queue.claim(task["task_id"], request_path=str(opening))
                resumed = wf.run_dir / "jobs/resumed/request.json"
                body = wf.request_body(task)
                atomic_json(resumed, body)
                wf.queue.change(task["task_id"], "running", patch={"request_path": str(resumed)})
                task = wf.queue.get(task["task_id"])
                context = body["spec"]["context"]
                visible = {c["id"]: c for c in context["memory"]["cards"]}
                self.assertEqual(task["inputs"]["context"]["exchange"], 1)
                self.assertTrue({cards[i]["id"] for i in (0, 1, 3)} <= visible.keys())
                self.assertIn(cards[2]["id"], self.index(wf, task))
                self.assertEqual(visible[cards[0]["id"]]["payload"], cards[0]["payload"])
                self.assertEqual(visible[cards[3]["id"]]["payload"], cards[3]["payload"])
                shown = visible[cards[1]["id"]]['payload']['observations']
                self.assertEqual({k:v for k,v in shown.items() if k != 'raw'},
                                 {k:v for k,v in cards[1]['payload']['observations'].items() if k != 'raw'})
                self.assertEqual(shown['raw']['read_card'], {'ref': cards[1]['id'], 'field': '/payload/observations/raw'})
                self.assertEqual(shown['raw']['sha256'], fingerprint(cards[1]['payload']['observations']['raw']))
                self.assertEqual(shown['raw']['numeric_summary']['/*']['maximum']['value'], 24999)
                self.assertIn(cards[1]['id'], context['memory']['read_manifest']['projected_refs'])
                self.assertNotIn(cards[1]["id"], {r["ref"] for r in context["memory"]["required_reads"]})
                full = self.full_card(wf, task, cards[1]["id"])
                self.assertEqual(full["payload"]["observations"]["raw"][-1], 24999)
                self.assertEqual(context["tool_results"][-1]["outputs"], [cards[1]["id"], cards[3]["id"]])

    def test_shared_index_preserves_positions_and_code_receives_original_request(self):
        wf = self.workflow(1)
        wf.budget.start_round(1)
        cards = self.science(wf, "isolation")
        own, other = self.debate_task(wf), self.debate_task(wf, side="B")
        wf.queue.ready()
        other = wf.queue.claim(other["task_id"])
        wf.submit(other, infrastructure.WorkflowSimulation.scripted(self, wf, other), "fixture")
        hidden = wf.result_card(wf.queue.get(other["task_id"]))
        older = make_card("ReviewDecisionCard", wf.journal.run_id, review("position", round=1,
                         pair=["H1", "H3"], exchange=1, side="B", arguments=["public earlier argument"]))
        wf.journal.commit("fixture/public-position", cards=[older])
        memory = wf.context(own)["memory"]
        self.assertIn(older["id"], self.index(wf, own))
        self.assertNotIn(hidden["id"], json.dumps(memory))
        for name, args in (("read_card", {"ref": hidden["id"]}),
                           ("read_asset", {"asset": str(wf.journal.card_path(hidden["id"]))})):
            with self.assertRaisesRegex(ContractError, "立场"):
                self.read(wf, own, name, **args)
        retrieved = self.read(wf, own, "retrieve_memory", query={"evidence_ids": [hidden["id"]]})
        self.assertNotIn(hidden["id"], retrieved["read_manifest"]["refs"])
        self.assertNotIn(hidden["id"], {c["id"] for c in retrieved["cards"]})
        joined = self.debate_task(wf, "arbitrate")
        self.assertIn(hidden["id"], self.index(wf, joined))
        code_id = wf.enqueue_role("fixture/code", "code", refs=[wf.brief["id"], cards[0]["id"]],
                                  context={"experiment_ref": cards[0]["id"], "request_ref": cards[0]["id"]})
        coding = wf.queue.get(code_id)
        context = wf.context(coding)
        self.assertEqual(context["research_brief"]["id"], wf.brief["id"])
        self.assertEqual(context["original_request"]["id"], cards[0]["id"])
        self.assertIn("directory", context["memory"])
        self.assertEqual(self.read(wf, coding, "read_card", ref=cards[1]["id"]), cards[1])
        from agents.tasks import TASKS
        self.assertNotIn('reference', TASKS)

    def test_late_evidence_cannot_replace_any_final_outcome_or_repeat_close(self):
        for outcome in ("A", "B", "tie", "undetermined"):
            with self.subTest(outcome=outcome):
                wf, task = self.drive(self.workflow_case(outcome, l3=False),
                                      stop=lambda t: t["inputs"]["purpose"] == "arbitrate")
                wf.submit(task, review("verdict", outcome=outcome, decisive_reason="fixture final judgment"), "fixture")
                with self.assertRaisesRegex(ContractError, "已提交"):
                    wf.submit(task, review("verdict", outcome="A", decisive_reason="duplicate receipt"), "fixture")
                wf.advance()
                before = deepcopy(wf.matches(1))
                old = wf.journal.card(before["H1-H2"]["verdict_ref"])
                cards = self.science(wf, "late")
                with self.assertRaisesRegex(ContractError, "已结案"):
                    wf.submit(task, review("verdict", outcome="A", decisive_reason="late evidence",
                                           evidence_refs=[cards[1]["id"]]), "fixture")
                sequence = wf.journal.sequence
                for _ in range(3):
                    wf = Workflow(wf.run_dir)
                    wf.advance()
                self.assertEqual(wf.journal.sequence, sequence)
                self.assertEqual(wf.matches(1), before)
                self.assertEqual(wf.journal.card(old["id"]), old)
                self.assertEqual(sum(e["type"] == "match_closed" for e in wf.journal.events), 1)

    def test_legacy_revision_cards_and_recorded_results_remain_readable(self):
        wf = self.workflow(1)
        old = make_card("ReviewDecisionCard", wf.journal.run_id, review("verdict", outcome="tie"))
        legacy = make_card("ReviewDecisionCard", wf.journal.run_id, review("revision", outcome="B"), previous=old)
        # 仅模拟已有历史数据，不经新的提交接口生成改判。
        closed = {"type": "match_closed", "round": 1, "pair": "H1-H2", "exchanges": 1}
        wf.journal.commit("fixture/legacy-original", cards=[old], events=[
            {**closed, "outcome": "tie", "verdict_ref": old["id"]}])
        wf.journal.commit("fixture/legacy", cards=[legacy], events=[
            {**closed, "outcome": "B", "verdict_ref": legacy["id"], "revision_of": old["id"]}])
        sequence = wf.journal.sequence
        wf = Workflow(wf.run_dir, read_only=True)
        self.assertEqual(wf.journal.card(old["id"]), old)
        self.assertEqual(wf.journal.card(legacy["id"]), legacy)
        self.assertEqual(wf.matches(1)["H1-H2"]["verdict_ref"], legacy["id"])
        self.assertEqual(wf.journal.sequence, sequence)


if __name__ == "__main__":
    unittest.main()
