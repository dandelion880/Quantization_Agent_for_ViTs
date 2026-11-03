"""只读审计真实/脚本运行痕迹；结构检查不能替代科研内容人工审查。"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.journal import Journal, read_json, atomic_json
from runtime.task_queue import GlobalTaskQueue
from core.memory import ResearchMemory


def audit(root):
    root = Path(root).resolve()
    manifest = read_json(root / "research_run.json")
    journal = Journal(root, manifest["run_id"])
    tasks = GlobalTaskQueue(journal).snapshot()
    hypotheses = journal.find_cards("HypothesisCard")
    evidence = journal.find_cards("EvidenceCard")
    matches = {}
    for event in journal.events:
        if event["type"] == "match_closed":
            matches[str(event["round"]) + "/" + event["pair"]] = event
    calls = []
    for path in sorted(root.glob("jobs/*/llm_audit.json")):
        record = read_json(path)
        calls.append({"job": path.parent.name, "role": record["role"], "requested": record["requested"],
                      "actual": record["actual"], "model_call_started": record["model_call_started"],
                      "completed_response": "response" in record, "elapsed_seconds": record.get("elapsed_seconds"),
                      "usage": record.get("usage"), "reported_usage": record.get("reported_usage"), "error": record.get("error")})
    interpretations = [c for c in journal.find_cards("ReviewDecisionCard") if c["payload"]["phase"] == "interpretation"]
    return {"schema_version": "qavit.validation_audit/2", "run": str(root), "journal": journal.audit(),
        "purpose": manifest["config"]["purpose"], "semantic_review_required": True,
        "tasks": {state: sum(t["state"] == state for t in tasks.values()) for state in {t["state"] for t in tasks.values()}},
        "pending_issues": [{k: t.get(k) for k in ("task_id", "key", "state", "error", "clarification")} for t in tasks.values() if t["state"] in ("failed", "needs_clarification")],
        "hypotheses": [{"id": c["id"], "round": c["payload"]["round"], "slot": c["payload"]["slot"],
                        "title": c["payload"]["title"], "mechanism": c["payload"]["mechanism"], "predictions": c["payload"]["predictions"],
                        "parents": c["parents"], "feedback_used": c["payload"]["feedback_used"],
                        "sources": c["payload"]["sources"], "asset": str(journal.card_path(c["id"]))} for c in hypotheses],
        "matches": matches, "all_recorded_exchanges_within_five": all(v["exchanges"] <= 5 for v in matches.values()),
        "evidence": [{"id": c["id"], "experiment_ref": c["payload"]["experiment_ref"], "source": c["payload"]["measurement_source"],
                      "execution_id": c["payload"]["execution_id"], "scope": c["payload"].get("scope"),
                      "resources": c["payload"]["resources"], "asset": str(journal.card_path(c["id"]))} for c in evidence],
        "judger_interpretations": [{"id": c["id"], **c["payload"]} for c in interpretations],
        "memory": ResearchMemory(journal).snapshot(), "llm_calls": calls,
        "started_real_call_attempts": sum(c["model_call_started"] for c in calls),
        "complete_real_responses": sum(c["completed_response"] for c in calls),
        "legacy_memory_imported": manifest["legacy_memory_imported"], "legacy_incumbent_imported": manifest["legacy_incumbent_imported"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.run)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("tasks", "started_real_call_attempts", "complete_real_responses", "pending_issues")}, ensure_ascii=False))
