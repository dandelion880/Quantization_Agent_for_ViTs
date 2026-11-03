"""真实 Coding/Safeguard 服务验证及明确标注的错误注入；不启动研究大循环。"""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import ROOT
from core.journal import read_json, atomic_json
from core.workflow import Workflow
from tools.safeguards import write_artifact, approval_receipt
from tests.support import validation_config, experiment_payload
from tests.test_infrastructure import NumericalSafeguard


def drive(workflow, ids):
    while True:
        workflow.budget.check()
        workflow.poll()
        tasks = [workflow.queue.get(i) for i in ids]
        if all(t["state"] == "completed" for t in tasks):
            return tasks
        if any(t["state"] in ("failed", "needs_clarification") for t in tasks):
            raise RuntimeError(str([{k: t.get(k) for k in ("key", "state", "error", "clarification")} for t in tasks]))
        workflow.dispatch()
        time.sleep(.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-budget", required=True)
    args = parser.parse_args()
    config = validation_config()
    config["budget"]["parent_deadline_unix"] = read_json(args.parent_budget)["deadline_unix"]
    workflow = Workflow(args.output, config)
    payload = experiment_payload()
    payload["execution"].update(requires_code=True, interface="quantizer_v1")
    payload["validation"] = {"mathematics": "FP32 per_tensor symmetric integer QDQ: s=max(max(abs(calibration))/(2^(bits-1)-1),1e-12); Q(x)=s*clip(round_ties_even(x/s),-2^(bits-1),2^(bits-1)-1). fit(calibration,parameters) returns frozen state; apply(x,state,parameters) cannot modify either input/state. bits comes from parameters, not a hidden constant. No training.",
        "atol": 1e-6, "rtol": 1e-6, "invariants": ["code_count", "frozen_state", "finite"], "cases": [
            {"seed": 1, "values": [-20, -8, -7.5, -1.5, -.5, 0, .5, 1.5, 6.5, 7, 20], "calibration_values": [-7, 7]},
            {"seed": 2, "shape": [32, 32], "distribution": "uniform", "scale": 3, "calibration_values": [-1, 1]},
            {"seed": 3, "values": [0, 0, 0], "calibration_values": [0, 0]}]}
    card = workflow.broker.freeze(payload, requester="user")
    workflow.journal.commit("validation/frozen_spec", cards=[card])
    refs = [workflow.brief["id"], card["id"]]
    result = {"real_llm_used": True, "scope": "Coding and Safeguard service test; explicit user test specification; no research round"}
    try:
        ids = [workflow.enqueue_role("candidate", "code", refs=refs, context={"experiment_ref": card["id"]}),
               workflow.enqueue_role("independent_reference", "reference", refs=refs, context={"experiment_ref": card["id"]})]
        candidate_task, reference_task = drive(workflow, ids)
        candidate, reference = candidate_task["result"]["artifact"], reference_task["result"]["artifact"]
        def verify(key, artifact):
            identifier = workflow.queue.enqueue(key, "verification", role="safeguard", inputs={"experiment_ref": card["id"], "candidate": artifact, "reference": reference}, resource_keys=workflow.experiment_resources(card))
            return drive(workflow, [identifier])[0]["result"]
        numerical = verify("numeric_candidate", candidate)
        identifier = workflow.enqueue_role("review_candidate", "safeguard", refs=refs, context={"experiment_ref": card["id"], "candidate": candidate, "reference": reference, "numerical": numerical})
        review_task = drive(workflow, [identifier])[0]
        review = workflow.journal.card(review_task["outputs"][0])
        approval = approval_receipt(card, candidate, reference, numerical, review)
        identifier = workflow.queue.enqueue("approved_experiment", "experiment", inputs={"experiment_ref": card["id"], "approval": approval}, resource_keys=workflow.experiment_resources(card))
        evidence = drive(workflow, [identifier])[0]["outputs"][0]
        wrong = write_artifact(workflow.run_dir, card, {"source": NumericalSafeguard.CANDIDATE.replace("torch.round", "torch.floor"),
            "interface": "quantizer_v1", "explanation": "TEST HARNESS deliberate floor fault; this source is NOT produced by Coding LLM"}, "program")
        rejection = verify("numeric_fault_injection", wrong)
        assert rejection["status"] == "rejected", rejection
        identifier = workflow.enqueue_role("review_fault_injection", "safeguard", refs=refs, context={"experiment_ref": card["id"], "candidate": wrong, "reference": reference, "numerical": rejection})
        rejected_task = drive(workflow, [identifier])[0]
        rejected = workflow.journal.card(rejected_task["outputs"][0])
        assert rejected["payload"]["approved"] is False
        result.update(status="passed", candidate=candidate, independent_reference=reference, numerical=numerical,
                      approved_review=review["id"], evidence_ref=evidence, fault_injection={"source": "explicit test harness", "artifact": wrong,
                      "numerical": rejection, "real_safeguard_review": rejected["id"], "experiment_launched": False})
    except BaseException as exc:
        result.update(status="incomplete", error={"code": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        workflow.runtime.stop()
        atomic_json(workflow.run_dir / "service_validation.json", result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
