"""独立真实Recorder与后续Generation服务检查，不冒充完成两轮科学辩论。"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.journal import atomic_json, read_json
from core.workflow import Workflow

def drive(workflow, ids):
    while True:
        workflow.budget.check()
        workflow.poll()
        tasks = [workflow.queue.get(i) for i in ids]
        if all(t['state'] == 'completed' for t in tasks):
            return tasks
        if any(t['state'] in ('failed', 'needs_clarification') for t in tasks):
            raise RuntimeError(str([{k: t.get(k) for k in ('key', 'state', 'error', 'clarification')} for t in tasks]))
        workflow.dispatch()
        time.sleep(.1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="已通过validate_code_service的运行目录")
    parser.add_argument("--intervention")
    args = parser.parse_args()
    workflow = Workflow(args.run, intervention=args.intervention)
    previous = read_json(workflow.run_dir / "service_validation.json")
    assert previous["status"] == "passed"
    refs = [workflow.brief["id"], previous["researcher_ref"],
            *previous["evidence_refs"], *previous["safeguard_refs"]]
    selected_review = previous["safeguard_refs"][-1]
    result = {"scope": "isolated real Recorder and subsequent Generation memory service; not a completed debate round", "real_llm_used": True}
    try:
        identifier = workflow.enqueue_role("memory_component/record", "record", refs=refs, context={
            "round": 1, "selection": None, "selection_ref": selected_review, "component_validation": True,
            "requirements": "这是独立服务验证，不是已完成的大循环。基于明确提供的真实量化器实验与Safeguard判断，生成有科学分析的Meta-review。如没有完整研究假设或因果对照，明确说明未知；不得编造完整轮次、假设胜负、跨模型规律或错误实现的实验结果。提出下一项最有价值的可证伪问题。"})
        metareview = drive(workflow, [identifier])[0]["outputs"][0]
        result["metareview_ref"] = metareview
        identifier = workflow.enqueue_role("memory_component/next_generation", "generate", refs=[workflow.brief["id"], metareview], context={
            "round": 2, "slot": "H2", "hypothesis_type": "engineering", "component_validation": True,
            "requirements": "验证读取Meta-review并据此提出后续工程假设。已有证据足以设计下一项可证伪研究；将尚缺的新增实验列入验证设计即可，本项服务测试不要求立即开展新实验。不得宣称已完成两轮研究。"})
        task = drive(workflow, [identifier])[0]
        hypothesis = workflow.journal.card(task["outputs"][0])
        assert any(row.get("metareview_ref") == metareview and row.get("change") for row in hypothesis["payload"]["feedback_used"])
        result.update(status="passed", next_hypothesis_ref=hypothesis["id"], feedback_used=hypothesis["payload"]["feedback_used"], memory=workflow.memory.snapshot())
    except BaseException as exc:
        result.update(status="incomplete", error={"code": type(exc).__name__, "message": str(exc)})
    finally:
        workflow.runtime.stop()
        atomic_json(workflow.run_dir / "memory_service_validation.json", result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
