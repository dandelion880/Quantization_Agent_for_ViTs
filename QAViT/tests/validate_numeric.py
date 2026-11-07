"""真实组件验证：程序请求实验并封装证据；不冒充真实Agent闭环。

仅显式小规模L1/L2，沿用调用者已有绝对预算。输出持久保存，不覆盖历史。
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.paths import ROOT
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from tools.experiments import execution_fingerprint
from tests.support import validation_config, experiment_payload


def measure(workflow, key, payload):
    card = workflow.broker.freeze(payload, requester="user")
    workflow.journal.commit(key + "/spec", cards=[card])
    cached = workflow.broker.cached(card)
    if cached:
        return card, cached, True
    identifier = workflow.queue.enqueue(key, "experiment", inputs={"experiment_ref": card["id"], "approval": None},
                                       resource_keys=workflow.experiment_resources(card))
    while True:
        workflow.budget.check(allow_numeric_drain=workflow.numeric_running())
        workflow.poll()
        workflow.update_numeric_wait()
        task = workflow.queue.get(identifier)
        if task["state"] == "completed":
            return card, workflow.journal.card(task["outputs"][0]), False
        if task["state"] == "failed":
            raise RuntimeError(str(task["error"]))
        if task["state"] == "retry":
            raise RuntimeError("组件验证不隐藏重试，保留原回执: " + str(task.get("error")))
        workflow.dispatch()
        workflow.update_numeric_wait()
        time.sleep(.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-budget", required=True)
    parser.add_argument("--model", choices=["tiny", "deit"], default="tiny")
    parser.add_argument("--gpus", help="可选授权 NVML 编号或 GPU UUID，逗号分隔；默认使用实际可见设备")
    args = parser.parse_args()
    parent = read_json(args.parent_budget)
    if parent["deadline_unix"] <= time.time():
        parser.error("原验证预算已耗尽，禁止重置")
    config = validation_config()
    config["budget"]["parent_deadline_unix"] = parent["deadline_unix"]
    payload = experiment_payload("L2")
    if args.model == "deit":
        config["runtime"]["allowed_gpus"] = ('auto' if args.gpus is None else
            [int(v) if v.strip().isdigit() else v.strip() for v in args.gpus.split(',')])
        config["research_brief"]["model"] = {"name": "DeiT-B comparison input only", "scope": {"model": "deit_base_patch16_224"},
            "adapter": {"kind": "server", "config_path": "QAViT/configs/deitb_backend.json",
                        "backend_overrides": {"device": "cuda:0"}}}
        config["research_brief"]["search_space"]["calibration_samples"] = [16]
        config["research_brief"]["evaluation_protocol"] = {"kind": "local_intervention_pilot", "split": "search", "count": 16}
        payload["execution"]["calibration"]["count"] = 16
        payload["execution"]["evaluation"]["count"] = 16
        payload["execution"]["resolved_choices"]["calibration_samples"] = 16
        payload["scope"]["model"] = "deit_base_patch16_224"
        payload["resources"].update(estimated_seconds=240)
    workflow = Workflow(Path(args.output), config)
    results = {"source": "real_numeric_component_validation", "real_llm_used": False,
               "purpose": "真实L1/L2组件、Evidence缓存和配置隔离；非Agent判断验证", "model": args.model}
    try:
        first, evidence, cached = measure(workflow, "local_l2", payload)
        same = deepcopy(payload)
        same["question"] = "相同测量能否支持另一个明确的科研解释？"
        second, reused, hit = measure(workflow, "same_measurement", same)
        assert hit and reused["id"] == evidence["id"]
        changed = deepcopy(payload)
        changed["execution"]["parameters"]["granularity"] = "per_channel"
        changed["execution"]["parameters"]["axis"] = -1
        changed["execution"]["resolved_choices"]["granularity"]["activation"] = "per_channel"
        third, varied, hit2 = measure(workflow, "changed_granularity", changed)
        assert not hit2 and execution_fingerprint(first) != execution_fingerprint(third)
        assert varied["payload"]["execution_id"] != evidence["payload"]["execution_id"]
        assert evidence["payload"]["observations"]["input_error"]["mse"] != varied["payload"]["observations"]["input_error"]["mse"], "异尺度通道fixture应产生不同数值；不能只检查配置标签"
        replication = deepcopy(payload)
        replication["replication"] = {"requested": True, "purpose": "验证明确的重复验证不会被缓存吞掉", "configuration_difference": {"execution": "independent invocation; same frozen inputs"}}
        rep = workflow.broker.freeze(replication, requester="user")
        assert workflow.broker.cached(rep) is None
        results.update(status="passed", cache_reused_evidence_ref=reused["id"], explicit_replication_bypasses_cache=True,
                       evidence=[{"ref": c["id"], "source": c["payload"]["measurement_source"], "observations": c["payload"]["observations"]}
                                 for c in (evidence, varied)], journal=workflow.journal.audit())
    except BaseException as exc:
        results.update(status="failed", error={"code": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        workflow.runtime.stop()
        atomic_json(workflow.run_dir / "numeric_validation.json", results)
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
