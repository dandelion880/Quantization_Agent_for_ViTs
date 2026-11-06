#!/usr/bin/env python3
"""QAViT V2 唯一命令入口；可用绝对路径从任意目录调用。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
CLI_STARTED_UNIX = time.time()
sys.dont_write_bytecode = True

from runtime.paths import ROOT
from core.journal import atomic_json, file_lock, read_json
from core.workflow import Workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="创建新的 V2 运行；模板须先填写")
    start.add_argument("--config", required=True)
    start.add_argument("--name")
    resume = sub.add_parser("resume", help="恢复整理后版本创建的运行；不重置预算")
    resume.add_argument("run")
    resume.add_argument("--intervention")
    resume.add_argument("--retry-failed", help="显式重新领取失败任务的原因")
    resume.add_argument("--managed-segment", help=argparse.SUPPRESS)
    for name in ("status", "report"):
        sub.add_parser(name).add_argument("run")
    extend = sub.add_parser("extend-budget", help="显式记录用户批准的墙钟预算延长")
    extend.add_argument("run")
    extend.add_argument("--seconds", type=float, required=True)
    extend.add_argument("--reason", required=True)
    answer = sub.add_parser("answer", help="回答具体科研澄清任务")
    answer.add_argument("run")
    answer.add_argument("--task", required=True)
    answer.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "start":
        task = read_json(Path(args.config).resolve())
        name = args.name or task.get("name", "qavit_v2") + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if "/" in name or name in (".", ".."):
            raise ValueError("name必须是单一目录名")
        run = ROOT / "EXP" / ("runs" if task["purpose"] == "research" else "validation") / name
        workflow = Workflow(run, task)
    else:
        workflow = Workflow(args.run, intervention=getattr(args, "intervention", None),
                            read_only=args.command in ("status", "report"))
        if args.command in ("status", "report"):
            result = workflow.summary()
            if args.command == "report":
                atomic_json(workflow.run_dir / "summary.json", result)
                print(str(workflow.run_dir / "summary.json"))
            else:
                print(json.dumps({k: result[k] for k in ("status", "workflow_completed", "formal_validation_success", "numerical_validation_status", "scientific_review_complete", "tasks", "clarifications", "failures")}, ensure_ascii=False, indent=2))
            return
        if args.command in ("extend-budget", "answer"):
            with file_lock(workflow.run_dir / "supervisor.lock"):
                if args.command == "extend-budget":
                    workflow.budget.extend(args.seconds, args.reason)
                else:
                    workflow.answer(args.task, args.text)
            print("已记录；使用 resume 继续，不会自动重置原预算。")
            return
        if args.retry_failed:
            with file_lock(workflow.run_dir / "supervisor.lock"):
                workflow.retry_failed(args.retry_failed)
    print("run_directory=" + str(workflow.run_dir), flush=True)
    active_clock = workflow.config.get("budget", {}).get("clock_mode") == "active_segments"
    segment = getattr(args, "managed_segment", None)
    if active_clock and not segment:
        from runtime.run_segments import run_managed
        result = run_managed(workflow, started_unix=CLI_STARTED_UNIX)
    else:
        if segment:
            current = workflow.budget.snapshot()["active_segment"]
            if not current or current["segment_id"] != segment:
                raise ValueError("受监控运行段不匹配，拒绝启动")
        result = workflow.run()
    print(json.dumps({k: result.get(k) for k in ("status", "workflow_completed", "scope_target_reached", "completed_rounds", "formal_validation_success", "numerical_validation_status", "scientific_review_complete", "error", "state_saved", "finalization_errors")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
