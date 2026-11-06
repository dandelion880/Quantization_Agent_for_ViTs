"""Monitor a cumulative-budget Supervisor until it and all owned workers stop.

The monitor never calls a model or schedules science. Stop receipts are written
after the Supervisor exits; a missing receipt is conservatively charged through
the next confirmed recovery. Legacy continuous-budget runs do not use this path.
"""
import os
import signal
import subprocess
import sys
import time

from core.cards import ContractError
from core.journal import atomic_json, file_lock, read_json
from runtime.paths import QAVIT
from runtime.task_queue import process_alive, process_identity


def run_managed(workflow, *, started_unix, readiness=None):
    with file_lock(workflow.run_dir / "segment_monitor.lock", timeout=0):
        state = workflow.budget.snapshot()
        previous = state["active_segment"]
        if previous:
            if process_alive(previous.get("owner")):
                raise ContractError("上一运行段监控仍活跃，拒绝并发恢复")
            # Child has a parent-death guard. Check its persisted identity as well.
            for event in workflow.journal.events:
                if event["type"] == "segment_supervisor_started" and event["segment_id"] == previous["segment_id"]:
                    workflow.runtime.stop_identity(event["process"])
                    if process_alive(event["process"]):
                        raise ContractError("旧 Supervisor 未停止，计时保持活跃")
            with file_lock(workflow.run_dir / "supervisor.lock", timeout=10):
                workflow.runtime.stop()
                workflow.poll()
                workflow.runtime.settle_abandoned_intents()
            workflow.budget.end_segment(processes_stopped=True, reason="recovery confirmation; unknown interval charged",
                                        recovered=True)
        segment = workflow.budget.begin_segment(started_unix=started_unix, owner=process_identity(os.getpid()))
        state = workflow.budget.snapshot()
        child = None
        interrupted = False
        try:
            workflow.event("run_status", "status/segment_start/" + segment, status="running")
            if state["remaining_seconds"] <= state["config"].get("shutdown_reserve_seconds", 0):
                workflow.event("run_status", "status/no_remaining/" + segment,
                               status="budget_exhausted", error="累计运行预算不足以安全启动")
            elif workflow.scope_target_reached():
                workflow.event("run_status", "status/target_reaffirmed/" + segment,
                               status="scope_target_reached")
            else:
                from runtime.codex_cli import parent_guard
                if readiness is not None:
                    old_registry = os.environ.get("QAVIT_PROCESS_REGISTRY")
                    os.environ["QAVIT_PROCESS_REGISTRY"] = str(workflow.run_dir / "jobs" / ("readiness-" + segment) / "processes")
                    try:
                        readiness(workflow, state["worker_deadline_unix"])
                    finally:
                        if old_registry is None:
                            os.environ.pop("QAVIT_PROCESS_REGISTRY", None)
                        else:
                            os.environ["QAVIT_PROCESS_REGISTRY"] = old_registry
                child = subprocess.Popen([sys.executable, "-B", str(QAVIT / "run.py"), "resume",
                    str(workflow.run_dir), "--managed-segment", segment],
                    start_new_session=True, preexec_fn=parent_guard(os.getpid()))
                identity = process_identity(child.pid)
                workflow.event("segment_supervisor_started", "segment/supervisor/" + segment,
                               segment_id=segment, process=identity)
                # Leave half the shutdown reserve for forced cleanup and durable
                # receipts if the Supervisor fails to honor the worker boundary.
                stop_at = state["deadline_unix"] - state["config"].get("shutdown_reserve_seconds", 0) / 2
                while child.poll() is None:
                    if time.time() >= stop_at:
                        workflow.runtime.stop_identity(identity)
                        break
                    try:
                        time.sleep(.1)
                    except KeyboardInterrupt:
                        interrupted = True
                        os.kill(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            workflow.runtime.stop_identity(identity)
                        break
                child.wait(timeout=5)
        except Exception as exc:
            workflow.event("run_status", "status/monitor_error/" + segment, status="blocked",
                           error=exc.to_dict() if hasattr(exc, "to_dict") else
                           {"code": type(exc).__name__, "message": str(exc)})
        finally:
            if child is not None and child.poll() is None:
                workflow.runtime.stop_identity(process_identity(child.pid))
                child.wait(timeout=5)
            with file_lock(workflow.run_dir / "supervisor.lock", timeout=10):
                workflow.runtime.stop()
                workflow.poll()
                workflow.runtime.settle_abandoned_intents()
                if workflow.mark_scope_target_if_ready():
                    workflow.event("run_status", "status/last_receipt/" + segment,
                                   status="scope_target_reached")
                report = workflow.summary()
                if report["status"] == "running":
                    status = ("budget_exhausted" if time.time() >= state["worker_deadline_unix"]
                              else "interrupted" if interrupted else "recovery_required")
                    workflow.event("run_status", "status/monitor_stop/" + segment,
                                   status=status, error="Supervisor 已停止；原始回执保留")
                workflow.budget.end_segment(processes_stopped=True, reason="Supervisor and registered CLI/workers exited")
                report = workflow.summary()
                saved = read_json(workflow.run_dir / "summary.json") if (workflow.run_dir / "summary.json").exists() else {}
                report.update(state_saved=True, stop_event_saved=True,
                              finalization_errors=saved.get("finalization_errors", []),
                              supervisor_exit_code=child.returncode if child else None,
                              monitor_cleanup_completed=True)
                atomic_json(workflow.run_dir / "summary.json", report)
                atomic_json(workflow.run_dir / "status.json", {"status": report["status"],
                    "sequence": workflow.journal.sequence, "active_segment": None})
        return report
