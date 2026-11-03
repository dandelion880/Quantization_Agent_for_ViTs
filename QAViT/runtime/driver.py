"""任务启动、回执接收、重试与停止。科研阶段由 Workflow.advance 决定。

这是 Workflow 的职责分区，不独立实例化；状态和生命周期统一由 Workflow 拥有。
"""


from pathlib import Path
from copy import deepcopy
import time
from core.cards import ContractError, fingerprint, PAYLOAD_SCHEMAS
from core.journal import atomic_json, file_lock, read_json
from runtime.budget import BudgetExhausted, RequestBudgetInsufficient
from runtime.subscription import structured_output_error
from agents.roles import role_spec
from tools.experiments import verify_assets



class WorkflowDriver:
    def request_body(self, task):
        inputs = task["inputs"]
        if task["kind"] == "llm":
            context = self.context(task)
            scoped = {**task, '_read_view': context['read_view']}
            permissions = self.authorized_assets(scoped)
            # 业务无状态续调的正文只在 context；消息保留调用身份，不重复巨型参数和工具结果。
            messages = [{'role': 'assistant', 'content': {'tool_calls': [
                {'id': c['id'], 'name': c['name']} for c in row['content'].get('tool_calls', [])]}}
                for row in task.get('conversation', [])]
            return {"kind": "llm", "spec": role_spec(task["role"], inputs["purpose"], context,
                        self.config["llm"], messages=messages), 'read_permissions': permissions,
                    "estimated_seconds": min(self.config["runtime"]["task_timeout_seconds"], self.budget.snapshot()["remaining_seconds"])}
        if task["kind"] in ("experiment", "verification"):
            experiment = self.journal.card(inputs["experiment_ref"])
            verify_assets(experiment["payload"]["execution"])
            resources = experiment["payload"]["resources"]
            return {"kind": task["kind"], "experiment": experiment, "approval": inputs.get("approval"),
                    "candidate": inputs.get("candidate"), "reference": inputs.get("reference"),
                    "gpu": resources.get("gpu"), "cpu_threads": resources["cpu_threads"],
                    "estimated_seconds": resources["estimated_seconds"]}
        if task["kind"] == "literature":
            return {"kind": "literature", **inputs, "estimated_seconds": 45}
        raise ContractError("此任务不由子进程执行")


    def dispatch(self):
        count = 0
        for candidate in self.queue.ready():
            task = self.queue.get(candidate["task_id"])
            held = {key for t in self.queue.snapshot().values() if t["state"] == "running" for key in t["resource_keys"]}
            if held.intersection(task["resource_keys"]):
                continue
            if task["kind"] in ("host_tool", "experiment_request"):
                self.queue.claim(task["task_id"])
                try:
                    if task["kind"] == "host_tool":
                        result = self.host_tool(task)
                        self.queue.finish(task["task_id"], result=result)
                    else:
                        self.progress_request(self.queue.get(task["task_id"]))
                except BudgetExhausted:
                    raise
                except ContractError as exc:
                    self.queue.fail(task["task_id"], {"code": type(exc).__name__, "message": str(exc)})
                count += 1
                continue
            active = [t for t in self.queue.snapshot().values() if t["state"] == "running"]
            if task["kind"] == "llm" and any(t["kind"] == "llm" for t in active):
                continue
            if task["kind"] in ("experiment", "verification") and sum(t["kind"] in ("experiment", "verification") for t in active) >= self.config["runtime"]["experiment_concurrency"]:
                continue
            refs = task["inputs"].get("refs", []) + ([task["inputs"]["experiment_ref"]] if task["inputs"].get("experiment_ref") else [])
            versions = {ref: self.journal.card(ref)["content_sha256"] for ref in refs}
            if task["kind"] == "experiment":
                experiment = self.journal.card(task["inputs"]["experiment_ref"])
                verify_assets(experiment["payload"]["execution"])
                approval = task["inputs"].get("approval")
                cached = self.broker.cached(experiment, code_artifact=approval["candidate"] if approval else None)
                if cached:
                    self.queue.claim(task["task_id"], input_versions=versions)
                    self.queue.finish(task["task_id"], outputs=[cached["id"]], result={"cache_reused": True})
                    count += 1
                    continue
            if task.get("input_versions") and any(versions.get(k) != v for k, v in task["input_versions"].items()):
                raise ContractError("任务引用版本发生静默变化")
            if task["kind"] == "llm":
                stale = self.retriever.relevant_delta(task["memory_version"], refs=refs,
                                                     scope=self.brief["payload"]["model"].get("scope", {}))
                if stale:
                    self.event("task_context_refreshed", f"refresh/{task['task_id']}/{self.journal.memory_version}",
                               task_id=task["task_id"], added_refs=stale, memory_version=self.journal.memory_version)
            try:
                self.runtime.launch(task, self.request_body(task), input_versions=versions)
            except RequestBudgetInsufficient as exc:
                self.queue.claim(task["task_id"], input_versions=versions)
                self.queue.fail(task["task_id"], {"code": type(exc).__name__, "message": str(exc)})
            count += 1
        return count


    def wait_for(self, task, ids, **patch):
        self.queue.change(task["task_id"], "waiting", patch={"waiting_for": list(ids), **patch})


    def wait_children_or_fail(self, task, ids):
        children = [self.queue.get(i) for i in ids]
        if any(t["state"] in ("failed", "cancelled") for t in children):
            self.queue.fail(task["task_id"], {"code": "service_failed", "children": [t["task_id"] for t in children],
                "causes": [{"task_id": t["task_id"], "state": t["state"], "error": deepcopy(t.get("error"))}
                           for t in children if t["state"] in ("failed", "cancelled")]})
        else:
            self.wait_for(task, ids)


    def accept(self, task, receipt, path):
        self.runtime.settle(receipt)
        if receipt["status"] != "completed":
            error = receipt.get("error", {"code": "unknown_worker_error"})
            code = error.get("code", "")
            if task["kind"] == "llm" and structured_output_error(error):
                # Reuse the same bounded contract-repair counter; no transport retry loop.
                raise ContractError(error["message"])
            retry = code in ("worker_lost", "InterruptedError", "llm_timeout", "llm_transport_error", "llm_format_error")
            audit = Path(path).parent / "llm_audit.json"
            if task["kind"] == "llm" and audit.exists() and read_json(audit).get("model_call_started") is False:
                self.transport_blocked, retry = error, False
            self.queue.fail(task["task_id"], error, retryable=retry)
            if code in ("llm_preflight_failed", "llm_authentication", "llm_quota_unknown", "llm_quota_exhausted", "llm_credit_policy",
                        "llm_model_mismatch", "llm_unsupported_model", "llm_unsupported_effort", "llm_cli_version", "llm_tool_isolation"):
                self.transport_blocked = error
            return
        data = receipt["data"]
        if task["kind"] == "llm":
            response = data["response"]
            if not isinstance(response, dict):
                raise ContractError("Agent response 必须是对象")
            if response.get("source") == "test_double" and self.config["purpose"] != "infrastructure_test":
                raise ContractError("真实运行拒绝模拟 Agent 输出")
            if response.get("tool_calls"):
                if response.get("decision"):
                    raise ContractError("一次响应不能同时提交最终决定与待执行工具")
                self.receive_tools(task, response, path)
            else:
                decision = response.get("decision") or {}
                if not isinstance(decision, dict) or not isinstance(decision.get("content"), dict):
                    raise ContractError("Agent decision/content 必须是对象")
                if decision.get("action") == "clarify":
                    value = decision["content"]
                    if any(not isinstance(value.get(k), str) or not value[k].strip() for k in ("question", "reason")):
                        raise ContractError("澄清必须说明具体缺失信息与科研影响")
                    self.queue.change(task["task_id"], "needs_clarification", patch={"clarification": value, "receipt": str(path)})
                elif decision.get("action") == "submit":
                    self.submit(task, decision["content"], path)
                else:
                    raise ContractError("缺少合法 submit/clarify 科研动作")
        elif task["kind"] == "experiment":
            experiment = self.journal.card(task["inputs"]["experiment_ref"])
            approval = task["inputs"].get("approval")
            evidence = self.broker.evidence(experiment, receipt, path, code_artifact=approval["candidate"] if approval else None)
            self.queue.finish(task["task_id"], outputs=[evidence["id"]], result={"receipt": str(path)}, cards=[evidence],
                              events=[self.memory.live_update(evidence_ref=evidence["id"])])
            if experiment["payload"]["level"] == "L3":
                provenance = evidence["payload"]["provenance"]
                if provenance.get("model_ref"):
                    try:
                        if self.brief["payload"]["evaluation_protocol"].get("formal_incumbent_allowed") is False:
                            raise ContractError("ResearchBrief.evaluation_protocol.formal_incumbent_allowed 禁止本次结果晋升 incumbent")
                        projection = self.memory.consider_incumbent({"model_ref": provenance["model_ref"],
                            "protocol_hash": provenance["protocol_hash"], "evidence_refs": [evidence["id"]]}, fingerprint(self.brief["payload"]["evaluation_protocol"]))
                    except ContractError as exc:
                        projection = None
                        self.event("incumbent_ineligible", "incumbent/ineligible/" + evidence["id"], evidence_ref=evidence["id"], reason=str(exc))
                    if projection:
                        self.journal.commit("incumbent/" + evidence["id"], events=[projection])
        else:
            self.queue.finish(task["task_id"], result=data)


    def poll(self):
        count = 0
        for task, result, path in self.runtime.poll():
            try:
                self.accept(task, result, path)
            except ContractError as exc:
                current = self.queue.get(task["task_id"])
                if current["state"] == "running":
                    if task["kind"] == "llm" and current.get("repair_count", 0) < 2:
                        feedback = {"error": str(exc), "rejected_receipt": str(path),
                                    "task_id": task["task_id"], "execution_id": result.get("execution_id"),
                                    "instruction": "仅修复合同错误；不得私改科学规格。规格不清时clarify。"}
                        response = result.get("data", {}).get("response", {})
                        # 所有可修复输入错误都返回原任务及原响应，不捕获未知程序异常。
                        feedback["rejected_response"] = response
                        format_error = result.get("error", {})
                        if structured_output_error(format_error):
                            feedback.update(format_error=format_error,
                                            instruction="原任务的结构化输出无效；按 stage 和错误定位重新输出严格 JSON，不得改写科研内容或规格。修复后仍需通过全部原合同校验。")
                            prior_feedback = current.get('repair_feedback', {})
                            prior_response = prior_feedback.get('rejected_response', {})
                            prior_decision = prior_response.get('decision') if isinstance(prior_response, dict) else None
                            prior_draft = self.rejected_draft(current)
                            if (task['inputs']['purpose'] == 'record'
                                    and prior_feedback.get('task_id', task['task_id']) == task['task_id']
                                    and isinstance(prior_decision, dict) and prior_decision.get('action') == 'submit'
                                    and isinstance(prior_draft, dict) and 'repair_patch' not in prior_draft):
                                # The undecodable attempt is separate from the last parsed draft.
                                # Keep the latter so a corrected patch can still merge within this task.
                                feedback['rejected_response'] = deepcopy(prior_feedback['rejected_response'])
                                feedback['rejected_response_receipt'] = prior_feedback.get('rejected_response_receipt', prior_feedback.get('rejected_receipt'))
                                feedback['instruction'] += ' rejected_response 保留本任务上次完整被拒稿；可提交 {repair_patch:{问题字段:修正值}}，其余字段沿用该稿。'
                        if (task['inputs']['purpose'] == 'record' and isinstance(response, dict)
                                and isinstance(response.get('decision'), dict)
                                and response['decision'].get('action') == 'submit'):
                            patch = response['decision'].get('content', {})
                            prior = self.rejected_draft(task)
                            if (isinstance(patch, dict) and set(patch) == {'repair_patch'} and isinstance(patch['repair_patch'], dict)
                                    and isinstance(prior, dict)):
                                feedback['rejected_response'] = deepcopy(response)
                                updates = patch['repair_patch']
                                if any(key.startswith(field + suffix) for key in updates
                                       for field in PAYLOAD_SCHEMAS['MetareviewCard']['properties'] for suffix in ('.', '[')):
                                    updates = {}  # 错误路径仍在原回执，不污染可恢复的完整草稿。
                                feedback['rejected_response']['decision']['content'] = {**prior, **updates}
                            feedback['instruction'] += ' repair_patch 只替换顶层字段；修复知识时提供 {repair_patch:{knowledge_updates:[完整修正数组]}}，不使用嵌套路径。其余字段沿用被拒正文。'
                        self.queue.change(task["task_id"], "retry", patch={"repair_count": current.get("repair_count", 0) + 1,
                            "repair_feedback": feedback, "retry_after": time.time()})
                    else:
                        self.queue.fail(task["task_id"], {"code": "contract_error", "message": str(exc), "receipt": str(path)})
                else:
                    raise
            count += 1
        return count


    def answer(self, task_id, text):
        task = self.queue.get(task_id)
        if task["state"] != "needs_clarification" or not text.strip():
            raise ContractError("需回答待澄清任务，不能改写已完成科研决定")
        answers = task.get("user_answers", []) + [{"question": task["clarification"], "answer": text, "at": time.time()}]
        self.queue.change(task_id, "ready", patch={"user_answers": answers, "clarification": None, "step": task["step"] + 1})


    def retry_failed(self, reason):
        if not reason.strip():
            raise ContractError("显式重试需说明原因，不重置预算或历史尝试")
        for task in self.queue.snapshot().values():
            if task["state"] == "failed":
                self.queue.change(task["task_id"], "ready", patch={"explicit_retry_reason": reason})


    def summary(self, status=None, error=None):
        events = [e for e in self.journal.events if e["type"] == "run_status"]
        tasks = self.queue.snapshot()
        counts = {state: sum(t["state"] == state for t in tasks.values()) for state in sorted({t["state"] for t in tasks.values()})}
        status = status or (events[-1]["status"] if events else "created")
        rounds = [{"round": n, "matches": self.matches(n), "l3_outcome": self.l3_outcome(n)}
                  for n in self.budget.snapshot()["rounds_started"]]
        attempts = [r["l3_outcome"] for r in rounds if r["l3_outcome"]["task_refs"]]
        value = {"schema_version": "qavit.report/2", "run_id": self.journal.run_id,
                 "status": status, "error": error, "workflow_completed": status == "completed",
                 "formal_validation_success": status == "completed" and bool(attempts) and all(r["formal_validation_success"] for r in attempts),
                 "tasks": counts, "budget": self.budget.snapshot(), "memory": self.memory.snapshot(),
                 "rounds": rounds,
                 "clarifications": [{"task_id": t["task_id"], **t["clarification"]} for t in tasks.values() if t["state"] == "needs_clarification"],
                 "failures": [{"task_id": t["task_id"], "key": t["key"], "error": t.get("error")} for t in tasks.values() if t["state"] == "failed"],
                 "evidence": [{"id": c["id"], "source": c["payload"]["measurement_source"],
                               "execution_id": c["payload"]["execution_id"]} for c in self.journal.find_cards("EvidenceCard")],
                 "legacy_memory_imported": False, "legacy_incumbent_imported": False,
                 "current_implementation_matches": self.current_implementation_matches,
                 "index_warning": getattr(self.journal, "index_error", None),
                 "pending_recording": self.memory.recording_batch(),
                 "recording_status": self.memory.recording_status(),
                 "scientific_review_complete": not self.memory.recording_batch()['source_refs'],
                 "journal": self.journal.audit()}
        return value


    def run(self):
        if self.read_only:
            raise ContractError("只读报告实例不能启动任务")
        with file_lock(self.run_dir / "supervisor.lock", timeout=0):
            status, error = "running", None
            try:
                self.event("run_status", f"status/start/{self.journal.sequence}", status=status)
                # 崩溃前已提交的工作结果优先接收；不消耗新的实验或模型调用。
                self.poll()
                while True:
                    budget = self.budget.check()
                    self.poll()
                    if self.transport_blocked:
                        status, error = "blocked", self.transport_blocked
                        break
                    if time.time() >= budget["worker_deadline_unix"]:
                        status, error = "budget_exhausted", "到达停机预留边界；保存现场，未完成阶段保持未完成"
                        break
                    closing = any(e["type"] == "budget_closeout_started" for e in self.journal.events)
                    if not closing and budget["config"].get("closeout_reserve_seconds", 0) and budget["research_remaining_seconds"] <= 0:
                        self.runtime.stop()
                        self.poll()
                        for task in self.queue.snapshot().values():
                            if task["state"] not in ("completed", "failed", "cancelled", "ineligible"):
                                self.queue.change(task["task_id"], "cancelled", patch={"error": {"code": "closeout_reserve"}})
                        self.event("budget_closeout_started", "budget/closeout", round=max(budget["rounds_started"], default=1))
                        closing = True
                    if closing:
                        number = next(e["round"] for e in self.journal.events if e["type"] == "budget_closeout_started")
                        batch = self.memory.recording_batch()
                        if self.recording_deferred(batch):
                            status, error = 'budget_stopped', '事实已归档；同一批科学复盘已延期，不重复消耗收尾额度'
                            break
                        # 独立收尾任务避免覆盖中断的 Recorder，也不伪造 round_finished。
                        key = f"r{number:03d}/budget_record"
                        if key not in self.tasks():
                            self.enqueue_recorder(number, incomplete=True, key=key)
                        recorded = self.tasks()[key]
                        if recorded["state"] == 'completed' or self.defer_recorder(recorded):
                            status, error = "budget_stopped", "科研预算结束；收尾 Recorder 状态: " + recorded["state"]
                            break
                    elif self.advance():
                        status = "completed"
                        break
                    dispatched = self.dispatch()
                    tasks = self.queue.snapshot().values()
                    if not dispatched and not any(t["state"] in ("running", "ready", "retry") for t in tasks):
                        status = "needs_clarification" if any(t["state"] == "needs_clarification" for t in tasks) else "blocked"
                        break
                    atomic_json(self.run_dir / "status.json", {"status": status, "sequence": self.journal.sequence,
                                                               "remaining_seconds": self.budget.snapshot()["remaining_seconds"]})
                    time.sleep(.1)
            except BudgetExhausted as exc:
                status, error = "budget_exhausted", str(exc)
            except KeyboardInterrupt:
                status, error = "interrupted", "user/process interruption"
            except OSError as exc:
                status, error = "storage_blocked", {"code": type(exc).__name__, "errno": exc.errno, "message": str(exc)}
            except BaseException as exc:
                status, error = "failed", {"code": type(exc).__name__, "message": str(exc)}
            finally:
                failures = []
                for step, action in (("release_workers", self.runtime.stop), ("accept_durable_receipts", self.poll)):
                    try:
                        action()
                    except Exception as exc:
                        failures.append({"step": step, "code": type(exc).__name__, "message": str(exc)})
                if failures and status == "completed":
                    status = "recovery_required"
                committed = False
                try:
                    pending = self.memory.recording_batch()
                    if pending['source_refs']:
                        self.event('recorder_pending', f'recorder/pending/{self.journal.sequence}', batch=pending, reason=status)
                    self.event("run_status", f"status/stop/{self.journal.sequence}", status=status, error=error)
                    committed = True
                except Exception as exc:
                    failures.append({"step": "save_stop_event", "code": type(exc).__name__, "message": str(exc)})
                try:
                    report = self.summary(status, error)
                except Exception as exc:
                    failures.append({"step": "rebuild_report", "code": type(exc).__name__, "message": str(exc)})
                    report = {"schema_version": "qavit.report/2", "run_id": self.journal.run_id,
                              "status": status, "error": error}
                report.update(stop_event_saved=committed, finalization_errors=failures)
                report["state_saved"] = committed and not failures
                try:
                    atomic_json(self.run_dir / "summary.json", report)
                    atomic_json(self.run_dir / "status.json", {"status": status, "error": error, "sequence": self.journal.sequence})
                except OSError as exc:
                    report["state_saved"] = False
                    failures.append({"step": "save_report", "code": type(exc).__name__, "errno": exc.errno, "message": str(exc)})
                # 磁盘不可写时仍向调用者返回真实失败。已提交卡片/完整worker回执
                # 保留为恢复依据，不能谎称失败的最终状态已经持久化。
            return report

