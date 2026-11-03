"""实验申请子流程：审批→冻结→Coding/独立参考→数值验证→Safeguard→执行→解释。

这是 Workflow 的职责分区，不独立实例化；状态和生命周期统一由 Workflow 拥有。
"""
from pathlib import Path
from agents.tasks import TASKS
from core.cards import ContractError, fingerprint
from runtime.identity import code_identity
from tools.safeguards import approval_receipt



class ExperimentRequests:
    def return_code_clarifications(self):
        """修订无法改变冻结规格时，将问题交回仍等待申请结果的科研任务。"""
        tasks = self.queue.snapshot()
        for task in tasks.values():
            parent = tasks.get(task["inputs"].get("parent_task"))
            if (task["kind"] != "experiment_request" or task["state"] != "waiting"
                    or not parent or parent["state"] != "waiting"
                    or task["task_id"] not in parent.get("waiting_for", [])):
                continue
            for identifier in task.get("waiting_for", []):
                child = tasks[identifier]
                if (child["state"] != "needs_clarification" or child["inputs"].get("purpose") != "code"
                        or not child["inputs"].get("context", {}).get("revision_feedback")):
                    continue
                clarification = child["clarification"]
                self.queue.change(identifier, "cancelled", patch={"error": {
                    "code": "frozen_specification_clarification", **clarification}})
                for sibling_id in task.get("waiting_for", []):
                    sibling = self.queue.get(sibling_id)
                    if (sibling["inputs"].get("purpose") == "reference"
                            and sibling["state"] in ("pending", "ready", "retry", "waiting", "needs_clarification")):
                        self.queue.change(sibling_id, "cancelled", patch={"error": {"code": "request_returned_to_author"}})
                self.queue.change(task["task_id"], "ready")
                self.queue.claim(task["task_id"])
                self.queue.finish(task["task_id"], result={"status": "rejected",
                    "reason": "frozen_specification_clarification", "experiment_ref": task["experiment_ref"],
                    "clarification": clarification, "coding_task_ref": self.queue.get(identifier)["task_card"],
                    "review_ref": task.get("code_revision_feedback", {}).get("review_ref"),
                    "instruction": "冻结规格不改写；由原申请任务根据澄清决定是否正常提交修订草案，或保留证据缺口。"
                                   "不得跳过数值验证或 Safeguard，不得将工程拒绝判作假说失败。"})
                break

    def l3_outcome(self, number):
        """由持久任务和证据重建；失败 TaskCard 不作为 Recorder 的成功依赖。"""
        tasks = self.tasks()
        key = f"r{number:03d}/L3"
        plan, trial = tasks.get(f"r{number:03d}/plan_l3"), tasks.get(key)
        related = [t for k, t in tasks.items() if k == key or k.startswith(key + "/")]
        if plan:
            related.insert(0, plan)
        facts = [{"task_id": t["task_id"], "key": t["key"], "task_ref": t["task_card"],
                  "state": t["state"], "result": t.get("result"), "error": t.get("error")}
                 for t in related]
        current = trial or plan
        result = (current.get("result") or {}) if current else {}
        status = (result.get("status", current["state"]) if current["state"] == "completed"
                  else current["state"]) if current else ("not_requested" if self.config["enable_l3"] else "disabled")
        revision = trial.get("code_revision", 0) if trial else 0
        execution = tasks.get(key + "/execute" + (f"/revision{revision}" if revision else ""))
        interpretation = tasks.get(key + "/interpret")
        cache_reused = result.get("cache_reused") or bool(interpretation and interpretation["inputs"]["context"].get("cache_reused"))
        code = tasks.get(key + "/code" + (f"/revision{revision}" if revision else ""))
        code_artifact = (code.get("result") or {}).get("artifact") if code else None
        refs = set(ref for t in related for ref in t["outputs"])
        if interpretation:
            refs.update(interpretation["inputs"]["refs"])
        if trial and (trial.get("result") or {}).get("evidence_ref"):
            refs.add(trial["result"]["evidence_ref"])
        experiment_ref = trial.get("experiment_ref") if trial else None
        valid, issues = [], []
        for ref in sorted(refs):
            try:
                evidence = self.journal.card(ref)
                if evidence["kind"] != "EvidenceCard":
                    continue
                if not experiment_ref:
                    raise ContractError("L3 请求缺少冻结实验引用")
                experiment = self.journal.card(experiment_ref)
                if experiment["payload"]["level"] != "L3":
                    raise ContractError("非 L3 证据不能作为正式实验验证")
                self.broker.validate_evidence_for(experiment, evidence, code_artifact=code_artifact)
                valid.append(ref)
            except (ContractError, OSError) as exc:
                issues.append({"evidence_ref": ref, "reason": str(exc)})
        return {"status": status, "request_state": trial["state"] if trial else "not_started",
                "execution_state": execution["state"] if execution else ("cache_reused" if cache_reused else "not_started"),
                "experiment_ref": experiment_ref, "tasks": facts,
                "task_refs": [t["task_card"] for t in related],
                "valid_evidence_refs": valid, "valid_evidence_missing": not bool(valid), "evidence_issues": issues,
                "formal_validation_success": bool(trial and trial["state"] == "completed" and status == "completed" and valid)}

    def progress_request(self, task):
        inputs, key = task["inputs"], task["key"]
        tasks = self.tasks()
        if not task.get("experiment_ref"):
            if inputs["requester"] != "judger":
                approval_key = key + "/scientific_approval"
                identifier = self.enqueue_role(approval_key, "experiment_approval", refs=inputs["refs"],
                                context={"experiment_draft": inputs["draft"], "requester": inputs["requester"]}, parent_task=task["task_id"])
                approval = self.result_card(tasks.get(approval_key))
                if not approval:
                    self.wait_children_or_fail(task, [identifier])
                    return
                if not approval["payload"]["approved"]:
                    self.queue.finish(task["task_id"], result={"status": "rejected", "review_ref": approval["id"]})
                    return
                refs = inputs["refs"] + [approval["id"]]
            else:
                refs = inputs["refs"]
            if inputs["draft"]["level"] == "L3" and not self.config["enable_l3"]:
                raise ContractError("本次验证明确禁用 L3")
            experiment = self.broker.freeze(inputs["draft"], requester=inputs["requester"], refs=refs)
            resources = experiment["payload"]["resources"]
            runtime = self.config["runtime"]
            if resources["cpu_threads"] > runtime["cpu_threads"]:
                raise ContractError("请求 CPU 线程超出本次资源上限")
            if resources.get("gpu") is not None and resources["gpu"] not in runtime.get("allowed_gpus", []):
                raise ContractError("请求 GPU 不在本次授权范围")
            self.queue.change(task["task_id"], "running", patch={"experiment_ref": experiment["id"]}, cards=[experiment])
            task = self.queue.get(task["task_id"])
        experiment = self.journal.card(task["experiment_ref"])
        execution, approval = experiment["payload"]["execution"], None
        suffix = ""
        if execution.get("requires_code"):
            revision = task.get("code_revision", 0)
            suffix = f"/revision{revision}" if revision else ""
            code_key, ref_key = key + "/code" + suffix, key + "/reference" + suffix
            failed_execution = tasks.get(key + "/execute" + suffix)
            if (failed_execution and failed_execution["state"] == "failed" and revision < 2
                    and (failed_execution.get("error") or {}).get("code") in ("TypeError", "AttributeError")):
                # 实现接口错误与审查拒绝共用两次修订额度；不重跑旧源码，不改冻结规格。
                request_path = failed_execution.get("request_path")
                self.queue.change(task["task_id"], "retry", patch={"retry_after": 0,
                    "code_revision": revision + 1, "code_revision_feedback": {
                    "execution_task_ref": failed_execution["task_card"], "error": failed_execution["error"],
                    "worker_log": str(Path(request_path).with_name("worker.log")) if request_path else None,
                    "previous_candidate": tasks[code_key]["result"]["artifact"],
                    "instruction": "依据实际执行错误和接口文档修订候选实现；仅修实现，不更改冻结数学、数据或容差。"
                                   "规格或独立参考有实质问题时 clarify。新源码必须重新数值验证及 Safeguard；"
                                   "执行失败和审查拒绝合计最多两次修订，失败执行原件保留。"}})
                return
            code_context = {"experiment_ref": experiment["id"]}
            reference_context = {"experiment_ref": experiment["id"]}
            if revision:
                code_context["revision_feedback"] = task["code_revision_feedback"]
                prior = task["code_revision_feedback"].get("numerical", {})
                failure = prior.get("failure", {})
                if (prior.get("status") == "rejected" and isinstance(failure, dict)
                        and failure.get("side") == "reference"):
                    # 仅反馈参考自身执行异常；不泄漏候选、对比数值或评审意见。
                    reference_context["self_execution_failure"] = {
                        "source_sha256": prior.get("reference_sha256"),
                        "error": {k: failure.get("error", {}).get(k) for k in ("type", "message")},
                        "instruction": "独立参考自身上次执行错误；依据冻结规格及公开接口修复，保持原验证门禁。"}
            ids = [self.enqueue_role(code_key, "code", refs=[self.brief["id"], experiment["id"]],
                                     context=code_context, parent_task=task["task_id"]),
                   self.enqueue_role(ref_key, "reference", refs=[self.brief["id"], experiment["id"]],
                                     context=reference_context, parent_task=task["task_id"]) ]
            tasks = self.tasks()
            if not all(tasks[k]["state"] == "completed" for k in (code_key, ref_key)):
                self.wait_children_or_fail(task, ids)
                return
            candidate, reference = tasks[code_key]["result"]["artifact"], tasks[ref_key]["result"]["artifact"]
            implementation = code_identity()["core_sha256"][:16]
            verify_key = key + "/numerical/" + implementation + suffix
            identifier = self.queue.enqueue(verify_key, "verification", role="safeguard", inputs={
                "experiment_ref": experiment["id"], "candidate": candidate, "reference": reference, "parent_task": task["task_id"]},
                resource_keys=self.experiment_resources(experiment), depends_on=ids)
            numerical_task = self.tasks()[verify_key]
            if numerical_task["state"] != "completed":
                self.wait_children_or_fail(task, [identifier])
                return
            numerical = numerical_task["result"]
            safeguard_key = key + "/safeguard/" + implementation + suffix
            identifier = self.enqueue_role(safeguard_key, "safeguard", refs=[self.brief["id"], experiment["id"]], context={
                "experiment_ref": experiment["id"], "candidate": candidate, "reference": reference, "numerical": numerical},
                depends=[numerical_task["task_id"]], parent_task=task["task_id"])
            review = self.result_card(self.tasks().get(safeguard_key))
            if not review:
                self.wait_children_or_fail(task, [identifier])
                return
            if not review["payload"]["approved"] or numerical["status"] != "passed":
                if revision < 2:
                    # 冻结规格不变；候选与参考分别重建并重新验证审查，参考不接收候选反馈。
                    self.queue.change(task["task_id"], "retry", patch={"retry_after": 0,
                        "code_revision": revision + 1, "code_revision_feedback": {
                        "review_ref": review["id"], "review": review["payload"], "numerical": numerical,
                        "previous_candidate": candidate,
                        "instruction": "依据拒绝原因修订候选源码；仅修实现，不更改冻结数学、数据或容差。"
                                       "原源码可按 previous_candidate.path 读取；规格或参考存在实质问题时 clarify。"
                                       "新源码将重新数值验证及 Safeguard，最多两次修订。"}})
                    return
                self.queue.finish(task["task_id"], outputs=[review["id"]], result={"status": "rejected", "reason": "Safeguard or numerical verification rejected"})
                return
            approval = approval_receipt(experiment, candidate, reference, numerical, review)
        trial_key = key + "/execute" + suffix
        prior_trial = self.tasks().get(trial_key)
        own_result = prior_trial and prior_trial["state"] == "completed"
        cache = self.journal.card(prior_trial["outputs"][0]) if own_result else self.broker.cached(experiment, code_artifact=approval["candidate"] if approval else None)
        cache_reused = bool(cache and not own_result) or bool(own_result and (prior_trial.get("result") or {}).get("cache_reused"))
        if cache:
            evidence = cache
        else:
            identifier = self.queue.enqueue(trial_key, "experiment", role=None, inputs={
                "experiment_ref": experiment["id"], "approval": approval, "parent_task": task["task_id"]}, resource_keys=self.experiment_resources(experiment))
            trial = self.tasks()[trial_key]
            if trial["state"] != "completed":
                if (execution.get("requires_code") and trial["state"] == "failed" and revision < 2
                        and (trial.get("error") or {}).get("code") in ("TypeError", "AttributeError")):
                    self.queue.change(task["task_id"], "retry", patch={"retry_after": 0})
                    return
                self.wait_children_or_fail(task, [identifier])
                return
            evidence = self.journal.card(trial["outputs"][0])
        interpretation_key = key + "/interpret"
        parent = self.queue.snapshot().get(inputs.get("parent_task"))
        if (inputs["requester"] == "judger" and parent and parent["kind"] == "llm"
                and parent["role"] == "judger" and parent["state"] == "waiting"
                and TASKS[parent["inputs"]["purpose"]]["output"] == "ReviewDecisionCard"
                and task["task_id"] in parent.get("waiting_for", [])
                and interpretation_key not in self.tasks()):
            # 原 Judger 已有问题与判断上下文，直接续调使用真实证据；已发布解释任务沿原链完成。
            self.queue.finish(task["task_id"], outputs=[evidence["id"]], result={
                "status": "completed", "experiment_ref": experiment["id"], "evidence_ref": evidence["id"],
                "cache_reused": cache_reused, "raw_fact_and_interpretation_separated": True})
            return
        identifier = self.enqueue_role(interpretation_key, "interpret", refs=[*inputs["refs"], experiment["id"], evidence["id"]],
                        context={"question": experiment["payload"]["question"], "decision_impact": experiment["payload"]["decision_impact"],
                                 "cache_reused": cache_reused, "request_experiment_ref": experiment["id"]}, parent_task=task["task_id"])
        interpretation = self.result_card(self.tasks().get(interpretation_key))
        if not interpretation:
            self.wait_children_or_fail(task, [identifier])
            return
        self.queue.finish(task["task_id"], outputs=[evidence["id"], interpretation["id"]], result={
            "status": "completed", "experiment_ref": experiment["id"], "evidence_ref": evidence["id"],
            "interpretation_ref": interpretation["id"], "cache_reused": cache_reused,
            "raw_fact_and_interpretation_separated": True})


    def experiment_resources(self, experiment):
        resources = experiment["payload"]["resources"]
        keys = ["model:" + fingerprint(experiment["payload"]["execution"]["backend"])]
        if resources.get("gpu") is not None:
            keys.append("gpu:" + str(resources["gpu"]))
        return keys

