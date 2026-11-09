"""Read committed history directly, without Journal/Workflow constructors.

Inputs delivered to the subject come ONLY from saved call requests. A referenced
card or asset is not expanded into subject knowledge. Later outcomes stay in the
program report, never in the quality Judge's input.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import VERSION
from .common import (QAVIT, EvalError, SourceStore, canonical, confined, digest,
                     fingerprint, loads, redact, write_json)
from .metrics import call_metrics, run_timing, task_timing, tool_metrics
from .rules import task_catalog


class FrozenRun:
    def __init__(self, root, output):
        self.root = Path(root).resolve()
        self.store = SourceStore(output)
        self.documents, self.file_rows, self.issues = {}, {}, []
        self.manifest = self.read("research_run.json", required=True)
        self.run_id = self.manifest["run_id"]
        self.tasks, self.versions, self.cards, self.events = {}, defaultdict(list), {}, []
        self.sequence, self.head = 0, None
        paths = sorted((self.root / "commits").glob("*.json"))
        if not paths:
            raise EvalError("no committed history")
        keys = set()
        for path in paths:
            commit = self.read(path, required=True)
            seq = self.sequence + 1
            if (path.name != f"{seq:09d}.json" or commit.get("sequence") != seq or
                    commit.get("run_id") != self.run_id or commit.get("previous_sha256") != self.head or
                    commit.get("sha256") != fingerprint({k: v for k, v in commit.items() if k != "sha256"}) or
                    commit.get("key") in keys):
                raise EvalError("invalid journal chain at " + path.name)
            keys.add(commit["key"])
            for entry in commit["cards"]:
                if entry["ref"] in self.cards:
                    raise EvalError("duplicate committed card")
                self.cards[entry["ref"]] = {**entry, "sequence": seq}
            for index, event in enumerate(commit["events"]):
                value = {**event, "commit_sequence": seq, "commit_time": commit.get("created_unix")}
                self.events.append(value)
                if event["type"] == "task_snapshot":
                    task = event["task"]
                    ref = task.get("task_card")
                    if ref not in self.cards:
                        raise EvalError("task snapshot has no committed TaskCard")
                    card = self.card(ref)
                    if card and card["payload"] != {k: v for k, v in task.items() if k != "task_card"}:
                        raise EvalError("TaskCard/event mismatch")
                    self.tasks[task["task_id"]] = task
                    self.versions[task["task_id"]].append({"task": task, "time": commit.get("created_unix"),
                        "sequence": seq, "path": str(path.relative_to(self.root)), "locator": f"#/events/{index}/task"})
            self.sequence, self.head = seq, commit["sha256"]
        self.calls, self.jobs, self.orphans = [], [], []
        self.read_jobs()

    def read(self, path, *, required=False):
        path = Path(path)
        path = confined(path if path.is_absolute() else self.root / path, self.root)
        name = path.relative_to(self.root).as_posix()
        if name in self.documents:
            return self.documents[name]
        try:
            row, raw = self.store.capture(path, self.root)
            value = loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise EvalError("expected object")
        except (OSError, ValueError, UnicodeError) as exc:
            if required:
                raise EvalError("unreadable required source: " + name) from exc
            self.issues.append({"path": name, "code": "missing_or_invalid_source", "detail": type(exc).__name__})
            return None
        self.documents[name], self.file_rows[name] = value, row
        return value

    def card(self, ref):
        if ref not in self.cards:
            raise EvalError("uncommitted card reference")
        path = "cards/" + digest(ref.encode()) + ".json"
        if self.cards[ref].get("availability") == "missing":
            return None
        value = self.read(path)
        if value is None:
            self.cards[ref]["availability"] = "missing"
            return None
        if (value.get("id") != ref or value.get("run_id") != self.run_id or
                value.get("content_sha256") != self.cards[ref]["sha256"] or
                value.get("content_sha256") != fingerprint({k: v for k, v in value.items() if k != "content_sha256"})):
            raise EvalError("committed card hash/identity mismatch")
        return value

    def source(self, path, locator, category, **metadata):
        name = str(Path(path).relative_to(self.root)) if Path(path).is_absolute() else str(path)
        row = self.file_rows[name]
        sid = "s-" + fingerprint([row["sha256"], locator, category, metadata])[:24]
        return {"source_id": sid, "immutable_ref": "sha256:" + row["sha256"],
                "sha256": row["sha256"], "object": row["object"], "origin": row["path"],
                "locator": locator, "source_type": category, "format": "json",
                "availability": "available", **metadata}

    def card_source(self, ref, category, **metadata):
        value = self.card(ref)
        if value is None:
            raise EvalError("committed card bytes are unavailable")
        return self.source("cards/" + digest(ref.encode()) + ".json", "#", category,
                           card_id=ref, commit_sequence=self.cards[ref]["sequence"], **metadata)

    def read_jobs(self):
        # Include claimed jobs whose directory was lost. Never use filename role guesses.
        directories = {p for p in (self.root / "jobs").glob("*") if p.is_dir()}
        claims = {}
        for versions in self.versions.values():
            for v in versions:
                raw_path = v["task"].get("request_path")
                if raw_path:
                    # A moved run can still be evaluated through its internal jobs/ path.
                    parts = Path(raw_path).parts
                    if "jobs" in parts:
                        directory = self.root.joinpath(*parts[parts.index("jobs"):]).parent
                        directories.add(directory)
                        claims.setdefault(directory.name, v["task"])
        seen = set()
        for directory in sorted(directories):
            if not directory.is_dir():
                self.issues.append({"path": str(directory.relative_to(self.root)), "code": "missing_job_directory"})
                claim = claims.get(directory.name)
                if claim:
                    call = {"execution_id": directory.name, "task_id": claim["task_id"], "kind": claim["kind"],
                        "attempt": claim.get("attempt"), "step": claim.get("step"), "status": "unknown",
                        "model_call_started": None, "response_observed": False, "requested": None,
                        "actual": {"model": None, "reasoning_effort": None, "source": "missing_job"},
                        "usage": None, "reported_usage": None, "elapsed_seconds": None, "started_unix": None,
                        "finished_unix": None, "resources": None, "error": {"code": "missing_job_directory"},
                        "binding_issues": ["missing_job_directory"], "request_path": None, "audit_path": None,
                        "receipt_path": None, "core_sha256": None, "input_versions": claim.get("input_versions", {}),
                        "read_view": None, "prompt_source_sha256": None}
                    self.jobs.append(call)
                    if claim["kind"] == "llm":
                        self.calls.append(call)
                continue
            if not directory.is_relative_to(self.root / "jobs") or directory.is_symlink():
                raise EvalError("job path must stay inside historical run")
            audit = self.read(directory / "llm_audit.json") if (directory / "llm_audit.json").exists() else None
            receipt_path = directory / ("result.json" if (directory / "result.json").exists() else "interrupted.json")
            receipt = self.read(receipt_path) if receipt_path.exists() else None
            started = self.read(directory / "started.json") if (directory / "started.json").exists() else None
            requests = [(p, self.read(p)) for p in sorted(directory.glob("request*.json"))]
            requests = [(p, r) for p, r in requests if r]
            expected = {r["request_sha256"] for r in (receipt, started) if r and r.get("request_sha256")}
            matches = [(p, r) for p, r in requests if self.file_rows[str(p.relative_to(self.root))]["sha256"] in expected]
            issues = []
            if len(expected) > 1 or len(matches) > 1:
                issues.append("ambiguous_request_binding")
            selected = matches[0] if len(matches) == 1 and len(expected) <= 1 else (requests[0] if not expected and len(requests) == 1 else None)
            request_path, request = selected if selected else (None, {})
            if not selected:
                issues.append("missing_or_ambiguous_worker_request")
            task_id = request.get("task_id") or (receipt or {}).get("task_id") or (audit or {}).get("request", {}).get("context", {}).get("task_id")
            execution_id = request.get("execution_id") or (receipt or {}).get("execution_id") or directory.name
            if execution_id != directory.name or execution_id in seen:
                raise EvalError("duplicate/misbound execution identity")
            seen.add(execution_id)
            task = self.tasks.get(task_id)
            if receipt and any(receipt.get(k) != request.get(k) for k in ("task_id", "execution_id", "core_sha256")):
                issues.append("receipt_identity_mismatch")
            spec = request.get("spec") or {}
            if audit:
                actual_request = audit.get("request") or {}
                if audit.get("request_sha256") != fingerprint(actual_request):
                    issues.append("audit_request_hash_mismatch")
                # Message(**m) -> asdict adds exactly the optional tool_call_id=None.
                # No other payload normalization is permitted at this integrity boundary.
                normalized = {**spec, "messages": [{"tool_call_id": None, **m} for m in spec.get("messages", [])]}
                if spec and any(actual_request.get(k) != normalized.get(k) for k in ("role", "system_prompt", "context", "tools", "messages", "response_schema")):
                    issues.append("audit_worker_request_mismatch")
                if task and actual_request.get("context", {}).get("task_id", task_id) != task_id:
                    issues.append("audit_task_mismatch")
            data = (receipt or {}).get("data") or {}
            response = data.get("response") or {}
            tool_calls = response.get("tool_calls") or []
            if not tool_calls and audit and audit.get("response"):
                returned = audit["response"]
                name = (returned.get("output") or {}).get("name")
                arguments = returned.get("resolved_arguments") or {}
                if name == "__tool_batch__":
                    tool_calls = arguments.get("calls", [])
                elif name and name != "__agent_decision__":
                    tool_calls = [{"name": name, "arguments": arguments}]
            observed = bool((audit or {}).get("response")) or bool(response)
            call_started = audit.get("model_call_started") if audit else (True if observed else None)
            usage = (audit or {}).get("usage")
            if usage is None:
                usage = data.get("usage")
            if usage is None and len((audit or {}).get("reported_usage") or []) == 1:
                usage = audit["reported_usage"][0]
            status = (receipt or {}).get("status") or ("failed" if (audit or {}).get("error") else "completed" if observed else "unfinished")
            versions = self.versions.get(task_id, [])
            claim = next((v for v in versions if Path(v["task"].get("request_path") or "").parent.name == directory.name
                          and v["task"]["state"] == "running"), None)
            call = {"execution_id": execution_id, "task_id": task_id, "kind": request.get("kind") or ("llm" if audit else (task or {}).get("kind")),
                    "attempt": claim["task"].get("attempt") if claim else None,
                    "step": claim["task"].get("step") if claim else None,
                    "status": status, "model_call_started": call_started, "response_observed": observed,
                    "requested": (audit or {}).get("requested") or data.get("requested"),
                    "actual": (audit or {}).get("actual") or data.get("actual") or {"model": None, "reasoning_effort": None, "source": "not_reported"},
                    "usage": usage if isinstance(usage, dict) else None,
                    "reported_usage": (audit or {}).get("reported_usage"),
                    "elapsed_seconds": (audit or {}).get("elapsed_seconds"),
                    "started_unix": (audit or {}).get("started_unix"),
                    "finished_unix": (audit or {}).get("finished_unix", (receipt or {}).get("finished_unix")),
                    "resources": (receipt or {}).get("resources"),
                    "error": redact((receipt or {}).get("error") or (audit or {}).get("error")),
                    "binding_issues": issues,
                    "request_path": str(request_path.relative_to(self.root)) if request_path else None,
                    "audit_path": str((directory / "llm_audit.json").relative_to(self.root)) if audit else None,
                    "receipt_path": str(receipt_path.relative_to(self.root)) if receipt else None,
                    "core_sha256": request.get("core_sha256"), "input_versions": request.get("input_versions", {}),
                    "read_view": ((audit or {}).get("request") or spec).get("context", {}).get("read_view"),
                    "tool_calls": [{"id": c.get("id"), "name": c["name"]} for c in tool_calls if isinstance(c, dict) and isinstance(c.get("name"), str)],
                    "prompt_source_sha256": ((audit or {}).get("request") or spec).get("context", {}).get("prompt_source_sha256")}
            self.jobs.append(call)
            if call["kind"] == "llm":
                if not audit:
                    call["binding_issues"].append("missing_llm_audit")
                self.calls.append(call)
            if not task:
                self.orphans.append(execution_id)

    def evidence(self):
        result = []
        for ref in self.cards:
            card = self.card(ref)
            if not card or card["kind"] != "EvidenceCard":
                continue
            p = card["payload"]
            exp_ref = p.get("experiment_ref")
            exp = (self.card(exp_ref) or {}).get("payload", {}) if exp_ref in self.cards else {}
            result.append({"id": ref, "commit_sequence": self.cards[ref]["sequence"],
                "experiment_ref": exp_ref, "execution_id": p.get("execution_id"),
                "measurement_source": p.get("measurement_source", "unknown"), "status": p.get("status", "unknown"),
                "scope": p.get("scope"), "level": exp.get("level"),
                "execution": exp.get("execution"), "resources": p.get("resources"),
                "provenance": p.get("provenance"), "source": self.card_source(ref, "posthoc_observation")})
        return result


def ancestry(task_id, tasks):
    result = []
    while task_id in tasks:
        if task_id in result:
            raise EvalError("cycle in parent_task lineage")
        result.append(task_id)
        inputs = tasks[task_id]["inputs"]
        # Program-created L3 requests bind their author with origin_task, while
        # interactive tool requests use parent_task. Both are explicit host links.
        task_id = inputs.get("parent_task") or inputs.get("origin_task")
    return result


def coverage_from_context(context):
    memory = context.get("memory") or {}
    return {k: memory.get(k, []) for k in ("required_reads", "required_assets", "read_manifest", "field_manifest")}


def build_case(run, task, calls, rules):
    purpose, role = task["inputs"]["purpose"], task["role"]
    context = task["inputs"].get("context", {})
    case = {"schema_version": VERSION, "case_id": "case-" + fingerprint([run.run_id, task["task_id"], purpose])[:24],
            "run_id": run.run_id, "task_id": task["task_id"], "task_key": task["key"],
            "role": role, "purpose": purpose, "rubric_id": rules.selected[purpose]["rubric_id"],
            "slot": context.get("slot"), "selection_stage": context.get("selection_stage"),
            "exchange": context.get("exchange"), "round": context.get("round"), "side": context.get("side"),
            "evaluated_task_state": task["state"], "parent_task": task["inputs"].get("parent_task"),
            "origin_task": task["inputs"].get("origin_task"),
            "depends_on": task.get("depends_on", []), "source_index": [], "trajectory": [],
            "data_coverage": {"missing_or_truncated_sources": [], "delivery_ranges": [], "blocking_issues": [],
                              "boundary_policy": "exact delivered requests per execution; no dereferencing menus/indexes; no posthoc in quality input"},
            "outputs": task.get("outputs", []), "calls": calls}
    sources, trajectory = case["source_index"], case["trajectory"]
    for call in calls:
        eid = call["execution_id"]
        row = {k: call[k] for k in ("execution_id", "attempt", "step", "status", "read_view")}
        row["source_ids"] = []
        issues = call["binding_issues"]
        case["data_coverage"]["missing_or_truncated_sources"].extend({"execution_id": eid, "code": i} for i in issues)
        case["data_coverage"]["blocking_issues"].extend(i for i in issues if i.endswith("mismatch") or i.startswith("ambiguous"))
        input_path, prefix = call["audit_path"], "#/request"
        if not input_path:
            input_path, prefix = call["request_path"], "#/spec"
        delivered = call["model_call_started"] is True
        if input_path and delivered:
            doc = run.documents[input_path]
            request = doc.get("request") if prefix == "#/request" else doc.get("spec")
            if request:
                visible = request.get("context", {})
                cutoff = (visible.get("read_view") or {}).get("sequence")
                if type(cutoff) is not int:
                    case["data_coverage"]["missing_or_truncated_sources"].append({"execution_id": eid, "code": "missing_visibility_sequence"})
                elif cutoff > run.sequence or any(ref not in run.cards or run.cards[ref]["sequence"] > cutoff for ref in call["input_versions"]):
                    case["data_coverage"]["blocking_issues"].append("future_or_uncommitted_input_version")
                for key in ("system_prompt", "context", "messages", "tools", "response_schema"):
                    if key in request:
                        src = run.source(input_path, prefix + "/" + key, "subject_input", execution_id=eid,
                                         visible_at_sequence=cutoff, delivery="recorded_request")
                        sources.append(src)
                        row["source_ids"].append(src["source_id"])
                case["data_coverage"]["delivery_ranges"].append({"execution_id": eid, **coverage_from_context(visible)})
        else:
            case["data_coverage"]["missing_or_truncated_sources"].append({"execution_id": eid, "code": "no_verified_delivered_request"})
        if call["audit_path"]:
            audit = run.documents[call["audit_path"]]
            if "resolved_arguments" in (audit.get("response") or {}):
                src = run.source(call["audit_path"], "#/response/resolved_arguments", "role_response", execution_id=eid)
                sources.append(src); row["source_ids"].append(src["source_id"])
            elif (audit.get("error") or {}).get("raw_output"):
                src = run.source(call["audit_path"], "#/error/raw_output", "unaccepted_response", execution_id=eid)
                sources.append(src); row["source_ids"].append(src["source_id"])
        if call["receipt_path"]:
            response = (run.documents[call["receipt_path"]].get("data") or {}).get("response") or {}
            for key in ("decision", "tool_calls"):
                if response.get(key):
                    src = run.source(call["receipt_path"], "#/data/response/" + key, "worker_response", execution_id=eid)
                    sources.append(src); row["source_ids"].append(src["source_id"])
        trajectory.append(row)
    for ref in task.get("outputs", []):
        if ref not in run.cards or run.card(ref) is None:
            case["data_coverage"]["missing_or_truncated_sources"].append({"card_id": ref, "code": "missing_committed_output"})
            continue
        sources.append(run.card_source(ref, "committed_output", attribution="host_bound_submission; compare with raw role response"))
    final = run.versions[task["task_id"]][-1]
    if task.get("result") is not None:
        sources.append(run.source(final["path"], final["locator"] + "/result", "task_result",
                                  attribution="host_receipt; not an independent scientific oracle"))
    artifact = (task.get("result") or {}).get("artifact")
    if isinstance(artifact, dict) and artifact.get("path"):
        raw_path = Path(artifact["path"])
        parts = raw_path.parts
        path = run.root.joinpath(*parts[parts.index("code"):]) if "code" in parts else raw_path
        try:
            entry, raw = run.store.capture(path, run.root)
            if digest(raw) != artifact.get("source_sha256"):
                raise EvalError("artifact_hash_mismatch")
            sources.append({**entry, "origin": entry["path"], "source_id": "s-" + fingerprint([entry["sha256"], "code_output"])[:24],
                "immutable_ref": "sha256:" + entry["sha256"], "locator": "#", "format": "text",
                "source_type": "committed_code", "availability": "available"})
        except (OSError, ValueError) as exc:
            case["data_coverage"]["missing_or_truncated_sources"].append({"code": "missing_or_changed_code_artifact", "detail": type(exc).__name__})
    own = task["task_id"]
    descendants = [t for t in run.tasks.values() if t["task_id"] != own and own in ancestry(t["task_id"], run.tasks)[1:]]
    direct = [t for t in descendants if (t["inputs"].get("parent_task") or t["inputs"].get("origin_task")) == own]
    ancestors = ancestry(own, run.tasks)[1:]
    request_ancestors = [a for a in ancestors if run.tasks[a]["kind"] == "experiment_request"]
    related = {t["task_id"] for t in descendants}
    for a in request_ancestors:
        related.update(t["task_id"] for t in run.tasks.values() if a in ancestry(t["task_id"], run.tasks))
    case["lineage"] = {"ancestors": ancestors, "direct_children": [t["task_id"] for t in direct],
                       "related_tasks": sorted(related), "note": "Relations are provenance, not additive attribution of child LLM quality/cost."}
    case["metrics"] = {**call_metrics(calls), **task_timing(run.versions[own]), "tools": tool_metrics(direct, calls)}
    case["posthoc_observations"] = {"related_task_ids": sorted(related), "channel": "run.json only; excluded from Judge"}
    case["input_version_refs"] = [{k: c.get(k) for k in ("execution_id", "core_sha256", "prompt_source_sha256", "input_versions", "read_view")} for c in calls]
    case["evidence_cutoff"] = {"per_execution": True, "journal_sequence": run.sequence, "journal_head": run.head}
    case["case_sha256"] = fingerprint(case)
    return case


def prepare(run_path, output, rules, pricing=None):
    run_path, output = Path(run_path).resolve(), Path(output).resolve()
    if output.is_relative_to(run_path) or run_path.is_relative_to(output):
        raise EvalError("evaluation output must not overlap historical run")
    run = FrozenRun(run_path, output)
    catalog = task_catalog()
    by_task = defaultdict(list)
    for call in run.calls:
        by_task[call["task_id"]].append(call)
    excluded, cases = [], []
    for task in run.tasks.values():
        purpose = task["inputs"].get("purpose")
        calls = by_task[task["task_id"]]
        reason = None
        if task["kind"] != "llm":
            continue
        if purpose not in rules.selected:
            reason = "supervisor_program_or_unsupported_purpose"
        elif task.get("role") != catalog[purpose]["role"]:
            reason = "role_purpose_mismatch"
        elif purpose == "rank" and any((run.card(ref) or {}).get("author") == "program" or
                    (run.card(ref) or {}).get("payload", {}).get("selection_source") == "existing_verdicts_and_screening"
                    for ref in task.get("outputs", []) if ref in run.cards):
            reason = "program_rank_not_llm_science"
        elif not calls:
            reason = "no_recorded_llm_execution"
        elif all(c["model_call_started"] is False for c in calls):
            reason = "preflight_only_no_subject_llm_call"
        if reason:
            excluded.append({"task_id": task["task_id"], "purpose": purpose, "reason": reason})
            continue
        calls.sort(key=lambda c: (c.get("attempt") or 0, c["execution_id"]))
        case = build_case(run, task, calls, rules)
        case["metrics"]["cost"] = call_metrics(calls, pricing)["cost"]
        case["case_sha256"] = fingerprint({k: v for k, v in case.items() if k != "case_sha256"})
        write_json(output / "cases" / (case["case_id"] + ".json"), case)
        cases.append({k: case[k] for k in ("case_id", "case_sha256", "task_id", "task_key", "role", "purpose", "slot", "selection_stage", "evaluated_task_state")})
    evidence = run.evidence()
    tasks = [{k: t.get(k) for k in ("task_id", "key", "kind", "role", "state", "outputs", "error")}
             | {"purpose": t["inputs"].get("purpose"), "parent_task": t["inputs"].get("parent_task"),
                "origin_task": t["inputs"].get("origin_task"), "previous_request_task": t["inputs"].get("previous_request_task"),
                "call_id": t["inputs"].get("call_id"),
                "result": t.get("result") if t["kind"] in ("experiment_request", "experiment") else None,
                "task_card": t.get("task_card"), "depends_on": t.get("depends_on", []),
                "source": run.card_source(t["task_card"], "task_provenance") if run.card(t["task_card"]) else None} for t in run.tasks.values()]
    run_result = {"schema_version": VERSION, "run_id": run.run_id, "source_run": str(run_path),
        "journal": {"sequence": run.sequence, "head": run.head}, "calls": run.calls, "jobs": run.jobs,
        "tasks": tasks, "evidence": evidence, "excluded_tasks": excluded, "orphan_executions": run.orphans,
        "data_coverage": run.issues, "metrics": call_metrics(run.calls, pricing), "timing": run_timing(run.events),
        "tools": tool_metrics(list(run.tasks.values()), run.calls),
        "experiment_states": dict(Counter(t["state"] for t in run.tasks.values() if t["kind"] == "experiment")),
        "cache_reused_requests": sum((t.get("result") or {}).get("cache_reused") is True for t in run.tasks.values() if t["kind"] == "experiment_request"),
        "unique_evidence_executions": len({e["execution_id"] for e in evidence if e["execution_id"]}),
        "posthoc_policy": "Observed outcomes only. No independent-repeat or causal quality claims inferred from execution counts."}
    write_json(output / "run.json", run_result)
    manifest = {"schema_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id, "source_run": str(run_path), "journal": run_result["journal"], "cases": cases,
        "rules_fingerprint": rules.version, "rules_provenance": rules.provenance,
        "implementation": {str(p.relative_to(QAVIT)): digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")},
        "task_catalog_sha256": digest((QAVIT / "agents/tasks.py").read_bytes()),
        "research_identity_source": run.file_rows["research_run.json"], "pricing": pricing,
        "source_files": list(run.store.files.values()), "run_sha256": fingerprint(run_result),
        "judge_config": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"},
        "isolation": "No production imports, queues, budgets, locks, Journal or GPU operations during extraction."}
    # Detect concurrent rewriting of a captured source; appended commits are outside this frozen horizon.
    for row in manifest["source_files"]:
        if digest((run_path / row["path"]).read_bytes()) != row["sha256"]:
            raise EvalError("history changed during extraction: " + row["path"])
    manifest["manifest_sha256"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest
