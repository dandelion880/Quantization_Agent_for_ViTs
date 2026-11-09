"""Independent, bounded, restartable LLM grading; results stay in the eval run."""
from pathlib import Path
import os
import re
import signal
import subprocess
import sys
import time

from .common import (EvalError, canonical, confined, digest, fingerprint, read_json,
                     redact, source_value, write_json)

DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_EFFORT = "medium"


def load_evaluation(root, rules):
    manifest = read_json(Path(root) / "manifest.json")
    if manifest["manifest_sha256"] != fingerprint({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise EvalError("evaluation manifest changed")
    if rules.version != manifest["rules_fingerprint"]:
        raise EvalError("rule version changed; prepare a new evaluation")
    if fingerprint(read_json(Path(root) / "run.json")) != manifest["run_sha256"]:
        raise EvalError("frozen run metrics changed")
    cases = []
    for row in manifest["cases"]:
        path = confined(Path(root) / "cases" / (row["case_id"] + ".json"), Path(root) / "cases")
        case = read_json(path)
        if case["case_sha256"] != row["case_sha256"] or case["case_sha256"] != fingerprint({k: v for k, v in case.items() if k != "case_sha256"}):
            raise EvalError("frozen case changed")
        cases.append(case)
    return manifest, cases


def supplied_sources(root, case):
    names = [str(v["model"]) for c in case["calls"] for v in (c.get("requested"), c.get("actual")) if isinstance(v, dict) and v.get("model")]
    result = []
    for source in case["source_index"]:
        if source["source_type"] == "posthoc_observation":
            continue
        result.append({k: source[k] for k in ("source_id", "immutable_ref", "locator", "availability", "source_type")}
                      | {"content": redact(source_value(root, source), names, blind=True),
                         "boundary": {k: source[k] for k in ("execution_id", "visible_at_sequence", "delivery", "attribution") if k in source}})
    return result


def render(root, case, rules):
    supplied = supplied_sources(root, case)
    identity = {k: case.get(k) for k in ("case_id", "run_id", "task_id", "task_key", "role", "purpose", "slot",
        "selection_stage", "exchange", "side", "round", "evaluated_task_state", "input_version_refs", "evidence_cutoff", "rubric_id")}
    # Data appears only in the untrusted case message. Original static system instructions stay unchanged.
    replacements = [
        ("case_identity:", identity),
        ("ResearchBrief /", {"instruction": "Original brief, constraints and role/task instructions are in the subject_input sources; do not infer unread originals."}),
        ("subject_visible_context", {"source_ids": [s["source_id"] for s in supplied if s["source_type"] == "subject_input"]}),
        ("task_trajectory：", {"executions": case["trajectory"], "output_source_ids": [s["source_id"] for s in supplied if s["source_type"] != "subject_input"]}),
        ("source_index:", supplied),
        ("selected_role_purpose_rubric:", rules.selected[case["purpose"]]),
        ("coverage_notes:", redact(case["data_coverage"]) | {"posthoc_observations": "excluded; only available in separate program report"}),
    ]
    template = (rules.root / "prompts/judge_case.md").read_text()
    # Locate placeholders before inserting untrusted text. Data that resembles a
    # template token must never be interpreted as a second template instruction.
    tokens = {}
    for prefix, value in replacements:
        matches = re.findall("<" + re.escape(prefix) + r"[^>]*>", template)
        if len(matches) != 1:
            raise EvalError("authoritative template placeholder mismatch: " + prefix)
        tokens[matches[0]] = canonical(value)
    template = re.sub("|".join(re.escape(token) for token in tokens), lambda m: tokens[m.group()], template)
    template += "\n\n输出 Schema 原文：\n" + canonical(rules.schema)
    template += ("\n\n来源 locator 使用 source_index 中的 #/ JSON pointer，可以继续向下定位已展示的 JSON 字段；"
                 "文本源码使用该 source 的根 locator，excerpt 必须是该位置的真实短摘录。每个 scored 项必须引用至少一个已展示来源；"
                 "不要引用未展开的卡片正文或事后材料。全量原始轨迹已冻结；缺失或未读材料使用 unknown。")
    return {"system_prompt": (rules.root / "prompts/judge_system.md").read_text(), "case_prompt": template,
            "supplied_sources": supplied}


def worker_environment(directory):
    names = {"PATH", "HOME", "CODEX_HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "HTTP_PROXY", "HTTPS_PROXY",
             "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {k: v for k, v in os.environ.items() if k in names}
    temp = Path(directory) / "tmp"
    temp.mkdir(exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TMPDIR=str(temp), NO_COLOR="1")
    return env


def invoke_worker(directory, timeout, cli_path=None):
    script = Path(__file__).with_name("transport_worker.py")
    command = [sys.executable, "-B", str(script), str(directory), "--timeout", str(timeout)]
    if cli_path:
        command += ["--cli-path", cli_path]
    with (directory / "worker.log").open("xb") as stream:
        proc = subprocess.Popen(command, cwd=directory, env=worker_environment(directory),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout + 15)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)


def accepted_result(root, case, rules):
    path = Path(root) / "results" / (case["case_id"] + ".json")
    if not path.exists():
        return None
    result = read_json(path)
    if result["case_sha256"] != case["case_sha256"] or result["rules_fingerprint"] != rules.version:
        raise EvalError("accepted score binding changed")
    attempt = confined(Path(root) / result["attempt"], Path(root) / "judge")
    request = read_json(attempt / "request.json")
    if request["case_sha256"] != case["case_sha256"] or request["rules_fingerprint"] != rules.version:
        raise EvalError("Judge request binding changed")
    if digest((attempt / "request.json").read_bytes()) != result["request_sha256"]:
        raise EvalError("Judge request bytes changed")
    raw = read_json(attempt / "judge_output.json")
    audit = read_json(attempt / "audit.json")
    if (audit.get("status") != "completed" or audit.get("tool_host_disabled_verified") is not True or
            audit.get("request_sha256") != result["request_sha256"]):
        raise EvalError("accepted Judge transport receipt changed")
    if raw != result["result"]:
        raise EvalError("raw/accepted Judge output mismatch")
    bundle = render(root, case, rules)
    if (request["bundle_sha256"] != fingerprint(bundle) or request.get("system_prompt") != bundle["system_prompt"] or
            request.get("case_prompt") != bundle["case_prompt"] or request.get("model") != DEFAULT_MODEL or
            request.get("reasoning_effort") != DEFAULT_EFFORT):
        raise EvalError("Judge evidence bundle changed")
    q = rules.validate(raw, case, bundle["supplied_sources"])
    if q != result["quality"]:
        raise EvalError("derived score mismatch")
    return result


def grade_case(root, case, rules, *, max_attempts=2, timeout=300, max_input_characters=600_000,
               cli_path=None, invoke=invoke_worker):
    root = Path(root)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3 or timeout <= 0 or max_input_characters <= 0:
        raise EvalError("invalid independent Judge limits")
    accepted = accepted_result(root, case, rules)
    if accepted:
        return {"case_id": case["case_id"], "status": "already_validated"}
    if case["data_coverage"]["blocking_issues"]:
        return {"case_id": case["case_id"], "status": "source_integrity_error", "issues": case["data_coverage"]["blocking_issues"]}
    bundle = render(root, case, rules)
    request = {"model": DEFAULT_MODEL, "reasoning_effort": DEFAULT_EFFORT,
               "case_id": case["case_id"], "case_sha256": case["case_sha256"], "rules_fingerprint": rules.version,
               "bundle_sha256": fingerprint(bundle), "system_prompt": bundle["system_prompt"], "case_prompt": bundle["case_prompt"],
               "implementation": {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}}
    base = root / "judge" / case["case_id"]
    base.mkdir(parents=True, exist_ok=True)
    if len(request["system_prompt"]) + len(request["case_prompt"]) > max_input_characters:
        write_json(base / "input_limit.json", {"case_sha256": case["case_sha256"], "characters": len(request["case_prompt"]),
                    "limit": max_input_characters, "status": "input_limit", "note": "No truncation, model call or fabricated score."})
        return {"case_id": case["case_id"], "status": "input_limit"}
    attempts = sorted(base.glob("attempt-*"))
    # Recover a completed attempt that crashed before validation; never pay for it again.
    for directory in attempts:
        if not (directory / "status.json").exists():
            status = validate_attempt(root, directory, case, rules, bundle)
            if status == "validated":
                return {"case_id": case["case_id"], "status": "recovered"}
    for number in range(len(attempts) + 1, max_attempts + 1):
        directory = base / f"attempt-{number:03d}"
        directory.mkdir(exist_ok=False)
        write_json(directory / "request.json", request)
        try:
            invoke(directory, timeout, cli_path)
        except (OSError, subprocess.TimeoutExpired) as exc:
            write_json(directory / "host_error.json", {"code": type(exc).__name__, "message": "Judge worker launch/timeout failure"})
        status = validate_attempt(root, directory, case, rules, bundle)
        if status == "validated":
            return {"case_id": case["case_id"], "status": status}
        audit = read_json(directory / "audit.json") if (directory / "audit.json").exists() else {}
        code = (audit.get("error") or {}).get("code", "")
        if any(s in code for s in ("authentication", "quota", "credit", "model_mismatch", "tool_isolation", "cli_version", "unsupported_model")):
            return {"case_id": case["case_id"], "status": status, "stop_session": True, "error_code": code}
    return {"case_id": case["case_id"], "status": "attempt_limit_reached"}


def validate_attempt(root, directory, case, rules, bundle):
    try:
        audit = read_json(directory / "audit.json")
        request = read_json(directory / "request.json")
        if (audit.get("status") != "completed" or audit.get("tool_host_disabled_verified") is not True or
                audit.get("request_sha256") != digest((directory / "request.json").read_bytes()) or
                request.get("bundle_sha256") != fingerprint(bundle) or request.get("case_sha256") != case["case_sha256"] or
                request.get("rules_fingerprint") != rules.version or request.get("model") != DEFAULT_MODEL or request.get("reasoning_effort") != DEFAULT_EFFORT or
                request.get("system_prompt") != bundle["system_prompt"] or request.get("case_prompt") != bundle["case_prompt"]):
            raise EvalError("incomplete or mismatched independent Judge receipt")
        raw = read_json(directory / "judge_output.json")
        q = rules.validate(raw, case, bundle["supplied_sources"])
        result = {"case_sha256": case["case_sha256"], "rules_fingerprint": rules.version,
                  "request_sha256": audit["request_sha256"], "attempt": str(directory.relative_to(root)),
                  "result": raw, "quality": q, "validated_unix": time.time()}
        write_json(Path(root) / "results" / (case["case_id"] + ".json"), result)
        status = {"status": "validated", "case_id": case["case_id"]}
    except (ValueError, OSError, KeyError) as exc:
        status = {"status": "invalid_or_incomplete", "case_id": case["case_id"], "error": redact(str(exc))[:3000]}
    write_json(directory / "status.json", status)
    return status["status"]
