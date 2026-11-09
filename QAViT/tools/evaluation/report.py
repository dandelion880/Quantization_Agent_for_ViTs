"""Separate scientific quality, resource accounting and posthoc observations."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
from pathlib import Path
import statistics

from .common import read_json, redact, write_json, write_text
from .judge import accepted_result, load_evaluation
from .metrics import call_metrics


def distribution(values):
    present = [x for x in values if x is not None]
    return {"n": len(present), "missing": len(values) - len(present),
            "mean": statistics.mean(present) if present else None,
            "median": statistics.median(present) if present else None,
            "min": min(present) if present else None, "max": max(present) if present else None}


def error_stratum(error):
    error = error or {}
    code = error.get("code", "")
    if error.get("origin") in ("host", "implementation"):
        return error["origin"]
    if any(x in code for x in ("structured_output", "inner_json", "format_error")):
        return "model_output_format"
    if any(x in code for x in ("timeout", "quota", "authentication", "transport", "worker_lost", "Interrupted", "launch")):
        return "harness_environment"
    return "unknown_origin" if error else "none"


def build_report(root, rules):
    root = Path(root)
    manifest, cases = load_evaluation(root, rules)
    run = read_json(root / "run.json")
    rows, grouped = [], defaultdict(list)
    for case in cases:
        accepted = accepted_result(root, case, rules)
        base = root / "judge" / case["case_id"]
        attempts = [{"path": str(p.parent.relative_to(root)), **read_json(p)} for p in sorted(base.glob("attempt-*/status.json"))]
        row = {k: case[k] for k in ("case_id", "task_id", "task_key", "role", "purpose", "slot", "selection_stage", "evaluated_task_state", "data_coverage", "lineage")}
        row.update(quality=accepted["quality"] if accepted else None,
                   judge_result=accepted["result"] if accepted else None, metrics=case["metrics"],
                   judge_status="validated" if accepted else "input_limit" if (base / "input_limit.json").exists() else "failed_or_pending" if attempts else "not_judged",
                   attempts=attempts, sources=case["source_index"],
                   error_strata=dict(Counter(error_stratum(c.get("error")) for c in case["calls"] if c.get("error"))))
        rows.append(row)
        key = (case["role"], case["purpose"], case["slot"] if case["purpose"] == "generate" else None,
               case["selection_stage"] if case["purpose"] == "rank" else None)
        grouped[key].append(row)
    groups = []
    for key, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        qualities = [m["quality"] for m in members if m["quality"] is not None]
        applicable = sum(q["applicable"] for q in qualities)
        scored = sum(q["scored"] for q in qualities)
        groups.append({"role": key[0], "purpose": key[1], "slot": key[2], "selection_stage": key[3],
            "task_count": len(members), "judged_task_count": len(qualities),
            "unjudged_task_count": len(members) - len(qualities),
            "Q_distribution": distribution([m["quality"]["Q"] if m["quality"] else None for m in members]),
            "criterion_counts": {k: sum(q[k] for q in qualities) for k in ("scored", "applicable", "total", "unknown", "not_applicable")},
            "effective_evidence_coverage": scored / applicable if applicable else None,
            "task_states": dict(Counter(m["evaluated_task_state"] for m in members)),
            "error_strata": dict(sum((Counter(m["error_strata"]) for m in members), Counter())),
            "critical_flags": dict(Counter(f["kind"] for m in members if m["judge_result"] for f in m["judge_result"]["critical_flags"])),
            "input_token_distribution_known_only": distribution([m["metrics"]["tokens"]["input_tokens"]["known_sum"] for m in members]),
            "router_elapsed_distribution_known_only": distribution([m["metrics"]["duration"]["router_elapsed_sum_seconds"] for m in members]),
            "cost_estimate_distribution": distribution([m["metrics"]["cost"]["amount"] for m in members]),
            "case_ids": [m["case_id"] for m in members], "difficulty": "not independently labeled"})
    judge_calls = []
    for path in sorted((root / "judge").glob("*/attempt-*/request.json")):
        audit_path = path.parent / "audit.json"
        audit = read_json(audit_path) if audit_path.exists() else {"status": "unknown", "model_call_started": None}
        judge_calls.append({"execution_id": str(path.parent.relative_to(root)), **audit})
    report = {"schema_version": "qavit.eval_l0.report/1", "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"], "manifest_sha256": manifest["manifest_sha256"],
        "rules_fingerprint": rules.version, "judge_config": manifest["judge_config"],
        "groups": groups, "cases": rows,
        "research_resources": {"metrics": run["metrics"], "timing": run["timing"], "tools": run["tools"],
                               "experiment_states": run["experiment_states"], "cache_reused_requests": run["cache_reused_requests"]},
        "judge_overhead": call_metrics(judge_calls, manifest.get("pricing")), "judge_attempts": judge_calls,
        "posthoc_observations": run["evidence"], "excluded_tasks": run["excluded_tasks"],
        "orphan_executions": run["orphan_executions"], "run_data_coverage": run["data_coverage"],
        "limits": ["Eval-L0 is observational, not a causal estimate of individual LLM contribution.",
                   "No expert gold labels, representative research task generation, FPR/FNR or pass threshold.",
                   "Unknown and not_applicable are separate; unjudged tasks never receive synthetic scores.",
                   "Independent means separate invocation/context; the Judge model family may overlap subjects.",
                   "Historical input context is exact delivered text; indexes and partial pages are not full-source evidence.",
                   "Counts of executions do not establish independent scientific replication."]}
    directory = root / "reports"
    directory.mkdir(exist_ok=True)
    n = len(list(directory.glob("report-*.json"))) + 1
    path = directory / f"report-{n:04d}"
    if path.with_suffix(".json").exists():
        raise ValueError("report filename collision")
    write_json(path.with_suffix(".json"), report)
    write_text(path.with_suffix(".md"), markdown(report))
    write_json(root / "latest_report.json", {"json": str(path.with_suffix(".json").relative_to(root)), "markdown": str(path.with_suffix(".md").relative_to(root))})
    return path.with_suffix(".md"), report


def markdown(report):
    def show(value):
        return "Unknown" if value is None else str(round(value, 3)) if isinstance(value, float) else str(value)
    def text(value):
        return html.escape(str(redact(value))).replace("|", "\\|").replace("\n", " ")
    lines = ["# QAViT Eval-L0 离线报告", "", f"运行：`{report['run_id']}`；规则：`{report['rules_fingerprint']}`。",
             "独立 Judge：gpt-6.1-sol / medium。质量与成本分列；未评分不代表低质量，未知用量不代表零消耗。",
             "无人工真值、无通过阈值；本报告不作单个 LLM 的因果贡献、误拒率或漏放率结论。", "",
             "## 科研质量（按角色和具体任务）", "",
             "| 角色 / purpose / 分层 | 任务 / 已评 | Q 均值 / 中位数 | 已评项 / 适用项 / 总项 | Unknown / N/A | 有效证据覆盖 |",
             "|---|---:|---:|---:|---:|---:|"]
    for g in report["groups"]:
        c, q = g["criterion_counts"], g["Q_distribution"]
        label = "/".join(str(v) for v in (g["role"], g["purpose"], g["slot"], g["selection_stage"]) if v)
        lines.append(f"| {label} | {g['task_count']} / {g['judged_task_count']} | {show(q['mean'])} / {show(q['median'])} | {c['scored']} / {c['applicable']} / {c['total']} | {c['unknown']} / {c['not_applicable']} | {show(g['effective_evidence_coverage'])} |")
    lines += ["", "分母仅覆盖已有有效 Judge 输出；未评任务单独列出。各角色及 Judger 子任务不混合成总分。", "",
              "## 资源统计（程序计算）", ""]
    for name, m in (("原科研运行", report["research_resources"]["metrics"]), ("独立评价开销", report["judge_overhead"])):
        lines += [f"### {name}", "", f"执行尝试 {m['execution_attempts']}；确认模型启动 {m['model_calls_started']}；完整响应 {m['model_calls_completed']}；失败尝试 {m['failed_attempts']}；启动状态未知 {m['model_calls_start_unknown']}。", "",
                  "| Token 字段 | 已知合计 | 缺用量的调用数 |", "|---|---:|---:|"]
        for k, v in m["tokens"].items():
            lines.append(f"| {k} | {show(v['known_sum'])} | {v['missing_calls']} |")
        lines += ["", f"Router 累计历时：{show(m['duration']['router_elapsed_sum_seconds'])} 秒（含预检/锁等待，非纯推理或端到端墙钟）。",
                  f"货币成本：{show(m['cost']['amount'])}；口径 `{m['cost']['basis']}`。订阅 Token 不自动折算账单。", ""]
    lines += ["科研运行闭合时间区间：`" + canonical_for_report(report["research_resources"]["timing"]) + "`。", "",
              "实验执行状态：`" + canonical_for_report(report["research_resources"]["experiment_states"]) + "`；缓存复用请求数：" + str(report["research_resources"]["cache_reused_requests"]) + "。", "",
              "工具请求及持久任务：`" + canonical_for_report(report["research_resources"]["tools"]) + "`。无关联任务表示未核实执行，不直接归责为模型错误。", "",
              "## 每项评分及原始证据", ""]
    for row in report["cases"]:
        lines += [f"### {text(row['task_key'])} · {row['case_id']}", "",
                  f"状态 `{row['evaluated_task_state']}`；评价 `{row['judge_status']}`；[冻结任务与全部来源](../cases/{row['case_id']}.json)。"]
        if row["judge_result"]:
            lines += [f"Q={show(row['quality']['Q'])}；总体置信度 {row['judge_result']['overall_confidence']}。", "",
                      "| 项目 | 状态 / 分数 | 置信度 | 理由 / 证据 |", "|---|---|---|---|"]
            sources = {s["source_id"]: s for s in row["sources"]}
            for c in row["judge_result"]["criteria"]:
                refs = []
                for ref in c["evidence_refs"]:
                    s = sources[ref["source_id"]]
                    refs.append(f"[{s['source_id']}](../{s['object']}) `{text(ref['locator'])}`")
                lines.append(f"| {c['criterion_id']} | {c['status']} / {show(c['score'])} | {c['confidence']} | {text(c['rationale'])} {'; '.join(refs)} |")
            lines += ["", "主要优点：" + text("；".join(row["judge_result"]["strengths"])),
                      "主要问题：" + text("；".join(row["judge_result"]["weaknesses"])),
                      "高严重性标记：" + text(canonical_for_report(row["judge_result"]["critical_flags"]))]
        else:
            lines += ["科研质量未得到有效评分；保留 Unknown，不自动归零。"]
        if row["data_coverage"]["missing_or_truncated_sources"]:
            lines += ["来源缺口：" + text(canonical_for_report(row["data_coverage"]["missing_or_truncated_sources"]))]
        lines += [""]
    lines += ["## 既有实验观察（posthoc，未传给质量 Judge）", "",
              "| Evidence | 测量来源 | 级别 | 执行身份 | 原件 |", "|---|---|---|---|---|"]
    for e in report["posthoc_observations"]:
        lines.append(f"| {e['id']} | {text(e['measurement_source'])} | {text(e['level'])} | {text(e['execution_id'])} | [完整条件与观测](../{e['source']['object']}) |")
    lines += ["", "数据范围、量化范围、分母和实际指标保留在原件；缓存不计独立重复，失败/局部诊断不冒充完整任务测量。",
              "", "## 边界与追溯", "", "[来源清单与实现版本](../manifest.json)；[原运行计量与关系](../run.json)。"]
    lines += ["- " + text(v) for v in report["limits"]]
    lines += [f"- 排除 {len(report['excluded_tasks'])} 个无可评分 LLM 产出的任务（含程序 rank、仅预检等）；孤立执行 {len(report['orphan_executions'])} 个。"]
    return "\n".join(lines) + "\n"


def canonical_for_report(value):
    from .common import canonical
    return canonical(redact(value))
