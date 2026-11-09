"""Objective accounting. No resource metric contributes to scientific Q."""
from collections import Counter
import math

from .common import EvalError

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens",
                "reasoning_output_tokens", "cache_write_input_tokens")
TERMINAL = {"completed", "failed", "cancelled", "ineligible"}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def union_seconds(intervals):
    valid = sorted((a, b) for a, b in intervals if number(a) and number(b) and b >= a)
    total, end = 0.0, None
    for start, stop in valid:
        total += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = max(stop, end if end is not None else stop)
    return total if valid else None


def usage_totals(calls):
    """Subfields are reported separately, never added to input/output totals."""
    result = {}
    for field in TOKEN_FIELDS:
        values = [(c.get("usage") or {}).get(field) for c in calls]
        known = [v for v in values if type(v) is int and v >= 0]
        result[field] = {"known_sum": sum(known) if known else None,
                         "known_calls": len(known), "missing_calls": len(values) - len(known)}
    return result


def price_estimate(calls, pricing=None):
    result = {"amount": None, "known_subtotal": None, "currency": None,
              "basis": "unknown_subscription_cost", "missing_calls": len(calls),
              "note": "Subscription tokens are usage, not a verified monetary charge."}
    if pricing is None:
        return result
    required = {"source", "as_of", "currency", "models", "basis"}
    if not required <= pricing.keys() or pricing["basis"] != "token_rate_estimate_not_subscription_charge":
        raise EvalError("pricing needs source/as_of/currency/models and explicit estimate basis")
    if not all(isinstance(pricing[k], str) and pricing[k].strip() for k in ("source", "as_of", "currency")):
        raise EvalError("pricing attribution is incomplete")
    costs, bases = [], Counter()
    for call in calls:
        actual = (call.get("actual") or {}).get("model")
        model = actual or (call.get("requested") or {}).get("model")
        rate = pricing["models"].get(model)
        if not rate:
            continue
        if (type(rate.get("input_includes_cached")) is not bool or
                any(not number(rate.get(k)) for k in ("input_per_million", "output_per_million", "cached_input_per_million"))):
            raise EvalError("pricing rates/usage semantics must be explicit finite nonnegative numbers")
        usage = call.get("usage") or {}
        i, o, cached = (usage.get(k) for k in TOKEN_FIELDS[:3])
        if any(type(v) is not int or v < 0 for v in (i, o, cached)):
            continue
        uncached = i - cached if rate["input_includes_cached"] else i
        if uncached < 0:
            raise EvalError("cached tokens exceed inclusive input tokens")
        write_tokens = usage.get("cache_write_input_tokens", 0)
        if write_tokens and not number(rate.get("cache_write_per_million")):
            continue
        # The price table must explicitly say whether cache writes are additional.
        if write_tokens and type(rate.get("cache_write_additional")) is not bool:
            raise EvalError("cache write price semantics missing")
        cost = (uncached * rate["input_per_million"] + o * rate["output_per_million"] +
                cached * rate["cached_input_per_million"])
        if write_tokens and rate["cache_write_additional"]:
            cost += write_tokens * rate["cache_write_per_million"]
        costs.append(cost / 1_000_000)
        bases["actual_model" if actual else "requested_model_estimate"] += 1
    return {"amount": sum(costs) if costs and len(costs) == len(calls) else None,
            "known_subtotal": sum(costs) if costs else None, "missing_calls": len(calls) - len(costs),
            "currency": pricing["currency"], "basis": pricing["basis"], "model_identity_basis": dict(bases),
            "source": pricing["source"], "as_of": pricing["as_of"],
            "note": "Rate estimate only; does not represent the subscription invoice."}


def call_metrics(calls, pricing=None):
    started = [c for c in calls if c.get("model_call_started") is True]
    duration = [c["elapsed_seconds"] for c in calls if number(c.get("elapsed_seconds"))]
    return {"execution_attempts": len(calls), "model_calls_started": len(started),
            "model_calls_start_unknown": sum(c.get("model_call_started") is None for c in calls),
            "preflight_only_attempts": sum(c.get("model_call_started") is False for c in calls),
            "model_calls_completed": sum(c.get("response_observed", False) for c in calls),
            "failed_attempts": sum(c.get("status") in ("failed", "interrupted") for c in calls),
            "model_calls_failed": sum(c.get("status") in ("failed", "interrupted") for c in started),
            "incomplete_attempts": sum(c.get("status") in ("unknown", "unfinished") for c in calls),
            "tokens": usage_totals(calls),
            "duration": {"router_elapsed_sum_seconds": sum(duration) if duration else None,
                         "missing_elapsed_attempts": len(calls) - len(duration),
                         "observed_model_interval_union_seconds": union_seconds(
                             [(c.get("started_unix"), c.get("finished_unix")) for c in started]),
                         "pure_inference_seconds": None, "subscription_lock_wait_seconds": None,
                         "note": "Router elapsed includes preflight/lock waits; sum is not run wall time. CLI inner reconnects are not observable."},
            "cost": price_estimate(calls, pricing)}


def task_timing(versions):
    seconds, retries, repairs = Counter(), 0, 0
    previous = None
    for version in versions:
        task, stamp = version["task"], version["time"]
        if previous:
            old = previous["task"]
            if number(stamp) and number(previous["time"]) and stamp >= previous["time"]:
                seconds[old["state"]] += stamp - previous["time"]
            retries += max(0, task.get("retry_count", 0) - old.get("retry_count", 0))
            repairs += max(0, task.get("repair_count", 0) - old.get("repair_count", 0))
        previous = version
    final = versions[-1]["task"] if versions else {}
    first = versions[0]["task"].get("created_unix") if versions else None
    end = final.get("finished_unix")
    if final.get("state") in TERMINAL and not number(end) and versions:
        end = versions[-1]["time"]
    return {"recorded_retry_events": retries, "recorded_format_repair_events": repairs,
            "task_wall_seconds": end - first if number(end) and number(first) and end >= first and final.get("state") in TERMINAL else None,
            "observed_state_seconds": dict(seconds), "open_state": final.get("state") if final.get("state") not in TERMINAL else None,
            "note": "State intervals are bounded by committed snapshots; waiting may include tools, code, numeric jobs and scheduling."}


def tool_metrics(tasks, calls=()):
    groups = {}
    for t in tasks:
        if t["kind"] not in ("host_tool", "literature", "experiment_request"):
            continue
        inputs = t["inputs"]
        name = inputs.get("name") or ("literature_" + inputs.get("action", "unknown") if t["kind"] == "literature" else "request_experiment")
        row = groups.setdefault(name, {"tasks": 0, "states": Counter(), "cache_reused": 0,
                                       "observed_model_requests": 0, "requests_without_linked_task": 0})
        row["tasks"] += 1
        row["states"][t["state"]] += 1
        row["cache_reused"] += (t.get("result") or {}).get("cache_reused") is True
    linked = {(t["inputs"].get("parent_task"), t["inputs"].get("call_id")) for t in tasks}
    for call in calls:
        for request in call.get("tool_calls", []):
            name = request["name"]
            row = groups.setdefault(name, {"tasks": 0, "states": Counter(), "cache_reused": 0,
                                           "observed_model_requests": 0, "requests_without_linked_task": 0})
            row["observed_model_requests"] += 1
            if not request.get("id") or (call.get("task_id"), request["id"]) not in linked:
                row["requests_without_linked_task"] += 1
    return {k: {**v, "states": dict(v["states"])} for k, v in groups.items()}


def run_timing(events):
    segments, waits = {}, {}
    for e in events:
        kind = e.get("type")
        if kind in ("budget_segment_started", "budget_segment_stopped"):
            segments.setdefault(e["segment_id"], {}).update(e)
        if kind == "budget_numeric_wait_started":
            waits.setdefault(e["wait_id"], {}).update(start=e.get("started_unix"))
        if kind == "budget_numeric_wait_stopped":
            waits.setdefault(e["wait_id"], {}).update(stop=e.get("stopped_unix"))
    return {"segment_count": len(segments),
            "active_segment_wall_union_seconds": union_seconds([(v.get("started_unix"), v.get("stopped_unix")) for v in segments.values()]),
            "numeric_wait_union_seconds": union_seconds([(v.get("start"), v.get("stop")) for v in waits.values()]),
            "open_segments": sum("stopped_unix" not in v for v in segments.values()),
            "open_numeric_waits": sum("stop" not in v for v in waits.values()),
            "note": "Only recorded closed intervals are measured. Pauses and parallel call sums are not run active time."}
