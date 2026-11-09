"""The uploaded package is the only scoring authority; code adds validation."""
import ast
from copy import deepcopy
from jsonschema import Draft202012Validator

from .common import QAVIT, RESOURCES, EvalError, digest, fingerprint, pointer, read_json


def task_catalog():
    tree = ast.parse((QAVIT / "agents/tasks.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets))


class Rules:
    def __init__(self, root=RESOURCES):
        self.root = root
        self.provenance = read_json(root / "provenance.json")
        for name, entry in self.provenance["files"].items():
            if digest((root / name).read_bytes()) != entry["sha256"]:
                raise EvalError("authoritative resource changed: " + name)
        self.schema = read_json(root / "contracts/judge_result.schema.json")
        Draft202012Validator.check_schema(self.schema)
        self.validator = Draft202012Validator(self.schema)
        self.selected = {}
        rubric_ids = set()
        for path in sorted((root / "rubrics").glob("*.json")):
            rubric = read_json(path)
            if rubric["rubric_id"] in rubric_ids:
                raise EvalError("duplicate rubric id")
            rubric_ids.add(rubric["rubric_id"])
            for purpose in rubric["purposes"]:
                if purpose in self.selected:
                    raise EvalError("duplicate purpose")
                selected = deepcopy(rubric)
                if isinstance(rubric["purposes"], dict):
                    # Preserve the parent instructions and the entire applicable sub-rubric.
                    selected["purposes"] = {purpose: deepcopy(rubric["purposes"][purpose])}
                self.selected[purpose] = selected
                criteria = self.criteria(purpose)
                if len({c["id"] for c in criteria}) != len(criteria) or not criteria:
                    raise EvalError("duplicate/empty criteria")
                for criterion in criteria:
                    if set(criterion["anchors"]) != {"0", "1", "2"}:
                        raise EvalError("incomplete anchors")
        catalog = task_catalog()
        expected = {p for p, t in catalog.items() if t["role"] != "supervisor"}
        if set(self.selected) != expected or len(rubric_ids) != 6:
            raise EvalError("rubrics do not cover the current six roles / LLM purposes")
        if any(catalog[p]["role"] != r["role"] for p, r in self.selected.items()):
            raise EvalError("role mapping mismatch")
        self.version = fingerprint(self.provenance)

    def criteria(self, purpose):
        r = self.selected[purpose]
        return r["purposes"][purpose]["criteria"] if isinstance(r["purposes"], dict) else r["criteria"]

    def validate(self, result, case, supplied):
        errors = sorted(self.validator.iter_errors(result), key=lambda e: str(list(e.path)))
        if errors:
            raise EvalError("judge schema: " + str(list(errors[0].path)) + " " + errors[0].message)
        for key in ("case_id", "role", "purpose", "rubric_id"):
            if result[key] != case[key]:
                raise EvalError("judge identity mismatch: " + key)
        ids = [c["id"] for c in self.criteria(case["purpose"])]
        if [c["criterion_id"] for c in result["criteria"]] != ids:
            raise EvalError("criterion set/order must exactly match the selected rubric")
        sources = {s["source_id"]: s for s in supplied}
        for row in [*result["criteria"], *result["critical_flags"]]:
            if not row.get("rationale", row.get("description", "")).strip():
                raise EvalError("empty rationale")
            if row.get("status") == "scored" and not row["evidence_refs"]:
                raise EvalError("scored criterion needs verifiable evidence; otherwise unknown")
            for ref in row["evidence_refs"]:
                src = sources.get(ref["source_id"])
                if not src or src["availability"] not in ("available", "partial"):
                    raise EvalError("unknown/unavailable source_id")
                base, loc = src["locator"], ref["locator"]
                if loc != base and not loc.startswith(base + "/"):
                    raise EvalError("locator outside the supplied source fragment")
                value = pointer(src["content"], "#" + loc[len(base):])
                excerpt = ref.get("excerpt")
                if excerpt and excerpt not in (value if isinstance(value, str) else __import__("json").dumps(value, ensure_ascii=False)):
                    raise EvalError("excerpt does not occur at the supplied locator")
        return quality(result)


def quality(result):
    rows = result["criteria"]
    scored = [r for r in rows if r["status"] == "scored"]
    na = sum(r["status"] == "not_applicable" for r in rows)
    applicable = len(rows) - na
    backed = sum(bool(r["evidence_refs"]) for r in scored)
    return {"Q": 100 * sum(r["score"] for r in scored) / (2 * len(scored)) if scored else None,
            "scored": len(scored), "applicable": applicable, "total": len(rows),
            "unknown": sum(r["status"] == "unknown" for r in rows), "not_applicable": na,
            "score_coverage": len(scored) / applicable if applicable else None,
            "effective_evidence_coverage": backed / applicable if applicable else None,
            "confidence": {c: sum(r["confidence"] == c for r in rows) for c in ("high", "medium", "low")}}
