"""python -B QAViT/tools/evaluation {rules,prepare,judge,report,verify}"""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "evaluation"

from .common import (EvalError, canonical, digest, evaluation_lock, output_directory,
                     read_json, write_json)
from .extract import prepare
from .judge import grade_case, load_evaluation
from .report import build_report
from .rules import Rules


def main(argv=None):
    parser = argparse.ArgumentParser(description="QAViT independent read-only historical Eval-L0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rules", help="verify the authoritative package and current role/purpose coverage")
    p = commands.add_parser("prepare", help="freeze a historical run and compute metrics; no model calls")
    p.add_argument("run")
    p.add_argument("--output", required=True)
    p.add_argument("--pricing", help="optional attributed rate estimate JSON, never subscription charges")
    p = commands.add_parser("judge", help="explicit existing cases only; independent model, no research tools")
    p.add_argument("evaluation")
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--case", action="append", help="case_id or exact task_id; may repeat")
    selector.add_argument("--all", action="store_true", help="grade every prepared case, consuming independent Judge quota")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--max-input-characters", type=int, default=600_000)
    p.add_argument("--cli-path")
    for command in ("report", "verify"):
        p = commands.add_parser(command)
        p.add_argument("evaluation")
    args = parser.parse_args(argv)
    rules = Rules()
    if args.command == "rules":
        print(canonical({"status": "verified", "rules_fingerprint": rules.version,
                         "files": len(rules.provenance["files"]), "purposes": list(rules.selected)}))
        return 0
    if args.command == "prepare":
        source, destination = Path(args.run).resolve(), Path(args.output).resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise EvalError("evaluation output must not overlap historical run")
    root = output_directory(args.output if args.command == "prepare" else args.evaluation, create=args.command == "prepare")
    with evaluation_lock(root):
        if args.command == "prepare":
            try:
                manifest = prepare(args.run, root, rules, read_json(args.pricing) if args.pricing else None)
            except Exception as exc:
                write_json(root / "prepare_error.json", {"status": "failed", "code": type(exc).__name__, "message": str(exc)})
                raise
            report, _ = build_report(root, rules)
            print(canonical({"status": "prepared", "cases": len(manifest["cases"]), "report": str(report)}))
        elif args.command == "judge":
            manifest, cases = load_evaluation(root, rules)
            selected = cases if args.all else [c for c in cases if c["case_id"] in args.case or c["task_id"] in args.case]
            if not args.all and any(not any(v in (c["case_id"], c["task_id"]) for c in selected) for v in args.case):
                raise EvalError("requested case/task is not in the frozen evaluation")
            statuses = []
            for case in selected:
                status = grade_case(root, case, rules, max_attempts=args.max_attempts, timeout=args.timeout,
                                    max_input_characters=args.max_input_characters, cli_path=args.cli_path)
                statuses.append(status)
                print(canonical(status), flush=True)
                if status.get("stop_session"):
                    break
            report, _ = build_report(root, rules)
            print(canonical({"report": str(report)}))
            return 0 if all(s["status"] in ("validated", "already_validated", "recovered") for s in statuses) else 2
        elif args.command == "verify":
            manifest, cases = load_evaluation(root, rules)
            for row in manifest["source_files"]:
                if digest((root / row["object"]).read_bytes()) != row["sha256"]:
                    raise EvalError("frozen raw source changed")
            from .judge import accepted_result
            for case in cases:
                accepted_result(root, case, rules)
            print(canonical({"status": "verified", "source_files": len(manifest["source_files"]), "cases": len(cases)}))
        else:
            report, _ = build_report(root, rules)
            print(canonical({"report": str(report)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvalError, OSError, ValueError, KeyError) as exc:
        print(canonical({"status": "evaluation_failed", "error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        sys.exit(2)
