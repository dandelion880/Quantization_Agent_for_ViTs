"""Strict serialization, immutable local evidence and output confinement."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

QAVIT = Path(__file__).resolve().parents[2]
PROJECT = QAVIT.parent
EVALUATIONS = PROJECT / "EXP/evaluations"
RESOURCES = QAVIT / "configs/evaluation/v0_1"


class EvalError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return digest(canonical(value).encode())


def loads(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise EvalError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise EvalError("nonfinite JSON: " + value)

    def number(value):
        n = float(value)
        return n if math.isfinite(n) else invalid(value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    write_text(path, canonical(value) + "\n")


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def confined(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise EvalError("path outside allowed root")
    return path


def output_directory(path, *, create=False):
    path = confined(path, EVALUATIONS)
    if path == EVALUATIONS:
        raise EvalError("use a distinct EXP/evaluations/<eval_id> directory")
    if create:
        path.mkdir(parents=True, exist_ok=False)
        path.chmod(0o700)
    if not path.is_dir():
        raise EvalError("evaluation directory does not exist")
    return path


@contextmanager
def evaluation_lock(root):
    with (Path(root) / ".evaluation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvalError("another offline evaluator owns this directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def pointer(value, locator):
    """RFC 6901 JSON pointers, with a literal # prefix (no URI decoding)."""
    if locator == "#":
        return value
    if not isinstance(locator, str) or not locator.startswith("#/"):
        raise EvalError("locator must be # or a #/ JSON pointer")
    try:
        for token in locator[2:].split("/"):
            if re.search(r"~(?![01])", token):
                raise EvalError("invalid JSON pointer escape")
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", token):
                    raise EvalError("invalid array locator")
                value = value[int(token)]
            elif isinstance(value, dict):
                value = value[token]
            else:
                raise EvalError("locator descends into scalar")
        return value
    except (KeyError, IndexError) as exc:
        raise EvalError("locator not present in supplied evidence") from exc


def redact(value, models=(), *, blind=False):
    """Keep scientific model identities (DeiT etc.); remove account/LLM identity."""
    if isinstance(value, dict):
        return {k: ("[BLINDED]" if blind and k in ("reasoning_effort", "agent_models") else "[REDACTED]" if re.fullmatch(
            r"(?i)(api[_-]?key|access_token|refresh_token|authorization|password|email)", k)
            else redact(v, models, blind=blind)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, models, blind=blind) for v in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r"(?i)(bearer\s+|(?:api[_-]?key|access_token|refresh_token|authorization)[\"']?\s*[:=]\s*[\"']?)[^\"'\s,;}]+", r"\1[REDACTED]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[KEY]", value)
    value = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[TOKEN]", value)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
    value = value.replace(str(PROJECT), "[PROJECT]")
    value = re.sub(r"/(?:root|home/[^/\s]+)/", "[HOME]/", value)
    if blind:
        for name in sorted(set(models), key=len, reverse=True):
            if name:
                value = value.replace(name, "[SUBJECT_MODEL]")
        value = re.sub(r"real/codex_subscription/[^\s\"']+", "real/codex_subscription/[BLINDED]", value)
    return value


class SourceStore:
    """Content-addressed exact bytes; originals are never opened for writing."""
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}

    def capture(self, path, allowed_root):
        path = confined(path, allowed_root)
        raw = path.read_bytes()
        sha = digest(raw)
        key = path.relative_to(Path(allowed_root).resolve()).as_posix()
        row = {"path": key, "sha256": sha, "bytes": len(raw), "object": "sources/" + sha}
        old = self.files.get(key)
        if old and old != row:
            raise EvalError("source changed while freezing: " + key)
        self.files[key] = row
        target = self.root / row["object"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            target.chmod(0o600)
        elif digest(target.read_bytes()) != sha:
            raise EvalError("frozen source object corrupted")
        return row, raw


def source_value(root, source):
    path = confined(Path(root) / source["object"], Path(root) / "sources")
    raw = path.read_bytes()
    if digest(raw) != source["sha256"]:
        raise EvalError("frozen source hash mismatch")
    value = loads(raw.decode()) if source.get("format", "json") == "json" else raw.decode()
    return pointer(value, source["locator"])
