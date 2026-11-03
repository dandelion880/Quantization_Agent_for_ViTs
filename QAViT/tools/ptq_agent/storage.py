"""原子 JSON 与运行锁；不负责经验选择、数值测量或 Agent 决策。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .schemas import JSON, PTQError


def atomic_json(path: Path, value: JSON | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    handle, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path: Path) -> JSON:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        raise PTQError("persistence_error", f"无法读取 {path}: {exc}") from exc


