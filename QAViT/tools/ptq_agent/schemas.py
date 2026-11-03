"""JSON 契约、可定位异常和持久化数据结构。仅支持本文声明的 schema 子集。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

JSON = dict[str, Any]


class PTQError(Exception):
    def __init__(self, code: str, message: str, *, path: str = "$",
                 status: str = "failure", retryable: bool = False):
        super().__init__(message)
        self.code, self.path, self.status = code, path, status
        self.retryable = retryable

    def to_dict(self) -> JSON:
        return {"code": self.code, "message": str(self), "path": self.path,
                "retryable": self.retryable}


def require(condition: bool, message: str, path: str = "$", code: str = "invalid_config") -> None:
    if not condition:
        raise PTQError(code, message, path=path)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def obj(properties: JSON, required: list[str] | None = None, *, additional: bool | JSON = False) -> JSON:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": additional}


def arr(items: JSON, **kwargs: Any) -> JSON:
    return {"type": "array", "items": items, **kwargs}


def validate(value: Any, schema: JSON, path: str = "$") -> None:
    """严格检查工具/配置边界；未知字段拒绝，bool 不作为整数，NaN 不合法。"""
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                return
            except PTQError as exc:
                errors.append(exc)
        # 例如 optional spec 的 null 分支与 object 分支均失败时，保留 object
        # 内 scale 等字段的错误；不能把可定位的错误抹成笼统的“类型不符”。
        if errors:
            detail = max(errors, key=lambda e: e.path.count(".") + e.path.count("["))
            if detail.path != path:
                raise detail
        raise PTQError("schema_error", "不符合任何允许的类型", path=path)
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    if kind:
        valid = isinstance(value, types[kind])
        if kind in ("integer", "number") and isinstance(value, bool):
            valid = False
        require(valid, f"需要 {kind}，实际为 {type(value).__name__}", path, "schema_error")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), "数值必须有限", path, "schema_error")
        for name, check in (("minimum", lambda x: value >= x), ("maximum", lambda x: value <= x)):
            if name in schema:
                require(check(schema[name]), f"不满足 {name}={schema[name]}", path, "schema_error")
    if "enum" in schema:
        require(value in schema["enum"], f"允许值为 {schema['enum']}", path, "schema_error")
    if isinstance(value, str) and "minLength" in schema:
        require(len(value) >= schema["minLength"], "字符串过短", path, "schema_error")
    if kind == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            require(key in value, f"缺少字段 {key}", f"{path}.{key}", "schema_error")
        for key, item in value.items():
            if key in props:
                validate(item, props[key], f"{path}.{key}")
            else:
                extra = schema.get("additionalProperties", False)
                require(extra is not False, f"未知字段 {key}", f"{path}.{key}", "schema_error")
                if isinstance(extra, dict):
                    validate(item, extra, f"{path}.{key}")
    if kind == "array":
        require(len(value) >= schema.get("minItems", 0), "列表过短", path, "schema_error")
        require(len(value) <= schema.get("maxItems", 100000), "列表过长", path, "schema_error")
        if schema.get("uniqueItems"):
            require(len({canonical(x) for x in value}) == len(value), "列表包含重复项", path, "schema_error")
        for index, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{index}]")


STR = {"type": "string", "minLength": 1}
INT = {"type": "integer", "minimum": 0}
POS = {"type": "integer", "minimum": 1}
NUMBER = {"type": "number", "minimum": 0}
NULL = {"type": "null"}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: JSON


@dataclass
class ToolResult:
    call_id: str
    name: str
    status: str
    data: JSON = field(default_factory=dict)
    error: JSON | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> JSON:
        return asdict(self)


@dataclass
class Message:
    role: str
    content: JSON
    tool_call_id: str | None = None


@dataclass
class LLMRequest:
    role: str
    system_prompt: str
    messages: list[Message]
    tools: list[JSON]
    context: JSON
    response_schema: JSON


@dataclass
class LLMResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    decision: JSON | None = None
    error: JSON | None = None
    usage: JSON | None = None
    source: str = "unspecified"

