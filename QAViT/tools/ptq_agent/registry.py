"""数值工具注册、权限和精确测量缓存；不包含旧搜索决策工具。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .schemas import JSON, STR, NULL, ToolCall, ToolResult, PTQError, arr, obj, require, validate, fingerprint
from .storage import atomic_json
from .profiling.evidence import read_measurement, seal
from .profiling import DEFINITION_VERSION

FREE_OBJECT = {"type": "object", "additionalProperties": True}

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: JSON
    returns: JSON
    roles: tuple[str, ...]
    handler: Callable[..., JSON]

    def schema(self) -> JSON:
        return {"name": self.name, "description": self.description, "parameters": self.parameters,
                "returns": {"anyOf": [obj({"call_id": STR, "name": STR, "status": {"enum": ["success"]},
                    "data": self.returns, "error": NULL, "artifacts": arr(STR)}),
                    obj({"call_id": STR, "name": STR, "status": {"enum": ["failure", "unsupported", "unconfigured"]},
                    "data": obj({}), "error": obj({"code": STR, "message": STR, "path": STR, "retryable": {"type": "boolean"}}),
                    "artifacts": arr(STR)})]},
                "error_codes": ["schema_error", "forbidden_tool", "unknown_tool", "invalid_config", "unsupported_format",
                                "unsupported_combination", "budget_exhausted", "numerical_error", "tool_exception", "not_configured"],
                "error_contract": "非 success 时 data={}，error={code,message,path,retryable}; artifacts 可为空"}


class ToolRegistry:
    def __init__(self, host):
        self.host, self.definitions = host, {}
        self.register(ToolDefinition("query_structure", "查询模块、保护范围及当前能力", obj({}),
            obj({"modules": arr(FREE_OBJECT), "capabilities": FREE_OBJECT}), ("profiling",), self.structure))
        self.register(ToolDefinition("read_trials", "读取搜索记录；不会返回最终评测", obj({"ids": arr(STR, maxItems=8)}),
            obj({"trials": arr(FREE_OBJECT)}), ("profiling",), self.read_trials))
        from .profiling.tools import register_profiling_tools
        register_profiling_tools(self)

    def structure(self):
        # 执行注册会替换为带实际 Conv/head 许可的闭包；不允许使用不完整初始化。
        raise RuntimeError("执行工具尚未注册")

    def read_trials(self, ids):
        raise RuntimeError("执行工具尚未注册")

    def register(self, definition: ToolDefinition) -> None:
        require(definition.name not in self.definitions, "工具重名")
        self.definitions[definition.name] = definition


    def schemas(self, role: str) -> list[JSON]:
        return [tool.schema() for tool in self.definitions.values() if role in tool.roles]


    def invoke(self, role: str, call: ToolCall) -> ToolResult:
        try:
            require(call.name in self.definitions, "未注册工具", code="unknown_tool")
            tool = self.definitions[call.name]
            require(role in tool.roles, f"{role} 无权调用 {call.name}", code="forbidden_tool")
            validate(call.arguments, tool.parameters, "$.arguments")
            data = tool.handler(**call.arguments)
            validate(data, tool.returns, "$.tool_output")
            artifacts = [data["evidence"]] if "evidence" in data else []
            if "trial" in data:
                artifacts.append(data["trial"]["evidence"])
            if "records" in data:
                artifacts.extend(r["memory_ref"] for r in data["records"])
                artifacts.extend(r["evidence"] for r in data["records"])
            return ToolResult(call.id, call.name, "success", data=data, artifacts=artifacts)
        except PTQError as exc:
            status = exc.status
            if exc.code.startswith("unsupported"):
                status = "unsupported"
            return ToolResult(call.id, call.name, status, error=exc.to_dict())
        except (FloatingPointError, OverflowError) as exc:
            return ToolResult(call.id, call.name, "failure", error={"code": "numerical_error", "message": str(exc),
                              "path": "$.tool", "retryable": False})
        except Exception as exc:
            # 工具异常不会伪装为成功；详细错误进入持久化日志并计入连续失败。
            return ToolResult(call.id, call.name, "failure", error={"code": "tool_exception", "message": f"{type(exc).__name__}: {exc}",
                              "path": "$.tool", "retryable": False})


    def _cached_measurement(self, kind, arguments, units, compute, *, adjustable_request=False):
        key = fingerprint({"kind": kind, "arguments": arguments, "identity": self.host.state["identity"],
                               "definition": DEFINITION_VERSION, "model_state": "immutable_original"})
        path = self.host.run_dir / "evidence" / f"{kind}-{key}.json"
        if path.exists():
            return read_measurement(path, self.host.state["identity"], arguments=arguments), str(path.relative_to(self.host.run_dir)), True
        self.host.consume("measurements", units)
        result = seal({"arguments": arguments, "identity": self.host.state["identity"],
                       "definition": DEFINITION_VERSION, "source": self.host.state["identity"]["source"],
                       "result": compute()})
        atomic_json(path, result)
        return result, str(path.relative_to(self.host.run_dir)), False

