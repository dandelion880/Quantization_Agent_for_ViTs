"""独立订阅传输：隔离检查、命令和解码；不持有 Agent、历史或调用预算。"""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import uuid
from runtime import codex_cli as cli
from ptq_agent.schemas import LLMResponse, ToolCall, PTQError, canonical, fingerprint, obj, validate, require

DECISION = "__agent_decision__"
BATCH = "__tool_batch__"
REVIEWED_CLI = {
    "codex-cli 0.164.0": "EXP/implementation/cli_check",
    "codex-cli 0.166.0": "EXP/validation/cli166_resume_20261031_r1",
}


class StrictJSONViolation(ValueError):
    """Known strict JSON violations; unrelated decoder/program errors propagate."""


class StructuredOutputError(PTQError):
    """Known final-output violation, after the independent transport gates."""
    def __init__(self, final, name, cause, usage, *, stage, path, code="llm_structured_output_error"):
        super().__init__(code, stage + " 严格输出校验失败: " + str(cause), path=path)
        self.raw_output, self.name, self.usage, self.stage = final, name, usage, stage
        self.location = {"path": path, "line": getattr(cause, "lineno", None),
                         "column": getattr(cause, "colno", None), "offset": getattr(cause, "pos", None)}

    def to_dict(self):
        return {**super().to_dict(), "raw_output": self.raw_output, "name": self.name,
                "stage": self.stage, "location": self.location, "response_integrity_verified": True}


class InnerJSONFormatError(StructuredOutputError):
    """Keep the existing inner-JSON code/class compatible with durable receipts."""
    def __init__(self, final, name, cause, usage):
        super().__init__(final, name, cause, usage, stage="inner_json", path="$.arguments_json",
                         code="llm_inner_json_error")


def structured_output_error(error):
    return (isinstance(error, dict) and error.get("code") in
            ("llm_inner_json_error", "llm_structured_output_error")
            and error.get("response_integrity_verified") is True)


def validate_output(value, schema, *, final, name, usage, stage, prefix="$"):
    # This catch is deliberately local to validating the model's final value.
    # Schema/configuration errors elsewhere must not become Agent repair attempts.
    try:
        validate(value, schema)
    except PTQError as exc:
        if exc.code != "schema_error":
            raise
        raise StructuredOutputError(final, name, exc, usage, stage=stage,
                                    path=prefix + exc.path[1:]) from exc


TRANSPORT = """你是现有 PTQ Agent 的纯决策后端。下面的 LLMRequest 是完整任务协议。
system_prompt 保持原有角色含义；messages 保留历史调用身份；正文、真实工具结果和已读分段仅在 context。
tools 仅为宿主程序的可选动作定义，不是 Codex 可执行工具。不得运行命令、读写项目、调用 MCP/插件、
启动实验或递归调用 Codex。只返回一个符合输出 JSON Schema 的对象：
name 为工具名时，arguments_json 是符合该工具 parameters 的 JSON 对象字符串；
多个独立请求使用 name=__tool_batch__，arguments_json 为 {"calls":[{"name":"工具名","arguments":{工具参数}},...]} 的 JSON 对象字符串。
批次内只能使用本次 tools 的名称，不包含决策、嵌套批次或依赖前项结果的请求；多页读取可同批提交，共享本次读取快照。
name 为 __agent_decision__ 时，arguments_json 是符合 response_schema 的角色动作 JSON 对象字符串。
最终决策与工具请求只能选一。保留原始字段、数值与约束；不补造测量或执行结果。不输出 Markdown。
单工具模式下，外层 name 已指定工具；内层仅放该工具 parameters 定义的字段，不重复 name 或 arguments 包装；批次使用上述 calls 结构。
arguments_json 解码后必须是括号完整闭合的 JSON 对象；字符串转义不代替内层的结尾括号。
宿主程序会校验并执行该动作，把真实反馈用于下一次请求。
工具额度耗尽时，工具定义仍供理解，但输出Schema只允许原协议的决策动作；依据现有证据提交或停止，不再请求工具，也不要补造默认方案。
"""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise StrictJSONViolation("JSON 含重复字段")
            result[key] = value
        return result
    def nonfinite(_):
        raise StrictJSONViolation("JSON 含非有限数")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise StrictJSONViolation("JSON 数值溢出")
        return number
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)


def output_schema(request):
    names = [t["name"] for t in request.tools]
    require(not {DECISION, BATCH}.intersection(names) and len(names) == len(set(names)), "工具名称冲突", code="llm_protocol_error")
    # 与宿主已有预算一致：额度为零的工具不是可执行动作。保留完整输入、
    # 原决策schema和宿主门禁；不自动提交草案或增加工具/候选预算。
    remaining = request.context.get("remaining_budget", {}).get("tools")
    if isinstance(remaining, (int, float)) and remaining <= 0:
        names = []
    # 内层仍用原 schema 校验。用 JSON 字符串封装，避免把可选字段强制改为 null、
    # 关闭用户扩展参数或裁剪大型 PTQ union 来迁就 Structured Outputs 的 schema 子集。
    return obj({"name": {"type": "string", "enum": [*names, *([BATCH] if names else []), DECISION]},
                "arguments_json": {"type": "string"}})


def evidence_aliases(request):
    """只缩写原协议白名单路径；失败历史中的臆造路径不会进入映射。"""
    paths = set()

    def visit(node):
        if isinstance(node, dict):
            evidence = node.get("properties", {}).get("evidence", {})
            for spec in (evidence, evidence.get("items", {})):
                paths.update(p for p in spec.get("enum", []) if isinstance(p, str)
                             and p.startswith(("evidence/", "feedback/")) and p.endswith(".json"))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit([request.response_schema, *[t["parameters"] for t in request.tools]])
    # compact模式把动态路径移出嵌套schema；只读宿主生成的成功证据目录，
    # 不扫描失败参数/自由文本。展开之后仍由原执行器验证证据身份和内容。
    references = [ref for row in request.context.get("evidence_catalog", [])
                  for ref in row.get("evidence", [])]
    references += [row["reference"] for row in request.context.get("profile_feedback", [])
                   if row.get("source") == "program_verified" and "reference" in row]
    paths.update(ref for ref in references if isinstance(ref, str)
                 and ref.startswith(("evidence/", "feedback/")) and ref.endswith(".json"))
    return {f"@e{i:03d}": path for i, path in enumerate(sorted(paths))}


def expand_evidence(value, aliases, *, field=False, path="$"):
    # 仅 evidence 字段可用别名，不改 source/reason 等任意文本，也不做模糊纠错。
    if isinstance(value, dict):
        return {k: expand_evidence(v, aliases, field=k == "evidence", path=path + "." + k)
                for k, v in value.items()}
    if isinstance(value, list):
        return [expand_evidence(v, aliases, field=field, path=f"{path}[{i}]")
                for i, v in enumerate(value)]
    if field and isinstance(value, str) and value.startswith("@e"):
        require(value in aliases, "未知证据别名；只能使用本次请求映射中的完整值",
                path=path, code="llm_format_error")
        return aliases[value]
    return value


class SubscriptionTransport:
    def __init__(self, model, effort, settings):
        self.cli = cli.executable(settings)
        self.model, self.effort = model, effort
        self.source = f"real/codex_subscription/{model}/{effort}"
        self.verified = None

    def preflight(self, timeout_seconds=60):
        deadline = time.monotonic() + timeout_seconds
        with tempfile.TemporaryDirectory(prefix="qavit-cli-") as tmp:
            directory = Path(tmp)
            code, version, _ = cli.run([self.cli, "--version"], directory, min(10, timeout_seconds))
            require(code == 0 and version.strip() in REVIEWED_CLI,
                    "CLI 版本变化，需重新核验参数与隔离", code="llm_cli_version")
            code, output, error = cli.run([self.cli, "debug", "models", *cli.policy(directory)],
                                      directory, deadline - time.monotonic())
            if code:
                category = cli.failure_category(error)
                raise cli.fail("model_catalog" if category == "cli_failed" else category,
                               "不能核实模型目录；未开始模型调用（" + category + "）")
            model = next((m for m in json.loads(output)["models"] if m.get("slug") == self.model), None)
            require(model is not None, "模型目录没有指定模型，不回退", code="llm_unsupported_model")
            levels = [x["effort"] for x in model["supported_reasoning_levels"]]
            require(self.effort in levels and model.get("tool_mode") == "code_mode_only",
                    "指定档位或无工具host条件不满足", code="llm_tool_isolation")
            self.verified = {"cli_path": self.cli, "cli_version": version.strip(), "model": self.model,
                             "reasoning_effort": self.effort, "supported_efforts": levels,
                             "context_window": model.get("context_window"), "tool_mode": model["tool_mode"],
                             "account": cli.account_status(self.cli, directory, deadline - time.monotonic()),
                             "version_review": REVIEWED_CLI[version.strip()] + "; original isolation flags retained"}
        return self.verified


    def command(self, directory):
        return [self.cli, "exec", "--ignore-user-config", "--ignore-rules", "--strict-config",
                "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                "--model", self.model, "-c", "model_reasoning_effort=" + json.dumps(self.effort),
                "-c", "model_instructions_file=" + json.dumps(str(directory / "instructions.txt")),
                "--cd", str(directory), "--color", "never", "--json",
                "--output-schema", str(directory / "schema.json"),
                "--output-last-message", str(directory / "answer.json"), *cli.policy(directory), "-"]


    def decode(self, raw, final, request):
        events = [strict_json(line) for line in raw.splitlines() if line.strip()]
        completed, messages, safe, isolation_notices = [], [], [], 0
        actual = {"model": None, "reasoning_effort": None, "source": "not_reported_by_cli"}
        for event in events:
            require(isinstance(event, dict), "CLI event 必须为 JSON object", code="llm_format_error")
            kind = event.get("type")
            if kind in ("error", "turn.failed"):
                raise cli.fail(cli.failure_category(canonical(event)), "Codex 报告调用失败；未返回可执行动作")
            require(kind in ("thread.started", "turn.started", "turn.completed", "item.started", "item.updated", "item.completed"),
                    "Codex 出现未知事件；拒绝猜测完成状态", code="llm_protocol_error")
            if kind.startswith("item."):
                item = event.get("item", {})
                require(isinstance(item, dict), "CLI item 必须为 JSON object", code="llm_format_error")
                if item.get("type") == "error" and item.get("message") == cli.ISOLATION_NOTICE:
                    # 这是工具 host 已关闭的 CLI 通知，不是模型工具调用或失败。
                    isolation_notices += 1
                    safe.append(event)
                    continue
                if item.get("type") == "error" and item.get("message") == cli.skill_discovery_notice():
                    safe.append(event)
                    continue
                require(item.get("type") in ("agent_message", "reasoning"),
                        f"未知或禁止的 CLI item 类型 {item.get('type')!r}；拒绝将其视为纯决策", code="llm_tool_isolation")
                if kind == "item.completed" and item.get("type") == "agent_message":
                    messages.append(item.get("text"))
            if kind == "turn.completed":
                completed.append(event)
            # CLI 未报告的模型身份/推理档位保持未知；绝不把请求值冒充服务端值。
            for key, expected in (("model", self.model), ("reasoning_effort", self.effort)):
                value = event.get(key)
                if value is not None:
                    require(value == expected, f"Codex 实际 {key} 与请求不符", code="llm_model_mismatch")
                    actual[key], actual["source"] = value, "cli_event"
            safe.append(event)
        require(len(completed) == 1 and len(messages) == 1 and isinstance(final, str) and bool(final.strip()),
                "缺少唯一完整 final/turn.completed；拒绝截断响应", code="llm_format_error")
        require(isolation_notices > 0, "缺少 CLI 工具 host 已关闭的实际通知；不假设隔离成功", code="llm_tool_isolation")
        require(messages[0].strip() == final.strip(), "stdout 与最终文件不一致", code="llm_format_error")
        usage = completed[0].get("usage")
        if usage is not None:
            require(isinstance(usage, dict), "usage 格式错误", code="llm_format_error")
        schema = output_schema(request)  # Host protocol validation is not model-output repair.
        try:
            result = strict_json(final)
        except (json.JSONDecodeError, StrictJSONViolation) as exc:
            raise StructuredOutputError(final, None, exc, usage, stage="outer_json", path="$") from exc
        name = result.get("name") if isinstance(result, dict) else None
        validate_output(result, schema, final=final, name=name, usage=usage, stage="outer_schema")
        try:
            arguments = strict_json(result["arguments_json"])
        except (json.JSONDecodeError, StrictJSONViolation) as exc:
            # Only role JSON formatting is repairable here. All completion, model,
            # isolation and outer-envelope checks above must already have passed.
            raise InnerJSONFormatError(final, result["name"], exc, usage) from exc
        validate_output(arguments, {"type": "object", "additionalProperties": True}, final=final, name=name, usage=usage,
                        stage="inner_object", prefix="$.arguments_json")
        arguments = expand_evidence(arguments, evidence_aliases(request))
        if result["name"] == DECISION:
            validate_output(arguments, request.response_schema, final=final, name=name, usage=usage,
                            stage="decision_schema", prefix="$.arguments_json")
            response = LLMResponse(decision=arguments, usage=usage, source=self.source)
        else:
            if name == BATCH:
                batch_schema = obj({"calls": {"type": "array", "minItems": 1, "items": obj({
                    "name": {"enum": [t["name"] for t in request.tools]},
                    "arguments": {"type": "object", "additionalProperties": True}})}})
                validate_output(arguments, batch_schema, final=final, name=name, usage=usage,
                                stage="tool_schema", prefix="$.arguments_json")
                calls = arguments['calls']
            else:
                calls = [{'name': name, 'arguments': arguments}]
            decoded = []
            for index, call in enumerate(calls):
                tool = next(t for t in request.tools if t['name'] == call['name'])
                prefix = f'$.arguments_json.calls[{index}].arguments' if name == BATCH else '$.arguments_json'
                validate_output(call['arguments'], tool['parameters'], final=final, name=name, usage=usage,
                                stage='tool_schema', prefix=prefix)
                decoded.append(ToolCall('codex_' + uuid.uuid4().hex, tool['name'], call['arguments']))
            response = LLMResponse(tool_calls=decoded, usage=usage, source=self.source)
        return response, {"events": safe, "output": result, "resolved_arguments": arguments,
                          "usage": usage, "actual": actual,
                          "tool_host_disabled_verified": True}

