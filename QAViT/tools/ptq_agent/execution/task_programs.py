"""LLM 的单次任务代码工作区：版本化 Python + 可审计的真实工具调用。

程序可适配模块名、布局对应的统计请求和 PTQ 组合，但不能改框架、数值插件、
数据划分或门槛。输出的方案仍完整展开为 v3 JSON，无须执行生成代码即可重载。
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
import uuid

from ..schemas import STR, ToolCall, obj, require, fingerprint
from ..storage import atomic_json, read_json
from ..registry import ToolDefinition
from .program_worker import validate_source, MAX_MESSAGE, BUILTINS, ATTRIBUTES


TOOLS = {"query_task_program_api", "write_task_program", "read_task_program", "run_task_program", "read_tool_source"}
SOURCE_FILES = (
    "profiling/programs.py", "profiling/adapters.py", "profiling/torch_profiler.py",
    "execution/plugin_tools.py", "execution/sites.py", "execution/planning.py",
    "execution/plugin_math.py", "execution/grids.py", "execution/calibration.py",
    "execution/coordinate.py", "execution/methods.py", "execution/plugin_contracts.py",
    "execution/objectives.py",
    "execution/adalog.py", "execution/adaround.py", "execution/gptaq.py",
    "execution/local_comparison.py", "execution/runtime_state.py", "execution/comparison.py",
)
FREE = {"type": "object", "additionalProperties": True}


class TaskPrograms:
    def __init__(self, session):
        self.session = session
        self.root = session.run_dir / "programs"

    def available_tools(self):
        return [d for name, d in self.session.registry.definitions.items()
                if name not in TOOLS and "profiling" in d.roles]

    def api(self, names=()):
        available = {d.name: d for d in self.available_tools()}
        require(set(names) <= set(available), "请求了未开放的工具")
        return {
            "scope": "本会话 Python 工具适配；不得修改共享源码、数值插件、模型或评价协议",
            "entry": "def run(inputs): 返回JSON对象；tool(name, arguments)返回完整status/data/error/artifacts envelope",
            "tool_names": sorted(available),
            # 直接复用注册表与执行器定义，避免 LLM 仅凭名字猜工具/语言能力。
            "tool_menu": [{"name": n, "description": available[n].description} for n in sorted(available)],
            "schemas": [available[n].schema() for n in names],
            "source_files": list(SOURCE_FILES),
            "limits": {"calls_per_run": 12, "cpu_seconds": 10, "memory_mib": 512,
                       "source_bytes": 65536, "versions": 64, "runs": 256},
            "language": "常规Python循环/分支/函数/推导式、JSON容器、有限内建函数；sqrt/log/log2/isfinite；无import/open/eval/exec/反射/shell/网络",
            "builtins": list(BUILTINS), "container_methods": sorted(ATTRIBUTES),
            "workflow": "查询接口→write_task_program→run_task_program；错误后写新版本并重试。所有内层工具仍计预算，仍执行原校验。",
            "evidence_rule": "程序自行计算/解释的JSON不是测量证据；引用内层工具真实返回的evidence。不能伪造profile/精度。",
            "example": "def run(inputs):\n    response = tool('query_structure', {})\n    if response['status'] != 'success':\n        return response\n    return {'modules': [m['name'] for m in response['data']['modules'] if m['name'].startswith(inputs['prefix'])]}\n",
        }

    def write(self, name, source, purpose):
        require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is not None, "程序名只能含字母/数字/_/-")
        try:
            validate_source(source)
        except (SyntaxError, ValueError) as exc:
            require(False, str(exc), code="invalid_task_program")
        digest = hashlib.sha256(source.encode()).hexdigest()
        reference = f"programs/{name}-{digest}.py"
        path = self.session.run_dir / reference
        if path.exists():
            require(path.read_text() == source, "程序源码 hash 冲突")
        else:
            require(len(list(self.root.glob("*.py"))) < 64, "本会话程序版本达到64个上限")
            self.root.mkdir(exist_ok=True)
            # 新版本使用独立名字，不覆盖已用于生成候选的代码。
            with path.open("x", encoding="utf-8") as stream:
                stream.write(source)
            atomic_json(path.with_suffix(".json"), {"program": reference, "sha256": digest, "purpose": purpose})
        return {"program": reference, "sha256": digest, "syntax_valid": True,
                "execution_verified": False, "next": "run_task_program；语法通过不等于程序执行成功"}

    def read(self, program):
        require(re.fullmatch(r"programs/[A-Za-z][A-Za-z0-9_-]{0,63}-[0-9a-f]{64}\.py", program) is not None,
                "只能读取本会话返回的程序路径")
        path = self.session.run_dir / program
        require(path.is_file() and not path.is_symlink(), "程序不存在或不是普通文件")
        source = path.read_text(encoding="utf-8")
        meta = read_json(path.with_suffix(".json"))
        require(hashlib.sha256(source.encode()).hexdigest() == meta["sha256"] == path.stem[-64:],
                "适配程序被修改", code="identity_mismatch")
        return {**meta, "source": source}

    def source(self, module, start_line=1, lines=100):
        require(module in SOURCE_FILES and 1 <= start_line and 1 <= lines <= 160, "源码读取范围不合法")
        path = Path(__file__).resolve().parents[1] / module
        raw = path.read_bytes()
        content = raw.decode().splitlines()
        return {"module": module, "sha256": hashlib.sha256(raw).hexdigest(), "total_lines": len(content),
                "start_line": start_line, "content": "\n".join(content[start_line - 1:start_line - 1 + lines]),
                "read_only": True}

    def invoke(self, name, arguments, origin):
        require(name in {d.name for d in self.available_tools()}, "程序无权调用此工具", code="forbidden_tool")
        host = self.session
        call = ToolCall("program-" + uuid.uuid4().hex, name, arguments)
        host.consume("tools")
        entry = {"role": "profiling", "epoch": host.state["analysis_epoch"], "call": asdict(call), "origin": origin}
        host.state["tool_calls"].append(entry)
        # 不覆盖外层 run_task_program 的 pending_call。中断的程序只可显式重新运行；
        # 已完成的内层测量仍持久化并可按原证据缓存复用。
        host.save()
        start = time.perf_counter()
        host.active_program = origin
        try:
            result = host.registry.invoke("profiling", call)
        finally:
            host.active_program = None
        record = {"role": "profiling", "epoch": host.state["analysis_epoch"],
                  "history_size": len(host.state["history"]), "arguments": arguments,
                  "result": result.to_dict(), "cost_seconds": time.perf_counter() - start, "origin": origin}
        atomic_json(host.run_dir / "calls" / f"{fingerprint(call.id)}.json", record)
        host.state["tool_results"].append(record)
        if result.status != "success":
            host._failure(result.error, f"program:{origin['program']}:{call.id}")
        host.save()
        return record

    def run(self, program, inputs):
        script = self.read(program)
        require(len(list(self.root.glob("run-*.json"))) < 256, "本会话程序执行达到256次上限")
        run_id = "run-" + uuid.uuid4().hex
        pending = self.session.state.get("pending_call") or {}
        origin = {"program": program, "sha256": script["sha256"], "run": f"programs/{run_id}.json",
                  "parent_call_id": pending.get("call", {}).get("id")}
        trace = {**origin, "inputs": inputs, "calls": [], "status": "running", "result": None}
        path = self.root / f"{run_id}.json"
        atomic_json(path, trace)
        process = None
        start = time.monotonic()
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", str(Path(__file__).with_name("program_worker.py"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.root, env={"LANG": "C.UTF-8", "PATH": os.defpath},
            )
            def send(value):
                raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() + b"\n"
                require(len(raw) <= MAX_MESSAGE, "工具消息超过4 MiB，需缩小请求", code="task_program_limit")
                process.stdin.write(raw)
                process.stdin.flush()

            send({"source": script["source"], "inputs": inputs})
            buffer = b""
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    require(selector.select(timeout=30), "适配程序30秒无响应", code="task_program_timeout")
                    chunk = os.read(process.stdout.fileno(), 65536)
                    require(chunk, f"适配程序异常退出，returncode={process.poll()}", code="task_program_error")
                    buffer += chunk
                    require(len(buffer) <= MAX_MESSAGE, "适配程序输出超过4 MiB", code="task_program_limit")
                    if b"\n" not in buffer:
                        continue
                    line, buffer = buffer.split(b"\n", 1)
                    message = json.loads(line)
                    if message["type"] == "tool":
                        require(len(trace["calls"]) < 12, "单程序最多12个工具调用", code="task_program_limit")
                        record = self.invoke(message["name"], message["arguments"], origin)
                        trace["calls"].append(record)
                        atomic_json(path, trace)
                        send({"type": "result", "result": record["result"]})
                    elif message["type"] == "done":
                        trace.update(status="completed", result=message["result"])
                        break
                    else:
                        require(False, message.get("error", "未知程序消息"), code="task_program_error")
            return {**origin, "status": "completed", "result": trace["result"],
                    "tool_calls": len(trace["calls"]),
                    "failed_calls": [r["result"] for r in trace["calls"] if r["result"]["status"] != "success"],
                    "note": "result是LLM程序生成的解释/适配输出；测量事实以各内层工具的原始evidence为准"}
        except BaseException as exc:
            trace.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if process:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)
            trace["elapsed_seconds"] = time.monotonic() - start
            atomic_json(path, trace)


def register_task_programs(session):
    workspace = TaskPrograms(session)
    session.task_programs = workspace
    definitions = [
        ("query_task_program_api", "查询本任务Python适配API和可用工具；names可取需查看完整schema的工具名",
         obj({"names": {"type": "array", "items": STR, "maxItems": 6}}, []), workspace.api),
        ("write_task_program", "创建/修改本任务Python适配程序（新版本，不改框架）；先查询API。入口def run(inputs)，通过tool调用真实统计/量化工具",
         obj({"name": STR, "source": STR, "purpose": STR}), workspace.write),
        ("read_task_program", "读取本会话已写入的Python适配程序及hash", obj({"program": STR}), workspace.read),
        ("run_task_program", "运行本会话Python适配程序；所有内层工具计入预算，完整记录输入输出；返回值本身不是测量证据",
         obj({"program": STR, "inputs": FREE}), workspace.run),
        ("read_tool_source", "只读查看独立统计/量化组件实现，便于编写本任务适配；不提供整套论文配方",
         obj({"module": {"enum": list(SOURCE_FILES)}, "start_line": {"type": "integer", "minimum": 1},
              "lines": {"type": "integer", "minimum": 1, "maximum": 160}}, ["module"]), workspace.source),
    ]
    for name, description, parameters, handler in definitions:
        session.registry.register(ToolDefinition(name, description, parameters, FREE, ("profiling",), handler))
