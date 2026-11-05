"""原始需求、唯一实现与实际执行条件的批准绑定。"""
import ast
import os
from pathlib import Path
import tempfile

from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256, read_json


INTERFACES = {"quantizer_v1": ("fit", "apply"), "executor_v1": ("build_plan",),
              "experiment_v1": ("execute",)}
NUMERIC_IMPORTS = {"torch", "numpy", "math", "statistics", "itertools", "functools", "collections",
                   "copy", "dataclasses", "typing", "random", "scipy", "ptq_agent",
                   "__future__", "abc", "contextlib", "operator", "bisect", "heapq", "array",
                   "enum", "numbers", "decimal", "fractions", "re", "struct", "json", "hashlib",
                   "time", "warnings"}
# Explicit numerical/structural standard libraries, not a blanket stdlib permission.
# Importing their implementation dependencies does not grant those capabilities.
CAPABILITY_MODULES = {"os", "sys", "subprocess", "socket", "ctypes", "importlib", "pathlib",
                      "shutil", "tempfile", "builtins", "multiprocessing", "urllib", "http"}
# 共享的只读数值接口；不包含本次实现或运行资产。
PUBLIC_INTERFACE_FILES = ('model.py', 'pipeline.py', 'contracts.py', 'plugin_contracts.py', 'calibration.py', 'plugin_math.py')
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
SAFE_IMPORTED_CALLS = {"re.compile"}  # Regex construction is not Python code compilation.


def inspect_source(source, interface):
    if interface not in INTERFACES:
        raise ContractError("未知代码接口")
    if not isinstance(source, str) or not source.strip():
        raise ContractError("代码 source 必须为非空字符串")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ContractError(f"代码语法错误 line {exc.lineno}: {exc.msg}") from exc
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not set(INTERFACES[interface]) <= functions:
        raise ContractError("代码没有实现冻结接口: " + ",".join(INTERFACES[interface]))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({item.asname or item.name.split('.')[0]: item.name for item in node.names})
        elif isinstance(node, ast.ImportFrom):
            aliases.update({item.asname or item.name: (node.module or '') + '.' + item.name for item in node.names})
    rebound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    rebound.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if (isinstance(node, ast.ImportFrom) and node.level) or name.split('.')[0] not in NUMERIC_IMPORTS:
                    capability = '文件/进程/网络能力须由宿主提供' if name.split('.')[0] in CAPABILITY_MODULES else '该依赖尚未声明支持'
                    raise ContractError(f"插件不允许导入模块 {name!r} (line {node.lineno})：{capability}")
            if isinstance(node, ast.ImportFrom):
                for item in node.names:
                    if item.name in CAPABILITY_MODULES or (item.name in FORBIDDEN_CALLS
                            and node.module + "." + item.name not in SAFE_IMPORTED_CALLS):
                        raise ContractError(f"插件不允许导入操作 {node.module}.{item.name} (line {node.lineno})：能力须由宿主提供")
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in aliases and node.attr in CAPABILITY_MODULES:
                raise ContractError(f"插件不允许访问能力 {ast.unparse(node)} (line {node.lineno})：不能经数值模块获取系统能力")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS
                and (aliases.get(node.func.id) not in SAFE_IMPORTED_CALLS or node.func.id in rebound)):
            raise ContractError(f"插件不允许调用 {node.func.id} (line {node.lineno})：执行/文件入口须使用宿主上下文")
    return {"interface": interface, "functions": sorted(functions), "syntax": "passed",
            "scope": "structural boundary only; not a proof of scientific correctness"}


def write_artifact(root, experiment, response, role):
    interface = experiment["payload"]["execution"].get("interface")
    if response.get("interface", interface) != interface:
        raise ContractError("代码接口与冻结实验不一致")
    source = response.get("source")
    structure = inspect_source(source, interface)
    identity = fingerprint({"experiment_sha256": experiment["content_sha256"], "source": source,
                            "interface": interface, "role": role, "execution_choices": response.get("execution_choices", {})})
    directory = Path(root) / "code" / identity
    if (directory / "manifest.json").exists():
        manifest = read_json(directory / "manifest.json")
        if file_sha256(directory / "implementation.py") != manifest["source_sha256"]:
            raise ContractError("已有代码资产被修改")
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "implementation.py"
    # 尚未发布 manifest 的目录可能来自中断；只原子补齐同一冻结输入的源码。
    fd, temporary = tempfile.mkstemp(prefix=".source-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    manifest = {"id": "code-" + identity, "path": str(path.resolve()), "source_sha256": file_sha256(path),
                "experiment_ref": experiment["id"], "experiment_sha256": experiment["content_sha256"],
                "interface": interface, "author": role, "structure": structure,
                "explanation": response.get("explanation", ""),
                "execution_choices": response.get("execution_choices", {})}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def verify_artifact(artifact, experiment):
    # Artifacts bind the immutable original request. Executable records additionally
    # bind that same request, so code delivery can precede execution finalization.
    payload = experiment["payload"]
    ref = payload.get("request_ref", experiment["id"])
    digest = payload.get("request_sha256", experiment["content_sha256"])
    if artifact["experiment_ref"] != ref or artifact["experiment_sha256"] != digest:
        raise ContractError("代码与原始实验需求不一致")
    if artifact["interface"] != payload["execution"].get("interface"):
        raise ContractError("代码接口与执行条件不一致")
    path = Path(artifact["path"])
    if file_sha256(path) != artifact["source_sha256"]:
        raise ContractError("批准的源码哈希已变化")
    inspect_source(path.read_text(), artifact["interface"])


def approval_receipt(experiment, candidate, review):
    from runtime.identity import code_identity
    verify_artifact(candidate, experiment)
    payload = review["payload"]
    if payload.get("approved") is not True or payload.get("phase") != "safeguard":
        raise ContractError("Safeguard 未批准")
    expected = {"experiment_ref": experiment["id"], "experiment_sha256": experiment["content_sha256"],
                "request_ref": candidate["experiment_ref"], "request_sha256": candidate["experiment_sha256"],
                "source_sha256": candidate["source_sha256"]}
    if payload.get("reviewed_binding") != expected:
        raise ContractError("Safeguard 审查未绑定当前原始需求、执行条件及源码")
    value = {**expected, "candidate": candidate,
             "host_implementation_sha256": code_identity()["core_sha256"],
             "safeguard_ref": review["id"], "safeguard_sha256": review["content_sha256"],
             "status": "approved"}
    return {**value, "approval_sha256": fingerprint(value)}


def validate_approval(approval, experiment):
    from runtime.identity import code_identity
    if not approval or approval.get("status") != "approved":
        raise ContractError("缺少 Safeguard 批准")
    if approval["approval_sha256"] != fingerprint({k: v for k, v in approval.items() if k != "approval_sha256"}):
        raise ContractError("批准记录损坏")
    if approval["experiment_ref"] != experiment["id"] or approval["experiment_sha256"] != experiment["content_sha256"]:
        raise ContractError("修改实验执行条件后批准自动失效")
    if approval.get("host_implementation_sha256") != code_identity()["core_sha256"]:
        raise ContractError("核心实现变化后必须重新通过 Safeguard")
    verify_artifact(approval["candidate"], experiment)
