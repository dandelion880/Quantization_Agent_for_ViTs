"""冻结规格、候选源码、独立参考和实际数值检查的批准绑定。"""
import ast
import os
from pathlib import Path
import tempfile

from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256, read_json


INTERFACES = {"quantizer_v1": ("fit", "apply"), "executor_v1": ("build_plan",),
              "experiment_v1": ("execute",)}
NUMERIC_IMPORTS = {"torch", "numpy", "math", "statistics", "itertools", "functools", "collections",
                   "copy", "dataclasses", "typing", "random", "scipy", "ptq_agent"}
# 共享的只读数值接口；不包含本次候选、独立参考或运行资产。
PUBLIC_INTERFACE_FILES = ('pipeline.py', 'contracts.py', 'plugin_contracts.py', 'calibration.py', 'plugin_math.py')
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}


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
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".")[0] not in NUMERIC_IMPORTS for name in names):
                raise ContractError("数值插件不能自行获取文件/进程/网络能力；请通过宿主声明所需资产或依赖")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ContractError("插件需使用宿主提供的上下文，禁止自建执行/文件入口")
    return {"interface": interface, "functions": sorted(functions), "syntax": "passed",
            "scope": "structural boundary only; not a proof of scientific correctness"}


def write_artifact(root, experiment, response, role):
    interface = experiment["payload"]["execution"].get("interface")
    if response.get("interface") != interface:
        raise ContractError("代码接口与冻结实验不一致")
    source = response.get("source")
    structure = inspect_source(source, interface)
    identity = fingerprint({"experiment_sha256": experiment["content_sha256"], "source": source,
                            "interface": interface, "role": role})
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
                "explanation": response.get("explanation", "")}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def verify_artifact(artifact, experiment):
    if artifact["experiment_ref"] != experiment["id"] or artifact["experiment_sha256"] != experiment["content_sha256"]:
        raise ContractError("代码与冻结 ExperimentCard 不一致")
    path = Path(artifact["path"])
    if file_sha256(path) != artifact["source_sha256"]:
        raise ContractError("批准的源码哈希已变化")
    inspect_source(path.read_text(), artifact["interface"])


def approval_receipt(experiment, candidate, reference, numerical, review):
    from runtime.identity import code_identity
    verify_artifact(candidate, experiment)
    verify_artifact(reference, experiment)
    if numerical.get("status") != "passed" or not numerical.get("cases"):
        raise ContractError("强制独立数值验证未通过，不得放行")
    if candidate["interface"] == "experiment_v1":
        expected = experiment["payload"]["validation"].get("interface_cases", [])
        if not expected or len(numerical.get("interface_cases", [])) != len(expected) or any(
                row.get("status") != "passed" for row in numerical["interface_cases"]):
            raise ContractError("真实 Data/Backend 接口案例未通过，不得只凭数学案例放行")
    if numerical.get("experiment_sha256") != experiment["content_sha256"]:
        raise ContractError("数值验证不是同一冻结规格")
    if numerical.get("candidate_sha256") != candidate["source_sha256"] or numerical.get("reference_sha256") != reference["source_sha256"]:
        raise ContractError("数值验证源码不匹配")
    if numerical.get("host_implementation_sha256") != code_identity()["core_sha256"]:
        raise ContractError("验证实现版本变化，必须重新数值验证和Safeguard审查")
    if review["payload"].get("approved") is not True or review["payload"].get("phase") != "safeguard":
        raise ContractError("Safeguard 未批准")
    for field in ("fidelity", "numerical_correctness", "experimental_validity"):
        if not review["payload"].get(field):
            raise ContractError("Safeguard 未完成三类审查: " + field)
    value = {"experiment_ref": experiment["id"], "experiment_sha256": experiment["content_sha256"],
             "candidate": candidate, "reference": reference, "verification_sha256": fingerprint(numerical),
             "host_implementation_sha256": numerical["host_implementation_sha256"],
             "safeguard_ref": review["id"], "safeguard_sha256": review["content_sha256"],
             "status": "approved"}
    return {**value, "approval_sha256": fingerprint(value)}


def validate_approval(approval, experiment):
    from runtime.identity import code_identity
    if not approval or approval.get("status") != "approved":
        raise ContractError("缺少 Safeguard 批准")
    if approval["approval_sha256"] != fingerprint({k: v for k, v in approval.items() if k != "approval_sha256"}):
        raise ContractError("批准记录损坏")
    if approval["experiment_sha256"] != experiment["content_sha256"]:
        raise ContractError("修改实验规格后批准自动失效")
    if approval.get("host_implementation_sha256") != code_identity()["core_sha256"]:
        raise ContractError("核心实现变化后必须重新通过数值验证及Safeguard")
    verify_artifact(approval["candidate"], experiment)
    verify_artifact(approval["reference"], experiment)
