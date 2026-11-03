"""任务内 Python 适配程序的受限子进程，不导入模型或框架。

只处理 JSON 和经父进程校验的工具 RPC。限制语法、内建函数、CPU 和内存，
不授予文件、网络、导入、对象反射或 shell 权限；不是通用 Python 沙箱。
"""

import ast
import builtins
import json
import math
import resource
import sys


MAX_SOURCE = 65536
MAX_MESSAGE = 4 * 1024**2
ATTRIBUTES = {
    "get", "items", "keys", "values", "copy", "update", "setdefault", "pop",
    "append", "extend", "insert", "remove", "sort", "reverse", "count", "index",
    "startswith", "endswith", "split", "rsplit", "join", "strip", "replace",
    "removeprefix", "removesuffix", "lower", "upper", "isdigit",
}
NODES = {
    "Module", "FunctionDef", "arguments", "arg", "Return", "Expr", "Constant",
    "Name", "Load", "Store", "Assign", "AnnAssign", "AugAssign", "If", "IfExp",
    "For", "While", "Break", "Continue", "Pass", "Raise", "Try", "ExceptHandler",
    "Assert", "List", "Tuple", "Dict", "Set", "Subscript", "Slice", "Call",
    "keyword", "Attribute", "BinOp", "UnaryOp", "BoolOp", "Compare", "Lambda",
    "ListComp", "DictComp", "SetComp", "GeneratorExp", "comprehension",
    "JoinedStr", "FormattedValue", "Add", "Sub", "Mult", "Div", "FloorDiv",
    "Mod", "Pow", "USub", "UAdd", "Not", "And", "Or", "Eq", "NotEq", "Lt",
    "LtE", "Gt", "GtE", "In", "NotIn", "Is", "IsNot",
}
BUILTINS = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "isinstance", "ValueError", "RuntimeError", "KeyError",
    "TypeError", "AssertionError",
)


def validate_source(source):
    if not isinstance(source, str) or not 0 < len(source.encode()) <= MAX_SOURCE:
        raise ValueError("程序必须为 1..65536 字节 UTF-8 源码")
    tree = ast.parse(source, filename="<task-program>")
    for node in ast.walk(tree):
        if type(node).__name__ not in NODES:
            raise ValueError(f"不支持语法 {type(node).__name__}，行 {getattr(node, 'lineno', 0)}")
        names = ([node.id] if isinstance(node, ast.Name) else
                 [node.arg] if isinstance(node, ast.arg) else
                 [node.name] if isinstance(node, (ast.FunctionDef, ast.ExceptHandler)) and node.name else [])
        if any(name.startswith("_") for name in names):
            raise ValueError("不允许私有名称/反射")
        if isinstance(node, ast.Attribute) and node.attr not in ATTRIBUTES:
            raise ValueError(f"不支持属性 {node.attr}，行 {node.lineno}")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.returns):
            raise ValueError("函数不接受装饰器或返回注解")
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if len(entries) != 1 or len(entries[0].args.args) != 1:
        raise ValueError("必须定义且仅定义一个入口 def run(inputs)")
    return tree


def send(message):
    data = json.dumps(message, ensure_ascii=False, allow_nan=False)
    if len(data.encode()) > MAX_MESSAGE:
        raise ValueError("程序消息超过 4 MiB；请缩小请求或分页读取证据")
    sys.stdout.write(data + "\n")
    sys.stdout.flush()


def receive():
    line = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ValueError("无效 RPC 消息")
    return json.loads(line)


def main():
    # 工具测量由父进程完成，不占用此进程 CPU 预算；不加载 torch/CUDA。
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    request = receive()
    try:
        tree = validate_source(request["source"])

        def tool(name, arguments):
            send({"type": "tool", "name": name, "arguments": arguments})
            response = receive()
            if response.get("type") != "result":
                raise RuntimeError(response.get("error", "工具 RPC 中断"))
            return response["result"]

        namespace = {"__builtins__": {name: getattr(builtins, name) for name in BUILTINS},
                     "tool": tool, "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
                     "isfinite": math.isfinite}
        exec(compile(tree, "<task-program>", "exec"), namespace)
        result = namespace["run"](request["inputs"])
        if not isinstance(result, dict):
            raise ValueError("run(inputs) 必须返回 JSON 对象")
        send({"type": "done", "result": result})
    except Exception as exc:
        send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
