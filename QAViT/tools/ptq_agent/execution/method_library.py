"""人可编辑的方法资料；只提供文档，执行权限始终来自数值注册表和契约。

会话开始时冻结一份小型文档快照，防止人工编辑改变运行中的 LLM 依据。
不从 Markdown 加载 Python，不提供整模型配方，也不自动选择/注入方法。
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from ..schemas import arr, obj, require
from ..storage import atomic_json, read_json
from ..registry import ToolDefinition
from .methods import plugin_versions


ROOT = Path(__file__).resolve().parents[3] / "docs" / "methods"


class MethodLibrary:
    def __init__(self, root=ROOT, *, snapshot=None):
        root = Path(root).resolve()
        index = snapshot["index"] if snapshot else json.loads((root / "index.json").read_text(encoding="utf-8"))
        require(index["schema_version"] == 1, "不支持的方法库版本")
        self.entries, self.documents = {}, {}
        registry = plugin_versions()
        covered = set()
        for entry in index["methods"]:
            key = entry["id"]
            require(key not in self.entries, f"方法库 ID 重复：{key}")
            path = (root / entry["document"]).resolve()
            require(path.is_relative_to(root) and path.suffix == ".md", "方法文档必须位于方法库内")
            content = snapshot["documents"][key]["content"] if snapshot else path.read_text(encoding="utf-8")
            require(0 < len(content) <= 16000, f"方法卡为空或过长：{key}")
            for target, names in entry["plugins"].items():
                require(target in registry and set(names) <= set(registry[target]), f"文档引用未注册插件：{key}/{target}")
                covered.update((target, name) for name in names)
            self.entries[key] = deepcopy(entry)
            self.documents[key] = {"id": key, "document": entry["document"], "content": content,
                                   "sha256": hashlib.sha256(content.encode()).hexdigest()}
        expected = {(target, name) for target, methods in registry.items() for name in methods}
        require(covered == expected, f"已注册插件缺少方法说明：{sorted(expected - covered)}")
        self.snapshot = {"index": index, "documents": self.documents}
        self.sha256 = hashlib.sha256(json.dumps(self.snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if snapshot:
            require(snapshot["sha256"] == self.sha256, "会话方法库快照被修改")

    def catalog(self, config, target="all", query=""):
        rows = []
        # 与实际位点命名兼容，例如 post_gelu / post-softmax；不是模型推荐器。
        words = [word for word in re.split(r"[\s_/-]+", query.lower()) if word]
        for entry in self.entries.values():
            if target != "all" and target not in entry["targets"]:
                continue
            searchable = json.dumps(entry, ensure_ascii=False).lower()
            if words and not any(word in searchable for word in words):
                continue
            row = deepcopy(entry)
            row["allowed_plugins"] = {
                t: [n for n in names if n in config["loop"].get(
                    "allowed_preprocess" if t == "preprocess" else "allowed_" + t + "_methods", names)]
                for t, names in entry["plugins"].items()}
            row["document_sha256"] = self.documents[entry["id"]]["sha256"]
            rows.append(row)
        return {"methods": rows, "library_sha256": self.sha256,
                "note": "allowed仅是白名单；适用位点/格式/顺序还须通过validate_experiment。文档不是测量证据。",
                "no_match_hint": "无匹配时用target=all, query=''查看全部索引；检索不执行方法选择" if not rows else None}

    def read(self, method):
        require(method in self.documents, f"未知方法文档：{method}；先查询方法索引")
        return {**deepcopy(self.documents[method]), "library_sha256": self.sha256,
                "kind": "method_documentation_not_measurement"}


def require_plan_documents(session, plan):
    """确认实际查阅过独立契约；不替 LLM 推荐插件或组合整模型配方。"""
    library = session.method_library
    results = [row["result"] for row in session.state["tool_results"]
               if row["result"]["status"] == "success"]
    require(any(r["name"] == "query_method_library"
                and r["data"].get("library_sha256") == library.sha256
                and r["data"].get("methods") for r in results),
            "提交前请用 query_method_library 查询本会话方法库；空query可查看全部索引",
            code="missing_method_documentation")
    read = {doc.get("id") for r in results if r["name"] == "read_method_document"
            for doc in r["data"].get("documents", [r["data"]])
            if doc.get("library_sha256") == library.sha256 and doc.get("id") in library.documents
            and doc.get("sha256") == library.documents[doc["id"]]["sha256"]}
    used = {(target, layer[target + "_method"]) for layer in plan["layers"]
            for target in ("weight", "activation") if layer.get(target)}
    used.update(("preprocess", row["method"]) for row in plan.get("preprocess", []))
    used.update(("operator", row["method"]) for row in plan.get("operators", []))
    missing = []
    for target, method in sorted(used):
        documents = {key for key, entry in library.entries.items()
                     if method in entry["plugins"].get(target, [])}
        if not documents.intersection(read):
            missing.append({"plugin": f"{target}/{method}", "documents": sorted(documents)})
    # 校准策略/阶段说明也来自可编辑索引，避免把默认值误当成已做尺度搜索。
    controls = {key for key, entry in library.entries.items() if "calibration" in entry["targets"]}
    missing.extend({"calibration_document": key} for key in sorted(controls - read))
    require(not missing, f"请先 read_method_document 阅读尚未查阅的独立契约：{missing}；"
            "阅读不等于选用所有方法，文档不能替代实测evidence",
            code="missing_method_documentation")


def register_method_library(session):
    path = session.run_dir / "method_library.json"
    library = MethodLibrary(snapshot=read_json(path) if path.is_file() else None)
    # 只保存约几十 KiB 的文本；真实请求日志同时保留实际返回内容。
    if not path.is_file():
        atomic_json(path, {**library.snapshot, "sha256": library.sha256})
    session.method_library = library
    free = {"type": "object", "additionalProperties": True}
    session.registry.register(ToolDefinition(
        "query_method_library", "按问题关键词/阶段检索独立PTQ组件说明；空query返回全部，可先profile再按需阅读",
        obj({"target": {"enum": ["all", "weight", "activation", "preprocess", "operator", "calibration"]},
             "query": {"type": "string"}}), free, ("profiling",),
        lambda target, query: library.catalog(session.config, target, query)))
    def read(method=None, methods=None):
        require((method is None) != (methods is None), "method/methods必须且只能指定一种")
        if method is not None:
            return library.read(method)
        return {"documents": [library.read(key) for key in methods], "library_sha256": library.sha256}
    session.registry.register(ToolDefinition(
        "read_method_document", "读取独立方法原理、伪代码、适用分布、参数、组合关系和局限；method单篇或methods批量1至8篇完整读取，不返回整模型配方",
        {"anyOf": [obj({"method": {"enum": list(library.entries)}}),
                   obj({"methods": arr({"enum": list(library.entries)}, minItems=1, maxItems=8, uniqueItems=True)})]},
        free, ("profiling",), read))
