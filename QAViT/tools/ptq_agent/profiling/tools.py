"""面向 LLM 的 profiling 工具：schema、权限、预算和证据分页。"""

from ..schemas import INT, STR, arr, obj, require, canonical
from .contracts import profile_spec_schema
from .evidence import read_returned_evidence
from .programs import ProfilePrograms
from .statistics import STATISTICS

VERSION = "profiling_programs_v1"
SAMPLE_FIELDS = {"count": {"type": "integer", "minimum": 1, "maximum": 32}, "seed": INT, "offset": INT}
MODULES = arr(STR, maxItems=8, uniqueItems=True)
FREE = {"type": "object", "additionalProperties": True}
EVIDENCE_SECTIONS = ("modules", "sites", "functional_sites", "operators", "module_times", "rows", "details")


def register_profiling_tools(registry):
    """仅向 profiling 角色注册；后续实现/决策侧接口不变。"""
    from ..registry import ToolDefinition
    programs = ProfilePrograms(registry.host)
    result_schema = obj({"summary": FREE, "shapes": FREE, "evidence": STR,
                         "cached": {"type": "boolean"}, "source": STR})

    def measured(kind, units, compute, arguments):
        # hardware 的 refresh 标签允许显式重测；完整 arguments/身份参与缓存键。
        args = {"definition": VERSION, "profile_adapter": programs.adapter.version, **arguments}
        detail, evidence, cached = registry._cached_measurement(kind, args, units, compute, adjustable_request=True)
        result = detail["result"]
        shapes = dict(result.get("shapes", {}))
        if kind == "profile_structure":
            for row in result["modules"]:
                if len(row["calls"]) == 1:
                    first = row["calls"][0]["input"]
                    if first and isinstance(first[0], dict) and "shape" in first[0]:
                        shapes[row["name"]] = first[0]["shape"]
        summary = {k: v for k, v in result.items() if k not in ("details", "shapes")}
        for key in ("modules", "sites", "functional_sites", "operators", "module_times"):
            if key in summary:
                summary[key + "_total"] = len(summary[key])
                summary[key] = summary[key][:24]
        return {"summary": summary, "shapes": shapes, "evidence": evidence, "cached": cached, "source": detail["source"]}

    def structure(**args):
        # 结构证据还包含执行许可；相同FP模型在不同保护/scope下不能共用该元数据缓存。
        provider = getattr(registry.host, "profile_module_permissions", None)
        dependencies = {**args, "module_protection": provider()} if provider else args
        return measured("profile_structure", args["count"], lambda: programs.structure(**args), dependencies)

    def distribution(**args):
        return measured("profile_distribution", len(args["modules"]) * args["count"], lambda: programs.distribution(**args), args)

    def sensitivity(**args):
        return measured("profile_sensitivity", len(args["modules"]) * len(args["variants"]) * args["count"], lambda: programs.sensitivity(**args), args)

    def hardware(**args):
        refresh = args.pop("refresh")
        profiler = programs.torch()
        runtime = programs.adapter.hardware_identity(profiler)
        return measured("profile_hardware", args["warmup"] + 2 * args["repeats"], lambda: programs.adapter.hardware(profiler, **args),
                        {**args, "refresh": refresh, "runtime": runtime})

    def read_evidence(evidence, section, offset, limit, path=None):
        payload = read_returned_evidence(registry.host, evidence)
        values = payload["result"].get(section)
        sections = [key for key, value in payload["result"].items()
                    if key in EVIDENCE_SECTIONS and isinstance(value, list)]
        require(isinstance(values, list), f"证据没有该分页字段；可用字段：{sections}")
        path = [] if path is None else path
        # 只在已核验 JSON 内选择字段；不解释文件路径或执行表达式。
        for key in path:
            if isinstance(values, dict):
                require(key in values, f"证据子字段不存在：{key}；可用字段：{list(values)}")
                values = values[key]
            else:
                require(isinstance(values, list) and key.isdigit() and int(key) < len(values),
                        f"证据数组索引无效：{key}")
                values = values[int(key)]
        indexed = isinstance(values, list)
        values = values if indexed else [values]
        rows, size = [], 0
        for index in range(offset, min(offset + limit, len(values))):
            row = values[index]
            chars = len(canonical(row))
            if size + chars > 6000 and rows:
                break
            # 不截断一行的数值；过大的复合行可按明确的子字段继续读取。
            keys = list(row) if isinstance(row, dict) else []
            row_path = path + [str(index)] if indexed else path
            require(chars <= 6000,
                    f"第{index}行超过单页大小；请使用 path 选择其子字段。"
                    f"当前path={path}；行内字段={keys}。例如 path={row_path + keys[:1]}",
                    code="evidence_row_too_large", path="$.arguments.path")
            rows.append(row)
            size += chars
        next_offset = offset + len(rows)
        return {"rows": rows, "total": len(values), "evidence": evidence, "section": section,
                "path": path, "offset": offset,
                "next_offset": next_offset if next_offset < len(values) else None}

    definitions = [
        # 真实 LLM 必须知道既有计费规则，才能在调用前规划；这里只补描述，不改变计费。
        ("profile_structure", "按需提取输入形状、结构、量化位点、参数量和限定公式 MACs；modules=[] 表示全部；首次消耗 measurements=count，缓存命中为0", obj({"modules": MODULES, **SAMPLE_FIELDS}), structure),
        ("profile_distribution", "定向跨 token/channel 分布；明确 CLS/special/patch 子集；首次消耗 measurements=count×模块数，缓存命中为0", obj({
            "modules": arr(STR, minItems=1, maxItems=8, uniqueItems=True), "target": {"enum": ["weight", "activation"]},
            "views": arr({"enum": ["tensor", "token", "channel"]}, minItems=1, uniqueItems=True),
            "tokens": {"enum": ["all", "cls", "special", "patch"]}, "statistics": arr({"enum": STATISTICS}, minItems=1, uniqueItems=True),
            "histogram_bins": {"type": "integer", "minimum": 0, "maximum": 64},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8}, **SAMPLE_FIELDS}), distribution),
        ("profile_hardware", "实际设备上测 FP 算子/模块时间、干净延迟和内存，不表示量化加速；首次消耗 measurements=warmup+2×repeats（批次），缓存命中为0；换 refresh 重测", obj({
            "modules": MODULES, **SAMPLE_FIELDS, "warmup": {"type": "integer", "minimum": 1, "maximum": 10},
            "repeats": {"type": "integer", "minimum": 1, "maximum": 20}, "refresh": STR}), hardware),
        ("profile_sensitivity", "选定 Linear 的 W-only/A-only/联合 QDQ 局部敏感性；模型不变，不是任务精度；首次消耗 measurements=count×模块数×变体数，缓存命中为0", obj({
            "modules": arr(STR, minItems=1, maxItems=8, uniqueItems=True),
            "variants": arr(obj({"name": STR, "method": {"enum": ["rtn"]}, "weight": profile_spec_schema("weight"),
                                 "activation": profile_spec_schema("activation")}), minItems=1, maxItems=8), **SAMPLE_FIELDS}), sensitivity),
    ]
    for name, description, parameters, handler in definitions:
        registry.register(ToolDefinition(name, description, parameters, result_schema, ("profiling",), handler))
    registry.register(ToolDefinition("read_profile_evidence", "分页读取实测明细，不重新forward；每页明细至多6000字符（不含分页元数据），按next_offset继续，null表示结束。可选path用键名/数组索引选择大行内的子字段，默认空数组；例如读取rows第0行的joint_output用path=['0','joint_output']", obj({
        "evidence": STR, "section": {"enum": list(EVIDENCE_SECTIONS)},
        "offset": INT, "limit": {"type": "integer", "minimum": 1, "maximum": 32},
        "path": arr(STR, maxItems=16)}, ["evidence", "section", "offset", "limit"]),
        obj({"rows": arr({}), "total": INT, "evidence": STR, "section": STR, "path": arr(STR),
             "offset": INT, "next_offset": {"anyOf": [INT, {"type": "null"}]}}), ("profiling",), read_evidence))
