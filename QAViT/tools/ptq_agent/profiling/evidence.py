"""测量证据的完整性检查；文件名命中不能替代依赖与内容核验。"""

from ..schemas import PTQError, fingerprint, require
from ..storage import read_json


def seal(payload):
    """校验和用于发现损坏，不是对外部反馈真实性的认证。"""
    return {**payload, "content_sha256": fingerprint(payload)}


def read_measurement(path, identity, *, arguments=None):
    try:
        payload = read_json(path)
    except PTQError as exc:
        raise PTQError("invalid_evidence", str(exc)) from exc
    require(isinstance(payload, dict), "测量证据必须为对象", code="invalid_evidence")
    content = {k: v for k, v in payload.items() if k != "content_sha256"}
    require(payload.get("content_sha256") == fingerprint(content),
            "证据内容校验失败；旧版无校验和证据不能作为新运行缓存", code="invalid_evidence")
    require(payload.get("identity") == identity, "证据模型/数据/实现身份不匹配", code="invalid_evidence")
    require(arguments is None or payload.get("arguments") == arguments,
            "证据测量参数不匹配", code="invalid_evidence")
    require(isinstance(payload.get("result"), dict), "证据缺少测量结果", code="invalid_evidence")
    return payload


def read_returned_evidence(host, reference):
    known = {a for row in host.state["tool_results"] if row["result"]["status"] == "success"
             for a in row["result"]["artifacts"] if a.startswith("evidence/")}
    require(reference in known, "只能读取本会话已成功返回的测量证据", code="invalid_evidence")
    path = (host.run_dir / reference).resolve()
    require(path.is_relative_to((host.run_dir / "evidence").resolve()),
            "证据路径越界", code="invalid_evidence")
    return read_measurement(path, host.state["identity"])


def measurement_index(host):
    """跨轮去重索引：校验实际文件，不从 LLM 自述推断测量已完成。"""
    index = {}
    for row in host.state["tool_results"]:
        if row["result"]["status"] != "success":
            continue
        for ref in row["result"]["artifacts"]:
            if ref.startswith("evidence/") and ref not in index:
                index[ref] = {"tool": row["result"]["name"],
                              "payload": read_returned_evidence(host, ref)}
    return index
