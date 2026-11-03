"""可追溯的公开文献读取。完整原文保留，LLM按页/分块继续读取。"""
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

from core.cards import ContractError
from core.journal import atomic_json


def fetch(url, directory, *, timeout=30):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise ContractError("只接受无凭据的公开HTTP(S)文献URL")
    if parsed.hostname in ("localhost", "127.0.0.1", "169.254.169.254"):
        raise ContractError("文献接口不访问本机/元数据服务")
    request = urllib.request.Request(url, headers={"User-Agent": "QAViT/2 scientific-literature-reader"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(32 * 1024 ** 2 + 1)
        if len(raw) > 32 * 1024 ** 2:
            raise ContractError("单份文献超过32MiB传输预算；保留URL并请求更明确资料")
        mime = response.headers.get("Content-Type", "")
        final_url = response.url
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".pdf" if raw.startswith(b"%PDF") else ".txt"
    original = directory / (digest + suffix)
    if not original.exists():
        original.write_bytes(raw)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ContractError("PDF原件已保存，但pypdf尚未安装；不得把PDF二进制当正文") from None
        pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages]
    else:
        pages = [raw.decode("utf-8", errors="replace")]
    record = {"url": url, "resolved_url": final_url, "mime": mime, "sha256": digest,
              "raw_asset": str(original.resolve()), "pages": pages, "source": "retrieved_public_source"}
    path = directory / (digest + ".json")
    atomic_json(path, record)
    return {"url": url, "sha256": digest, "asset": str(path.resolve()), "page_count": len(pages),
            "text_preview": pages[0][:6000], "presentation": "preview only; full source retained in asset"}


def search(query, directory, *, timeout=30):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query, "rows": 5})
    request = urllib.request.Request(url, headers={"User-Agent": "QAViT/2 scientific-literature-reader"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(4 * 1024 ** 2)
    data = json.loads(raw)
    records = [{"title": " ".join(row.get("title", [])), "doi": row.get("DOI"), "url": row.get("URL"),
                "published": row.get("published"), "links": row.get("link", []),
                "scope": "bibliographic metadata; not full-text verification"} for row in data["message"]["items"]]
    directory = Path(directory)
    path = directory / (hashlib.sha256(raw).hexdigest() + ".json")
    atomic_json(path, {"query": query, "url": url, "response": data})
    return {"records": records, "asset": str(path.resolve()), "source": "crossref_metadata"}
