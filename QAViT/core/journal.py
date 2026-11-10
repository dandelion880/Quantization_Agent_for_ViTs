"""单写者的文件提交日志。提交文件是事实源；索引可随时重建。

先落卡片，最后原子发布一个 commit。未被 commit 引用的孤立文件不算事实。
不依赖 NFS 上的 SQLite WAL，也不把多次 rename 称作跨文件事务。
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time

from .cards import ContractError, canonical, fingerprint, strict_loads, validate_card


def read_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical(value) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def file_lock(path, *, timeout=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ContractError("资源锁忙: " + path.name)
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        yield
    finally:
        fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


class ConcurrentCommit(ContractError):
    pass


def journal_read(method):
    """Reuse verification only within one host operation, including nested helpers."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.journal.read_session():
            return method(self, *args, **kwargs)
    return wrapped


class Journal:
    def __init__(self, root, run_id):
        self.root, self.run_id = Path(root).resolve(), run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self.commits, self.events, self.cards, self.keys = [], [], {}, {}
        self.memory_version = 0
        self.card_versions = {}
        self._read_cache = None
        self.refresh()

    @property
    def sequence(self):
        return len(self.commits)

    def card_path(self, ref):
        return self.root / "cards" / (hashlib.sha256(ref.encode()).hexdigest() + ".json")

    def card(self, ref, *, field=()):
        if ref not in self.cards:
            raise ContractError("未提交或未知卡片引用: " + str(ref))
        path = self.card_path(ref)
        signature = None
        if self._read_cache is not None:
            stat = path.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = self._read_cache.get(ref)
            if cached and cached[0] == signature:
                value = cached[1]
                for key in field:
                    value = value[key]
                return deepcopy(value)
        value = read_json(path)
        validate_card(value)
        if value["content_sha256"] != self.cards[ref]["sha256"]:
            raise ContractError("已提交卡片被改写: " + ref)
        if self._read_cache is not None:
            # The freshly parsed object is owned by this operation. Copy only
            # the requested return value, not a second full card for field reads.
            self._read_cache[ref] = (signature, value)
        if field:
            for key in field:
                value = value[key]
            return deepcopy(value)
        return deepcopy(value) if self._read_cache is not None else value

    @contextmanager
    def read_session(self):
        """A bounded operation cache; never a second fact source or permission cache.

        Every new operation revalidates file content; a changed inode/size/mtime/ctime
        also forces revalidation inside the operation. Returned values are independent.
        """
        with self._mutex:
            nested = self._read_cache is not None
            if not nested:
                self._read_cache = {}
            try:
                yield
            finally:
                if not nested:
                    self._read_cache = None

    def refresh(self):
        with self._mutex:
            directory = self.root / "commits"
            files = sorted(directory.glob("*.json")) if directory.exists() else []
            if len(files) < len(self.commits):
                raise ContractError("已提交日志被删除")
            for path in files[len(self.commits):]:
                commit = read_json(path)
                seq = len(self.commits) + 1
                if path.name != f"{seq:09d}.json" or commit["sequence"] != seq:
                    raise ContractError("提交序列存在缺口")
                expected = self.commits[-1]["sha256"] if self.commits else None
                if commit["previous_sha256"] != expected or commit["run_id"] != self.run_id:
                    raise ContractError("提交链或运行身份不匹配")
                if commit["sha256"] != fingerprint({k: v for k, v in commit.items() if k != "sha256"}):
                    raise ContractError("提交哈希损坏")
                if commit["key"] in self.keys:
                    raise ContractError("重复提交键")
                refs = set(self.cards) | {c["ref"] for c in commit["cards"]}
                for entry in commit["cards"]:
                    card = read_json(self.card_path(entry["ref"]))
                    validate_card(card)
                    if card["content_sha256"] != entry["sha256"] or card["run_id"] != self.run_id:
                        raise ContractError("提交的卡片身份不一致")
                    if set(card["refs"]) - refs:
                        raise ContractError("卡片引用缺失")
                    if card["kind"] != "TaskCard":
                        self.memory_version += 1
                        self.card_versions[card["id"]] = self.memory_version
                    self.cards[entry["ref"]] = dict(entry, sequence=seq, kind=card["kind"])
                    if card['kind'] == 'TaskCard':
                        self.cards[entry['ref']].update(task_kind=card['payload']['kind'], state=card['payload']['state'])
                self.commits.append(commit)
                self.keys[commit["key"]] = commit
                self.events.extend(dict(event, commit_sequence=seq) for event in commit["events"])
            return self

    def commit(self, key, *, cards=(), events=(), expected_sequence=None, fault=None):
        with self._mutex, file_lock(self.root / "commit.lock", timeout=10):
            self.refresh()
            if key in self.keys:
                return deepcopy(self.keys[key])
            if expected_sequence is not None and expected_sequence != self.sequence:
                raise ConcurrentCommit("提交前状态已变化；重新读取")
            available = set(self.cards) | {c["id"] for c in cards}
            if len({c["id"] for c in cards}) != len(cards):
                raise ContractError("同一提交重复卡片 ID")
            pending = {c["id"]: c for c in cards}
            def referenced(ref):
                return pending[ref] if ref in pending else self.card(ref)
            for card in cards:
                validate_card(card)
                if card["id"] in self.cards:
                    raise ContractError("已提交卡片不能再次发布；复用引用或相同幂等键")
                if card["run_id"] != self.run_id or set(card["refs"]) - available:
                    raise ContractError("卡片运行身份或引用不合法")
                if card["id"] in card["refs"]:
                    raise ContractError("卡片不能引用自身")
                previous = card.get("previous")
                if card["version"] > 1:
                    prior = self.card(previous)
                    if prior["logical_id"] != card["logical_id"] or prior["kind"] != card["kind"] or prior["version"] + 1 != card["version"]:
                        raise ContractError("卡片版本链不一致")
                elif previous:
                    raise ContractError("初始版本不能声称覆盖前一版本")
                payload = card["payload"]
                if card["kind"] == "EvidenceCard":
                    ref = payload["experiment_ref"]
                    if ref not in card["refs"] or referenced(ref)["kind"] != "ExperimentCard":
                        raise ContractError("原始证据必须引用同一已冻结 ExperimentCard")
                for ref in payload.get("evidence_refs", []):
                    if ref not in card["refs"] or referenced(ref)["kind"] != "EvidenceCard":
                        raise ContractError("科学判断的证据引用类型或封套引用不完整")
                if card["kind"] == "HypothesisCard":
                    for ref in card["parents"]:
                        if referenced(ref)["kind"] != "HypothesisCard":
                            raise ContractError("假设父关系不能指向其它卡片种类")
                path = self.card_path(card["id"])
                if path.exists():
                    if read_json(path) != card:
                        raise ContractError("不可覆盖已有卡片")
                else:
                    atomic_json(path, card)
            if fault:
                fault("cards_written")
            commit = {"sequence": self.sequence + 1, "run_id": self.run_id,
                      "key": key, "created_unix": time.time(),
                      "previous_sha256": self.commits[-1]["sha256"] if self.commits else None,
                      "cards": [{"ref": c["id"], "sha256": c["content_sha256"]} for c in cards],
                      "events": deepcopy(list(events))}
            commit["sha256"] = fingerprint(commit)
            atomic_json(self.root / "commits" / f"{commit['sequence']:09d}.json", commit)
            if fault:
                fault("commit_published")
            self.refresh()
            # 此索引只是便于人读；恢复永远核验提交日志。
            try:
                atomic_json(self.root / "checkpoint.json", {
                    "run_id": self.run_id, "sequence": self.sequence,
                    "commit_sha256": commit["sha256"], "memory_version": self.memory_version,
                    "card_count": len(self.cards)})
                self.index_error = None
            except OSError as exc:
                # commit 已落盘，此时不能向调用者伪报事务未完成而触发重复科研动作。
                # 索引不是事实源，下一次提交或恢复可重建；记录可见的内存警告。
                self.index_error = {"errno": exc.errno, "message": str(exc)}
            return deepcopy(commit)

    def find_cards(self, kind=None):
        return [self.card(ref) for ref, entry in self.cards.items() if kind is None or entry["kind"] == kind]

    def audit(self):
        # 从头验证链，包含已载入卡片；防止长进程漏掉事后篡改。
        fresh = Journal(self.root, self.run_id)
        return {"status": "passed", "commits": fresh.sequence, "cards": len(fresh.cards),
                "memory_version": fresh.memory_version, "head": fresh.commits[-1]["sha256"] if fresh.commits else None}
