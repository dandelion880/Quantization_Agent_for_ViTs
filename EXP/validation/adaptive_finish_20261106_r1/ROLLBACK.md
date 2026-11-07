# 本轮修改前的回退副本

`before_fix.tar.gz` 保存了本轮开始时的 220 个源码、提示词、配置和说明文件，包含当时尚未提交的改动。它不是 Git HEAD：三场辩论、Judger 并发、单实现链、预算改进等最新能力均在其中。`BEFORE.json` 保存逐文件 SHA256 和原源码身份 `16a91db97f9ff6456510becae97892dfd48aee6a5b07932c43d7a533bd1c00a1`。归档已逐文件校验，未改动数据、权重或历史运行。

优先采用配置级数值回退：独立实验保持默认 `gpu_count=1`；若还需串行，将新研究输入 `runtime.experiment_concurrency=1`。这不改变 LLM 通道、科学流程和预算模式。本轮真实检查通过，当前未全局停用这些能力。

如需恢复本轮开始的源码，先使本项目运行完全停止，避免运行中改写实现身份。以下从项目根执行；会先保存当前文件，新增的测试文件也移入备份，不删除研究产物。此步骤未自动执行。

```bash
python -B - <<'PY'
from pathlib import Path
import hashlib, json, shutil, tarfile
root = Path.cwd()
out = root / 'EXP/validation/adaptive_finish_20261106_r1'
before = json.loads((out / 'BEFORE.json').read_text())
archive = out / 'before_fix.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == before['archive_sha256']
saved = out / 'restore_saved_current'
saved.mkdir(exist_ok=False)
for name in before['files']:
    path = root / name
    if path.exists():
        target = saved / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
for name in json.loads((out / 'CHANGE_SCOPE.json').read_text())['added']:
    path = root / name
    if path.exists():
        target = saved / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
with tarfile.open(archive, 'r:gz') as source:
    assert {m.name for m in source.getmembers()} == set(before['files'])
    for member in source.getmembers():
        raw = source.extractfile(member).read()
        assert hashlib.sha256(raw).hexdigest() == before['files'][member.name]
        (root / member.name).write_bytes(raw)
print('Restored this round’s baseline; current files retained in', saved)
PY
```

恢复旧科研运行仍遵守既有源码干预与冻结资产校验，不重置预算或篡改包身份。新旧版本不应同时运行；本地归档和原始日志不纳入 Git 的大型产物提交。
