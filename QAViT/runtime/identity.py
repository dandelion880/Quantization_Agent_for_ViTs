"""递归源码身份与可恢复快照；不扫描 EXP、依赖安装目录或缓存。

大型 imagenet_manifest.json 是只读数据输入，实际选择的文件由
experiments.resolve_execution/verify_assets 逐个冻结并复核 SHA256。
第三方安装版本见 docs/runtime_dependencies.json；模型构造源码在 tools 内参与身份。
"""
import hashlib
import os
from pathlib import Path
from runtime.paths import QAVIT
from core.cards import ContractError, fingerprint
from core.journal import file_sha256


def source_files(root=None):
    """所有自有 Python、Prompt、配置模板与长期接口/方法说明，支持任意子目录。"""
    root = QAVIT if root is None else Path(root)
    paths = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [name for name in subdirs if not name.startswith('.') and name != '__pycache__']
        for name in files:
            path = Path(directory) / name
            if name.startswith('.') or (Path(directory) == root / 'configs/assets' and name == 'imagenet_manifest.json'):
                continue
            if path.suffix in ('.py', '.md', '.json', '.txt') or name == 'LICENSE':
                paths.append(path)
    return sorted(paths)


def code_identity():
    files = {str(p.relative_to(QAVIT)): file_sha256(p) for p in source_files()}
    library = {n: sha for n, sha in files.items() if n.startswith('tools/ptq_agent/')}
    own = {n: sha for n, sha in files.items() if n not in library}
    return {'core_sha256': fingerprint(files), 'numerical_library_sha256': fingerprint(library),
            'core_files': own, 'numerical_files': library}


def measurement_implementation():
    identity = code_identity()
    # 不手写移动文件清单；后续新增 worker、数值/模型子目录自动纳入。
    files = {n: sha for n, sha in identity['core_files'].items()
             if n.startswith(('tools/', 'runtime/', 'configs/')) or n in ('core/cards.py', 'core/journal.py')}
    return {'files': files, 'numerical_library_sha256': identity['numerical_library_sha256']}


def code_snapshot(identity):
    result = {}
    for name, expected in {**identity['core_files'], **identity['numerical_files']}.items():
        raw = (QAVIT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ContractError('保存运行源码快照期间实现发生变化')
        result[name] = raw.decode('utf-8')
    return result
