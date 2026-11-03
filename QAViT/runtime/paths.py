"""项目根路径与依赖初始化；所有缓存和临时文件只写 EXP。"""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT.parent
QAVIT = ROOT / 'QAViT'
LIBRARY = QAVIT / 'tools/ptq_agent'


def bootstrap():
    for path in (QAVIT / 'tools', QAVIT / '.runtime_pkgs/common', QAVIT / '.runtime_pkgs/classification'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    tmp = ROOT / "EXP/cache/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    os.environ["TMPDIR"] = str(tmp)
    # 依赖库缓存也留在 EXP，且位于执行工具允许写入的临时目录内。
    for key, name in {
        'XDG_CACHE_HOME': 'xdg', 'MPLCONFIGDIR': 'matplotlib',
        'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
        'HF_HOME': 'huggingface', 'CUDA_CACHE_PATH': 'cuda',
        'NUMBA_CACHE_DIR': 'numba', 'TRITON_CACHE_DIR': 'triton',
    }.items():
        cache = tmp / name
        cache.mkdir(exist_ok=True)
        os.environ[key] = str(cache)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


bootstrap()
