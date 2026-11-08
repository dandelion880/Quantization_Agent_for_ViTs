"""Regenerate the standalone report's literal tree without opening artifacts."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

BASE = Path(__file__).parent.resolve()
ROOT = BASE.parents[2]
REPORT = BASE / 'PROJECT_RESOURCE_INVENTORY.md'
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
            '.cache', '.venv', 'venv', '.runtime_pkgs'}
RAW = {'jobs', 'cards', 'commits', 'contexts', 'direct_calls', 'scripted_requests',
       'tools', 'trials', 'evidence', 'models', 'model', 'checkpoints', 'weights'}
COPIES = {'before', 'backup', 'before_source', 'source_before', 'source_snapshot',
          'snapshots', 'originals', 'original', 'historical_numeric', 'rollback_source'}
BINARY = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.gguf', '.npy', '.npz',
          '.pkl', '.pickle', '.onnx', '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z',
          '.so', '.pyc', '.png', '.jpg', '.jpeg', '.webp', '.mp4'}


def main():
    manifest = {'generated_at': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT),
                'files': [], 'directories': [], 'omitted': [], 'symlinks': []}
    def visit(path, prefix=''):
        entries, skipped = [], []
        def order(child):
            if path == ROOT:
                return (0, {'QAViT': 0, 'EXP': 1, 'reference': 2}.get(child.name, 3), child.name.lower())
            return (not child.is_dir(), 0, child.name.lower())
        for child in sorted(path.iterdir(), key=order):
            rel = child.relative_to(ROOT)
            in_exp = rel.parts[0] == 'EXP'
            if child.is_symlink():
                manifest['symlinks'].append(str(rel))
                entries.append((child, ' [符号链接；不追入]', False))
                continue
            reason = None
            if child.is_dir():
                if child.name in EXCLUDED or rel == Path('EXP/cache'):
                    reason = '版本库、缓存、第三方安装或虚拟环境'
                elif in_exp and child.name in RAW:
                    reason = '批量实验运行产物'
                elif in_exp and (child.name in COPIES or child.name.startswith('before_')):
                    reason = '历史实现副本/快照'
                if reason:
                    manifest['omitted'].append({'path': str(rel), 'kind': 'subtree', 'reason': reason})
                    entries.append((child, ' [省略：' + reason + ']', False))
                else:
                    manifest['directories'].append(str(rel))
                    entries.append((child, '', True))
            else:
                size = child.stat().st_size
                if child.suffix.lower() in BINARY:
                    reason = '二进制/模型/归档产物'
                elif child.suffix in {'.log', '.out', '.err', '.pid'} or '.log.' in child.name:
                    reason = '运行日志/进程痕迹'
                elif in_exp and size > 1024 * 1024:
                    reason = '大于1MiB的实验结果/快照'
                if reason:
                    manifest['omitted'].append({'path': str(rel), 'kind': 'file', 'bytes': size, 'reason': reason})
                    skipped.append(reason)
                else:
                    manifest['files'].append({'path': str(rel), 'bytes': size})
                    entries.append((child, '', False))
        if skipped:
            note = '；'.join(f'{reason} {count} 项' for reason, count in sorted(Counter(skipped).items()))
            entries.append((None, '[省略文件：' + note + ']', False))
        lines = []
        for index, (child, label, recurse) in enumerate(entries):
            last = index == len(entries) - 1
            name = (child.name + ('/' if child.is_dir() else '') if child else '')
            lines.append(prefix + ('└── ' if last else '├── ') + name + label)
            if recurse:
                lines.extend(visit(child, prefix + ('    ' if last else '│   ')))
        return lines
    lines = [ROOT.name + '/', *visit(ROOT)]
    manifest['counts'] = {k: len(manifest[k]) for k in ('files', 'directories', 'omitted', 'symlinks')}
    summary = ('扫描时间：' + manifest['generated_at'] + '；逐项列出 ' + str(len(manifest['files']))
               + ' 个文件、' + str(len(manifest['directories'])) + ' 个展开目录。省略项见规则及 manifest。\n\n')
    original = REPORT.read_text()
    left, rest = original.split('<!-- INVENTORY_START -->', 1)
    _, right = rest.split('<!-- INVENTORY_END -->', 1)
    REPORT.write_text(left + '<!-- INVENTORY_START -->\n' + summary + '```text\n' + '\n'.join(lines)
                      + '\n```\n<!-- INVENTORY_END -->' + right)
    (BASE / 'inventory_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
