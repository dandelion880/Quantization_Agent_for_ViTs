"""Offline package/link/hash audit; developer evidence, never a production gate."""
from pathlib import Path
import hashlib
import json
import re
import sys
import unicodedata
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'QAViT'))
from runtime.paths import QAVIT
from runtime.identity import code_identity, code_snapshot
from tests.audit_knowledge import audit

BASE = Path(__file__).resolve().parent
PACKAGE = BASE / 'source_package/qavit_skill_system_20261116'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def slug(text):
    text = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', text).replace('`', '').lower()
    return ''.join(c for c in text if c in ('-', '_', ' ') or unicodedata.category(c)[0] in ('L', 'N')).replace(' ', '-')


def anchors(path):
    found, counts = set(), {}
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*$', path.read_text(), re.M):
        key = slug(title)
        occurrence = counts.get(key, 0)
        counts[key] = occurrence + 1
        found.add(key + ('-' + str(occurrence) if occurrence else ''))
    return found


def main():
    errors, files, links = [], [], []
    manifest = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        p = PACKAGE / row['path']
        if sha(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            errors.append('package mismatch: ' + row['path'])
    identity = code_identity()
    snapshot = code_snapshot(identity)
    paths = sorted([*(QAVIT / 'docs/skills').rglob('*.md'), *(QAVIT / 'knowledge/research-skills').rglob('*.md'),
                    QAVIT / 'docs/SCIENTIFIC_SKILLS_COMPATIBILITY.md'])
    for path in paths:
        name = str(path.relative_to(QAVIT))
        if snapshot.get(name) != path.read_text() or identity['core_files'].get(name) != sha(path):
            errors.append('snapshot mismatch: ' + name)
        files.append({'path': name, 'sha256': sha(path), 'snapshot_matches': snapshot.get(name) == path.read_text()})
        body = re.sub(r'```[\s\S]*?```|`[^`\n]*`', '', path.read_text())
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if '://' in target:
                continue  # Source links preserved, external originals not re-verified here.
            relative, _, anchor = target.partition('#')
            dest = (path.parent / relative).resolve() if relative else path
            if not dest.is_file() or (anchor and anchor not in anchors(dest)):
                errors.append(name + ': invalid link/anchor ' + target)
            links.append({'from': name, 'target': target})
    mapping = json.loads((BASE / 'integration_map.json').read_text())
    for row in mapping['files']:
        p = QAVIT.parent / row['installed']
        row['installed_sha256'] = sha(p)
        row['content_note'] = ('relative links adapted; frozen propagation API link retained' if row['source'].endswith('error-propagation-and-intervention.md')
            else 'relative links adapted; current training permission documentation linked' if row['source'].endswith('reference-construction.md')
            else 'relative links adapted; package-author provenance wording clarified' if row['source'].endswith('ptq4dino.md')
            else 'relative links adapted where needed; research meaning preserved')
    (BASE / 'integration_map.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    navigation = audit()
    errors += navigation['errors']
    result = {'status': 'passed' if not errors else 'failed', 'errors': errors,
        'package_manifest_files_verified': len(manifest['files']), 'installed_markdown_snapshot_files': len(files),
        'local_links_and_anchors_checked': len(links), 'external_originals_reverified': False,
        'navigation': navigation, 'source_core_sha256': identity['core_sha256'], 'files': files, 'links': links}
    (BASE / 'asset_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'links')}, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
