"""Offline knowledge/link consistency check and Skill index regeneration; no LLM."""
import argparse
import json
from pathlib import Path
import re
import sys

QAVIT = Path(__file__).resolve().parents[1]
SKILLS = QAVIT / 'docs/skills'
START, END = '<!-- SKILL_CATALOG_START -->', '<!-- SKILL_CATALOG_END -->'


def skill_table(root=SKILLS):
    rows = ['| Skill | 何时使用 / 何时不需要 |', '|---|---|']
    for path in sorted(root.glob('*/SKILL.md')):
        metadata = path.read_text(encoding='utf-8').split('---', 2)[1]
        name = re.search(r'^name: ([a-z0-9-]+)$', metadata, re.M).group(1)
        if name != path.parent.name:
            raise ValueError(f'Skill name 与目录不同: {path}')
        # The pack uses plain name and a folded multiline description (>-).
        block = re.search(r'^description: >-\n((?:  [^\n]*\n?)+)', metadata, re.M).group(1)
        description = ' '.join(line.strip() for line in block.splitlines()).replace('|', '\\|')
        rows.append(f'| [{name}]({name}/SKILL.md) | {description} |')
    return '\n'.join(rows)


def generated_catalog(root=SKILLS):
    left, rest = (root / 'INDEX.md').read_text(encoding='utf-8').split(START, 1)
    _, right = rest.split(END, 1)
    return left + START + '\n' + skill_table(root) + '\n' + END + right


def audit():
    errors = []
    if (SKILLS / 'INDEX.md').read_text(encoding='utf-8') != generated_catalog():
        errors.append('Skill INDEX 与 frontmatter 不一致；运行 --write-skill-index')
    # Current navigation and knowledge links; historical EXP is deliberately not rewritten.
    pages = [QAVIT.parent / name for name in ('README.md', 'STRUCTURE.md')]
    pages += [QAVIT / 'README.md', *QAVIT.joinpath('knowledge').rglob('*.md'),
              *QAVIT.joinpath('docs').rglob('*.md')]
    for page in pages:
        body = re.sub(r'(?ms)^```.*?^```[^\n]*$', '', page.read_text(encoding='utf-8'))
        body = re.sub(r'`[^`\n]*`', '', body)
        for link in re.findall(r'\]\(([^)]+)\)', body):
            if '://' in link or link.startswith('#'):
                continue
            target = link.split('#', 1)[0]
            if not (page.parent / target).exists():
                errors.append(f'{page.relative_to(QAVIT.parent)}: missing {link}')
    index = json.loads((QAVIT / 'docs/methods/index.json').read_text())
    for row in index['methods']:
        if not (QAVIT / 'docs/methods' / row['document']).is_file():
            errors.append('missing method document: ' + row['id'])
    return {'status': 'failed' if errors else 'passed', 'errors': errors,
            'skill_count': len(list(SKILLS.glob('*/SKILL.md'))),
            'method_count': len(index['methods']), 'checked_pages': len(pages)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-skill-index', action='store_true')
    args = parser.parse_args()
    if args.write_skill_index:
        (SKILLS / 'INDEX.md').write_text(generated_catalog(), encoding='utf-8')
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result['status'] == 'passed' else 1)
