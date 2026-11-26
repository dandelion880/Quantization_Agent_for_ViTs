"""Versioned procedure packages. The Journal owns Recorder intent and each run's read view.
A shared release file is the publication point; unfinished package files grant no visibility.
"""
from hashlib import sha256
import json
from pathlib import Path
import re

import yaml

from core.cards import ContractError, canonical, fingerprint
from core.journal import atomic_bytes, atomic_json, file_lock, read_json
from core.memory import lexical_tokens
from runtime.paths import ROOT


SHARED_ROOT = ROOT / 'EXP/skills'
NAME = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*')
HASH = re.compile(r'[0-9a-f]{64}')


def metadata(text):
    match = re.match(r'\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)', text, re.S)
    if not match:
        raise ContractError('Missing Skill YAML header')
    try:
        return yaml.safe_load(match[1])
    except yaml.YAMLError as exc:
        raise ContractError('Invalid Skill frontmatter') from exc


def digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


class SkillLibrary:
    def __init__(self, journal):
        self.journal = journal
        self.root = SHARED_ROOT

    def package(self, name, files, *, parent=None, retired=False):
        info = metadata(files['SKILL.md'])
        if info['name'] != name:
            raise ContractError('Skill package name differs from frontmatter')
        manifest = {'name': name, 'description': info['description'], 'supersedes': parent,
                    'retired': retired, 'files': {k: digest(v) for k, v in files.items()}}
        version = fingerprint(manifest)
        folder = self.root / 'versions' / version
        if not (folder / 'manifest.json').is_file():
            for member, body in files.items():
                atomic_bytes(folder / name / member, body.encode())
            atomic_json(folder / 'manifest.json', manifest)
        return version

    def manifest(self, version):
        if not HASH.fullmatch(version):
            raise ContractError('Invalid Skill version')
        value = read_json(self.root / 'versions' / version / 'manifest.json')
        if fingerprint(value) != version:
            raise ContractError('Skill manifest hash mismatch')
        return value

    def member(self, version, member):
        manifest = self.manifest(version)
        if member not in manifest['files']:
            raise ContractError('Unknown member in this Skill version')
        data = (self.root / 'versions' / version / manifest['name'] / member).read_bytes()
        if sha256(data).hexdigest() != manifest['files'][member]:
            raise ContractError('Skill member hash mismatch')
        return data.decode('utf8')

    def releases(self):
        return [read_json(path) for path in (self.root / 'releases').glob('*.json')]

    @staticmethod
    def heads(manual, releases):
        heads = dict(manual)
        conflicts = {}
        # Explicit ancestry only. The shared lock permits one successor per base.
        children = {(r['name'], r['base_sha256']): r['version'] for r in releases}
        for name in sorted({r['name'] for r in releases}):
            tip = heads.get(name)
            while (name, tip) in children:
                tip = children[name, tip]
            if tip:
                heads[name] = tip
            other = {r['version'] for r in releases if r['name'] == name and
                     (name, r['version']) not in children and r['version'] != tip}
            if other:
                conflicts[name] = sorted(other)
        return heads, conflicts

    def initialize(self, source):
        if any(e['type'] == 'skill_library_opened' for e in self.journal.events):
            return
        groups = {}
        for path, text in source.items():
            parts = Path(path).parts
            if parts[:2] == ('docs', 'skills') and len(parts) >= 4:
                groups.setdefault(parts[2], {})['/'.join(parts[3:])] = text
        with file_lock(self.root / 'publish.lock', timeout=10):
            manual = {name: self.package(name, files) for name, files in groups.items() if 'SKILL.md' in files}
            heads, conflicts = self.heads(manual, self.releases())
            self.journal.commit('skills/open', events=[{'type':'skill_library_opened',
                'manual':manual, 'heads':heads, 'conflicts':conflicts}])

    def view(self, sequence):
        heads, conflicts = {}, {}
        for event in self.journal.events:
            if event['commit_sequence'] > sequence:
                break
            if event['type'] == 'skill_library_opened':
                heads, conflicts = dict(event['heads']), event['conflicts']
            elif event['type'] == 'skill_update_result' and event['status'] == 'published':
                heads[event['name']] = event['version']
        return heads, conflicts

    def catalog(self, sequence, query=None):
        heads, conflicts = self.view(sequence)
        terms = lexical_tokens(query or '')
        rows = []
        for name, version in heads.items():
            info = self.manifest(version)
            if info['retired'] and query is not None:
                continue
            description = info['description']
            score = len(terms & lexical_tokens(name.replace('-', ' ') + ' ' + description))
            rows.append((-score, name, description, version, info['retired']))
        rows.sort()
        if query is not None:
            rows = rows[:7]
        lines = []
        for _, name, description, version, retired in rows:
            if query is None:
                lines.append(f'{name} {version} {"retired" if retired else "active"} {description}')
            else:
                line=f'skills/{name}/SKILL.md — {description[:150]}'
                if sum(len(item)+1 for item in lines)+len(line)>3600:
                    break
                lines.append(line)
        if query is None:
            lines += [f'{name}@{v} conflicting lineage; exact read only' for name in conflicts for v in conflicts[name]]
            return '# QAViT Scientific Skill Catalog\n' + '\n'.join(lines)
        return '\n'.join([*lines, '完整目录及版本：read_asset skills/INDEX.md（find 或 offset/limit）。']) if lines else ''

    def read(self, asset, sequence):
        if asset == 'skills/INDEX.md':
            return asset, self.catalog(sequence)
        parts = asset.split('/', 2)
        if len(parts) != 3:
            raise ContractError('Use skills/<name>/SKILL.md or skills/INDEX.md')
        name, _, requested = parts[1].partition('@')
        heads, conflicts = self.view(sequence)
        version = heads.get(name)
        if requested:
            for tip in [version, *conflicts.get(name, [])]:
                while tip and tip != requested:
                    tip = self.manifest(tip)['supersedes']
                if tip:
                    version = tip
                    break
            else:
                raise ContractError('Skill version outside the call snapshot')
        if not version:
            raise ContractError('Unknown Skill in this snapshot')
        info = self.manifest(version)
        if info['retired'] and not requested:
            raise ContractError('Retired Skill; use its explicit version')
        return f'skills/{name}@{version}/{parts[2]}', self.member(version, parts[2])

    def prepare(self, submissions, batch, sequence, resolve, field_value):
        rows = []
        heads, _ = self.view(sequence)
        for draft in submissions if isinstance(submissions, list) else [submissions]:
            row = {'submission': draft}
            try:
                if not isinstance(draft, dict):
                    raise ContractError('Skill update must be an object')
                name = draft.get('name')
                if not isinstance(name, str) or len(name) > 64 or not NAME.fullmatch(name):
                    raise ContractError('Invalid Skill name')
                action = draft.get('action')
                if action not in ('create', 'revise', 'retire'):
                    raise ContractError('Unknown Skill action')
                base = draft.get('base_sha256')
                if (action == 'create') != (base is None) or heads.get(name) != base:
                    raise ContractError('Skill name exists or base_sha256 differs from the visible version')
                fields = ['rationale'] if action == 'retire' else ['rationale','description','procedure','scope','counterexamples']
                if any(not isinstance(draft.get(k), str) or not draft[k].strip() for k in fields):
                    raise ContractError('Missing Skill text: ' + ', '.join(fields))
                if action != 'retire' and len(draft['description']) > 1024:
                    raise ContractError('Skill description exceeds 1024 characters')
                sources = draft.get('sources')
                if not isinstance(sources, list) or not sources or len(sources) > 12:
                    raise ContractError('Bind 1–12 compact source fields from this Recorder batch')
                attachments = []
                for source in sources:
                    if not isinstance(source, dict) or not isinstance(source.get('ref'), str):
                        raise ContractError('Skill source needs a card ref and field')
                    card = resolve(source['ref'])
                    field = source['field']
                    if (card['id'] not in batch['source_refs'] or not isinstance(field, str)
                            or not field.startswith('/payload/') or len(field) > 500):
                        raise ContractError('Source outside Recorder batch or missing exact payload pointer')
                    if card['kind'] == 'TaskCard' and field.split('/')[2] not in ('error','state','kind','purpose'):
                        raise ContractError('Engineering source must identify state or failure')
                    value = field_value(card, field)
                    if len(canonical(value).encode()) > 4096:
                        raise ContractError('Choose a narrower source field for the portable attachment')
                    attachments.append({'origin_run':card['run_id'], 'card_id':card['id'], 'kind':card['kind'],
                        'card_sha256':card['content_sha256'], 'field':field, 'value':value,
                        'value_sha256':fingerprint(value), **{k:card['payload']['provenance'][k] for k in
                            ('protocol_hash','model_ref') if k in card['payload'].get('provenance', {})},
                        **{k:card['payload'][k] for k in
                            ('status','measurement_source','execution_id') if k in card['payload']}})
                row.update(status='queued', sources=attachments)
            except (ContractError, KeyError, TypeError, IndexError, OSError) as exc:
                row.update(status='invalid', issue=str(exc))
            rows.append(row)
        return rows

    def publish(self, meta, index, row, manual):
        draft = row['submission']
        name, base = draft['name'], draft.get('base_sha256')
        operation = fingerprint([meta['content_sha256'], index])
        receipt = self.root / 'releases' / (operation + '.json')
        with file_lock(self.root / 'publish.lock', timeout=10):
            if receipt.is_file():
                return read_json(receipt)
            heads, conflicts = self.heads(manual, self.releases())
            if heads.get(name) != base or name in conflicts:
                return {'status':'conflict', 'name':name, 'issue':'Shared head diverged from submitted base'}
            retired = draft['action'] == 'retire'
            files = {k:self.member(base, k) for k in self.manifest(base)['files']} if base else {}
            origin = {'origin_run':meta['run_id'], 'metareview_ref':meta['id'],
                      'metareview_sha256':meta['content_sha256'], 'index':index,
                      'rationale':draft['rationale'], 'sources':row['sources']}
            files['references/origin.json'] = canonical(origin) + '\n'
            if not retired:
                description = json.dumps(draft['description'], ensure_ascii=False)
                files['SKILL.md'] = (f'---\nname: {name}\ndescription: {description}\nmetadata:\n'
                    f'  origin: recorder\n---\n\n# {name}\n\n{draft["scope"]}\n\n{draft["procedure"]}\n\n'
                    f'## 反例与边界\n\n{draft["counterexamples"]}\n\n[来源与修订依据](references/origin.json)\n')
            version = self.package(name, files, parent=base, retired=retired)
            result = {'status':'published', 'name':name, 'version':version, 'base_sha256':base,
                      'operation':operation, 'origin_run':meta['run_id'], 'metareview_ref':meta['id']}
            atomic_json(receipt, result)
            return result

    def reconcile(self):
        completed = {e['key'] for e in self.journal.events if e['type'] == 'skill_update_result'}
        opened = next((e for e in self.journal.events if e['type'] == 'skill_library_opened'), None)
        if not opened:
            return
        for event in list(self.journal.events):
            if event['type'] != 'recorder_batch_completed' or not event.get('skill_updates_queued'):
                continue
            ref = event['metareview_ref']
            if all(f'skills/result/{ref}/{i}' in completed for i in event.get('skill_update_indices', [])) and 'skill_update_indices' in event:
                continue
            meta = self.journal.card(ref)
            for index, row in enumerate(meta['payload']['skill_updates']):
                key = f'skills/result/{ref}/{index}'
                if row['status'] != 'queued' or key in completed:
                    continue
                try:
                    result = self.publish(meta, index, row, opened['manual'])
                except (OSError, ContractError) as exc:
                    # Keep the committed outbox pending; retry on restart or next checkpoint.
                    self.journal.commit(key + '/deferred/' + fingerprint(str(exc)), events=[
                        {'type':'skill_update_deferred', 'key':key, 'metareview_ref':ref, 'index':index, 'issue':str(exc)}])
                    continue
                self.journal.commit(key, events=[{'type':'skill_update_result', 'key':key,
                    'index':index, 'metareview_ref':ref, **result}])
