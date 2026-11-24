"""Task-owned immutable edits. A retry can reproduce the same source revision."""
import ast
from hashlib import sha256
from pathlib import Path
from core.cards import ContractError
from core.journal import atomic_json, read_json


def revision(directory, digest):
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ContractError('invalid source version')
    path = directory / (digest + '.json')
    if not path.is_file():
        raise ContractError('source revision unavailable in this task')
    source = read_json(path)['source']
    if sha256(source.encode()).hexdigest() != digest:
        raise ContractError('source revision content changed')
    return source


def edit(directory, args, allowed):
    if 'asset' in args:
        path = Path(args['asset']).resolve()
        if path.suffix != '.py' or not allowed.get(str(path)):
            raise ContractError('source edit base must be an authorized pinned Python file')
        if not args.get('base'):
            raise ContractError('edit_source.asset requires base from read_asset.sha256')
        if args['base'] != allowed[str(path)]:
            raise ContractError('edit_source.base differs from pinned file sha256; use read_asset edit_base')
        raw = path.read_bytes()
        source = raw.decode('utf-8')
        if sha256(raw).hexdigest() != args['base']:
            raise ContractError('source edit base changed')
    else:
        source = revision(directory, args['base']) if 'base' in args else ''
    old = args['old']
    if (source and not old) or source.count(old) != 1:
        raise ContractError('source edit needs one exact matching range')
    source = source.replace(old, args['new'], 1)
    digest = sha256(source.encode()).hexdigest()
    atomic_json(directory / (digest + '.json'), {'source': source})
    return {'revision': digest, 'characters': len(source), 'base': args.get('base')}


def locate(text, args):
    start = args.get('offset', 0)
    if 'symbol' in args:
        try:
            nodes = ast.walk(ast.parse(text))
            lines = text.splitlines(keepends=True)
            matches = sorted(n.lineno for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                             and n.name == args['symbol'])
        except SyntaxError as exc:
            raise ContractError('symbol lookup requires Python source') from exc
        offsets = [sum(map(len, lines[:n-1])) for n in matches]
        return next((i for i in offsets if i >= start), len(text))
    if 'find' in args:
        index = text.find(args['find'], start)
        return index if index >= 0 else len(text)
    return start


def asset_text_page(text, args, default_limit):
    start = locate(text, args)
    stop = start + args.get('limit', default_limit)
    page = {'offset': start, 'text': text[start:stop], 'total_characters': len(text),
            'next_offset': stop if stop < len(text) else None}
    if 'find' in args or 'symbol' in args:
        page['matched'] = start < len(text)
        if not page['matched']:
            page['read_asset'] = {k: args[k] for k in ('asset', 'snapshot_field', 'member') if k in args}
    return page


def engineering_context(context):
    """Keep the scientific request and current evidence. Older bodies stay addressable."""
    from difflib import unified_diff
    ctx = dict(context)
    memory = dict(ctx.get('memory', {}))
    cards = memory.get('cards', [])
    # Return exact references for historical cards; current review/failure feedback
    # remains in the task and bound feedback fields. Keep reviews for diff-based repair.
    def compact(card):
        if not card.get('id') or card.get('kind') in ('ReviewDecisionCard','EvidenceCard') or 'payload' not in card:
            return card
        body = card['payload']
        row = {k:card[k] for k in ('id','kind','content_sha256') if k in card}
        row['read_card'] = {'ref':card['id']}
        row['payload'] = {k:body[k] for k in ('status','experiment_ref','question','level','measurement_source') if k in body}
        provenance = body.get('provenance', {})
        row['payload'].update({k:provenance[k] for k in ('model_ref','training_declared','protocol_hash') if k in provenance})
        return row
    memory['cards'] = [compact(card) for card in cards]
    ctx['memory'] = memory
    task = ctx.get('task', {})
    history = task.get('implementation_history', [])
    history = history if isinstance(history, list) else []
    current = task.get('candidate', {})
    previous = next((row['candidate'] for row in reversed(history)
                     if row['candidate']['source_sha256'] != current.get('source_sha256')), None)
    if previous and isinstance(ctx.get('candidate_source'), str):
        path = Path(previous['path'])
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == previous['source_sha256']:
            delta = ''.join(unified_diff(path.read_text().splitlines(True), ctx['candidate_source'].splitlines(True),
                                        fromfile=previous['source_sha256'], tofile=current['source_sha256']))
            if len(delta) < len(ctx['candidate_source']):
                ctx['candidate_source'] = {'diff': delta, 'base_sha256': previous['source_sha256'],
                                           'source_sha256': current['source_sha256'], 'read_asset': {'asset': current['path']}}
    memory['asset_locations'] = {p:v for p,v in memory.get('asset_locations',{}).items()
                                 if isinstance(ctx.get(v.get('location','').removeprefix('context.')),str)}
    return ctx
