"""Coding-only native workspace, gated by an actual kernel sandbox probe."""
from pathlib import Path
import hashlib
import json
import os
import stat
from runtime import codex_cli as cli
from core.cards import ContractError, fingerprint

NATIVE_FEATURES = {'shell_tool', 'unified_exec', 'code_mode_host', 'code_mode'}


def profile(work, sources):
    rules = {':root':'deny', ':minimal':'read', str(work):'write', str(sources):'read',
             '/dev/nvidiactl':'deny', '/dev/nvidia-uvm':'deny'}
    rules.update({str(p):'deny' for p in Path('/dev').glob('nvidia[0-9]*')})
    filesystem = ','.join(json.dumps(k)+'='+json.dumps(v) for k,v in rules.items())
    return ['-c','default_permissions="qavit-coding"', '-c','permissions.qavit-coding.filesystem={'+filesystem+'}',
            '-c','permissions.qavit-coding.network.enabled=false']


def prepare(client, spec, job, deadline):
    import time
    task_id = spec['context']['task_id']
    workspace = job.parent.parent / 'code/native' / fingerprint(task_id)
    work, sources = workspace/'work', workspace/'sources'
    work.mkdir(parents=True, exist_ok=True)
    sources.mkdir(exist_ok=True)
    options = profile(work.resolve(),sources.resolve())
    marker = job/'native-private-probe'
    marker.touch(mode=0o600,exist_ok=True)
    # Commands run with exactly the same filesystem and network profile as the tool.
    probe = ['test ! -r "$1" && test ! -r /dev/nvidiactl && test ! -w "$2" && test -w .']
    args = [client.cli,'sandbox',*options,'--','/bin/sh','-c',*probe,'probe',str(marker.resolve()),str(sources.resolve())]
    code, _, stderr = cli.run(args,work,min(10,deadline-time.time()))
    if code:
        return None, {'status':'fallback','reason':stderr.strip()[:500] or 'filesystem probe failed'}
    # Copy only the bound implementation interfaces and task-owned parent Python files.
    from tools.safeguards import PUBLIC_INTERFACE_FILES
    from runtime.paths import QAVIT
    allowed = spec.get('native_read_permissions', {})
    imports = [QAVIT/'tools/ptq_agent/execution'/f for f in PUBLIC_INTERFACE_FILES]
    context = spec['context']['task']
    history = context.get('implementation_history', [])
    if isinstance(history,list):
        imports += [Path(r['candidate']['path']) for r in history]
    parent = context.get('revision_feedback',{}).get('previous_candidate')
    if parent:
        imports.append(Path(parent['path']))
    for src in imports:
        src = src.resolve()
        digest = allowed.get(str(src))
        if not digest:
            continue
        raw = src.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ContractError('native source changed: '+src.name)
        dest = sources/(digest+'-'+src.name)
        if not dest.exists():
            dest.write_bytes(raw)
        if dest.read_bytes() != raw:
            raise ContractError('staged native source differs from its frozen origin')
    return (work,options), {'status':'isolated','workspace':str(work),'sources':str(sources)}


def delivered(work):
    fd = os.open(work/'implementation.py',os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 2**20:
            raise ContractError('native delivery must be a regular task-owned Python file under 1 MiB')
        return stream.read(2**20+1).decode('utf-8')


def usage_total(reports):
    if not reports or any(not isinstance(row,dict) for row in reports):
        return None
    keys = set().union(*reports)
    if any(type(row.get(k)) is not int or row[k] < 0 for k in keys for row in reports):
        return None
    return {k:sum(row[k] for row in reports) for k in keys}

def interrupted_usage(path):
    reports = []
    if path.is_file():
        with path.open('rb') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if isinstance(event, dict) and event.get('type') == 'turn.completed':
                    reports.append(event.get('usage'))
    return {'completed_turn_usage': usage_total(reports), 'reported_turns': len(reports),
            'interrupted_turn_usage_unknown': True}
