"""Bounded live role turns from one frozen request. No Journal or numerical dispatch."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.paths import QAVIT
from evaluation.common import EVALUATIONS, EvalError, confined, digest, output_directory, read_json, write_json
from evaluation.judge import evaluation_manifest
from agents.roles import role_spec, submission_schema
from agents.source_workspace import edit, revision, asset_text_page
from core.cards import ContractError, validate_card, validate_schema
from runtime.identity import code_identity
from runtime.model_router import call_once
from tools.safeguards import write_artifact, verify_artifact


def start_request(root, case):
    manifest = evaluation_manifest(root)
    if manifest['manifest_sha256'] != case['manifest_sha256']:
        raise EvalError('evaluation release differs from case')
    name = 'jobs/' + case['execution_id'] + '/request.json'
    row = next(r for r in manifest['source_files'] if r['path'] == name)
    raw = confined(root / row['object'], root / 'sources').read_bytes()
    if digest(raw) != row['sha256'] or row['sha256'] != case['request_sha256']:
        raise EvalError('request differs from its frozen object')
    import json
    request = json.loads(raw)
    if request['kind'] != 'llm' or request['execution_id'] != case['execution_id']:
        raise EvalError('expected a role-call breakpoint')
    if request['input_versions'] != case['input_versions'] or request['spec']['context']['read_view'] != case['read_view']:
        raise EvalError('breakpoint parent or read view differs from the frozen request')
    return request


def live_tool(call, spec, allowed, destination):
    name, args = call['name'], call['arguments']
    schema = next(t['parameters'] for t in spec['tools'] if t['name'] == name)
    validate_schema(args, schema, name)
    if name == 'edit_source':
        return edit(destination / 'revisions', args, allowed)
    if name == 'read_asset' and not {'member','snapshot_field'}.intersection(args):
        from agents.interaction import RoleInteraction
        path = RoleInteraction.asset_path(args['asset'])
        # Only unchanged, pinned research text. Other tools are visible choices,
        # but offline replay cannot simulate their scientific results.
        if str(path) not in allowed or path.suffix not in ('.md','.py'):
            raise NotImplementedError('asset not served by frozen-text replay')
        raw = path.read_bytes()
        if digest(raw) != allowed[str(path)]:
            raise NotImplementedError('file differs from the historical read permission')
        text = raw.decode('utf-8')
        result = {'asset':str(path),'sha256':digest(raw), **asset_text_page(text, args, RoleInteraction.PAGE_CHARACTERS)}
        if path.suffix == '.py' and spec['role'] == 'coding':
            result['edit_base'] = {'asset':str(path),'base':digest(raw)}
        return result
    raise NotImplementedError('action recorded, downstream result unmeasured: ' + name)


def replay(request, destination, turns):
    old = request['spec']
    allowed = request['read_permissions']
    spec = role_spec(old['role'],old['purpose'],old['context'],dict(old['settings']),messages=old['messages'])
    # Tool-channel replay, including Coding when native tools exist on this host.
    spec['settings']['coding_native_tools'] = False
    ctx = spec['context']
    parent = ctx['task'].get('revision_feedback', {}).get('previous_candidate')
    if parent and allowed.get(parent['path']) == parent['source_sha256']:
        ctx['previous_candidate_source_edit_base'] = {'asset': parent['path'], 'base': parent['source_sha256']}
    for key, item in ctx.items():
        if key.startswith('prior_implementation_') and isinstance(item,dict) and 'read_asset' in item:
            path = item['read_asset']['asset']
            if allowed.get(path) == item['sha256']:
                item['edit_base'] = {'asset':path, 'base':item['sha256']}
    trace = {'start':request['execution_id'],'parent_versions':request['input_versions'],
             'read_view':ctx['read_view'],'current_core':code_identity()['core_sha256'],
             'channel':'host tools','status':'turn_limit','steps':[]}
    begin = time.monotonic()
    try:
        for step in range(turns):
            job = destination / 'steps' / str(step+1)
            entry = {'job':str(job.relative_to(destination))}
            trace['steps'].append(entry)
            response = call_once(spec,job,time.time()+300)['response']
            write_json(job/'response.json', response)
            entry['usage'] = response['usage']
            if not response['tool_calls']:
                decision = response['decision']
                entry['action'] = decision['action']
                if decision['action'] == 'submit':
                    content = dict(decision['content'])
                    validate_schema(content, submission_schema(old['purpose']), old['purpose'])
                    if old['purpose'] == 'code':
                        if 'revision' in content:
                            content['source'] = revision(destination/'revisions',content.pop('revision'))
                        experiment = validate_card(ctx['original_request'])
                        artifact = write_artifact(destination, experiment, content, 'coding')
                        verify_artifact(artifact, experiment)
                        entry['artifact'] = artifact
                trace['status'] = 'submitted' if decision['action'] == 'submit' else 'clarification'
                break
            delivered = []
            for call in response['tool_calls']:
                try:
                    result = live_tool(call,spec,allowed,destination)
                except ContractError as exc:
                    result = {'error':str(exc)}
                delivered.append({'call_id':call['id'],'name':call['name'],'result':result})
            write_json(job/'delivered.json',delivered)
            ctx['tool_results'] = [*ctx.get('tool_results',[]), *delivered]
            spec['messages'].append({'role':'assistant','content':{'tool_calls':response['tool_calls']}})
    except NotImplementedError as exc:
        trace.update(status='unmeasured',reason=str(exc))
    except Exception as exc:
        trace.update(status='blocked',error=type(exc).__name__,reason=str(exc))
    finally:
        trace['elapsed_seconds'] = time.monotonic()-begin
        write_json(destination/'trace.json', trace)
    return trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('collection',type=Path)
    p.add_argument('case')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--held-out', action='store_true', help='explicitly run a reserved case for independent assessment')
    p.add_argument('--turns',type=int,choices=range(1,5),default=1)
    args = p.parse_args()
    collection = read_json(args.collection)
    case = collection['cases'][args.case]
    if (case['partition'] != 'development') != args.held_out:
        p.error('reserved cases require --held-out; development cases use the default mode')
    root = confined(EVALUATIONS / collection['evaluation'],EVALUATIONS)
    request = start_request(root,case)
    dest = confined(args.output,EVALUATIONS)
    if dest.parent != EVALUATIONS:
        p.error('use a fresh sibling evaluation directory')
    destination = output_directory(dest,create=True)
    result = replay(request,destination,args.turns)
    print(result['status'], destination/'trace.json')
    return int(result['status'] == 'blocked')


if __name__ == '__main__':
    sys.exit(main())
