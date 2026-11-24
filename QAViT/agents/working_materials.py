"""Within-task delivery views. Exact cards, immutable tool results and asset permissions remain authoritative."""
from core.cards import canonical

RESEARCH = ('focus', 'generate', 'select', 'plan_l3')
READ_TOOLS = ('read_card', 'read_asset', 'retrieve_memory', 'read_topic', 'read_method', 'read_inspiration')


def material_selection(task, values):
    """A fresh read batch survives one continuation. Explicit pins survive later calls."""
    pins = set()
    recent_labels = {}
    for row, child, _ in values:
        label = row.get('call_id', child['inputs'].get('call_id'))
        if label and child.get('task_id'):
            recent_labels[label] = child['task_id']
        if 'active_materials' in child['inputs']:
            pins = {recent_labels.get(key, key) for key in child['inputs']['active_materials']}
    messages = task.get('conversation') or []
    last = messages[-1].get('content') if messages else None
    latest = {c['id'] for c in last.get('tool_calls', [])} if isinstance(last, dict) else set()
    batch = set()
    if latest:
        # Tool call labels are only unique inside one response. A later call may use '0' again.
        for i in reversed(range(len(values))):
            row, child, _ = values[i]
            label = row.get('call_id', child['inputs'].get('call_id'))
            if label in latest:
                batch.add(i)
                latest.remove(label)
            if not latest:
                break
    elif values:
        batch.add(len(values) - 1)
    purpose = task.get('inputs', {}).get('purpose')
    if purpose and purpose not in RESEARCH:
        return set(range(len(values))), pins
    recent_trials = [i for i, (_,child,_) in enumerate(values) if child['kind'] == 'experiment_request'][-2:]
    def page(result):
        if isinstance(result, dict) and 'text' in result and result.get('sha256'):
            return (result.get('asset') or result.get('ref'), result.get('member'), result.get('field'), result['sha256'])
    open_pages = set()
    keep = set()
    for i, (row, child, result) in enumerate(values):
        ids = {row.get('call_id'), child.get('task_id'), child.get('task_card')} - {None}
        bound = {result.get(k) for k in ('ref','id') if isinstance(result.get(k), str)} if isinstance(result, dict) else set()
        if i in batch or ids & pins or pins.intersection([*child.get('outputs', []), *bound]):
            keep.add(i)
            if page(result):
                open_pages.add(page(result))
        elif child['kind'] == 'experiment_request':
            if i in recent_trials or child.get('state') != 'completed' or (isinstance(result, dict)
                    and result.get('status') in ('failed','blocked','engineering_blocked')):
                keep.add(i)
        elif child.get('state') != 'completed' or child['inputs'].get('name') not in READ_TOOLS:
            keep.add(i)
    keep.update(i for i, (_,_,result) in enumerate(values) if page(result) in open_pages)
    return keep, pins


def research_card(view):
    """Keep the measured outcome and experimental conditions beside their exact original."""
    p = view.get('payload', {})
    ref = view['id']
    def source(key):
        return {'read_card': {'ref': ref, 'field': '/payload/' + key}, 'delivery': 'on_demand'}
    if view.get('kind') == 'ExperimentCard':
        e = p.get('execution', {})
        if len(canonical(e)) > 2000:
            p['execution'] = {k: e[k] for k in ('service','tool','method','site','calibration','evaluation',
                'evaluations','resolved_choices','source_ref','parent_ref','accept_trial','training','seed') if k in e}
            for k in ('arguments', 'parameters'):
                if k in e:
                    p['execution'][k] = e[k] if len(canonical(e[k])) <= 1600 else source('execution/' + k)
            p['execution_source'] = source('execution')
        if 'original_request' in p:
            p['original_request'] = source('original_request')
    if view.get('kind') == 'EvidenceCard':
        for key in ('provenance', 'artifacts', 'resources'):
            if len(canonical(p.get(key))) > 1000:
                val = p[key]
                p[key] = source(key)
                if key == 'provenance':
                    p[key].update({k: val[k] for k in ('model_ref','training_declared','execution_id',
                                  'limitations','warnings','failure','error') if k in val})
        obs = p.get('observations', {})
        if isinstance(obs, dict) and isinstance(obs.get('coverage'), dict) and len(canonical(obs['coverage'])) > 1400:
            obs['coverage'] = source('observations/coverage')
    return view


def research_recovery(context):
    """Keep exact code/config handles and failures; recovery tools expand whole source records."""
    for key in ('research_topic', 'inspiration_continuation'):
        recovery = context.get(key) or {}
        for entry in recovery.get('implementation_history', []):
            h = entry.get('host_source') or {}
            if h:
                entry['host_source'] = {k:h[k] for k in ('source_version','source_archive_sha256','source_snapshot') if k in h}
            entry.pop('meaning', None)
    # Identical values can have several independent consumers (topic/inspiration).
    # Only point to bodies present in this very delivery.
    seen = {}
    def share(value, location):
        if isinstance(value, dict):
            for k,v in list(value.items()):
                if k in ('host_source','candidate','research_note','configuration') and len(canonical(v)) > 120:
                    signature = canonical(v)
                    if signature in seen:
                        value[k] = {'body_location':seen[signature]}
                    else:
                        seen[signature] = location + '.' + k
                else:
                    share(v,location+'.'+k)
        if isinstance(value,list):
            for i,v in enumerate(value):
                share(v,location+'['+str(i)+']')
    for key in ('inspiration_continuation','research_topic'):
        share(context.get(key), 'context.'+key)
