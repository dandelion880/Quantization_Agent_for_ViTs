"""Compare exact stdin byte prefixes on adjacent real Coding calls, without billing another turn."""
from os.path import commonprefix
from pathlib import Path
import json
from runtime.model_router import transport_input
from core.cards import canonical
from core.journal import atomic_json, read_json
out = Path(__file__).resolve().parent
rows = []
for job in sorted((out/'engineering_matched/jobs').glob('*/llm_audit.json'), key=lambda p:read_json(p).get('started_unix',0)):
    audit = read_json(job)
    if audit['role'] != 'coding' or not audit.get('usage'):
        continue
    data = dict(audit['request'])
    data.pop('system_prompt')
    old, new = canonical(data).encode(), transport_input(data).encode()
    assert json.loads(old) == json.loads(new)
    rows.append((job.parent.name,old,new,audit['usage'].get('cached_input_tokens')))
results = [{'previous':a[0],'current':b[0],'old_prefix_bytes':len(commonprefix([a[1],b[1]])),
            'new_prefix_bytes':len(commonprefix([a[2],b[2]])), 'observed_old_order_cached_tokens':b[3]}
           for a,b in zip(rows,rows[1:]) if a[0].rsplit('-a',1)[0] == b[0].rsplit('-a',1)[0]]
atomic_json(out/'prefix.json', {'pairs':results,'scope':'actual adjacent requests; canonical prompt bytes only; no observed new-order paid cache usage'})
