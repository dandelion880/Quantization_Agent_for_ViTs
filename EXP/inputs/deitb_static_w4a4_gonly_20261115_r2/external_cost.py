"""Extract reported usage only from this Codex thread and its delegated children."""
import json,os
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT_ID='01a1dfbd-5076-7410-86a3-90409f026ac7'
rows=[]
for path in Path('/root/.codex/sessions').rglob('*.jsonl'):
 with path.open() as stream:
  try:meta=json.loads(next(stream)).get('payload',{})
  except (StopIteration,ValueError):continue
  source=meta.get('source');spawn=source.get('subagent',{}).get('thread_spawn',{}) if isinstance(source,dict) else {}
  if meta.get('id')!=ROOT_ID and spawn.get('parent_thread_id')!=ROOT_ID:continue
  first=None;last=None;contexts=[]
  for line in stream:
   try:item=json.loads(line)
   except ValueError:continue
   p=item.get('payload',{})
   if item.get('type')=='turn_context':
    value={k:p.get(k) for k in ['model','effort']}
    if value not in contexts:contexts.append(value)
   if item.get('type')=='event_msg' and p.get('type')=='token_count' and p.get('info',{}).get('total_token_usage'):
    value={'as_of':item['timestamp'],'total':p['info']['total_token_usage'],'last':p['info'].get('last_token_usage',{})}
    first=first or value;last=value
  if last:
   baseline={k:first['total'][k]-first['last'].get(k,0) for k in first['total']}
   usage={k:last['total'][k]-baseline[k] for k in last['total']}
   rows.append({'thread':meta['id'],'agent':spawn.get('agent_path','/root'),'session_contexts':contexts,'as_of':last['as_of'],'usage':usage,'usage_initial_baseline':baseline})
keys=('input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens','total_tokens')
result={'as_of_utc':datetime.now(timezone.utc).isoformat(),'scope':'external preparation, engineering, Codex monitoring and reporting; separate from QAViT scientific CLI and isolated preparation probes/Generation','sessions':rows,'totals':{k:sum(row['usage'].get(k,0) for row in rows) for k in keys},'notes':['cached input is a subset of input; reasoning output is a subset of output; do not add twice','reported repeated-context consumption, not unique evidence volume or a subscription invoice','session/request model metadata is visible; service actual identity remains unreported','snapshot excludes later reporting/final-answer usage']}
(HERE/'external_monitor_usage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result['totals']))
