"""Read-only post-formal context reconstruction; never resume or write the source run."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'QAViT'))
from core.journal import Journal,file_sha256
from core.memory import ResearchMemory
from core.solutions import SolutionLibrary
from agents.interaction import RoleInteraction
from runtime.identity import code_identity
run=ROOT/'EXP/runs/deitb_static_w4a4_gonly_20261115_r2'
out=Path(__file__).resolve().parent
protected=[run/x for x in ('research_run.json','summary.json','checkpoint.json','status.json')]
before={str(p.relative_to(run)):file_sha256(p) for p in protected}
files_before={str(p.relative_to(run)):(p.stat().st_size,p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
journal=Journal(run,run.name)
brief=journal.find_cards('ResearchBrief')[-1]['payload']
host=SimpleNamespace(journal=journal,memory=ResearchMemory(journal),solutions=SolutionLibrary(journal,brief,shared=False))
records={}
for event in journal.events:
 if event['type']=='task_snapshot' and event['task']['inputs'].get('purpose')=='record':
  records[event['task']['inputs']['context']['round']]=event['task']
transitions={str(number):RoleInteraction.recorder_B_transition(host,task,journal.sequence) for number,task in records.items()}
diagnosis=json.loads((ROOT/'EXP/inputs/deitb_static_w4a4_gonly_20261115_r2/scientific_memory_diagnosis.json').read_text())
for row in diagnosis['findings']:
 view=transitions[str(row['round'])]
 assert view['before_l3']['candidate']['model_ref']==row['pre_round_B_model_ref'],view
 assert view['after_l3']['candidate']['model_ref']==row['new_candidate_model_ref'],view
 assert view['before_l3']['candidate']['model_ref']!=view['after_l3']['candidate']['model_ref']
after={str(p.relative_to(run)):file_sha256(p) for p in protected}
files_after={str(p.relative_to(run)):(p.stat().st_size,p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
assert before==after and files_before==files_after
result={'purpose':'post-formal engineering followup; not a formal intervention or scientific rerun',
 'formal_frozen_core_sha256':json.loads((run/'research_run.json').read_text())['code_identity']['core_sha256'],
 'current_core_sha256':code_identity()['core_sha256'],'sequence':journal.sequence,'head_sha256':journal.commits[-1]['sha256'],
 'formal_run_file_stat_inventory_unchanged':True,'formal_run_files_checked':len(files_before),
 'protected_original_hashes_unchanged':before,'transitions':transitions,
 'LLM_calls':0,'numeric_executions':0,'history_or_Memory_written':False}
(out/'historical_context_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','rounds_checked':list(transitions),'run_files_unchanged':len(files_before),'sequence':journal.sequence},ensure_ascii=False))
