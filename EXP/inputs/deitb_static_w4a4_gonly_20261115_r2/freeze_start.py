"""Freeze this authorized three-round run before starting its managed segment."""
from pathlib import Path
import sys,json,time,hashlib,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import read_json,atomic_json,file_sha256
from core.cards import fingerprint
from runtime.identity import code_identity
config=read_json(HERE/'input.json')
assert config['budget']=={'max_rounds':3,'wall_seconds':14400,'clock_mode':'active_segments','exclude_numeric_wait':True}
assert not config.get('scope_target') and config['enable_l3']
assert config['llm']['agent_models']['generation']==['gpt-6-astra','high']
assert [r['count'] for r in config['research_brief']['evaluation_protocol']['evaluations']]==[50000,2000]
run=ROOT/'EXP/runs'/HERE.name
wf=Workflow(run,config)
initial={'source_identity':code_identity(),'input_sha256':file_sha256(HERE/'input.json'),
 'resolved_config_sha256':fingerprint(wf.config),'manifest_sha256':file_sha256(run/'research_run.json'),
 'initial_memory':wf.memory.snapshot(),'initial_solution_library':wf.solutions.snapshot(),
 'initial_B':wf.solutions.fallback(),'initial_journal_sequence':wf.journal.sequence,
 'frozen_unix':time.time(),'preparation_included_in_formal_budget':False,
 'monitor_requested':{'model':'gpt-6.1-sol','reasoning_effort':'ultra'},
 'monitor_actual':{'model':None,'reasoning_effort':None,'source':'not_reported_by_cli'}}
atomic_json(HERE/'experiment_identity.json',initial)
atomic_json(HERE/'config_resolved.json',wf.config)
print(str(run),flush=True)
