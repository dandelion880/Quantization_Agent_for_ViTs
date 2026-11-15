"""Read-only local Generation replay; no research task dispatch or historical writes."""
from pathlib import Path
import sys, json, time, hashlib, os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'QAViT'))
from core.workflow import Workflow
from core.journal import atomic_json
from core.cards import canonical
from agents.roles import role_spec
from runtime.model_router import call_once
from runtime.identity import code_identity

source=ROOT/'EXP/runs/deitb_static_w4a4_gonly_20261115_r1'
out=Path(__file__).parent/os.environ.get('QAVIT_REPLAY_OUTPUT','generation_replay_compact')
if out.exists():
    raise SystemExit('Replay output already exists; set QAVIT_REPLAY_OUTPUT to a new directory name.')
wf=Workflow(source, read_only=True)
task=wf.queue.get_by_key('r002/select')
# Include the actually measured counterexample returned during that selection.
context=wf.context(task)
evidence=wf.journal.find_cards('EvidenceCard')
context['replay_validation']={'mode':'local_read_only','historical_run':str(source),
 'instruction':'本次仅验证更新后的交付与局部科学决策。分析直接交付的真实证据及FP/Q差距，指出多实验臂和关键反证；说明若改变已提交H2位点/干预应使用哪个版本修订入口。不要执行实验或调用工具；提交selected_slot/rationale/evidence_refs/uncertainties，选择仅为局部回放，不发布到历史运行。',
 'evidence_origin':'prior_real_run_read_now_not_new_measurements'}
# Also present the real dual-task outcomes as explicit subsequent diagnostic material,
# never pretend these were known to the historical R2 decision.
context['replay_validation']['later_dual_task_evidence']=[{'id': c['id'], 'key_facts': wf.present_card(c)['payload']['key_facts']} for c in evidence
 if wf.journal.card(c['payload']['experiment_ref'])['payload']['level']=='L3']
current_gpu=Path(__file__).parent/'gpu/GPU_RESULTS.json'
if current_gpu.exists():
    check=json.loads(current_gpu.read_text())
    context['replay_validation']['current_real_gpu_check']={k:check[k] for k in
        ('scope','gpu_bindings','parallel','equality','failure')}
    context['replay_validation']['instruction'] += '同时说明本轮真实GPU工程检查支持什么、不能支持什么；它不验证科学候选收益。'
spec=role_spec('generation','select',context,wf.config['llm'])
out.mkdir(parents=True,exist_ok=True)
atomic_json(out/'request.json',spec)
head=(wf.journal.sequence, wf.journal.commits[-1]['sha256'])
result=call_once(spec,out/'call',time.time()+600)
response=result['response']
atomic_json(out/'response.json',response)
atomic_json(out/'receipt.json',{'source_run':str(source),'source_head_before':head,
 'source_head_after':(wf.journal.sequence,wf.journal.commits[-1]['sha256']),'implementation':code_identity(),
 'request_sha256':hashlib.sha256(canonical(spec).encode()).hexdigest(),
 'kind':'real_generation_call_on_real_historical_evidence_not_new_scientific_execution'})
print(json.dumps(response,ensure_ascii=False))
