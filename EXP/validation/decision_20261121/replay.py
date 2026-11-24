"""Decision-only replay. A receipt is released only for an exact accepted frozen experiment."""
from pathlib import Path
import sys,time,subprocess
from runtime.paths import ROOT
from core.cards import object_schema,TEXT
from core.journal import read_json,atomic_json
from agents.roles import role_spec
from runtime.model_router import call_once

out=Path(__file__).resolve().parent
run=ROOT/'EXP/runs/dinov3b_static_w4a4_gonly_next4_20261118_r1'
arm,case=sys.argv[1:]
dest=out/arm/case
dest.mkdir(parents=True,exist_ok=False)
def call(spec,step):
    reply=call_once(spec,dest/step,time.time()+210)['response']
    atomic_json(dest/(step+'.json'),reply)
    return reply

if case=='focus':
    prefix='task-d99dfca9*'
else:
    prefix='task-58d6bc51*'
job=sorted((run/'jobs').glob(prefix))[-1 if case=='focus' else 1]
spec=read_json(job/'request.json')['spec']
purpose=spec['purpose']
# The operator approved up to five calls for this independent, zero-GPU replay.
spec['settings']={**spec['settings'],'max_transport_retries':0}
spec['settings'].pop('subscription_max_used_percent', None)
if arm=='after':
    spec['system_prompt']=role_spec('generation',purpose,{},spec['settings'])['system_prompt']
    spec['context']['skill_catalog']=(ROOT/'QAViT/docs/skills/INDEX.md').read_text()
if case=='focus':
    call(spec,'focus')
    sys.exit()

# This moment follows approval of the historical design, before its numeric worker.
job=next((run/'jobs').glob('task-8a8e4*/request.json'))
frozen=read_json(job)['experiment']
p=frozen['payload']
ctx={k:spec['context'][k] for k in ('task','model_metadata','research_topic','skill_catalog','fallback_B')}
ctx['pending_design']={k:p[k] for k in ('question','controls','measurements','decision_impact')}
ctx['experiment_ref']=frozen['id']
ctx['replay_task']='研究决策回放。判断是否采用这项已冻结的实验，解释它能否区分当前解释。采用原设计则 experiment_ref 返回其精确 ID；若应改测或直接提交，请留空并给出替代决定。说明值得保留或调整的实验条件和什么结果会改变方案。可选一个目录内 Skill 随结果读取，也可留空。只预算一次已有数值结果读取，不执行新规格。'
spec.update(context=ctx,messages=[],tools=[],response_schema=object_schema({'experiment_ref':{'type':'string'},'reason':TEXT,'skill':{'type':'string'}},['experiment_ref','reason','skill']))
decision=call(spec,'design')['decision']
if decision['experiment_ref']!=frozen['id']:
    sys.exit()
evidence=next(c for f in (run/'cards').glob('*.json') if (c:=read_json(f))['kind']=='EvidenceCard' and c['payload']['experiment_ref']==frozen['id'])
assert evidence['payload']['status']=='completed' and evidence['payload']['measurement_source']=='real'
obs=evidence['payload']['observations']
skill=decision.get('skill','')
skills=ROOT/'QAViT/docs/skills'
for path in skills.glob('*/SKILL.md'):
    if skill in (path.parent.name, 'skills/'+path.parent.name+'/SKILL.md', str(path.relative_to(ROOT/'QAViT'))):
        ctx['requested_skill']=(path.read_text() if arm=='after' else subprocess.check_output(
            ['git','show','6aa8f285fd5a6840050f47e77a3eff8522f67a7c:'+str(path.relative_to(ROOT))],cwd=ROOT,text=True))
ctx.update(previous_decision=decision,receipt={'ref':evidence['id'],'experiment_ref':frozen['id'],'execution_id':evidence['payload']['execution_id'],
    # Freeze exactly the per-arm fields supplied during both live calls.
    'selectors':obs['selectors'],'arm_results':{name:{'metrics':arm['metrics']} for name,arm in obs['arms'].items()},
    'calibration_count':len(obs['calibration_sample_ids']),'search_count':len(obs['search_sample_ids'])})
ctx['replay_task']='已收到所选实验的真实结果。现在说明判断和下一步，给出当前可回退的完整候选构造或说明还缺哪项测量及为何值得，清楚标记预测。'
spec['response_schema']=object_schema({'judgment':TEXT,'next_step':TEXT},['judgment','next_step'])
call(spec,'receipt')
