"""Read-only historical and explicit synthetic history costs; no scientific evidence."""
from pathlib import Path
import argparse, json, statistics, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'QAViT'))
from core.workflow import Workflow
from core.cards import canonical
from core.journal import atomic_json
from tests.support import validation_config, topic_decision

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]

def measure(fn, n=5):
    values=[]
    for _ in range(n):
        start=time.perf_counter(); result=fn(); values.append(time.perf_counter()-start)
    return {'median_seconds':statistics.median(values),'samples_seconds':values}, result

def costs(wf, task):
    result={'commits':wf.journal.sequence,'events':len(wf.journal.events),'cards':len(wf.journal.cards)}
    for name, fn, n in [('journal_refresh',wf.journal.refresh,10),
                        ('queue_projection',lambda:wf.queue.snapshot(fields=('state','kind')),10),
                        ('budget_projection',wf.budget.snapshot,10),
                        ('runtime_poll',wf.runtime.poll,5),
                        ('directory_size_scan',lambda:sum(p.stat().st_size for p in wf.run_dir.rglob('*') if p.is_file() and not p.is_symlink()),3),
                        ('context',lambda:wf.context(task),3),
                        ('permissions',lambda:wf.authorized_assets(task),3)]:
        result[name], value=measure(fn,n)
        if name=='context':
            result['context_characters']=len(canonical(value))
            result['context_fields']={k:len(canonical(v)) for k,v in value.items()}
        if name=='directory_size_scan':result['directory_bytes']=value
    return result

def main(stage):
    out=BASE/(stage+'_costs.json')
    if out.exists():raise FileExistsError(out)
    historical=Workflow(ROOT/'EXP/runs/deitb_static_w4a4_gonly_20261115_r2',read_only=True)
    task=historical.queue.get_by_key('r003/generate/H3')
    result={'stage':stage,'history':costs(historical,task),'synthetic_scope':'200 topics, 120 ideas, 800 tasks; state fixtures only'}
    cfg=validation_config();cfg.update(purpose='infrastructure_test')
    cfg['budget'].update(min_free_bytes=0)
    wf=Workflow(BASE/('perf_'+stage),cfg)
    events=[]
    for i in range(200):
        row={**topic_decision(wf),'id':f'topic-{i}','ref':f'topic-{i}@1','version':1,
             'status':'active' if i==199 else 'paused','actions':[],
             'decision_ref':wf.brief['id'],'source_refs':[],'method_ids':[],'inspiration_ids':[],
             'unresolved_questions':['paired counterevidence unresolved']}
        events.append({'type':'topic_version','entry':row})
    for i in range(120):
        events.append({'type':'inspiration_updated','entry':{'id':f'idea-{i}','revision':1,
             'status':'active' if i==119 else 'paused','title':f'fixture idea {i}',
             'idea':'Synthetic mechanism note. '*80,'basis':'fixture, not measured',
             'progress':'No new evidence. '*30,'open_questions':['retain counterevidence'],
             'next_step':'repair paired measurement','source_refs':[],'absorbed_parts':[]}})
    wf.journal.commit('synthetic-history',events=events)
    # Many real committed queue transitions, no worker launches.
    for i in range(800):
        tid=wf.queue.enqueue(f'history/{i}','host_tool',inputs={'fixture':True})
        wf.queue.change(tid,'cancelled')
    tid=wf.enqueue_role('benchmark/focus','focus',context={'round':2})
    result['synthetic']=costs(wf,wf.queue.get(tid))
    result['status_write'],_=measure(lambda:atomic_json(BASE/('status_'+stage+'.json'),{'status':'fixture','sequence':wf.journal.sequence}),20)
    atomic_json(out,result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage');main(parser.parse_args().stage)
