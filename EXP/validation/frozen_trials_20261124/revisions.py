"""Run the actual worker and Evidence path, then independently evaluate the sealed exports on paired dev prefixes."""
from pathlib import Path
from copy import deepcopy
import os
import subprocess
import sys
import time
import torch
from numerical import OUT, RUN, REQUEST, OBS, OBJECTIVES, P
from runtime.paths import QAVIT
from runtime.identity import code_identity, code_snapshot
from core.cards import make_card, fingerprint
from core.journal import Journal, read_json, atomic_json, file_sha256
from tools.experiments import ExperimentBroker, asset, preflight_experiment, resolve_execution
from tools.frozen import _materialize_source, bind_parent_initialization, FrozenData
from tools.experiment_worker import backend_and_data, evaluate_task, evaluation_summary
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.calibration_search import state_identity
from core.static_w4a4 import check_plan, native_audit
from ptq_agent.execution.sites import site_catalog
from trajectory import tasks
V = 'blocks.8.attn::attention_value'
DIR = OUT/'workers'
def era():
    path=OUT/'era/source.json'
    if path.exists(): return read_json(OUT/'era/binding.json')
    ident=code_identity()
    atomic_json(path,{'files':code_snapshot(ident)})
    b={'source_version':ident['core_sha256'],'source_archive_sha256':ident['archive_sha256'],
       'source_snapshot':{**asset(path),'field':'files'}, 'source_cache':str(OUT/'era/cache')}
    _materialize_source(b)
    atomic_json(OUT/'era/binding.json',b)
    return b

def main():
    d=read_json(OUT/'dev/results.json')['results']
    base=d['P08']
    source=era()
    original=read_json(REQUEST)['experiment']['payload']['execution']
    binding={'model_ref':base['model_ref'],'model_dir':base['package'],'assets':[asset(Path(base['package'])/k) for k in base['files']],
             'evidence_ref':str(OUT/'dev/results.json'),'experiment_ref':str(OUT/'dual_feedback.py'),
             'verified_increment':base['candidate']['child_state_sha256'],'candidate':None, **source}
    brief=deepcopy(read_json(RUN/'research_run.json')['config']['research_brief'])
    journal=Journal(DIR/'journal','incremental-validation')
    br=next((k for k in journal.cards if k.startswith('ResearchBrief-')),None)
    if not br:
        card=make_card('ResearchBrief',journal.run_id,brief)
        journal.commit('brief',cards=[card]);br=card['id']
    broker=ExperimentBroker(journal,br,None)
    def worker(name,execution,ev_refs=()):
        job=DIR/name
        marker=job/'evidence.json'
        if marker.exists(): return journal.card(read_json(marker)['ref'])
        draft={'level':'L2','question':name,'decision_impact':'compare exactly paired candidate and unchanged child under the same development budget',
               'measurements':['model or grid identity and measured response'], 'resources':{'cpu_threads':2,'estimated_seconds':600},
               'evidence_refs':list(ev_refs),'execution':execution}
        payload=preflight_experiment(draft,brief,executable=True)
        spec=payload['execution']=resolve_execution(payload['execution'],brief,level='L2')
        spec['frozen_sources']=[binding]
        bind_parent_initialization(journal,payload)
        e=make_card('ExperimentCard',journal.run_id,payload,author='generation',refs=list(ev_refs))
        journal.commit(name+'/'+e['id'],cards=[e])
        if (job/'request.json').exists():
            job=DIR/(name+'-retry-'+str(int(time.time())))
        uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        request={'kind':'experiment','experiment':e,'approval':None,'task_id':name,'execution_id':name+'-'+str(int(time.time())),
                 'core_sha256':code_identity()['core_sha256'],'resource_keys':['gpu:'+uuid],'gpu':1,'gpus':[1],
                 'gpu_bindings':[{'index':1,'uuid':uuid,'device_node':'/dev/nvidia1'}],'deadline_unix':time.time()+600}
        atomic_json(job/'request.json',request)
        with (job/'worker.log').open('w') as f:
            subprocess.run([sys.executable,'-B',str(QAVIT/'runtime/task_worker.py'),str(job/'request.json')],stdout=f,stderr=f,check=True)
        r=read_json(job/'result.json')
        if r['status']!='completed': raise RuntimeError((job,r['error']))
        evidence=broker.evidence(e,r,job/'result.json')
        broker.validate_evidence_for(e,evidence)
        journal.commit(name+'/observation',cards=[evidence])
        atomic_json(marker,{'ref':evidence['id'],'actual_job':str(job)})
        return evidence
    actions=[read_json(OUT.parent.parent/'evaluations'/('frozen_trials_20261124_chain_'+k)/'state.json')['revision'] for k in ('old','new')]
    assert all(a['base_id']=='P08' and a['target']=='P' and a['factor']==actions[0]['factor'] for a in actions)
    fixed=read_json(OUT/'fixed_plan.json')
    candidates=[{'id':'unchanged','scales':{}},{'id':'adaptive','scales':{P:{'activation':actions[0]['factor']}}},
                {'id':'fixed','scales':fixed['next_grid_relative_to_first_child']}]
    score=worker('score',{'service':'frozen_helper','tool':'score_frozen_candidates','source_ref':base['model_ref'],'training':False,
           'calibration':{'split':'calibration','count':4},'arguments':{'candidates':candidates,'observations_spec':OBS,'objectives':OBJECTIVES}})
    for name in ('adaptive','fixed'):
        evidence=worker(name,{'service':'quantize_evaluate','parent_ref':base['model_ref'],'training':False,
            'accept_trial':{'evidence_ref':score['id'],'candidate_id':name},
            'evaluations':[{'kind':'classification','split':'search','count':1}]},[score['id']])
        ev=evidence['payload']; pkg=Path(next(a['path'] for a in ev['artifacts'] if Path(a['path']).name=='weights.pt')).parent
        path=DIR/name/'paired_dev.json'
        if path.exists():continue
        t=time.monotonic()
        backend,cfg,data=backend_and_data(original)
        cfg['_development_view']=brief['execution_guidance']['development']['manifest']
        rec={'source':evidence['id'],'model_ref':ev['provenance']['model_ref'],'candidate':ev['observations']['refinement'],'tasks':{}}
        for kind,split,n in [('classification','dev',24),('ade_frozen','ade_dev',6)]:
            model,m=QuantizationExecutor(backend,cfg,FrozenData(data)).load(pkg)
            assert state_identity(model)==rec['candidate']['child_state_sha256']
            plan=m['plan']
            audit=native_audit(model,plan,site_catalog(backend),check_plan(plan,site_catalog(backend)))
            atomic_json(path.parent/'static.json',audit)
            req={'kind':kind,'split':split,'count':n}
            val=evaluate_task(backend,cfg,data,model,plan,req,path.parent,True,m['audit'])
            key,summary=evaluation_summary(req,val)
            atomic_json(path.parent/(key+'_dev.json'),val)
            assert summary['sample_ids']==base['tasks'][key]['sample_ids']
            rec['tasks'][key]=summary
            del model
        rec['seconds']=time.monotonic()-t
        atomic_json(path,rec)
        if name=='adaptive':
            feedback={'model_ref':rec['model_ref'],'child_state_sha256':rec['candidate']['child_state_sha256'],
                      'parent_model_ref':base['model_ref'],'parent_state_sha256':base['candidate']['child_state_sha256'],
                      'selection_source':score['id'],'worker_evidence':evidence['id'],'change':rec['candidate']['changes'],
                      'source_sha256':file_sha256(path),'tasks':tasks(rec),'cold_state_matches':True}
            for k in ('old','new'): atomic_json(OUT/('revision_'+k)/'result.json',feedback)
        del backend
if __name__=='__main__':
    if sys.argv[1:] == ['era']: era()
    else:
        torch.set_num_threads(2)
        main()
