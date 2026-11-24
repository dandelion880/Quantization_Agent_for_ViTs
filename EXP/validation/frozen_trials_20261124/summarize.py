"""Summarize actual receipts and exact repeated page bodies. No judge or actor gets these aggregates."""
from collections import Counter
from pathlib import Path
import hashlib
import json
from core.journal import atomic_json, read_json, file_sha256
from core.cards import fingerprint
from runtime.paths import ROOT
OUT=Path(__file__).resolve().parent
E=ROOT/'EXP/evaluations'
def h(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def audits(d):
    seen=set(); totals=Counter(); pages=Counter(); details=[]
    for fp in sorted(d.glob('*/llm_audit.json')):
        r=read_json(fp); u=r.get('usage') or {}
        tokens={k:u.get(k,0) for k in ('input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens')}
        totals.update(tokens)
        ctx=r['request']['context']
        # Count exact surfaced page texts in known scientific read slots, not strings from the system prompt or JSON schemas.
        bodies=[x['result'] for x in ctx.get('tool_results',[]) if isinstance(x.get('result'),dict)]
        bodies.extend([ctx.get('local_study',{}).get('original_page',{}),ctx.get('strategy_page',{})])
        counts=Counter()
        for b in bodies:
            if not isinstance(b.get('text'),str): continue
            body=b['text']; key=(b.get('sha256'),b.get('offset'),h(body))
            byte_count=len(body.encode())
            counts.update(page_exposures=1,page_body_bytes=byte_count)
            if key in seen:counts.update(repeated_pages=1,repeated_page_bytes=byte_count)
            seen.add(key)
        pages.update(counts)
        resp=read_json(fp.parent/'response.json') if (fp.parent/'response.json').exists() else {}
        calls=resp.get('tool_calls',[])
        read_calls=[c['name'] for c in calls if c['name'].startswith(('read','retrieve'))]
        totals.update(voluntary_read_calls=len(read_calls),elapsed_seconds=r['elapsed_seconds'])
        details.append({'job':fp.parent.name,'request_sha256':r['request_sha256'],'audit_path':str(fp.relative_to(ROOT)),
            'response_sha256':fingerprint(resp),'requested_model':r['requested'],'actual_model':r['actual'],'fallback':r['automatic_model_fallback'],'usage':tokens,'seconds':r['elapsed_seconds'],
            'read_tools':read_calls,'actual_action':resp.get('decision',{}).get('action') if resp.get('decision') else None,**counts})
    return {'calls':len(details),'totals':dict(totals),**pages,'details':details}

def main():
    focus={k:audits(E/('frozen_trials_20261124_focus_'+k+'_complete')) for k in ('old','new','direct','skill')}
    trajectory={k:audits(E/('frozen_trials_20261124_chain_'+k)) for k in ('old','new')}
    for k,r in trajectory.items():
        s=read_json(E/('frozen_trials_20261124_chain_'+k)/'state.json')
        r['choices']=[s['selection']['candidate_id'],{f:s['revision'][f] for f in ('base_id','target','factor')},
                    {f:s['disposition'][f] for f in ('decision','state_sha256','next_step')}]
    d=read_json(OUT/'dev/results.json')
    rec={**d['results'],**{k:read_json(OUT/'workers'/k/'paired_dev.json') for k in ('adaptive','fixed')}}
    arms={k:{'state':r['candidate']['child_state_sha256'],'model_ref':r['model_ref'],'top1':r['tasks']['classification']['metrics']['top1'],
        'mIoU':r['tasks']['segmentation']['metrics']['mIoU'],'seconds':r['seconds']} for k,r in rec.items()}
    cal,search=(read_json(OUT/'paired'/f) for f in ('calibration.json','confirmation.json'))
    worker={k:{'evidence':read_json(OUT/'workers'/k/'evidence.json')['ref'],'resources':read_json(OUT/'workers'/k/'result.json')['resources'],
                'experiment_sha256':read_json(OUT/'workers'/k/'result.json')['experiment_sha256']} for k in ('score','adaptive','fixed')}
    host=audits(E/'frozen_trials_20261124_focus_host_r3')
    host['ordering_bug_rehearsal']=audits(E/'frozen_trials_20261124_focus_host_r2')
    host_dir=E/'frozen_trials_20261124_focus_host_r3'
    ev=read_json(host_dir/'numeric/evidence.json')
    job=read_json(host_dir/'numeric/result.json')
    measured=job['data']['observations']['result']
    host['completed_experiment']={**ev,'core':job['core_sha256'],'request_sha256':job['request_sha256'],
        'result_sha256':file_sha256(host_dir/'numeric/result.json'),'source':measured['source'],
        'split':measured['split'],'sample_ids':measured['sample_ids'],'state':measured['numeric_state_sha256'],
        'classification_by_arm':measured['classification_diagnostic']}
    # Release scientific requests and decisions. Full transport audits stay in the local evaluation directories.
    host_calls=[read_json(host_dir/x['job']/'response.json') for x in host['details']]
    actor_actions={'focus':{'voluntary_request':next(t['arguments'] for a in host_calls for t in a['tool_calls']
            if t['name']=='request_experiment'),'evidence':ev,'final_decision':host_calls[-1]['decision'],
            'topic_sha256':file_sha256(host_dir/'final_topic.json')},
        'trajectories':{k:read_json(E/('frozen_trials_20261124_chain_'+k)/'state.json') for k in trajectory}}
    atomic_json(OUT/'actions.json',actor_actions)
    origin=read_json(OUT/'paired/run.json')
    brief=read_json(Path(origin['request']).parents[2]/'research_run.json')['config']['research_brief']
    manifest=brief['execution_guidance']['development']['manifest']
    assert file_sha256(Path(manifest['path']))==d['dev_manifest']==manifest['sha256']
    development=read_json(Path(manifest['path']))
    samples=d['results']['parent']['tasks']
    dev_cls=development['classification'][:samples['classification']['sample_count']]
    dev_ade=development['ade_frozen'][:samples['segmentation']['sample_count']]
    assert [r['sample_id'] for r in dev_cls]==samples['classification']['sample_ids']
    assert [r['name'] for r in dev_ade]==[Path(k).name for k in samples['segmentation']['sample_ids']]
    groups={'calibration':set(x['source']['file_sha256'] for x in cal['views']),
            'confirmation':set(x['source']['file_sha256'] for x in search['views']),
            'dev_classification':{x['file_sha256'] for x in dev_cls},'dev_ADE':{x['file_sha256'] for x in dev_ade}}
    assert sum(map(len,groups.values()))==len(set.union(*groups.values()))
    for arm in rec.values():
        for task,res in samples.items():
            assert arm['tasks'][task]['sample_ids']==res['sample_ids']
    split_check={'disjoint_original_file_hashes':True,'all_arms_have_same_ordered_dev_ids':True,
            'distinct_originals':{k:len(v) for k,v in groups.items()},
            'dev_classification_prefix_sha256':fingerprint(dev_cls),'dev_ADE_prefix_sha256':fingerprint(dev_ade),
            'ADE_head_exposure':development['ade_head_overlap']}
    summary={'scope':'development breakpoints and small verified dev prefixes, both held-out cases unopened',
        'focus':focus,'host_focus':host,'data_boundaries':split_check,'three_decisions':trajectory,'arms':arms,'seed':d['seed'],'development_manifest_sha256':d['dev_manifest'],
        'calibration':{'sample_ids':cal['sample_ids'],'views':len(cal['views']),'seconds':cal['seconds'],'winners':cal['winners']},
        'confirmation':{'sample_ids':search['sample_ids'],'views':len(search['views']),'seconds':search['seconds']},
        'workers':worker,'totals_seconds':{'initial_pairing_and_confirmation':read_json(OUT/'paired/run.json')['seconds'],'four_initial_dev_packages':d['seconds']},
        'first_failed_old_harness':audits(E/'frozen_trials_20261124_focus_old/steps'),
        'non_llm_setup_failures':['focus_direct: KeyError(execution) before first call','focus_skill: KeyError(execution) before first call','host first run: missing clock field before first call'],
        'scientific_page_counting':'Exact same source hash, offset and actual UTF-8 text in tool_results.result, local_study.original_page or strategy_page. Excludes prompts/schemas, summaries and material elsewhere; lower bound.'}
    files=['paired/run.json','paired/calibration.json','paired/confirmation.json','dev/results.json','fixed_plan.json',
           'parity.json','custom_parent.json','workers/adaptive/paired_dev.json','workers/fixed/paired_dev.json',
           'revision_new/result.json','revision_old/result.json','production_binding.json','parity_final.json','actions.json','custom_export.json','verification.json']
    if (OUT/'custom_export.json').is_file():
        h2=read_json(OUT/'custom_export.json')
        summary['H2_package']={'model_ref':h2['package_model_ref'],'export_seconds':h2['setup']['seconds'],
            'cold_seconds':h2['cold_seconds'],'source':h2['source'],
            'arms':{k:{'state':r['state_sha256'],'top1':r['tasks']['classification']['metrics']['top1'],
                       'mIoU':r['tasks']['segmentation']['metrics']['mIoU']} for k,r in h2['results'].items()}}
    summary['artifacts']={k:file_sha256(OUT/k) for k in files if (OUT/k).is_file()}
    atomic_json(OUT/'measurements.json',summary)
if __name__=='__main__': main()
