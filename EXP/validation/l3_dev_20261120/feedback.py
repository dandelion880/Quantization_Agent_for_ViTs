"""Import completed validation worker receipts, then obtain a bounded research handoff."""
from pathlib import Path
import time
from runtime.paths import ROOT
from core.cards import make_card, fingerprint, TEXT, object_schema
from core.journal import atomic_json, read_json, Journal, file_sha256
from core.memory import ResearchMemory
from tools.experiments import asset, ExperimentBroker
from tools.development import stage_protocol
from runtime.model_router import call_once
from agents.roles import fragment

out = Path(__file__).resolve().parent
config = read_json(next((ROOT/'EXP/inputs').glob('dino*next4/input.json')))
view = read_json(out/'development.json')
binding = asset(out/'development.json')
requests = [dict(kind=kind,split=split,count=len(view[kind]),metrics=metrics,full_dataset=False)
            for kind,split,metrics in [('classification','dev',['top1','top5']),('ade_frozen','ade_dev',['mIoU','mAcc','aAcc'])]]
brief = config['research_brief']
brief['execution_guidance']['development'] = dict(manifest=binding,evaluations=requests)
j = Journal(out/'journal','development_20261120')
memory = ResearchMemory(j)
if not j.find_cards('ResearchBrief'):
    j.commit('brief',cards=[make_card('ResearchBrief',j.run_id,brief,author='user')])
b = j.find_cards('ResearchBrief')[0]
broker = ExperimentBroker(j,b['id'],None)
facts = {}
for code,pattern in [('B32','7c18'),('H1','7f70')]:
    solution = next((ROOT/'EXP/runs').glob('dino*/solution_library/'+pattern+'*.json'))
    row = read_json(solution)
    directory = out/'measurements'/row['id']
    observations, refs, workers = {}, [], []
    for kind,name in [('classification','classification'),('ade_frozen','segmentation')]:
        r = read_json(directory/kind/'receipt.json')
        ref = asset(directory/kind/(name+'.json'))
        assert (r['solution_sha256']==file_sha256(solution) and r['sha256']==ref['sha256'] and
                r['summary']['development_sha256']==binding['sha256'] and r['cold_load'])
        refs.extend([ref,asset(directory/kind/'receipt.json')])
        observations[name]=r['summary']
        workers.append(r)
    assert workers[0]['backbone_hashes']==workers[1]['backbone_hashes']
    assert workers[0]['implementation']==workers[1]['implementation']
    old = next(a for a in row['artifacts'] if Path(a['path']).name=='result.json' and '/dual_tasks/' not in a['path'])
    assert file_sha256(old['path'])==old['sha256']
    parent = read_json(old['path'])['data']
    frozen = read_json(Path(old['path']).with_name('request.json'))['experiment']['payload']
    execution = dict(service='evaluate_artifact',resolved_choices={'training':False},backend=frozen['execution']['backend'],
                     seed=frozen['execution']['seed'],evaluations=requests,requires_code=False,model_dir=parent['model_dir'],
                     model_scope=frozen['scope'],quantization_contract=frozen['execution']['quantization_contract'],
                     frozen_solution=asset(solution),validation_runner=asset(out/'measure.py'),
                     measurement_implementation=workers[0]['implementation'])
    execution['result_protocol']=stage_protocol(brief,execution,'L3')
    execution['protocol_hash']=fingerprint(execution['result_protocol'])
    payload=dict(level='L3',stage='execution',scope=frozen['scope'],execution=execution,validation={},
                 question='在同一冻结骨干上测定两个完整开发视图的成对 FP 和量化指标。',
                 controls={'reference':'同视图、任务头和推理几何的 FP'},decision_impact='比较同视图保持率并反馈下一步研究。',measurements=['Top-1','mIoU'],
                 resources=frozen['resources'],replication={'requested':False,'purpose':'','configuration_difference':{}})
    existing=[c for c in j.find_cards('ExperimentCard') if c['payload']['execution'].get('model_dir')==parent['model_dir']]
    experiment=existing[0] if existing else make_card('ExperimentCard',j.run_id,payload)
    if not existing: j.commit(code+'/import-spec',cards=[experiment])
    original = parent['observations']
    results=dict(artifacts=refs,measurement_source='real',observations={
        'static_w4a4':original['static_w4a4'],'coverage':original['coverage'],
        'coverage_origin':{'historical_evidence':row['evidence_ref'],'verified_same_model_digest':workers[0]['backbone_hashes'][1]},
        'reload':{'host_verified':True,'evaluated_instance':'reloaded','passed':True,
                  'independent_workers':[r['backbone_hashes'] for r in workers]},
        'evaluations':observations}, provenance={
        'model_identity':parent['provenance']['model_identity'],'model_ref':row['id'],
        'protocol_hash':execution['protocol_hash'],'data_identity':{'batch_size':1,'development_sha256':binding['sha256']},
        'execution_facts':{'measurement_kind':'full_model','data':{k:{f:v[f] for f in
          ('split','sample_count','sample_ids','full_dataset')} for k,v in observations.items()}}})
    receipt=dict(experiment_ref=experiment['id'],experiment_sha256=experiment['content_sha256'],
                 execution_id=code+'-cold-load',status='completed',data=results,resources={
                 'leased_gpu_seconds':sum(r['finished']-r['started'] for r in workers)})
    ep=directory/'dual_task_result.json'
    existing=[c for c in j.find_cards('EvidenceCard') if c['payload']['experiment_ref']==experiment['id']]
    if existing: evidence=existing[0]
    else:
        atomic_json(ep,receipt)
        evidence=broker.evidence(experiment,receipt,ep)
        j.commit(code+'/evidence',cards=[evidence],events=[memory.live_update(round_number=1,evidence_ref=evidence['id'])])
    broker.validate_evidence_for(experiment,evidence)
    facts[code]={'evidence_ref':evidence['id'],'official_observed':row['metrics'],'calibration_samples':len(
        read_json(Path(parent['model_dir'])/'manifest.json')['calibration_ids']), 'metrics':{
        k:{'FP':v['metrics']['baseline_'+m],'Q':v['metrics'][m],'retention_percent':100*v['metrics'][m]/v['metrics']['baseline_'+m]}
        for k,v,m in [('top1',observations['classification'],'top1'),('mIoU',observations['segmentation'],'mIoU')]}}
atomic_json(out/'facts.json',facts)
context={'paired_development_results':facts,'view_sha256':binding['sha256'],'counts':[len(view[r['kind']]) for r in requests],
         'ADE_head_training_overlap':view['ade_head_overlap'],'method':'H1 在完整父方案上采用 FC2 双条件行选择。B32 与 H1 校准预算不同。',
         'research_request':'依据双开发反馈作下一步科学判断，说明对照、可改变判断的观察。保留独立 B；本次只记录，不启动实验。'}
for role in ('generation','recorder'):
    destination=out/'feedback'/role
    schema=(object_schema({'judgment':TEXT,'next_comparison':TEXT},['judgment','next_comparison']) if role=='generation' else
            object_schema({'scientific_analysis':TEXT,'generation_guidance':TEXT,'next_questions':{'type':'array','items':TEXT}},
                          ['scientific_analysis','generation_guidance','next_questions']))
    spec={'role':role,'settings':config['llm'],'system_prompt':fragment(role+'.md'),'messages':[],
          'tools':[],'context':context,'response_schema':schema}
    if not (destination/'decision.json').exists():
        result=call_once(spec,destination,time.time()+150)
        atomic_json(destination/'decision.json',result['response']['decision'])
    decision=read_json(destination/'decision.json')
    if role=='generation':
        context['generation_judgment']=decision
    elif not j.find_cards('MetareviewCard'):
        decision['evidence_refs']=[v['evidence_ref'] for v in facts.values()]
        batch=memory.recording_batch()
        card=make_card('MetareviewCard',j.run_id,memory.prepare_metareview(decision,batch,1),
                       author='recorder',refs=decision['evidence_refs'])
        j.commit('record',cards=[card],events=[memory.apply_metareview(card)])
print('recorded',memory.snapshot()['L3'])
