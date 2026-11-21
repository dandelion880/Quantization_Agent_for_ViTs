"""Read-only projection of completed historical records and independent view receipts."""
from pathlib import Path
from core.workflow import Workflow
from core.evidence import evidence_facts
from core.cards import canonical
from core.journal import atomic_json, read_json,file_sha256
from runtime.configuration import config_check
from runtime.paths import ROOT

out=Path(__file__).resolve().parent
run=ROOT/'EXP/runs/dinov3b_static_w4a4_gonly_next4_20261118_r1'
wf=Workflow(run,read_only=True)
event=next(e for e in wf.journal.events if e['type']=='round_finished' and e['round']==1)
task={'task_id':'read-only-next-focus','kind':'role','role':'generation','inputs':{'purpose':'focus','context':{'round':2}},
      '_read_view':{'sequence':event['commit_sequence']}}
rows=wf.previous_candidate_results(task,event['commit_sequence'],set())
assert not wf.previous_candidate_results(task,event['commit_sequence']-1,set())
assert len(rows)==3
for row in rows:
    result=event['l3_outcome']['candidates'][row['slot']]
    assert row['candidate_ref']==result['hypothesis_ref']
    assert [m['ref'] for m in row['measurements']]==result['valid_evidence_refs']
    for m in row['measurements']:
        assert m['calibration_count']==64
        for name,metric in [('segmentation','mIoU'),('classification','top1')]:
            raw=wf.journal.card(m['ref'])['payload']['observations']['evaluations'][name]
            assert raw['metrics'][metric]==m['tasks'][name]['paired_metrics'][metric]['Q']
card=wf.journal.card('EvidenceCard-a5f8a7d2c8ba4f7089e88c0677f19149@1')
arm=evidence_facts(card)['arm_results']['local']
assert arm['paired_difference_vs_B64']==card['payload']['observations']['arms']['local']['paired_difference_vs_B64']
config=config_check(read_json(ROOT/'EXP/inputs/dinov3b_static_w4a4_independent_dev/input.json'), resolve_resources=True)
handoff=config['research_brief']['research_handoff']['paired_development_results']
assert file_sha256(ROOT/handoff['path'])==handoff['sha256']
paired=read_json(ROOT/handoff['path'])
from core.journal import Journal
j=Journal(ROOT/'EXP/validation/l3_dev_20261120/journal','development_20261120')
for model,entry in paired.items():
    measurements=evidence_facts(j.card(entry['evidence_ref']))['evaluations']
    for name,key,count in [('classification','top1',5000),('segmentation','mIoU',502)]:
        m=measurements[name]
        assert m['purpose']=='development' and m['sample_count']==count and m['complete_view'] and not m['full_dataset']
        assert m['development_sha256']==config['research_brief']['execution_guidance']['development']['manifest']['sha256']
        assert all(m['paired_metrics'][key][field]==entry['metrics'][key][field] for field in ('FP','Q','retention_percent'))
atomic_json(out/'projection.json',{'previous_results':rows,'projection_characters':len(canonical(rows)),
    'before_round_close':[],'verified_paired_local_difference':arm['paired_difference_vs_B64'],
    'development_evidence_refs':{k:v['evidence_ref'] for k,v in paired.items()},'new_input_checked':True})
