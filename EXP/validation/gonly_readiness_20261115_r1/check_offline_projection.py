"""Read-only rendering check; does not re-score or rewrite frozen Eval-L0 cases."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'QAViT'))
from tools.evaluation.judge import supplied_sources,project_sources
from tools.evaluation.common import canonical,read_json
out=Path(__file__).parent
source=ROOT/'EXP/evaluations/deitb_static_w4a4_gonly_20261115_r1_l0'
results=[]
for path in sorted((source/'cases').glob('*.json')):
 case=read_json(path)
 if case['purpose']!='record':continue
 raw=path.read_bytes();sources=supplied_sources(source,case)
 projected=project_sources(sources)
 assert path.read_bytes()==raw
 results.append({'case_id':case['case_id'],'role':case['role'],
    'source_characters':len(canonical(sources)),'projected_characters':len(canonical(projected)),
    'original_case_sha256':hashlib.sha256(raw).hexdigest(),'original_case_unchanged':True})
assert len(results)==2
(out/'offline_projection.json').write_text(json.dumps({'kind':'read_only_real_recorder_cases_no_judge_call_no_new_score','cases':results},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False))
