from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from runtime.paths import ROOT
from core.journal import read_json,atomic_json,file_sha256
out=Path(__file__).resolve().parent
view=read_json(out/'development.json')
rows=view['classification']
names={r['sample_id'] for r in rows}
hashes={r['file_sha256'] for r in rows}
seen={}
for path in [ROOT/'QAViT/configs/assets/imagenet_manifest.json',
             *ROOT.glob('EXP/validation/dinov3b_preparation*/role_source/QAViT/configs/assets/imagenet_manifest.json'),
             *ROOT.glob('EXP/inputs/dino*/*64.json')]:
    key=file_sha256(path)
    if key in seen: continue
    src=read_json(path)
    seen[key]={s:{'same_id':len(names & {r['sample_id'] for r in pool}),
                  'same_file_hash':len(hashes & {r['file_sha256'] for r in pool})}
               for s,pool in src['splits'].items()}
settings=read_json(ROOT/'QAViT/configs/dinov3_ade_evaluation.json')
ds=Path(settings['data_root'])
with ThreadPoolExecutor(max_workers=4) as pool:
    val={h for h in pool.map(file_sha256,sorted((ds/'images/validation').glob('*.jpg')))}
head=Path(view['classified_head_pool'])
head_ids={f'train/{p.parent.name}/{p.name}' for c in head.iterdir() if c.is_dir() for p in c.iterdir()}
atomic_json(out/'overlap.json',dict(imagenet=seen,classification_head_pool_ids=len(head_ids),
    head_pool_same_ids=len(names & head_ids),ade_official_hash_overlap=len(val & {r['file_sha256'] for r in view['ade_frozen']}),
    ade_head=view['ade_head_overlap'],ade_exclusions=view['exclusions'],
    scope='bound QAViT run and preparation; undocumented external historical exposure unknown'))
