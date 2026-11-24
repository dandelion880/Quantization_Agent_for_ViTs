"""Freeze this experiment's two training views before comparing any candidate."""
from pathlib import Path
import json
import hashlib
import random
import re
import numpy as np
from PIL import Image
import torch
from runtime.paths import ROOT
from runtime.files import file_hash
from core.journal import atomic_json
from ptq_agent.backends.image_data import image_transform

out = Path(__file__).resolve().parent
torch.set_num_threads(1)
source = next((ROOT/'EXP/inputs').glob('dino*/*64.json'))
base = json.loads(source.read_text())
rng = random.Random(20261120)
excluded = {r['file_sha256'] for rows in base['splits'].values() for r in rows}
transform = image_transform(base['preprocessing'])
root = Path(base['selection']['search_train_root'])
head_pool = ROOT.parent/'ptq_training_free/data/imagenet_subset_50k_10k/train'
rows = []
for cls, label in base['class_to_idx'].items():
    head_names = {p.resolve().name for p in (head_pool/cls).iterdir()} | {p.name for p in (head_pool/cls).iterdir()}
    paths = sorted(p for p in (root/cls).iterdir() if p.suffix.lower() in ('.jpg','.jpeg') and p.name not in head_names)
    rng.shuffle(paths)
    count = 0
    for path in paths:
        digest = file_hash(path)
        if digest in excluded:
            continue
        excluded.add(digest)
        with Image.open(path) as im:
            x = transform(im.convert('RGB')).unsqueeze(0)
        rows.append(dict(path=str(path), file_sha256=digest, input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                         sample_id='/'.join(path.parts[-3:]), dataset_split='train', label=label))
        count += 1
        if count == 5:
            break
    assert count == 5
ade_settings = json.loads((ROOT/'QAViT/configs/dinov3_ade_evaluation.json').read_text())
ade_root = Path(ade_settings['data_root'])
exclude_files = [ROOT.parent/'ptq_training_free/manifests'/f for f in ('activation_images.json','calibration_crops.json')]
excluded_names = set(re.findall(r'ADE_train_\d+\.jpg', ''.join(f.read_text() for f in exclude_files)))
paths = sorted(p for p in (ade_root/'images/training').glob('*.jpg') if p.name not in excluded_names)
rng.shuffle(paths)
seg = []
classes = set()
required_labels = set(range(1, ade_settings["model"]["decode_head"]["num_classes"] + 1))
for p in paths:
    ann = ade_root/'annotations/training'/p.with_suffix('.png').name
    with Image.open(ann) as img:
        labels = set(map(int,np.unique(np.asarray(img))))
    if len(seg) >= 500 and not (labels & (required_labels - classes)):
        continue
    classes.update(labels)
    seg.append(dict(name=p.name, file_sha256=file_hash(p), annotation_sha256=file_hash(ann)))
    if len(seg) >= 500 and required_labels <= classes:
        break
result = dict(schema='qavit.development/1', seed=20261120, classification=rows, ade_frozen=seg,
              manifest_sha256=file_hash(source), preprocessing=base['preprocessing'], class_to_idx=base['class_to_idx'],
              exclusions={str(p):file_hash(p) for p in [source,*exclude_files]}, classified_head_pool=str(head_pool),
              ade_labels=sorted(classes), ade_head_overlap='training split used by frozen head; individual exposures unknown')
target = out/'development.json'
if target.exists() and json.loads(target.read_text()) != result:
    raise ValueError('development view changed; use a fresh directory')
atomic_json(target,result)
print(len(rows),len(seg),len(classes-{0,255}),file_hash(target),flush=True)
