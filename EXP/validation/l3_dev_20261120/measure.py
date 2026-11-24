"""Bounded validation using a previously verified full package and its archived loader."""
from pathlib import Path
import argparse
import os
import time
from runtime.paths import ROOT
from core.journal import read_json, atomic_json, file_sha256
from runtime.gpu_resources import acquire, available_inventory, ResourceBusy
from core.cards import validate_card
from tools.frozen import load_frozen, _materialize_source
from tools.experiment_worker import backend_and_data, evaluate_task, evaluation_summary
from runtime.guards import protect_files
from runtime.identity import measurement_implementation
from ptq_agent.execution.evaluation import model_digest
import torch

parser = argparse.ArgumentParser()
parser.add_argument('solution', type=Path)
parser.add_argument('kind', choices=['classification','ade_frozen'])
parser.add_argument('gpu', type=int)
parser.add_argument('view_sha256')
parser.add_argument('--count', type=int)
args = parser.parse_args()
out = Path(__file__).resolve().parent
row = read_json(args.solution)
artifacts = row['artifacts']
seg = next(a for a in artifacts if a['path'].endswith('/segmentation.json'))
assert file_sha256(seg['path']) == seg['sha256']
prior = read_json(seg['path'])
package = Path(seg['path']).parent/'model'
binding = {k:row[k] for k in ('source_version','source_snapshot','source_archive_sha256','evidence_ref')}
binding.update(model_dir=str(package),candidate=row.get('code'),model_ref=row['id'],
               assets=[a for a in artifacts if package in Path(a['path']).parents],
               source_cache=str(out/'sources'/row['source_version']))
directory = out/('smoke' if args.count else 'measurements')/row['id']/args.kind
directory.mkdir(parents=True, exist_ok=True)
inventory = available_inventory({})
gpu = next(r for r in inventory if r['index'] == args.gpu)
for attempt in range(30):
    try:
        _, leases = acquire([args.gpu],bindings=[gpu])
        break
    except ResourceBusy:
        if attempt == 29:
            raise
        time.sleep(2)
os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],QAVIT_RESOURCE_ASSIGNED='1')
started = time.time()
spec = read_json(package.parent/'request.json')['experiment']['payload']['execution']
_materialize_source(binding)
backend,cfg,data = backend_and_data(spec)
torch.set_num_threads(4)
torch.manual_seed(spec['seed'])
protect_files(directory,cuda_device_nodes=[gpu['device_node']])
model, receipt = load_frozen(binding,backend,cfg,data)
identity = [model_digest(backend.model),model_digest(model)]
assert identity == prior['backbone_hashes']
cfg['_development_view'] = dict(path=str(out/('initial_view.json' if args.count else 'development.json')),sha256=args.view_sha256)
requested = dict(kind=args.kind,split='dev' if args.kind=='classification' else 'ade_dev',count=args.count)
coverage = read_json(package.parent/'result.json')
print('frozen package verified',identity,flush=True)

value = evaluate_task(backend,cfg,data,model,receipt['manifest']['plan'],requested,directory,False,
                      coverage['data']['observations']['coverage'])
assert [model_digest(backend.model),model_digest(model)] == identity
key,summary = evaluation_summary(requested,value)
atomic_json(directory/(key+'.json'),value)
atomic_json(directory/'receipt.json',dict(summary=summary,sha256=file_sha256(directory/(key+'.json')),
    solution_sha256=file_sha256(args.solution),evidence_ref=row['evidence_ref'],model_ref=row['id'],
    backbone_hashes=identity,gpu=gpu,started=started,finished=time.time(),cold_load=True,
    implementation=measurement_implementation()))
