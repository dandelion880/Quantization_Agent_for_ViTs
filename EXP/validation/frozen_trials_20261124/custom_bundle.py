"""Real H2 regression. Copy the exact archived package, persist the measured delta and reopen in a new process."""
from pathlib import Path
from copy import deepcopy
import shutil
import sys
import time
import torch

from numerical import OUT, RUN, REQUEST, P
from core.cards import fingerprint
from core.journal import read_json, atomic_json, file_sha256
from tools.experiment_worker import backend_and_data, evaluate_task, evaluation_summary
from tools.frozen import load_refinement_parent, package_identity
from ptq_agent.execution.calibration_search import _quantizers, state_identity
from ptq_agent.execution.model import tensor_digest, forward
from ptq_agent.execution.frozen_trials import accepted_copy
from runtime.identity import code_identity

PACKAGE = OUT / 'custom_export'
RESULT = OUT / 'custom_export.json'


def inputs():
    request = next(RUN.glob('jobs/task-7353*/request.json'))
    src = read_json(OUT / 'custom_parent.json')
    ref = src['custom_source']['model_ref']
    binding = next(s for s in read_json(request)['experiment']['payload']['execution']['frozen_sources'] if s['model_ref'] == ref)
    data_spec = read_json(REQUEST)['experiment']['payload']['execution']
    return binding, src, data_spec


def export():
    start = time.monotonic()
    parent, prior, data_spec = inputs()
    backend, cfg, data = backend_and_data(data_spec)
    model, manifest = load_refinement_parent(parent, backend, cfg, data)
    receipt = prior['scores']['candidates'][1]['receipt']
    child, accepted = accepted_copy(model, receipt, source_ref=parent['model_ref'], input_shapes=manifest.get('input_shapes', {}))
    assert accepted == prior['receipt'] and state_identity(child) == accepted['child_state_sha256']
    original = Path(parent['model_dir'])
    files = {str(Path(a['path']).relative_to(original)): a['sha256'] for a in parent['assets']}
    PACKAGE.mkdir()
    shutil.copytree(original, PACKAGE / 'parent')
    assert {str(f.relative_to(PACKAGE/'parent')): file_sha256(f) for f in (PACKAGE/'parent').rglob('*') if f.is_file()} == files
    sealed = {'format':'development H2 existing nested package plus measured grid delta', 'parent_files':files,
              'loader_sha256':parent['candidate']['source_sha256'],'source_version':parent['source_version'],
              'parent_ref':parent['model_ref'],'parent_identity':package_identity(parent,manifest),
              'calibration_ids':manifest['calibration_ids'],'preparation_training':False,
              'scored_ids':prior['scores']['sample_ids'],'scored_receipt':receipt,
              'child_state_sha256':accepted['child_state_sha256']}
    atomic_json(PACKAGE/'manifest.json', {**sealed, 'content_sha256':fingerprint(sealed)})
    atomic_json(OUT/'custom_export_setup.json',{'seconds':time.monotonic()-start,'child':accepted,'core':code_identity()['core_sha256']})


def cold_evaluate():
    start = time.monotonic()
    parent, prior, data_spec = inputs()
    backend, cfg, data = backend_and_data(data_spec)
    brief = read_json(RUN/'research_run.json')['config']['research_brief']
    cfg['_development_view'] = brief['execution_guidance']['development']['manifest']
    sealed = read_json(PACKAGE/'manifest.json')
    digest = sealed.pop('content_sha256')
    assert digest == fingerprint(sealed)
    assert sealed['loader_sha256'] == parent['candidate']['source_sha256'] and sealed['source_version'] == parent['source_version']
    assert sealed['parent_ref'] == parent['model_ref'] and sealed['preparation_training'] is False
    origin = Path(parent['model_dir'])
    files = {str(Path(a['path']).relative_to(origin)):a['sha256'] for a in parent['assets']}
    assert files == sealed['parent_files']
    bound = {**deepcopy(parent), 'model_dir':str(PACKAGE/'parent'),'model_identity':sealed['parent_identity'],
             'assets':[{'path':str(PACKAGE/'parent'/k),'sha256':v} for k,v in files.items()]}
    result = {}
    for arm in ('parent','child'):
        model, manifest = load_refinement_parent(bound, backend, cfg, data)
        assert state_identity(model) == sealed['scored_receipt']['parent_state_sha256']
        if arm == 'child':
            model, accepted = accepted_copy(model,sealed['scored_receipt'],source_ref=bound['model_ref'],input_shapes=manifest.get('input_shapes',{}))
            assert accepted == prior['receipt'] and state_identity(model) == sealed['child_state_sha256']
        q = _quantizers(model)
        assert tensor_digest(q[P].h2_levels) == prior['verified_custom_levels']
        effective_state = state_identity(model)
        x, _ = next(data.batches('calibration'))
        logits = tensor_digest(forward(backend, model, x))
        if arm == 'child':
            assert logits == prior['logits_sha256']
        directory = OUT/'custom_dev'/arm
        directory.mkdir(parents=True)
        cov = {'declared_sites':[{'site_id':sid,'module':m.layer['module']} for sid,m in q.items()]}
        measured = {}
        for kind,split,n in (('classification','dev',4),('ade_frozen','ade_dev',2)):
            selection = {'kind':kind,'split':split,'count':n}
            row = evaluate_task(backend,cfg,data,model,manifest['plan'],selection,directory,False,cov)
            key, fields = evaluation_summary(selection,row)
            measured[key] = fields
            assert state_identity(model) == effective_state
        result[arm] = {'state_sha256':effective_state,'decoded_classification_sha256':logits,'tasks':measured}
        del q,model
    for key in ('classification','segmentation'):
        assert result['parent']['tasks'][key]['sample_ids'] == result['child']['tasks'][key]['sample_ids']
    atomic_json(RESULT,{'measurement':'two-process physical package regression and paired development prefixes; no formal L3',
        'setup':read_json(OUT/'custom_export_setup.json'),'cold_seconds':time.monotonic()-start,
        'source':prior['custom_source'],'child_manifest_sha256':file_sha256(PACKAGE/'manifest.json'),
        'package_model_ref':fingerprint({'manifest.json':file_sha256(PACKAGE/'manifest.json'),**{'parent/'+k:v for k,v in files.items()}}),
        'data_view':cfg['_development_view']['sha256'],'results':result,'core':code_identity()['core_sha256']})


if __name__ == '__main__':
    torch.set_num_threads(2)
    {'export':export,'cold':cold_evaluate}[sys.argv[1]]()
