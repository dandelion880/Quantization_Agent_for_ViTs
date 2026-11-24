"""Check both production inputs at the same role/planning entry point without running L3."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'QAViT'))
from runtime.paths import ROOT
from core.cards import ContractError, fingerprint
from core.journal import read_json, atomic_json, file_sha256
from core.workflow import Workflow
from agents.roles import role_spec
from tools.experiments import preflight_experiment, resolve_execution
from tools.profiling import inspect_run_model
from tools.experiment_worker import backend_and_data


def check(label, source, numeric_dir, tag):
    config = read_json(ROOT / 'EXP/inputs' / source / 'input.json')
    config.update(purpose='validation', enable_l3=False, name=label + '_context' + tag)
    path = HERE / config['name']
    if path.exists():
        wf = Workflow(path, read_only=True)
        task = wf.queue.get_by_key('context/focus')
    else:
        wf = Workflow(path, config)
        task = wf.queue.get(wf.enqueue_role('context/focus','focus',refs=[wf.brief['id']],context={'round':1}))
    packet = wf.context(task)
    system = role_spec('generation','focus',packet,{})['system_prompt']
    metadata = packet['model_metadata']
    detail = read_json(Path(metadata['read_asset']['asset']))
    assert detail['summary']['model_id'] == metadata['model_id']
    assert file_sha256(Path(metadata['read_asset']['asset'])) == wf.model_metadata['asset']['sha256']
    brief = wf.brief['payload']
    evaluations = brief['evaluation_protocol']['evaluations']
    prior = read_json(numeric_dir / 'input.json')['payload']['execution']
    research = {'level':'L3','question':'Bind the declared final protocol (dry planning only)',
                'measurements':['top1','mIoU'],'execution':{
                    'service':'quantize_evaluate','plan':prior['plan'],
                    'calibration':{'split':'calibration','count':brief['evaluation_protocol']['calibration_samples']},
                    'evaluations':evaluations,'resolved_choices':{'training':False}}}
    plan = preflight_experiment(research,brief,executable=True)
    frozen = resolve_execution(plan['execution'], brief, level='L3')
    assert frozen['research_protocol_sha256'] == wf.solutions.key
    assert frozen['backend']['segmentation_assets'] == detail['asset_binding']['segmentation_assets']
    return {'model_id':metadata['model_id'],'user_input':source,'runtime_summary':metadata,
            'focus_prompt_sha256':fingerprint(system),'binding':brief['model']['binding'],
            'calibration_count':frozen['calibration']['count'],'requested_final_tasks':evaluations,
            'dry_l3_plan_checked':True,'protocol_hash':wf.solutions.key,'B':wf.solutions.fallback(),
            'role_called':False}


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--deit-dir', type=Path, default=HERE/'deit_verified')
    parser.add_argument('--dino-dir', type=Path, default=HERE/'dino_verified')
    parser.add_argument('--tag', default='', help='optional suffix for a fresh pair of context snapshots')
    args = parser.parse_args()
    rows = {'deit':check('deit','deitb_static_w4a4_gonly_20261115_r1', args.deit_dir, args.tag),
            'dino':check('dino','dinov3b_static_w4a4', args.dino_dir, args.tag)}
    assert rows['deit']['focus_prompt_sha256'] == rows['dino']['focus_prompt_sha256']
    for key in ('protocol_hash','binding','model_id'):
        assert rows['deit'][key] != rows['dino'][key]
    brief = read_json(ROOT/'EXP/inputs/dinov3b_static_w4a4/input.json')['research_brief']
    brief['model']['adapter']['backend_overrides'] = {'data_path':None, 'task_head':None, 'segmentation_config':None}
    brief['evaluation_protocol'] = {'kind':'structure_only'}
    other = {'kind':'server','config_path':'QAViT/configs/deitb_backend.json',
             'backend_overrides':{'device':'cpu','data_path':None}}
    brief['model']['allowed_adapters'] = [other]
    available = inspect_run_model(brief, cpu_threads=3)
    assert available['summary']['pool_counts'] is None and available['structure_profile'] is None
    assert available['summary']['model_id'] == rows['dino']['model_id'] and available['summary']['sequence']['patch_start']==5
    rows['weights_only'] = {'raw_structure_measured':False, 'model_id':available['summary']['model_id'],
                            'pool_counts':None, 'available_sites':available['summary']['site_counts']}
    brief['model']['binding'] = available['binding']
    request = {'service':'tensor_probe','backend':other,'site':'blocks.0.mlp.fc2',
               'method':'uniform_minmax','parameters':{'bits':4,'granularity':'per_tensor','signed':True},
               'resolved_choices':{'activation_bits':4,'training':False},'calibration':{'split':'calibration','count':1}}
    bound = resolve_execution(request, brief, level='L1')
    try:
        backend_and_data(bound, inspect_without_data=True)
    except ContractError as exc:
        assert '备选执行适配器加载的模型' in str(exc)
        rows['alternate_provider'] = {'same_research_protocol':True,'wrong_backbone_rejected':True,'reason':str(exc)}
    else:
        raise AssertionError('foreign backbone accepted under a permitted provider')
    atomic_json(HERE/'context_checks.json',rows)
