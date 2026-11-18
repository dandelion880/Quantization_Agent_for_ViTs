"""V2 Profiling 服务目录、工具会话与已验证候选绑定。"""
from pathlib import Path
from core.cards import ContractError, fingerprint, validate_card
from core.journal import atomic_json, read_json, file_sha256
from tools.experiments import asset, verify_assets
from ptq_agent.profiling.host import ProfileHost

PROFILE_NAMES = ('query_structure', 'query_execution_capabilities', 'query_ptq_plugins', 'query_ptq_sites', 'profile_structure', 'profile_distribution', 'profile_hardware', 'profile_sensitivity', 'profile_plugin_sites', 'profile_ptq_candidates', 'profile_trial_diagnostics', 'profile_trial_ablation', 'read_profile_evidence', 'query_method_library', 'read_method_document')


# 这些服务的实现实际从 calibration split 捕获样本；目录/能力/历史读取不捕获。
CALIBRATION_PROFILE_NAMES = frozenset({
    'profile_structure', 'profile_distribution', 'profile_hardware', 'profile_sensitivity',
    'profile_plugin_sites', 'profile_ptq_candidates', 'profile_trial_diagnostics', 'profile_trial_ablation',
})


def uses_calibration(execution):
    return (execution.get('tool') in CALIBRATION_PROFILE_NAMES
            and not (execution.get('tool') == 'profile_distribution'
                     and execution.get('arguments', {}).get('target') == 'weight'))


def profile_host(backend, cfg, directory):
    """建立完整纯工具会话；计数只计量，时间行为保留。"""
    return ProfileHost(backend, cfg, directory)


def bind_profile_evidence(journal, reference):
    """从本次运行已提交的成功测量授权读取；不扫描孤立作业或相信调用方绑定。"""
    path = Path(reference)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "evidence" or ".." in path.parts:
        raise ContractError("Profiling 证据引用必须是已授权的 evidence/ 文件")
    for card in journal.find_cards("EvidenceCard"):
        facts = card["payload"]
        if (card["run_id"] != journal.run_id or facts["status"] != "completed"
                or facts["measurement_source"] == "test_double"):
            continue
        source = journal.card(facts["experiment_ref"])
        if (source["payload"]["execution"]["service"] != "profile"
                or facts["observations"].get("result", {}).get("evidence") != reference):
            continue
        assets = {Path(a["path"]).resolve(): a for a in facts["artifacts"]}
        for result_path, result_asset in assets.items():
            if result_path.name != "profile_result.json" or not result_path.is_relative_to(journal.root / "jobs"):
                continue
            measurement_path = (result_path.parent / "profile" / path).resolve()
            if measurement_path not in assets or not measurement_path.is_relative_to(result_path.parent / "profile/evidence"):
                continue
            binding = {"reference": reference, "evidence_ref": card["id"],
                       "evidence_asset": asset(journal.card_path(card["id"])),
                       "result_asset": result_asset, "measurement_asset": assets[measurement_path]}
            verify_assets(binding)
            result = read_json(result_path)
            if (result["status"] == "success" and reference in result["artifacts"]
                    and result["data"] == facts["observations"]["result"]):
                return [binding]
    raise ContractError("只能读取本次运行已提交并绑定原始资产的成功 Profiling 测量")


def restore_profile_evidence(host, bindings, run_id):
    """把冻结授权的测量复制入新作业，并恢复成功返回账目；保留底层权限检查。"""
    from ptq_agent.profiling.evidence import read_measurement
    for binding in bindings:
        verify_assets(binding)
        source = read_json(binding["evidence_asset"]["path"])
        validate_card(source)
        result = read_json(binding["result_asset"]["path"])
        reference = binding["reference"]
        path = Path(reference)
        if (source["kind"] != "EvidenceCard" or source["run_id"] != run_id or source["id"] != binding["evidence_ref"]
                or source["payload"]["status"] != "completed" or source["payload"]["measurement_source"] == "test_double"
                or any(binding[key] not in source["payload"]["artifacts"] for key in ("result_asset", "measurement_asset"))
                or result["status"] != "success" or reference not in result["artifacts"]
                or result["data"] != source["payload"]["observations"]["result"]
                or path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "evidence" or ".." in path.parts):
            raise ContractError("Profiling 恢复的证据归属或成功登记不匹配")
        measurement = read_measurement(Path(binding["measurement_asset"]["path"]), host.state["identity"])
        atomic_json(host.run_dir / path, measurement)
        host.state["tool_results"].append({"restored_from": source["id"], "result": result})
    host.save()


def register_trials(host, bindings):
    """把明确引用的 v2 实测候选投影为旧诊断接口的只读索引；不导入旧 Memory。"""
    from ptq_agent.profiling.evidence import seal
    for binding in bindings:
        verify_assets(binding)
        model = Path(binding["model_dir"])
        manifest, evaluation = read_json(model / "manifest.json"), read_json(binding["evaluation_asset"]["path"])
        if manifest["identity"] != host.state["identity"] or manifest["weights_sha256"] != file_sha256(model / "weights.pt"):
            raise ContractError("候选模型/数据/源码身份与当前诊断基准不一致")
        if evaluation.get("status") != "completed" or evaluation.get("split") != "search":
            raise ContractError("不完整或非search候选不能进入旧诊断接口")
        tid = binding["trial_id"]
        directory = host.run_dir.parent / "trials" / tid
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / "artifact"
        if not link.exists():
            link.symlink_to(model, target_is_directory=True)
        elif link.resolve() != model.resolve():
            raise ContractError("候选只读索引发生冲突")
        trial = seal({"trial_id": tid, "status": "completed", "candidate": manifest["plan"], "artifact": manifest,
                      "evaluation": evaluation, "metrics": evaluation["metrics"], "error": None,
                      "source_evidence_ref": binding["evidence_ref"], "source_evidence_asset": binding["evidence_asset"],
                      "scope": evaluation["scope"]})
        atomic_json(directory / "result.json", trial)
        feedback = {"id": "execution-" + tid, "trial_id": tid, "split": "search", "metrics": [], "difficulties": [],
                    "summary": "本次v2运行已完成测量，实际样本数与subset标记保留；不是正式完整评测。",
                    "evidence": [str((directory / "result.json").resolve())]}
        reference = "feedback/" + fingerprint(feedback) + ".json"
        host.state.setdefault("execution_feedback", []).append({"source": "program_verified", "trial": tid,
            "evidence": [reference], "configuration": manifest["plan"], "input_shapes": manifest["input_shapes"],
            "metrics": evaluation["metrics"], "gate": None, "error": None})
        if host.feedback_source(feedback) != "program_verified":
            raise ContractError("v2 候选桥接未通过旧诊断来源检查")
        record = {"source": "program_verified", "data": feedback, "reference": reference}
        atomic_json(host.run_dir / reference, record)
        host.state["profile_feedback"].append(record)
    host.save()



def inspect_run_model(brief, cpu_threads=4):
    if brief['model'].get('adapter', {}).get('kind') not in ('server', 'factory'):
        return None
    import torch
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        return _inspect_run_model(brief)
    finally:
        torch.set_num_threads(previous)


def _inspect_run_model(brief):
    """One CPU-only binding for a new run. No search, calibration fit or performance metric."""
    from collections import Counter
    from copy import deepcopy
    from tools.experiments import bind_backend
    from tools.experiment_worker import backend_and_data
    from tools.vision_models import segmentation_assets
    from ptq_agent.profiling.torch_profiler import TorchProfiler
    from ptq_agent.execution.sites import site_catalog

    model = brief['model']
    adapter = model.get('adapter', {})
    seg = any(e.get('kind') == 'ade_frozen' for e in brief['evaluation_protocol'].get('evaluations', []))
    bound = bind_backend(adapter, segmentation=seg)
    probe = deepcopy(bound)
    probe.setdefault('backend_overrides', {})['device'] = 'cpu'
    backend, cfg, data = backend_and_data({'backend': probe}, inspect_without_data=True)
    data_identity = data.identity if data is not None else {}
    spec = getattr(backend, 'spec', {})
    actual_id = spec.get('model_id', cfg['backend'].get('model_id'))
    declared = model.get('model_id', model.get('scope', {}).get('model'))
    if adapter.get('kind') == 'server' and any(i and i != actual_id for i in
            (model.get('model_id'), model.get('scope', {}).get('model'))):
        raise ContractError(f'ResearchBrief model identity {declared} differs from bound model {actual_id}')
    catalog = site_catalog(backend, cfg)
    blocks = spec.get('block_paths', [])
    summary = {k: spec[k] for k in ('family', 'architecture', 'patch_size', 'depth', 'hidden_dim', 'num_heads',
               'cls_token_count', 'special_token_type', 'special_token_count', 'prefix_token_count',
               'block_forward', 'block_type', 'task_forward', 'classification_head') if k in spec}
    summary.update(model_id=actual_id, checkpoint_sha256=backend.identity().get('checkpoint_sha256'),
                   site_counts=dict(Counter(row['kind'] for row in catalog.values())),
                   site_examples={kind: next(s for s, r in catalog.items() if r['kind'] == kind)
                                  for kind in sorted({r['kind'] for r in catalog.values()})},
                   classification_head=getattr(backend, 'task_head', None) or spec.get('classification_head'),
                   preprocessing=data_identity.get('preprocessing'), pool_counts=data_identity.get('counts'),
                   split_sources=data_identity.get('split_sources'))
    if isinstance(summary.get('classification_head'), dict):
        summary['classification_head'] = {k: summary['classification_head'][k] for k in
            ('kind', 'feature', 'protected_module', 'sha256') if k in summary['classification_head']}
    profile = None
    if blocks and getattr(backend, 'data_adapter', None) and callable(getattr(backend, 'samples', None)):
        # Use the existing FP structure profiler on one identified calibration input.
        # No quantizer fitting and no per-role profiling; keep the module tree on disk.
        profiler = TorchProfiler(backend, cfg['constraints'])
        profile = profiler.structure([blocks[0]], 1, cfg['backend']['seed'], 0)
        row = next(r for r in profile['modules'] if r['name'] == blocks[0])
        summary.update(first_block=blocks[0], last_block=blocks[-1], example_block_call=row['calls'][0],
                       input=profile['input'], example_sample_ids=profile['sample_ids'])
        summary['task_output'] = profile['output']
    if spec.get('prefix_token_count') is not None:
        prefix = spec['prefix_token_count']
        summary['sequence'] = {'cls': [0,spec['cls_token_count']], 'special': [spec['cls_token_count'],prefix],
                               'patch_start':prefix, 'intervals':'half-open, image sequence'}
    if bound.get('segmentation_assets'):
        _, settings, _, receipt = segmentation_assets(cfg['backend'])
        if receipt.get('backbone_sha256', receipt.get('deit_backbone_sha256')) != summary['checkpoint_sha256']:
            raise ContractError("分割 head 与指定模型身份不同")
        summary['segmentation'] = {'feature_interface': settings['model']['backbone']['type'],
            'out_indices': settings['model']['backbone']['out_indices'], 'test_cfg': settings['model']['test_cfg'],
            'head_sha256': receipt['checkpoint_sha256'], 'preprocessing': settings['pipeline']}
    summary['profile_tools'] = ['query_ptq_sites', 'profile_structure', 'profile_distribution', 'read_profile_evidence']
    binding = {'sha256': fingerprint(bound), 'segmentation': bool(bound.get('segmentation_assets'))}
    if model.get('allowed_adapters'):
        from tools.experiments import model_task_identity
        binding['alternate_source'] = model_task_identity(backend, cfg['backend'])
        binding['task_assets'] = [a['sha256'] for a in bound.get('segmentation_assets', [])]
    # Preview the existing evaluator order without loading extra images. Profiles
    # with their own seed report their actual IDs in their Evidence.
    previews = {s: data.ids(s)[:32] for s in ('calibration', 'search')} if data is not None else {}
    summary['sample_selection'] = {'detail_field': 'sample_views', 'ordering_seeds': data_identity.get('ordering_seeds'),
        'meaning': 'evaluation/fit use the fixed per-split order; seeded profiling reports its own actual IDs'}
    return {'summary': summary, 'sample_views': previews, 'asset_binding': bound, 'module_inventory': backend.modules(),
            'site_catalog': catalog, 'structure_profile': profile,
            'binding': binding}
