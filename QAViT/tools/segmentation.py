"""ADE20K冻结head评价：复用旧MMSeg协议，显式记录分类量化尺度迁移。"""
from pathlib import Path
import sys
from contextlib import contextmanager
from runtime.paths import ROOT, QAVIT
from runtime.files import read, file_hash



@contextmanager
def segmentation_dependencies(seed=42):
    # 两份已验证timm在独立命名空间加载；不替换任何forward/权重/上游文件。
    saved = {k: v for k, v in sys.modules.items() if k == 'timm' or k.startswith('timm.')}
    paths = list(sys.path)
    # PyTorch按模块名缓存TorchScript重载；两份timm的DenseNet同名但行号不同。
    # 仅隔离导入注册表，退出恢复。该工具使用eager forward，不启用JIT。
    import torch._jit_internal as jit
    registries = (jit._overloaded_methods, jit._overloaded_method_class_fileno)
    saved_jit = [{k:v for k,v in registry.items() if 'timm.' in str(k)} for registry in registries]
    for registry, entries in zip(registries, saved_jit):
        for k in entries:
            del registry[k]
    for k in saved:
        del sys.modules[k]
    try:
        sys.path = [str(QAVIT / '.runtime_pkgs/segmentation')] + [
            p for p in sys.path if p != str(QAVIT / '.runtime_pkgs/classification')]
        from tools.vision_models import setup_segmentation
        setup_segmentation(seed=seed)
        yield
    finally:
        for k in list(sys.modules):
            if k == 'timm' or k.startswith('timm.'):
                del sys.modules[k]
        sys.modules.update(saved)
        for registry, entries in zip(registries, saved_jit):
            for k in list(registry):
                if 'timm.' in str(k):
                    del registry[k]
            registry.update(entries)
        sys.path[:] = paths


def evaluate(backbone, backend, directory, count=None):
    """Use the selected task's trained head, preprocessing and frozen backbone."""
    import torch
    from runtime.files import save
    from ptq_agent.execution.evaluation import model_digest
    from tools.vision_models import build_ade, segmentation_assets, state_sha
    reference, device = backend.model, backend.device
    settings_path, settings, head_path, receipt = segmentation_assets(backend.config)
    with segmentation_dependencies(backend.config['seed']):
        segmentor, features, heads = build_ade(backend, settings, head_path, receipt)
    fp_adapter, q_adapter = features(reference), features(backbone)
    frozen = [state_sha(h) for h in heads]
    from mmengine.dataset import Compose
    from tools.vision_models import checked_segmentation_step
    ADE = Path(settings['data_root'])
    modern = segmentor.segmentor.backbone
    # JSON将原配置tuple保存为list；MMSeg Resize要求tuple，恢复容器类型而不改数值。
    pipeline = Compose([{**row, 'scale':tuple(row['scale'])} if isinstance(row.get('scale'), list)
                        else row for row in settings['pipeline']])
    files = sorted((ADE / 'images/validation').glob('*.jpg'))
    total_files = len(files)
    expected_images = settings.get('expected_images')
    if not files or (expected_images is not None and total_files != expected_images):
        raise ValueError('分割 validation 图像数量与任务资产不符')
    classes = settings['model']['decode_head']['num_classes']
    ignore_index = segmentor.segmentor.decode_head.ignore_index
    if count is not None:
        if not 0 < count <= total_files:
            raise ValueError('ADE count越界')
        files = files[:count]
    histograms = {name: torch.zeros((classes,classes), dtype=torch.int64) for name in ('fp', 'quantized')}
    model_hashes = [model_digest(reference), model_digest(backbone)]
    with torch.inference_mode():
        for i, p in enumerate(files):
            annotation = ADE / 'annotations/validation' / (p.stem + '.png')
            packed = pipeline(dict(img_path=str(p), seg_map_path=str(annotation), reduce_zero_label=True, seg_fields=[]))
            if i == 0:
                from mmengine.dataset import pseudo_collate
                prepared = segmentor.segmentor.data_preprocessor(pseudo_collate([packed]), training=False)
                h, w = settings['model']['test_cfg']['crop_size']
                x = prepared['inputs'][:, :, :h, :w]
                expected, actual = modern(x), fp_adapter(x)
                if len(actual) != len(expected):
                    raise ValueError('分割特征层数量不同')
                errors = [{'max_abs': float((a-b).abs().max()),
                           'relative_mse': float((a-b).double().square().sum()/b.double().square().sum().clamp_min(1e-30))}
                          for a,b in zip(actual, expected)]
                if any(e['relative_mse'] > 1e-8 for e in errors):
                    raise ValueError('分割 FP adapter 数值一致性检查失败: ' + str(errors))
                del modern, expected, actual, prepared, x
            for name, adapter in (('fp', fp_adapter), ('quantized', q_adapter)):
                segmentor.segmentor.backbone = adapter
                output = checked_segmentation_step(segmentor, packed)
                pred, gt = output.pred_sem_seg.data.cpu().long().reshape(-1), output.gt_sem_seg.data.cpu().long().reshape(-1)
                valid = gt != ignore_index
                if not ((gt[valid] >= 0) & (gt[valid] < classes)).all():
                    raise ValueError('ADE标签越界')
                histograms[name] += torch.bincount(classes * gt[valid] + pred[valid], minlength=classes**2).reshape(classes,classes)
            if (i+1) % 10 == 0:
                save(directory / 'ade_progress.json', {'completed_images': i+1, 'requested':len(files)})
                print('ADE', i+1, '/', len(files), flush=True)
    if [state_sha(h) for h in heads] != frozen or [model_digest(reference), model_digest(backbone)] != model_hashes:
        raise ValueError('ADE评价改变骨干或冻结Neck/Head')
    results = {}
    for name, hist in histograms.items():
        tp = hist.diag().double(); total = hist.sum(1); predicted = hist.sum(0)
        results[name] = {'mIoU':float((tp/(total+predicted-tp)).nanmean()*100),
                         'mAcc':float((tp/total).nanmean()*100), 'aAcc':float(tp.sum()/hist.sum()*100),
                         'confusion_matrix':hist.tolist()}
    return {'images':len(files), 'full_dataset_evaluation':expected_images is not None and len(files)==expected_images, 'classes':classes,
            'metrics':results, 'delta_mIoU_pp':results['quantized']['mIoU']-results['fp']['mIoU'],
            'head_receipt':str(head_path), 'head_receipt_sha256':file_hash(head_path), 'head_checkpoint':receipt['checkpoint'],
            'checkpoint_sha256':backend.checkpoint_hash, 'head_checkpoint_sha256':receipt['checkpoint_sha256'],
            'head_hashes':frozen, 'head_unchanged':True, 'backbones_unchanged':True,
            'backbone_hashes':model_hashes,
            'backbone_hash_scope':'all named parameters and buffers, including nonpersistent decoded/reference weights',
            'fp_adapter_parity':errors, 'sample_ids':[str(p) for p in files],
            'protocol':{**settings['model']['test_cfg'], 'feature_layers':settings['model']['backbone']['out_indices'],
                        'config':str(settings_path), 'config_sha256':file_hash(settings_path), 'preprocessing':settings['pipeline']},
            'scope':'frozen quantized backbone transfer (quantization scope in experiment plan); original classification head unused; frozen FP32 Neck/Head',
            'recalibrated':False, 'trained':False, 'native_low_precision_kernel':False}
