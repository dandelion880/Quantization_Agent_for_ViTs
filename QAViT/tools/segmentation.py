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


def task_pipeline(settings, *, annotations=True):
    from mmengine.dataset import Compose
    return Compose([{**row, 'scale': tuple(row['scale'])} if isinstance(row.get('scale'), list) else row
                    for row in settings['pipeline'] if annotations or row['type'] != 'LoadAnnotations'])


class DenseTask:
    """Original image -> bound MMSeg preprocessing -> individual inference calls."""
    def __init__(self, backend):
        from tools.vision_models import build_ade, segmentation_assets
        path, settings, head, receipt = segmentation_assets(backend.config)
        with segmentation_dependencies(backend.config['seed']):
            wrapped, self.features, _ = build_ade(backend, settings, head, receipt)
            self.pipeline = task_pipeline(settings, annotations=False)
        self.segmentor = wrapped.segmentor
        self.indices = tuple(settings['model']['backbone']['out_indices'])
        self.test_cfg = settings['model']['test_cfg']
        self.identity = {'settings_sha256': file_hash(path), 'head_sha256': receipt['checkpoint_sha256'],
                         'feature_layers': list(self.indices), 'test_cfg': self.test_cfg,
                         'pipeline': [r for r in settings['pipeline'] if r['type'] != 'LoadAnnotations'],
                         'data_preprocessor': settings['model']['data_preprocessor']}

    def windows(self, row):
        from mmengine.dataset import pseudo_collate
        path = Path(row['path'])
        if file_hash(path) != row['file_sha256']:
            raise ValueError('dense view original differs from the selected manifest')
        packed = self.pipeline(dict(img_path=str(path), seg_fields=[]))
        prepared = self.segmentor.data_preprocessor(pseudo_collate([packed]), training=False)
        meta = prepared['data_samples'][0].metainfo
        image = prepared['inputs']
        h, w = image.shape[-2:]
        mode = self.test_cfg.get('mode', 'whole')
        if mode not in ('slide', 'whole'):
            raise ValueError('unsupported bound segmentation inference mode ' + mode)
        ch, cw = self.test_cfg['crop_size'] if mode == 'slide' else (h, w)
        sh, sw = self.test_cfg['stride'] if mode == 'slide' else (h, w)
        for iy in range(1 + max(0, h - ch + sh - 1) // sh):
            for ix in range(1 + max(0, w - cw + sw - 1) // sw):
                bottom, right = min(iy * sh + ch, h), min(ix * sw + cw, w)
                top, left = max(bottom - ch, 0), max(right - cw, 0)
                crop = image[:, :, top:bottom, left:right]
                yield crop, {**meta, 'img_shape': tuple(crop.shape[-2:])}, {
                    'original_hw': list(meta['ori_shape']), 'resized_hw': list(meta['img_shape']),
                    'prepared_shape': list(image.shape), 'box_yxyx': [top, left, bottom, right]}

    def observe(self, model, x, meta, *, logits):
        # Features are exactly the tensors passed to the frozen neck by extract_feat.
        features = self.features(model)(x)
        result = {str(i): t for i, t in zip(self.indices, features)}
        if logits:
            neck = self.segmentor.neck(features) if self.segmentor.with_neck else features
            result['logits'] = self.segmentor.decode_head.predict(neck, [meta], self.segmentor.test_cfg)
        return result


def _fp_identity(backend, files, annotation_dir, *, settings, head, weights, classes, ignore):
    from importlib import import_module
    import torch
    from core.cards import fingerprint
    from runtime.identity import measurement_implementation
    pairs = [(str(p), file_hash(p), file_hash(annotation_dir / (p.stem + '.png'))) for p in files]
    device = torch.device(backend.device)
    return fingerprint({'schema': 1, 'assets': [file_hash(settings), file_hash(head), *weights],
        'ordered_images_and_annotations': pairs, 'classes': classes, 'ignore': int(ignore),
        'implementation': measurement_implementation(), 'seed': backend.config['seed'],
        'dependency_versions': {name: str(import_module(name).__version__) for name in
            ('torch','torchvision','mmseg','mmengine','mmcv','numpy','cv2','PIL')},
        'pinned_dependencies': file_hash(QAVIT / 'docs/runtime_dependencies.json'),
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'device': [torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)] if device.type == 'cuda' else 'cpu',
        'threads': torch.get_num_threads(), 'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
        'deterministic': [torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark]})


def evaluate(backbone, backend, directory, count=None, *, fp_reference=None, development=None):
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
    from tools.vision_models import checked_segmentation_step
    ADE = Path(settings['data_root'])
    modern = segmentor.segmentor.backbone
    # JSON将原配置tuple保存为list；MMSeg Resize要求tuple，恢复容器类型而不改数值。
    pipeline = task_pipeline(settings)
    expected_images = settings.get('expected_images')
    if development is None:
        files = sorted((ADE / 'images/validation').glob('*.jpg'))
        annotation_dir = ADE / 'annotations/validation'
        if not files or (expected_images is not None and len(files) != expected_images):
            raise ValueError('分割 validation 图像数量与任务资产不符')
    else:
        from tools.development import segmentation_files
        files, annotation_dir = segmentation_files(ADE, development)
    total_files = len(files)
    classes = settings['model']['decode_head']['num_classes']
    ignore_index = segmentor.segmentor.decode_head.ignore_index
    if count is not None:
        if not 0 < count <= total_files:
            raise ValueError('ADE count越界')
        files = files[:count]
    histograms = {name: torch.zeros((classes,classes), dtype=torch.int64) for name in ('fp', 'quantized')}
    model_hashes = [model_digest(reference), model_digest(backbone)]
    def identity():
        return _fp_identity(backend, files, annotation_dir, settings=settings_path, head=head_path,
                            weights=[model_hashes[0], receipt['checkpoint_sha256'], *frozen],
                            classes=classes, ignore=ignore_index)
    identity_before = identity()
    reused = None
    if fp_reference and file_hash(fp_reference['path']) == fp_reference['sha256']:
        prior = read(fp_reference['path'])
        if identity_before and prior.get('fp_identity') == identity_before:
            counts = torch.tensor(prior['metrics']['fp']['confusion_matrix'])
            if counts.dtype != torch.int64 or counts.shape != (classes, classes) or (counts < 0).any():
                raise ValueError('invalid reference FP confusion counts')
            histograms['fp'] = counts
            reused = fp_reference['sha256']
    adapters = [('quantized', q_adapter)] if reused else [('fp', fp_adapter), ('quantized', q_adapter)]
    with torch.inference_mode():
        for i, p in enumerate(files):
            annotation = annotation_dir / (p.stem + '.png')
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
            for name, adapter in adapters:
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
    if identity() != identity_before:
        raise ValueError('ADE reference inputs or implementation changed during evaluation')
    results = {}
    for name, hist in histograms.items():
        tp = hist.diag().double(); total = hist.sum(1); predicted = hist.sum(0)
        results[name] = {'mIoU':float((tp/(total+predicted-tp)).nanmean()*100),
                         'mAcc':float((tp/total).nanmean()*100), 'aAcc':float(tp.sum()/hist.sum()*100),
                         'confusion_matrix':hist.tolist()}
    return {'images':len(files), 'split': 'ade_dev' if development is not None else 'ade_validation',
            'complete_view':len(files)==total_files, 'full_dataset_evaluation':development is None and expected_images is not None and len(files)==expected_images, 'classes':classes,
            'fp_identity': identity_before, 'fp_reference_sha256': reused,
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
