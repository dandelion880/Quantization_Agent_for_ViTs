"""ADE20K冻结head评价：复用旧MMSeg协议，显式记录分类量化尺度迁移。"""
from pathlib import Path
import sys
from contextlib import contextmanager
from runtime.paths import ROOT, QAVIT
from runtime.files import read, file_hash

SETTINGS = QAVIT / 'configs/ade_evaluation.json'
HEAD = QAVIT / 'configs' / read(SETTINGS)['head_receipt']


@contextmanager
def segmentation_dependencies():
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
        setup_segmentation(seed=42, threads=4)
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


def build_adapter(backbone, device):
    import torch
    from torch import nn
    with segmentation_dependencies():
        from tools.vision_models import build_ade, state_sha
        from timm.layers import resample_abs_pos_embed
        original, _, heads, _ = build_ade(device, HEAD)
        original.segmentor.backbone.freeze_backbone = True
        frozen = [state_sha(h) for h in heads]

    class Features(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            b, _, h, w = x.shape
            if h % 16 or w % 16:
                raise ValueError('ADE输入须patch对齐；不静默改变padding')
            tokens = self.model.patch_embed.proj(x).flatten(2).transpose(1, 2)
            pos = resample_abs_pos_embed(self.model.pos_embed, (h // 16, w // 16),
                                        num_prefix_tokens=1, interpolation='bicubic', antialias=True)
            tokens = self.model.pos_drop(torch.cat((self.model.cls_token.expand(b, -1, -1), tokens), dim=1) + pos)
            features = []
            for index, block in enumerate(self.model.blocks):
                tokens = block(tokens)
                if index in (2, 5, 8, 11):
                    patches = self.model.norm(tokens)[:, 1:]
                    if patches.shape[1] != h * w // 256:
                        raise ValueError('CLS/patch索引错误')
                    features.append(patches.transpose(1, 2).reshape(b, 768, h // 16, w // 16))
            return tuple(features)
    return original, Features(backbone), heads, frozen, state_sha


def evaluate(backbone, reference, directory, count=None):
    """冻结ImageNet校准尺度迁移评价；不将其称作ADE任务校准或head训练。"""
    import torch
    from runtime.files import save
    from ptq_agent.execution.evaluation import model_digest
    device = next(reference.parameters()).device
    segmentor, fp_adapter, heads, frozen, state_sha = build_adapter(reference, device)
    from mmengine.dataset import Compose
    from tools.vision_models import checked_segmentation_step
    settings = read(SETTINGS)
    ADE = Path(settings['data_root'])
    modern = segmentor.segmentor.backbone
    q_adapter = type(fp_adapter)(backbone)
    # 先验证精确官方FP骨干在同一冻结head上的512特征一致性。
    generator = torch.Generator(device='cpu').manual_seed(42)
    x = torch.randn((1, 3, 512, 512), generator=generator).to(device)
    with torch.inference_mode():
        expected, actual = modern(x), fp_adapter(x)
        errors = [{'max_abs': float((a-b).abs().max()),
                   'relative_mse': float((a-b).double().square().sum() / b.double().square().sum().clamp_min(1e-30))}
                  for a, b in zip(actual, expected)]
    if any(e['relative_mse'] > 1e-8 for e in errors):
        raise ValueError('ADE FP adapter未通过数值一致性验证: ' + str(errors))
    segmentor.segmentor.backbone = fp_adapter
    del modern, expected, actual, x
    # JSON将原配置tuple保存为list；MMSeg Resize要求tuple，恢复容器类型而不改数值。
    pipeline = Compose([{**row, 'scale':tuple(row['scale'])} if isinstance(row.get('scale'), list)
                        else row for row in settings['pipeline']])
    files = sorted((ADE / 'images/validation').glob('*.jpg'))
    if len(files) != 2000:
        raise ValueError('ADE20K validation必须2000图')
    if count is not None:
        if not 0 < count <= 2000:
            raise ValueError('ADE count越界')
        files = files[:count]
    histograms = {name: torch.zeros((150,150), dtype=torch.int64) for name in ('fp', 'quantized')}
    model_hashes = [model_digest(reference), model_digest(backbone)]
    with torch.inference_mode():
        for i, p in enumerate(files):
            annotation = ADE / 'annotations/validation' / (p.stem + '.png')
            packed = pipeline(dict(img_path=str(p), seg_map_path=str(annotation), reduce_zero_label=True, seg_fields=[]))
            for name, adapter in (('fp', fp_adapter), ('quantized', q_adapter)):
                segmentor.segmentor.backbone = adapter
                output = checked_segmentation_step(segmentor, packed)
                pred, gt = output.pred_sem_seg.data.cpu().long().reshape(-1), output.gt_sem_seg.data.cpu().long().reshape(-1)
                valid = gt != 255
                if not ((gt[valid] >= 0) & (gt[valid] < 150)).all():
                    raise ValueError('ADE标签越界')
                histograms[name] += torch.bincount(150 * gt[valid] + pred[valid], minlength=22500).reshape(150,150)
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
    return {'images':len(files), 'full_dataset_evaluation':len(files)==2000, 'classes':150,
            'metrics':results, 'delta_mIoU_pp':results['quantized']['mIoU']-results['fp']['mIoU'],
            'head_receipt':str(HEAD), 'head_receipt_sha256':file_hash(HEAD), 'head_checkpoint':read(HEAD)['checkpoint'],
            'head_hashes':frozen, 'head_unchanged':True, 'backbones_unchanged':True,
            'backbone_hashes':model_hashes,
            'backbone_hash_scope':'all named parameters and buffers, including nonpersistent decoded/reference weights',
            'fp_adapter_parity':errors, 'sample_ids':[str(p) for p in files],
            'protocol':{'mode':'slide','crop':[512,512],'stride':[341,341], 'resize':[2048,512],
                        'config':str(SETTINGS), 'preprocessing':settings['pipeline']},
            'scope':'ImageNet-calibrated static W4A4 backbone transfer; original classification head unused; frozen FP32 Neck/Head',
            'recalibrated':False, 'trained':False, 'native_low_precision_kernel':False}
