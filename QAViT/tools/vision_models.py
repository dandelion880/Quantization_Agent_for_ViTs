"""本地DeiT模型与冻结ADE头适配；沿用已验证构造、加载和推理数学。"""
from pathlib import Path
import collections.abc
import importlib.util
import os
import sys
import types
from runtime.paths import ROOT
from runtime.files import read, file_hash

QAVIT = ROOT / 'QAViT'


def load_model(spec):
    """官方timm 0.3.2 DeiT，strict加载并逐张量核对；不下载或训练。"""
    import torch
    if spec['family'] != 'DeiT' or spec['model_id'] != 'deit_base_patch16_224':
        raise ValueError('本版已验证本地模型入口仅支持指定non-distilled DeiT-B')
    shim = types.ModuleType('torch._six')
    shim.container_abcs = collections.abc
    sys.modules.setdefault('torch._six', shim)
    import timm
    if timm.__version__ != '0.3.2':
        raise ValueError('分类模型必须使用已验证timm 0.3.2')
    path = Path(spec['code_path']) / 'models.py'
    module_spec = importlib.util.spec_from_file_location('qavit_official_deit', path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    model = module.deit_base_patch16_224(pretrained=False).float().eval()
    payload = torch.load(spec['checkpoint_path'], map_location='cpu', weights_only=True)
    state = payload.get('model', payload)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('checkpoint加载不完整')
    for key, value in model.state_dict().items():
        if not torch.equal(value, state[key]):
            raise ValueError('加载张量与checkpoint不同: ' + key)
    return model, dict(strict=True, missing_keys=[], unexpected_keys=[],
                       loaded_tensor_count=len(state), loaded_tensors_bitwise_equal=True,
                       excluded_checkpoint_keys=[])


def state_sha(model):
    import hashlib
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        value = hashlib.sha256(str((t.dtype, tuple(t.shape))).encode() + t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        h.update((name + value).encode())
    return h.hexdigest()


def setup_segmentation(seed=42, threads=None):
    import random
    import numpy as np
    import torch
    # Placement/authorization belongs to the host UUID lease. Here the supplied
    # model device is worker-local (normally cuda:0); CPU remains a valid path.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if threads is not None:
        torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_ade(device, receipt_path):
    import torch
    from torch import nn
    from contextlib import nullcontext
    import timm
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    from mmengine.config import ConfigDict
    settings = read(QAVIT / 'configs/ade_evaluation.json')
    asset = read(QAVIT / 'configs/assets/deit_model_manifest.json')['models'][0]
    checkpoint = (QAVIT / 'configs/assets' / asset['checkpoint_path']).resolve()
    if file_hash(checkpoint) != asset['checkpoint_hash']['value']:
        raise ValueError('DeiT母模型hash改变')

    @MODELS.register_module(force=True)
    class DeiTDualTaskBackbone(nn.Module):
        def __init__(self, freeze_backbone=True, out_indices=(2, 5, 8, 11)):
            super().__init__()
            self.model = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=False, dynamic_img_size=True)
            self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
            if self.model.num_prefix_tokens != 1 or len(self.model.blocks) != 12:
                raise ValueError('非预期DeiT架构')
            self.model.head = nn.Identity()
            self.model.float().eval().requires_grad_(False)
            self.freeze_backbone = freeze_backbone
            self.out_indices = tuple(out_indices)
            if self.out_indices != (2, 5, 8, 11):
                raise ValueError('固定四级patch特征层')
        def forward(self, x):
            if x.ndim != 4 or x.shape[-1] % 16 or x.shape[-2] % 16:
                raise ValueError('输入必须是patch对齐的BCHW')
            with torch.no_grad() if self.freeze_backbone else nullcontext():
                out = self.model.forward_intermediates(x, indices=self.out_indices, norm=True,
                    intermediates_only=True, output_fmt='NCHW')
            shape = (x.shape[0], 768, x.shape[-2] // 16, x.shape[-1] // 16)
            if len(out) != 4 or any(tuple(t.shape) != shape for t in out):
                raise ValueError('patch特征索引/形状不匹配')
            return tuple(out)
        def train(self, mode=True):
            super().train(mode)
            self.model.eval()
            return self

    register_all_modules(init_default_scope=True)
    segmentor = MODELS.build(ConfigDict(settings['model']))
    backbone = segmentor.backbone.model
    heads = [segmentor.neck, segmentor.decode_head]
    before = state_sha(backbone)
    receipt = read(receipt_path)
    if receipt['status'] != 'formal_head_complete' or receipt['task'] != 'ade':
        raise ValueError('缺少正式ADE head来源')
    if receipt['training_budget'] != 160000 or not receipt['backbone_unchanged']:
        raise ValueError('head训练来源不符合已验证协议')
    if receipt['deit_backbone_sha256'] != asset['checkpoint_hash']['value']:
        raise ValueError('head母模型来源不匹配')
    if file_hash(receipt['checkpoint']) != receipt['checkpoint_sha256']:
        raise ValueError('head checkpoint hash不匹配')
    payload = torch.load(receipt['checkpoint'], map_location='cpu')
    for name, head in zip(('neck', 'decode_head'), heads):
        head.load_state_dict(payload[name], strict=True)
    if before != receipt['backbone_state_sha256'] or state_sha(backbone) != before:
        raise ValueError('head覆盖了骨干，或骨干来源不匹配')

    class SegLogits(nn.Module):
        def __init__(self, segmentor):
            super().__init__()
            self.segmentor = segmentor
        def to(self, *args, **kwargs):
            self.segmentor.to(*args, **kwargs)
            return self
        def forward(self, x):
            return self.segmentor.decode_head(self.segmentor.extract_feat(x))
        @property
        def encode_decode(self):
            return self.segmentor.encode_decode
        @encode_decode.setter
        def encode_decode(self, value):
            self.segmentor.encode_decode = value
        def test_step(self, data):
            return self.segmentor.test_step(data)

    model = SegLogits(segmentor).float().to(device).eval().requires_grad_(False)
    return model, backbone, heads, 'segmentor.backbone.model'


def checked_segmentation_step(model, packed):
    import torch
    from mmengine.dataset import pseudo_collate
    original = model.encode_decode
    def checked(*args, **kwargs):
        out = original(*args, **kwargs)
        if not torch.isfinite(out).all():
            raise RuntimeError('Nonfinite segmentation crop logits')
        return out
    model.encode_decode = checked
    try:
        return model.test_step(pseudo_collate([packed]))[0]
    finally:
        model.encode_decode = original
