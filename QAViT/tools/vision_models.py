"""绑定本地官方构造器及冻结任务头。模型源码、权重和前向语义来自所选资产。"""
from pathlib import Path
import collections.abc
import importlib.util
import sys
import types
from runtime.paths import ROOT
from runtime.files import read, file_hash

QAVIT = ROOT / 'QAViT'


def construct_model(spec):
    """Dispatch on the bound source API. Constructor names belong to the asset."""
    root = Path(spec['code_path']).resolve()
    source = spec['code_source']['repo_url'].removesuffix('.git').rstrip('/')
    if source == 'https://github.com/facebookresearch/deit':
        shim = types.ModuleType('torch._six')
        shim.container_abcs = collections.abc
        sys.modules.setdefault('torch._six', shim)
        import timm
        if timm.__version__ != '0.3.2':
            raise ValueError('此 DeiT 源码须使用 timm 0.3.2')
        entry = importlib.util.spec_from_file_location('qavit_official_deit', root / 'models.py')
        module = importlib.util.module_from_spec(entry)
        sys.modules[entry.name] = module
        entry.loader.exec_module(module)
    elif source == 'https://github.com/facebookresearch/dinov3':
        import torch._dynamo.config as dynamo
        # Official block.py sets this compiler cache option at import time.
        if not hasattr(dynamo, 'accumulated_cache_size_limit'):
            dynamo._config['accumulated_cache_size_limit'] = 1024
            dynamo._default['accumulated_cache_size_limit'] = 1024
            dynamo._allowed_keys.add('accumulated_cache_size_limit')
        sys.path.insert(0, str(root))
        try:
            from dinov3.hub import backbones as module
        finally:
            sys.path.remove(str(root))
        if Path(module.__file__).resolve() != root / 'dinov3/hub/backbones.py':
            raise ValueError('DINOv3 constructor import 与绑定源码目录不一致')
    else:
        raise ValueError('模型构造接口尚未适配，可绑定显式 factory: ' + source)
    return getattr(module, spec['constructor'])(pretrained=False)


def load_model(spec):
    import torch
    model = construct_model(spec).float().eval()
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


def segmentation_settings(config):
    """Resolve task-specific settings. Only exact archived assets have legacy bindings."""
    path = config.get('segmentation_config')
    legacy_settings = {}
    if not path and config.get('asset_manifest'):
        legacy = read(QAVIT / 'configs/assets/legacy_task_bindings.json')
        rows = legacy.get(file_hash(Path(config['asset_manifest'])), {})
        legacy_settings = rows.get(config.get('model_id'), {})
        relative = legacy_settings.get('segmentation_config')
        if relative:
            path = QAVIT / 'configs' / relative
    if not path:
        raise ValueError('ade_frozen 需要绑定该骨干的 segmentation_config')
    path = Path(path).resolve()
    settings = read(path)
    if "expected_images" in legacy_settings:
        settings.setdefault("expected_images", legacy_settings["expected_images"])
    return path, settings


def segmentation_assets(config):
    path, settings = segmentation_settings(config)
    receipt_path = (path.parent / settings['head_receipt']).resolve()
    receipt = read(receipt_path)
    if receipt.get('task_definition_sha256'):
        from core.cards import fingerprint
        if fingerprint({k: settings[k] for k in ('model', 'pipeline')}) != receipt['task_definition_sha256']:
            raise ValueError('任务特征层、头或预处理与冻结 head 的训练配置不符')
    return path, settings, receipt_path, receipt


def build_ade(backend, settings, receipt_path, receipt):
    import torch
    from torch import nn
    import timm
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    from mmengine.config import ConfigDict
    from copy import deepcopy
    from timm.layers import resample_abs_pos_embed

    device = backend.device
    config = deepcopy(settings['model'])
    binding = config['backbone']
    indices = tuple(binding['out_indices'])
    if not indices or tuple(sorted(set(indices))) != indices or indices[0] < 0 or indices[-1] >= backend.spec['depth']:
        raise ValueError('分割任务头的特征层超出绑定骨干')
    checkpoint = (receipt_path.parent / receipt['checkpoint']).resolve()
    parent_hash = receipt.get('backbone_sha256', receipt.get('deit_backbone_sha256'))
    if receipt['task'] != 'ade' or parent_hash != backend.checkpoint_hash or file_hash(checkpoint) != receipt['checkpoint_sha256']:
        raise ValueError('分割 head 与实际母模型或 checkpoint 身份不符')
    from ptq_agent.profiling.adapters import ProfileAdapter
    patch = backend.spec['patch_size']
    ph, pw = (patch, patch) if isinstance(patch, int) else patch

    class Features(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            b, _, h, w = x.shape
            if h % ph or w % pw:
                raise ValueError('分割输入须 patch 对齐')
            if binding['type'] == 'DINOv3MMSegBackbone':
                features = self.model.get_intermediate_layers(x, n=indices, reshape=True, norm=True,
                    return_class_token=False, return_extra_tokens=False)
            elif binding['type'] == 'DeiTDualTaskBackbone':
                # timm 0.3.2 PatchEmbed asserts fixed image size. Use the original
                # projection and the trained MMSeg reference's position interpolation.
                m = self.model
                tokens = m.patch_embed.proj(x).flatten(2).transpose(1, 2)
                tokens = torch.cat((m.cls_token.expand(b, -1, -1), tokens), 1)
                tokens = getattr(m.pos_drop, '_ptq_embedding_tokens', nn.Identity())(tokens)
                pos = resample_abs_pos_embed(m.pos_embed, (h // ph, w // pw),
                    num_prefix_tokens=backend.spec['prefix_token_count'], interpolation='bicubic', antialias=True)
                pos = getattr(m.pos_drop, '_ptq_position_embedding', nn.Identity())(pos)
                tokens = m.pos_drop(getattr(m.pos_drop, '_ptq_embedding_sum', nn.Identity())(tokens + pos))
                features = []
                for i, block in enumerate(m.blocks):
                    tokens = block(tokens)
                    if i in indices:
                        start, stop = ProfileAdapter().token_layout(backend, backend.spec['block_paths'][i],
                            tokens.shape, x.shape)['patch_slice']
                        patches = m.norm(tokens)[:, start:stop]
                        features.append(patches.transpose(1, 2).reshape(b, -1, h // ph, w // pw))
            else:
                raise ValueError('未适配的任务特征接口: ' + binding['type'])
            shape = (b, backend.spec['hidden_dim'], h // ph, w // pw)
            if len(features) != len(indices) or any(tuple(f.shape) != shape for f in features):
                raise ValueError('patch 特征数量/形状与任务头不符')
            return tuple(features)

    if binding['type'] == 'DeiTDualTaskBackbone':
        # Reference consumer used to train this specific archived head.
        if receipt.get('status') != 'formal_head_complete' or not receipt.get('backbone_unchanged'):
            raise ValueError('缺少冻结 head 来源')
        modern = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=False, dynamic_img_size=True)
        modern.load_state_dict(backend.model.state_dict(), strict=True)
        modern.head = nn.Identity()
        if state_sha(modern) != receipt['backbone_state_sha256']:
            raise ValueError('分割参考骨干与 head 训练来源不匹配')
        class Reference(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = modern
            def forward(self, x):
                return self.model.forward_intermediates(x, indices=indices, norm=True,
                    intermediates_only=True, output_fmt='NCHW')
        reference = Reference()
        payload = torch.load(checkpoint, map_location='cpu')
    elif binding['type'] == 'DINOv3MMSegBackbone' and receipt['format'] == 'mmseg_state_dict':
        # Compare all backbone tensors in the original MMSeg checkpoint, then
        # load only neck/head. An attached classifier is a separate frozen asset.
        saved = torch.load(checkpoint, map_location='cpu')['state_dict']
        source = {k.removeprefix('backbone.model.'): v for k, v in saved.items() if k.startswith('backbone.model.')}
        state = {k: v for k, v in backend.model.state_dict().items()
                 if not (backend.task_head and k.startswith(backend.task_head['protected_module'] + '.'))}
        if source.keys() != state.keys() or any(not torch.equal(v.cpu(), source[k]) for k, v in state.items()):
            raise ValueError('MMSeg checkpoint 骨干与绑定 FP 骨干不同')
        payload = {n: {k[len(n)+1:]: v for k,v in saved.items() if k.startswith(n + '.')}
                   for n in ('neck', 'decode_head')}
        class Reference(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                from ptq_agent.profiling.adapters import boundary_tensor
                layers, handles = {}, []
                try:
                    for i in indices:
                        def capture(_module, _args, output, index=i):
                            layers[index] = boundary_tensor(output)
                        handles.append(self.model.blocks[i].register_forward_hook(capture))
                    self.model.forward_features(x)
                finally:
                    for handle in handles:
                        handle.remove()
                prefix = backend.spec['prefix_token_count']
                return tuple(self.model.norm(layers[i][:, prefix:]).transpose(1, 2).reshape(
                    x.shape[0], -1, x.shape[-2] // ph, x.shape[-1] // pw) for i in indices)
        reference = Reference(backend.model)
    else:
        raise ValueError('未适配的冻结分割 backbone/head 接口')

    register_all_modules(init_default_scope=True)
    # MMSeg builds the configured neck, head and preprocessor normally.
    config['backbone'] = {'type': lambda: reference}
    segmentor = MODELS.build(ConfigDict(config))
    heads = [segmentor.neck, segmentor.decode_head]
    for name, head in zip(('neck', 'decode_head'), heads):
        head.load_state_dict(payload[name], strict=True)

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
    return model, Features, heads


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
