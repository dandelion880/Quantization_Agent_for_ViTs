"""真实 torch 运算的随机小型 ViT；只验证流程/数值，不代表任务精度。"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch

from ptq_agent.backends.server import ServerBackend, state_sha256


class TinyBackend:
    def __init__(self, cfg):
        import types, sys, collections.abc
        shim = types.ModuleType("torch._six")
        shim.container_abcs = collections.abc
        sys.modules.setdefault("torch._six", shim)
        from timm.models.vision_transformer import VisionTransformer
        torch.manual_seed(17)
        self.torch, self.config, self.device = torch, deepcopy(cfg), cfg.get("device", "cpu")
        fixture = cfg.get("fixture_shape", {})
        self.model = VisionTransformer(img_size=32, patch_size=16, embed_dim=fixture.get("embed_dim", 24), depth=fixture.get("depth", 2),
                                       num_heads=fixture.get("heads", 3), mlp_ratio=2, num_classes=1000).to(self.device).eval().requires_grad_(False)
        self.spec = dict(family="synthetic-ViT-validation-fixture", classification_head=None, patch_size=16,
                         cls_token_count=1, special_token_count=0, smoke_input_size=32)
        self.task_head = self.head_class_mapping = None
        self._named = dict(self.model.named_modules())
        self._modules = [dict(name=n, type=type(m).__name__, block=None,
                              protected=not isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)),
                              weight_shape=list(m.weight.shape), activation_shape=None)
                         for n, m in self.model.named_modules() if hasattr(m, "weight")]
        self.original_state = state_sha256(self.model)
        self.rows, splits, self.images = [], {}, {}
        for split in ("calibration", "search", "final"):
            splits[split] = []
            g = torch.Generator().manual_seed(cfg["data_seeds"][split])
            for i in range(cfg[split + "_samples"]):
                sid = f"synthetic-{split}-{i}"
                row = {"sample_id": sid, "path": sid}
                self.images[sid] = torch.randn(1, 3, 32, 32, generator=g)
                splits[split].append(row)
                self.rows.append(row)
        self.data_adapter = SimpleNamespace(splits=splits, PREPROCESS="seeded synthetic RGB32; no task accuracy",
                                            full_validation=False, load_image=lambda r: self.images[r["sample_id"]].clone())

    def identity(self):
        return {"source": "synthetic_torch_vit_validation_only", "model_sha256": self.original_state,
                "config": self.config}

    def forward_model(self, model, x):
        return model(x)

    modules = ServerBackend.modules
    reset = ServerBackend.reset
    samples = ServerBackend.samples
    _forward = ServerBackend._forward
    tensor = ServerBackend.tensor


def tiny_backend(config):
    return TinyBackend(config)
