"""只读 FP 资产适配；复用已验证加载器，无下载、训练或量化参数拟合。

开放结构、权重/输入统计和 forward preflight，可显式绑定已有冻结分类头。
搜索/终评及PTQ由execution层执行；Backend旧trial接口仍拒绝，不能回退到toy。
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .base import Backend
from ..schemas import JSON, PTQError, fingerprint, require


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_local(name: str, path: Path):
    """只导入固定项目入口；用户配置的代码路径不开放给 Agent。"""
    require(path.is_file(), f"缺少本地适配入口：{path}", code="asset_missing")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ServerBackend(Backend):
    def __init__(self, config: JSON):
        self.config = deepcopy(config)
        required = ("asset_manifest", "model_id", "model_path")
        if not all(config.get(k) for k in required):
            raise PTQError("backend_not_configured", "请填写 asset_manifest/model_id/model_path",
                           status="unconfigured")
        import torch
        self.torch = torch
        manifest_path = Path(config["asset_manifest"])
        manifest = json.loads(manifest_path.read_text())
        records = [x for x in manifest["models"] if x["model_id"] == config["model_id"]]
        require(len(records) == 1, "模型 ID 在资产 manifest 中必须唯一", code="asset_mismatch")
        self.spec = deepcopy(records[0])
        # 本地资产清单允许相对路径；仍逐文件核对原来源hash。
        for key in ('checkpoint_path', 'code_path'):
            self.spec[key] = str((manifest_path.parent / self.spec[key]).resolve())
        self.spec['code_source']['local_path'] = str((manifest_path.parent / self.spec['code_source']['local_path']).resolve())
        self.spec['code_source']['source_file_sha256'] = {
            str((manifest_path.parent / path).resolve()): value
            for path, value in self.spec['code_source']['source_file_sha256'].items()}
        require(self.spec.get("preparation_status") == "ready", "资产尚未验证完成", code="asset_mismatch")
        if self.spec["family"] == "ViT":
            raise PTQError("unsupported_backend", "原始 Google ViT 需要独立 JAX 适配器，不能替换为 timm ViT", status="unsupported")
        self.checkpoint = Path(config["model_path"])
        require(self.checkpoint.resolve() == Path(self.spec["checkpoint_path"]).resolve(),
                "配置 checkpoint 与已验证资产不一致", code="asset_mismatch")
        self.checkpoint_hash = file_sha256(self.checkpoint)
        require(self.checkpoint_hash == self.spec["checkpoint_hash"]["value"],
                "checkpoint SHA256 不匹配", code="asset_mismatch")
        for path, digest in self.spec["code_source"]["source_file_sha256"].items():
            require(file_sha256(Path(path)) == digest, f"已验证模型源码发生变化：{path}", code="asset_mismatch")
        self.device = config.get("device", "cpu")
        device = torch.device(self.device)
        require(device.type in ("cpu", "cuda"), "只支持 CPU/CUDA")
        if device.type == "cuda":
            require(torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count(), "请求的 CUDA 设备不可用")
        loader_path = Path(__file__).resolve().parents[2] / 'vision_models.py'
        loader = import_local("ptq_agent_asset_loader", loader_path)
        # strict=True + 逐张量相等校验；不创建随机下游分类头。
        self.model, self.load_report = loader.load_model(self.spec)
        self.model = self.model.to(device).eval().requires_grad_(False)
        require(sum(p.numel() for p in self.model.parameters()) == self.spec["num_params"],
                "实际参数量与资产 manifest 不一致", code="asset_mismatch")
        self.task_head, self.head_class_mapping = None, None
        if config.get("task_head"):
            from .task_heads import attach_head
            self.task_head, self.head_class_mapping = attach_head(
                self.model, self.spec, config["task_head"], self.checkpoint_hash)
            self.spec["classification_head"] = self.task_head["kind"]
        self.original_state = state_sha256(self.model)
        self._named = dict(self.model.named_modules())
        self._modules = []
        for name, module in self._named.items():
            if not name or not list(module.parameters(recurse=False)):
                continue
            linear = isinstance(module, torch.nn.Linear)
            protected = not linear or name == "head" or name.startswith(("head.", "neck.", "decode_head."))
            weight = getattr(module, "weight", None)
            parts = name.split(".")
            block = int(parts[1]) if len(parts) > 1 and parts[0] == "blocks" and parts[1].isdigit() else None
            self._modules.append({"name": name, "type": type(module).__name__, "block": block,
                "protected": protected, "weight_shape": list(weight.shape) if weight is not None else [],
                "activation_shape": None, "activation_layout": "captured at runtime; no assumed feature axis"})
        source_root = Path(__file__).parents[1]
        self._identity = {"source": "real/pretrained_fp", "backend": "server_readonly_v1",
            "config": deepcopy(config), "model_id": config["model_id"], "checkpoint_sha256": self.checkpoint_hash,
            "asset_manifest_sha256": file_sha256(manifest_path), "loader_sha256": file_sha256(loader_path),
            "model_source": self.spec["code_source"], "torch_version": torch.__version__,
            "numpy_version": np.__version__, "python_version": platform.python_version(),
            "agent_code_sha256": fingerprint({str(p.relative_to(source_root)): file_sha256(p)
                                              for p in sorted(source_root.rglob("*.py"))}),
            "quantization": "none", "task_head": "not attached; official head retained if present"}
        if self.task_head:
            self._identity["task_head"] = deepcopy(self.task_head)
        self.data_adapter, self.rows = None, []
        if config.get("data_path"):
            if config.get("data_protocol") in ("imagefolder_manifest_v1", "imagefolder_manifest_v2"):
                from .image_data import ImageManifest
                self.data_adapter = ImageManifest(Path(config["data_path"]), config.get("data_sha256"))
                require(self.data_adapter.full_validation == (config["data_protocol"] == "imagefolder_manifest_v2"),
                        "配置与图像清单协议版本不符")
                data_module = Path(__file__).with_name("image_data.py")
            else:
                raise ValueError("请显式配置 imagefolder_manifest_v1/v2；框架不加载旧项目的数据适配源码")
            self.rows = self.data_adapter.records()
            require(config["calibration_samples"] <= len(self.rows), "输入池超过 manifest 大小")
            self._identity["data"] = {"manifest_sha256": file_sha256(Path(config["data_path"])),
                "preprocessing": self.data_adapter.PREPROCESS, "adapter_sha256": file_sha256(data_module),
                "use": ("explicit disjoint calibration/search/final manifests" if hasattr(self.data_adapter, "splits")
                        else "read-only profiling pool; NOT search/final task evaluation")}

    def identity(self) -> JSON:
        return deepcopy(self._identity)

    def forward_model(self, model, inputs):
        # 不包inference_mode，梯度统计与推理共用相同任务forward，参数始终冻结。
        if self.task_head:
            from .task_heads import forward_head
            return forward_head(model, inputs)
        return model.forward_features(inputs) if self.spec["family"].startswith(("DINOv2", "DINOv3")) else model(inputs)

    def modules(self) -> list[JSON]:
        return deepcopy(self._modules)

    def reset(self) -> None:
        require(not self.model.training and all(not p.requires_grad for p in self.model.parameters()),
                "FP 模型冻结状态改变", code="protection_failed")
        require(state_sha256(self.model) == self.original_state, "模型权重/buffer 发生变化", code="protection_failed")

    def baseline(self) -> JSON:
        raise PTQError("evaluation_not_configured", "FP 读取/统计已接入；任务 head、互斥 search/final manifest 和指标协议尚未绑定。请先运行 preflight",
                       status="unconfigured")

    def hardware_metrics(self) -> JSON:
        return {"gpu_memory_bytes": {"value": None, "reason": "尚未绑定候选任务的显存测量协议"},
                "native_low_precision_latency_ms": {"value": None, "reason": "当前仅 FP forward，无原生低精度 kernel"}}

    def samples(self, split, count=None, seed=0, offset=0):
        require(split == "calibration", "真实数据仅用于只读统计；search/final 尚未配置", code="data_protocol_mismatch")
        require(self.data_adapter is not None, "缺少只读输入 manifest", code="backend_not_configured")
        total = self.config["calibration_samples"]
        count = total if count is None else count
        require(type(count) is int and count > 0 and type(offset) is int and offset >= 0
                and count + offset <= total, "样本范围越界")
        indices = np.random.default_rng(seed).permutation(total)[offset:offset + count]
        rows = [self.rows[int(i)] for i in indices]
        inputs = self.torch.cat([self.data_adapter.load_image(row) for row in rows], dim=0)
        return inputs, [r["sample_id"] for r in rows]

    def _forward(self, inputs):
        with self.torch.inference_mode():
            return self.forward_model(self.model, inputs)

    def tensor(self, module, target, count, seed, offset=0):
        require(module in self._named and bool(module), "未知模块", code="unknown_module")
        require(target in ("weight", "activation"), "target 必须为 weight/activation")
        item = self._named[module]
        if target == "weight":
            weight = getattr(item, "weight", None)
            require(weight is not None, "模块没有 weight")
            return weight.detach().cpu().numpy().copy(), []
        inputs, ids = self.samples("calibration", count, seed, offset)
        values = []
        def capture(_module, args):
            require(args and self.torch.is_tensor(args[0]), "模块第一个输入必须为张量")
            values.append(args[0].detach().cpu().numpy().copy())
        handle = item.register_forward_pre_hook(capture)
        try:
            for image in inputs:
                # 沿用已验证预处理，检查原图和输入张量 SHA256。
                self._forward(image.unsqueeze(0).to(self.device))
        finally:
            handle.remove()
            self.reset()
        require(len(values) == len(ids), "模块必须每张图恰好调用一次")
        value = np.concatenate(values, axis=0)
        require(bool(np.isfinite(value).all()), "输入激活非有限", code="numerical_error")
        return value, ids

    def preflight(self) -> JSON:
        """最小合法合成输入验证加载/token 布局，不作为任务指标。"""
        torch = self.torch
        self.reset()
        size = self.spec["smoke_input_size"]
        generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])
        x = torch.randn(1, 3, size, size, generator=generator).to(self.device)
        try:
            outputs = self._forward(x)
            if self.task_head:
                require(tuple(outputs.shape) == (1, 1000), "冻结分类头输出形状错误")
                with torch.inference_mode():
                    outputs = {"output": outputs, **self.model.forward_features(x)}
            outputs = outputs if isinstance(outputs, dict) else {"output": outputs}
            tensors = {k: v for k, v in outputs.items() if torch.is_tensor(v)}
            require(bool(tensors) and all(torch.isfinite(v).all().item() for v in tensors.values()),
                    "forward 输出为空或非有限", code="numerical_error")
            tokens = None
            if self.spec["family"].startswith(("DINOv2", "DINOv3")):
                storage, width = self.spec["special_token_count"], self.spec["hidden_dim"]
                patch_count = (size // self.spec["patch_size"]) ** 2
                require(tuple(tensors["x_norm_patchtokens"].shape) == (1, patch_count, width), "patch token 形状错误")
                require(tuple(tensors["x_norm_clstoken"].shape) == (1, width), "CLS token 形状错误")
                token_key = "x_storage_tokens" if "x_storage_tokens" in tensors else "x_norm_regtokens"
                require(tuple(tensors[token_key].shape) == (1, storage, width), "storage/register token 形状错误")
                require(tuple(tensors["x_prenorm"].shape) == (1, 1 + storage + patch_count, width), "完整 token 序列形状错误")
                tokens = {"cls_index": 0, "special_slice": [1, 1 + storage],
                          "patch_slice": [1 + storage, 1 + storage + patch_count], "slice_convention": "[start, stop)"}
            else:
                width = 1000 if self.spec["family"] in ("DeiT", "Swin") else self.spec["hidden_dim"]
                require(tuple(tensors["output"].shape) == (1, width), "官方 forward 输出形状不符")
            result = {"status": "passed", "kind": "synthetic_forward_preflight", "dataset_images": 0,
                "input_shape": list(x.shape), "output_shapes": {k: list(v.shape) for k, v in tensors.items()},
                "all_outputs_finite": True, "token_layout": tokens, "load_report": self.load_report,
                "device": self.device, "num_params": sum(p.numel() for p in self.model.parameters()),
                "quantized_modules": [], "training": False, "calibration": False,
                "task_metrics": None, "task_metrics_reason": "预检未绑定任务评测，不能报告 Top-1/mIoU",
                "hardware_metrics": self.hardware_metrics(), "identity": self.identity(), "modules": self.modules()}
        finally:
            self.reset()
            require(file_sha256(self.checkpoint) == self.checkpoint_hash, "checkpoint 文件变化", code="protection_failed")
        result["protection"] = {"state_before": self.original_state, "state_after": state_sha256(self.model),
                                "checkpoint_unchanged": True, "all_parameters_frozen": True}
        return result
