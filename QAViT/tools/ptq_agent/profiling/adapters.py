"""模型语义/硬件的可插入边界；统计核心不依赖特定 ViT 或 GPU 型号。

宿主可继承 ProfileAdapter，并通过 ProfilingSession(adapter=...) 注入。
自定义适配必须使用新的 version；它参与缓存和恢复检查，不能由 LLM 动态导入。
"""

import platform

from ..schemas import PTQError, require


class ProfileAdapter:
    version = "torch_vit_profile_v1"

    def token_layout(self, backend, name, shape, image_shape):
        """返回已验证的 token 切片；window 下标仅表示窗口内位置。"""
        require(len(shape) == 3 and len(image_shape) == 4, "token 统计要求 [B,N,C] 特征与图像输入")
        spec = getattr(backend, "spec", {})
        family = spec.get("family", "")
        cls, special = spec.get("cls_token_count", 0), spec.get("special_token_count", 0)
        scope, batch_semantics = "image_sequence", "images"
        extra = {}
        if family.startswith(("DINO", "MAE", "DeiT")):
            require(name.startswith("blocks.") or name == "norm", "该位点尚未验证全局 token 排列")
            patch = spec["patch_size"]
            ph, pw = (patch, patch) if isinstance(patch, int) else patch
            patches = (image_shape[-2] // ph) * (image_shape[-1] // pw)
            require(shape[0] == image_shape[0], "全局 token 的 batch 维不匹配")
        elif family == "Swin":
            # 从实际 stage/window 对象读取分辨率；不硬编码层数、图像或窗口大小。
            parts = name.split(".")
            require(len(parts) >= 5 and parts[0] == "layers" and parts[2] == "blocks",
                    "Swin token 统计目前限定已验证的 block 内输入")
            block = backend.model.get_submodule(".".join(parts[:4]))
            h, w = block.input_resolution
            cls, special = 0, 0
            if parts[4] == "attn":
                wh, ww = block.attn.window_size
                require(h % wh == 0 and w % ww == 0, "窗口/padding 布局需专用适配")
                patches = wh * ww
                require(shape[0] == image_shape[0] * (h // wh) * (w // ww), "window batch 数不匹配")
                scope, batch_semantics = "window_local", "images_times_windows"
                extra = {"window_size": [wh, ww], "shift_size": block.shift_size}
            else:
                patches = h * w
                require(shape[0] == image_shape[0], "stage token 的 batch 维不匹配")
                scope = "stage_grid"
            extra["stage_resolution"] = [h, w]
        else:
            raise PTQError("unsupported_token_layout", "未知 token 语义；请注入 ProfileAdapter.token_layout，不能按模型名猜测", status="unsupported")
        prefix = cls + special
        require(shape[1] == prefix + patches, "实际 token 数与声明布局不一致")
        return {"token_axis": 1, "channel_axis": 2, "cls_slice": [0, cls],
                "special_slice": [cls, prefix], "patch_slice": [prefix, prefix + patches],
                "token_index_scope": scope, "batch_semantics": batch_semantics, **extra}

    def hardware_identity(self, profiler):
        """实际设备/运行设置参与缓存；不把某型号的性能表用于其它芯片。"""
        torch, device = profiler.torch, profiler.device
        if device.type not in ("cpu", "cuda"):
            raise PTQError("unsupported_profile_device", "该设备需要独立硬件适配器与同步/计时协议", status="unsupported")
        info = {"adapter": self.version, "device": str(device), "host": platform.node(),
                "machine": platform.machine(), "processor": platform.processor(),
                "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "hip_runtime": getattr(torch.version, "hip", None),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        if device.type == "cuda":
            p = torch.cuda.get_device_properties(device)
            info["accelerator"] = {"name": p.name, "compute_capability": [p.major, p.minor],
                "total_memory": p.total_memory, "multiprocessors": p.multi_processor_count,
                "uuid": str(getattr(p, "uuid", "unavailable"))}
        return info

    def hardware(self, profiler, **arguments):
        self.hardware_identity(profiler)  # 未适配的异步设备不能按同步 CPU 计时。
        return profiler.hardware(**arguments)
