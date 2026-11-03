"""按需 profiling 程序与工具注册；不改模型，不启动任务量化或终评。

所有统计明细落到 evidence。LLM 默认只读摘要，可分页读取已有证据，避免
每次把数百个 channel 或 profiler 的所有事件塞进上下文。
"""

import numpy as np

from .adapters import ProfileAdapter
from .contracts import is_linear, linear_forward_supported, validate_profile_spec
from .statistics import STATISTICS, tensor_statistics
from ..quantization.integer import quantize
from ..schemas import require


def error_metrics(reference, result):
    before, after = np.asarray(reference, dtype=np.float64), np.asarray(result, dtype=np.float64)
    delta = after - before
    mse = float(np.mean(delta ** 2))
    energy = float(np.mean(before ** 2))
    norm = float(np.linalg.norm(before.reshape(-1)) * np.linalg.norm(after.reshape(-1)))
    return {"mse": mse, "relative_mse": mse / energy if energy else None,
            "relative_mse_reason": None if energy else "zero reference energy",
            "max_abs_error": float(np.max(np.abs(delta))),
            "cosine": float(np.clip(np.sum(before * after) / norm, -1, 1)) if norm else None}


class ProfilePrograms:
    def __init__(self, host):
        self.host, self.backend = host, host.backend
        self.constraints = host.state["config"]["constraints"]
        self.adapter = getattr(host, "profile_adapter", None) or ProfileAdapter()

    def torch(self):
        from .torch_profiler import TorchProfiler
        return TorchProfiler(self.backend, self.constraints)

    def module_info(self, names):
        known = {m["name"]: m for m in self.backend.modules()}
        require(bool(names) and len(names) == len(set(names)) and set(names) <= set(known), "请提供已知且不重复的模块")
        provider = getattr(self.host, "profile_module_permissions", None)
        permissions = provider() if provider else {}
        return {n: {**known[n], "protected": permissions.get(n, known[n]["protected"])} for n in names}

    def capture(self, names, count, seed, offset):
        self.module_info(names)
        if hasattr(self.backend, "model"):
            return self.torch().capture(names, count, seed, offset)
        x, ids = self.backend.samples("calibration", count, seed, offset)
        _, activations = self.backend._forward(x)
        self.backend.reset()
        return {n: activations[n] for n in names}, ids, list(x.shape)

    def structure(self, modules, count, seed, offset):
        if hasattr(self.backend, "model"):
            result = self.torch().structure(modules, count, seed, offset)
            # 由宿主提供许可，保持统计模块不依赖执行模块；只修正元数据，
            # 不改变原模型冻结检查、forward、统计值或执行端保护门禁。
            provider = getattr(self.host, "profile_module_permissions", None)
            permissions = provider() if provider else {}
            for row in result["modules"]:
                row["protected"] = permissions.get(row["name"], row["protected"])
            for site in result["sites"]:
                site["protected"] = permissions.get(site["module"], site["protected"])
            return result
        known = self.backend.modules()
        require(set(modules) <= {m["name"] for m in known}, "未知模块")
        names = modules or [m["name"] for m in known]
        inputs, ids, shape = self.capture(names, count, seed, offset)
        rows = []
        for m in known:
            n = m["name"]
            if n not in names:
                continue
            w, b = self.backend._weights[n], self.backend._biases[n]
            rows.append({"name": n, "type": m["type"], "protected": m["protected"] or n in self.constraints["protected_modules"] + self.constraints["excluded_modules"],
                "parameter_count": w.size + b.size, "parameter_bytes": w.nbytes + b.nbytes,
                "calls": [{"input": [{"shape": list(inputs[n].shape), "dtype": str(inputs[n].dtype), "device": "cpu"}],
                           "macs": count * w.size if m["type"] == "Linear" else None}]})
        return {"modules": rows, "functional_sites": [], "sites": [
            {"id": r["name"] + "." + t, "module": r["name"], "target": t, "protected": r["protected"]}
            for r in rows if r["type"] == "Linear" for t in ("weight", "input")],
            "parameter_count": sum(x.size for x in (*self.backend._weights.values(), *self.backend._biases.values())),
            "parameter_bytes": sum(x.nbytes for x in (*self.backend._weights.values(), *self.backend._biases.values())),
            "counted_macs": sum(r["calls"][0]["macs"] or 0 for r in rows),
            "macs_scope": "selected toy Linear only, bias/norm/activation excluded; entire batch",
            "input": {"shape": shape, "device": "cpu"}, "sample_ids": ids}

    def token_view(self, name, x, image_shape, subset, need_tokens):
        if not need_tokens and subset == "all":
            return x, None
        require(x.ndim == 3 and len(image_shape) == 4,
                f"{name}: token 统计需要 [B,N,C]，实测 shape={list(x.shape)}、"
                f"image_shape={image_shape}。请把此模块单独用 views=['tensor','channel']、"
                "tokens='all' 重试；三维模块可继续请求 token 统计。",
                code="unsupported_token_layout")
        layout = self.adapter.token_layout(self.backend, name, list(x.shape), image_shape)
        require(x.ndim == 3 and layout["token_axis"] == 1 and layout["channel_axis"] == 2,
                "当前归约要求 token/channel 分别为轴 1/2")
        start, stop = [0, x.shape[1]] if subset == "all" else layout[subset + "_slice"]
        require(0 <= start < stop <= x.shape[1], "请求的 token 子集为空或越界")
        return x[:, start:stop, :], {**layout, "selected_original_indices": list(range(start, stop))}

    def distribution(self, modules, target, views, tokens, statistics, histogram_bins, top_k, count, seed, offset):
        self.module_info(modules)
        if target == "activation":
            values, ids, image_shape = self.capture(modules, count, seed, offset)
        else:
            require(tokens == "all" and "token" not in views, "权重没有图像 token 轴")
            values = {n: self.backend.tensor(n, "weight", count, seed, offset)[0] for n in modules}
            ids, image_shape = [], []
        summary, details, shapes = [], [], {}
        for name, value in values.items():
            if target == "activation":
                shapes[name] = list(value.shape)
            x, layout = self.token_view(name, value, image_shape, tokens, "token" in views)
            row = {"module": name, "target": target, "original_shape": list(value.shape),
                   "selected_shape": list(x.shape), "token_layout": layout, "views": {}}
            if "tensor" in views:
                row["views"]["tensor"] = tensor_statistics(x, statistics)
            if histogram_bins:
                counts, edges = np.histogram(x, bins=histogram_bins)
                row["histogram"] = {"counts": counts.tolist(), "edges": edges.tolist(), "scope": "selected tensor, exact"}
            for view in (v for v in views if v != "tensor"):
                if view == "token":
                    axis = 1
                elif target == "weight":
                    axis = 0
                else:
                    if x.ndim == 4 and hasattr(self.backend, "model"):
                        import torch
                        module = self.backend.model.get_submodule(name)
                        # 标准 Conv2d 的逻辑布局明确为 NCHW；不把宽度轴误当通道。
                        require(type(module) is torch.nn.Conv2d and x.shape[1] == module.in_channels,
                                "4D channel 统计只接受已核对的标准 Conv2d NCHW 输入")
                        axis = 1
                    else:
                        require(x.ndim in (2, 3), "channel 输入统计支持 [B,C]/[B,N,C]；其它布局需显式适配")
                        axis = x.ndim - 1
                grouped = tensor_statistics(x, list(dict.fromkeys([*statistics, "range", "rms"])), "channel", axis)
                groups = grouped["grouping"]["groups"]
                full = [{"module": name, "target": target, "view": view,
                         "index": layout["selected_original_indices"][g["start"]] if view == "token" else g["start"],
                         "statistics": g["statistics"]} for g in groups]
                details.extend(full)
                ranked = sorted(full, key=lambda g: max(abs(v) for v in g["statistics"]["range"]), reverse=True)
                row["views"][view] = {"preserved_axis": axis, "reduced_axes": [a for a in range(x.ndim) if a != axis],
                    "group_count": len(full), "largest_abs_groups": ranked[:top_k],
                    "rms_min": min(g["statistics"]["rms"] for g in full), "rms_max": max(g["statistics"]["rms"] for g in full)}
            summary.append(row)
        return {"rows": summary, "details": details, "shapes": shapes, "sample_ids": ids,
                "definition": "existing float64 tensor_statistics; population std; exact requested token/channel reductions"}

    def sensitivity(self, modules, variants, count, seed, offset):
        infos = self.module_info(modules)
        for name, info in infos.items():
            require(is_linear(self.backend, info) and not info["protected"] and name not in
                    self.constraints["protected_modules"] + self.constraints["excluded_modules"], "探测位点受保护或不是 Linear")
        require(len({v["name"] for v in variants}) == len(variants), "variant 名重复")
        inputs, ids, _ = self.capture(modules, count, seed, offset)
        rows = []
        for name in modules:
            x = inputs[name].astype(np.float64)
            w = self.backend.tensor(name, "weight", count, seed, offset)[0].astype(np.float64)
            if hasattr(self.backend, "model"):
                module = self.backend.model.get_submodule(name)
                require(linear_forward_supported(module),
                        "该 Linear 子类的 forward 语义尚未验证，不能用通用矩阵乘代替", code="unsupported_profile_site")
                bias = module.bias.detach().cpu().numpy() if module.bias is not None else 0
                # DINOv3 qkv 的 K bias 被官方 bias_mask 屏蔽，不能直接用裸 bias。
                if hasattr(module, "bias_mask"):
                    bias = bias * module.bias_mask.detach().cpu().numpy()
            else:
                bias = self.backend._biases[name]
            before = x @ w.T + bias
            for variant in variants:
                require(variant["weight"] or variant["activation"], "至少探测 W/A 之一")
                for target, value in (("weight", w), ("activation", x)):
                    spec = variant[target]
                    if spec:
                        validate_profile_spec(spec, target, list(value.shape))
                        require(spec["format"] in self.constraints["allowed_formats"], "探测位宽违反硬约束")
                qw = quantize(w, variant["weight"]).dequantize() if variant["weight"] else w
                qx = quantize(x, variant["activation"]).dequantize() if variant["activation"] else x
                rows.append({"module": name, "variant": variant, "weight": error_metrics(w, qw),
                    "activation": error_metrics(x, qx), "weight_only_output": error_metrics(before, x @ qw.T + bias),
                    "activation_only_output": error_metrics(before, qx @ w.T + bias),
                    "joint_output": error_metrics(before, qx @ qw.T + bias)})
        self.backend.reset()
        return {"rows": rows, "shapes": {n: list(x.shape) for n, x in inputs.items()}, "sample_ids": ids,
            "scope": "isolated Linear on identical FP inputs; ephemeral minmax diagnostic grids fitted on these inputs",
            "task_accuracy": None, "speedup": None, "model_mutated": False,
            "representation": "existing NumPy float64 Q/DQ; not native integer kernel; no fitted scales exported"}
