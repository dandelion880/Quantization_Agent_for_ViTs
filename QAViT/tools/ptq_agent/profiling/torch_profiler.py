"""PyTorch 只读结构/硬件测量。hooks 和 profiler 均按请求安装并在 finally 清理。"""

from contextlib import contextmanager
import math
import os
from pathlib import Path
import time

from ..schemas import PTQError, require
from .contracts import linear_forward_supported


class TorchProfiler:
    def __init__(self, backend, constraints):
        if not hasattr(backend, "model"):
            raise PTQError("unsupported_backend", "该测量需要 PyTorch 模型；不把 NumPy 时间冒充目标硬件算子时间", status="unsupported")
        import torch
        self.torch, self.backend, self.model = torch, backend, backend.model
        self.device = next(self.model.parameters()).device
        self.named = dict(self.model.named_modules())
        self.protected = {m["name"] for m in backend.modules() if m["protected"]}
        self.protected.update(constraints["protected_modules"] + constraints["excluded_modules"])

    @contextmanager
    def protected_run(self):
        from ..backends.server import state_sha256
        require(all(not m.training for m in self.model.modules()), "profiling 必须使用 eval 模型")
        before = state_sha256(self.model)
        try:
            with self.torch.inference_mode():
                yield
        finally:
            require(before == state_sha256(self.model), "profiling 改变了模型参数或 buffer", code="protection_failed")
            self.backend.reset()

    def inputs(self, count, seed, offset):
        x, ids = self.backend.samples("calibration", count, seed, offset)
        require(self.torch.is_tensor(x), "PyTorch profiling 需要 tensor 输入")
        return x.to(self.device), ids

    def forward(self, x):
        y = self.backend._forward(x)
        return y

    def check_output(self, output):
        """有限值检查不放进干净计时区，避免把检查同步算作模型延迟。"""
        if self.torch.is_tensor(output):
            require(bool(self.torch.isfinite(output).all()), "profiling forward 输出非有限", code="numerical_error")
        elif isinstance(output, dict):
            for value in output.values():
                self.check_output(value)
        elif isinstance(output, (list, tuple)):
            for value in output:
                self.check_output(value)

    def select(self, names, *, descendants=False):
        require(len(names) == len(set(names)) and all(n in self.named for n in names), "未知/重复模块")
        if not names:
            return self.named
        return {n: m for n, m in self.named.items() if n in names or
                (descendants and any(n.startswith(parent + ".") or parent == "" for parent in names))}

    def describe(self, value):
        if self.torch.is_tensor(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if isinstance(value, (list, tuple)):
            return [self.describe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self.describe(v) for k, v in value.items()}
        return None

    def structure(self, names, count, seed, offset):
        torch = self.torch
        selected = self.select(names, descendants=True)
        x, ids = self.inputs(count, seed, offset)
        rows, calls, stack, functional = [], {}, [], []
        for name, m in selected.items():
            calls[name] = []
            rows.append({"name": name or "<root>", "type": type(m).__name__,
                "parent": name.rpartition(".")[0] or "<root>",
                "parameter_count": sum(p.numel() for p in m.parameters(recurse=False)),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in m.parameters(recurse=False)),
                "buffer_bytes": sum(p.numel() * p.element_size() for p in m.buffers(recurse=False)),
                "protected": name in self.protected, "calls": calls[name]})

        def before(name):
            def hook(m, args):
                stack.append(name)
            return hook

        def after(name):
            def hook(m, args, out):
                if name in calls:
                    macs = None
                    if torch.is_tensor(out) and isinstance(m, torch.nn.Linear) and linear_forward_supported(m):
                        macs = out.numel() * m.in_features
                    elif torch.is_tensor(out) and type(m).forward is torch.nn.Conv2d.forward:
                        macs = out.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
                    calls[name].append({"input": self.describe(args), "output": self.describe(out), "macs": macs})
                require(stack.pop() == name, "模块调用栈不一致")
            return hook

        # 沿用已有 FP audit 的 TorchFunctionMode 只读观测模式，只记录形状。
        # 不展开/替换 fused SDPA，避免为了统计改变真实算子与数值行为。
        from torch.overrides import TorchFunctionMode
        owner = self
        class ShapeDispatch(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                name = getattr(func, "__name__", "")
                module = stack[-1] if stack else "<root>"
                if name == "scaled_dot_product_attention":
                    q, k, v = args[:3]
                    leading = math.prod(torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2]))
                    qk = leading * q.shape[-2] * k.shape[-2] * q.shape[-1]
                    pv = leading * q.shape[-2] * k.shape[-2] * v.shape[-1]
                    functional.append({"module": module, "operator": name, "inputs": owner.describe(args[:3]),
                        "output": owner.describe(result), "macs": qk + pv,
                        "logical_sites": ["q", "k", "v", "qk_matmul", "pv_matmul"],
                        "instrumentation": "fused operator observed unchanged; internal probabilities not materialized"})
                elif name in ("matmul", "__matmul__", "bmm", "mm") and len(args) >= 2 and all(torch.is_tensor(a) and a.ndim >= 2 for a in args[:2]):
                    functional.append({"module": module, "operator": name, "inputs": owner.describe(args[:2]),
                        "output": owner.describe(result), "macs": result.numel() * args[0].shape[-1],
                        "logical_sites": ["lhs", "rhs", "output"]})
                return result

        handles = []
        try:
            with self.protected_run():
                for name, m in self.named.items():
                    handles.extend((m.register_forward_pre_hook(before(name)),
                                    m.register_forward_hook(after(name), always_call=True)))
                with ShapeDispatch():
                    self.check_output(self.forward(x))
        finally:
            for handle in handles:
                handle.remove()
        sites = []
        for name, m in selected.items():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                for target in ("weight", "input"):
                    sites.append({"id": name + "." + target, "module": name, "target": target,
                        "protected": name in self.protected,
                        "sensitivity_supported": isinstance(m, torch.nn.Linear) and linear_forward_supported(m),
                        "execution_ready": False})
        relevant = [f for f in functional if not names or f["module"] in names
                    or any(f["module"].startswith(n + ".") for n in names)]
        return {"modules": rows, "functional_sites": relevant, "sites": sites, "sample_ids": ids,
            "input": self.describe(x), "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
            "buffer_bytes": sum(p.numel() * p.element_size() for p in self.model.buffers()),
            "counted_macs": sum(c["macs"] or 0 for row in rows for c in row["calls"]) + sum(f["macs"] for f in relevant),
            "macs_scope": "selected modules (all if names empty); verified Linear/Conv2d + functional matmul + SDPA; bias/norm/softmax and unknown custom forwards excluded",
            "macs_convention": "one multiplication plus accumulation = 1 MAC; entire requested batch",
            "count_note": "global parameters are unique; per-module direct parameters may share storage"}

    def capture(self, names, count, seed, offset, max_bytes=128 << 20):
        torch = self.torch
        selected = self.select(names)
        require(bool(names), "张量采集必须明确限定模块")
        x, ids = self.inputs(count, seed, offset)
        values, handles, used = {}, [], 0
        def capture(name):
            def hook(m, args):
                nonlocal used
                require(args and torch.is_tensor(args[0]), "仅支持模块第一个 tensor 输入")
                value = args[0]
                require(name not in values, "重复调用模块需要独立调用位点，不能静默拼接")
                used += value.numel() * value.element_size()
                require(used <= max_bytes, "张量采集超过 128 MiB；请显式缩小位点或样本范围", code="profile_memory_budget")
                values[name] = value.detach().cpu().numpy().copy()
            return hook
        try:
            with self.protected_run():
                for name, m in selected.items():
                    handles.append(m.register_forward_pre_hook(capture(name)))
                self.check_output(self.forward(x))
        finally:
            for handle in handles:
                handle.remove()
        require(set(values) == set(names), "指定模块未在实际 forward 中执行")
        return values, ids, list(x.shape)

    def hardware(self, modules, count, seed, offset, warmup, repeats):
        torch = self.torch
        if self.device.type not in ("cpu", "cuda"):
            raise PTQError("unsupported_profile_device", "请为该设备提供同步/计时适配器", status="unsupported")
        selected = self.select(modules)
        x, ids = self.inputs(count, seed, offset)
        cuda = self.device.type == "cuda"
        sync = lambda: torch.cuda.synchronize(self.device) if cuda else None
        elapsed, device_elapsed = [], []
        with self.protected_run():
            for _ in range(warmup):
                self.check_output(self.forward(x))
            sync()
            allocated = torch.cuda.memory_allocated(self.device) if cuda else None
            reserved = torch.cuda.memory_reserved(self.device) if cuda else None
            if cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            # 无 hooks/profiler 的计时和峰值；不要把 profiler 开销混进延迟。
            for _ in range(repeats):
                start_event = torch.cuda.Event(enable_timing=True) if cuda else None
                end_event = torch.cuda.Event(enable_timing=True) if cuda else None
                sync()
                start = time.perf_counter()
                if cuda:
                    start_event.record(torch.cuda.current_stream(self.device))
                self.forward(x)
                if cuda:
                    end_event.record(torch.cuda.current_stream(self.device))
                sync()
                elapsed.append((time.perf_counter() - start) * 1000)
                if cuda:
                    device_elapsed.append(start_event.elapsed_time(end_event))
            memory = {"cuda_allocated_before_bytes": allocated, "cuda_reserved_before_bytes": reserved,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else None,
                "note": "CUDA peak is this process, clean timed forwards incl resident model/input; CPU peak includes loading; not per-layer occupancy"}
            try:
                import resource
                memory["cpu_process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.sys.platform == "darwin" else 1024)
            except ImportError:
                memory["cpu_process_lifetime_peak_rss_bytes"] = None
            if cuda:
                memory["cuda_peak_increment_bytes"] = memory["cuda_peak_allocated_bytes"] - allocated
            try:
                memory["cpu_current_rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except OSError:
                memory["cpu_current_rss_bytes"] = None

            handles, ranges = [], {}
            def before(name):
                def hook(m, args):
                    span = torch.profiler.record_function("ptq_profile::" + name)
                    span.__enter__()
                    ranges.setdefault(name, []).append(span)
                return hook
            def after(name):
                def hook(m, args, output):
                    ranges[name].pop().__exit__(None, None, None)
                return hook
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            try:
                for name, m in selected.items():
                    handles.extend((m.register_forward_pre_hook(before(name)),
                                    m.register_forward_hook(after(name), always_call=True)))
                with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                    for _ in range(repeats):
                        self.forward(x)
                    sync()
            finally:
                for handle in handles:
                    handle.remove()
        operators, module_times = [], []
        for event in prof.key_averages(group_by_input_shape=True):
            if event.key.startswith("aten::"):
                operators.append({"operator": event.key, "calls": event.count, "input_shapes": event.input_shapes,
                    "self_cpu_ms": event.self_cpu_time_total / 1000,
                    "self_cuda_ms": event.self_cuda_time_total / 1000 if cuda else None,
                    "cpu_net_alloc_bytes": event.self_cpu_memory_usage,
                    "cuda_net_alloc_bytes": event.self_cuda_memory_usage if cuda else None})
            elif event.key.startswith("ptq_profile::"):
                module_times.append({"module": event.key.removeprefix("ptq_profile::"), "calls": event.count,
                    "inclusive_cpu_ms": event.cpu_time_total / 1000,
                    "inclusive_cuda_ms": event.cuda_time_total / 1000 if cuda else None})
        cuda_observed = cuda and any(r["self_cuda_ms"] > 0 for r in operators)
        if cuda and not cuda_observed:
            for row in operators:
                row["self_cuda_ms"] = None
            for row in module_times:
                row["inclusive_cuda_ms"] = None
        field = "self_cuda_ms" if cuda_observed else "self_cpu_ms"
        operators.sort(key=lambda row: row[field], reverse=True)
        total = sum(r[field] for r in operators)
        for row in operators:
            row["self_time_fraction"] = row[field] / total if total else None
        return {"device": str(self.device), "device_name": torch.cuda.get_device_name(self.device) if cuda else "CPU",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "dtype": str(x.dtype), "input_shape": list(x.shape), "sample_ids": ids,
            "warmup": warmup, "repeats": repeats, "wall_latency_ms": elapsed,
            "cuda_event_latency_ms": device_elapsed if cuda else None, "memory": memory,
            "operators": operators, "module_times": module_times,
            "operator_scope": "whole forward; requested modules add inclusive attribution ranges",
            "operator_cuda_available": cuda_observed,
            "operator_cuda_reason": None if cuda_observed else "CPU device or CUDA profiler did not capture events",
            "native_low_precision": False, "model_state": "original_fp",
            "protocol": "same resident batch; data/CPU->GPU transfer excluded; clean latency separate from instrumented trace; op totals over repeats; nested module times must NOT be summed"}
