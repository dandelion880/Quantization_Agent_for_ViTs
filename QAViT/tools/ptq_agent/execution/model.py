"""受约束的执行 Agent：编译已校验的配置，不生成或运行任意代码。"""

from copy import deepcopy
import hashlib
from pathlib import Path
import os
import tempfile
import torch

from ..schemas import require, fingerprint
from ..storage import atomic_json, read_json
from ..backends.server import file_sha256
from ..profiling.evidence import seal
from .codecs import DTYPES, ranges, view, validate_record, validate_params
from .contracts import RAW_FLOATS, normalize_plan
from .capabilities import detect, validate_plan
from .methods import get_method, versions


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str(value.dtype).encode()
        + str(tuple(value.shape)).encode()
        + value.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def forward(backend, model, inputs):
    """注入模型而不替换 backend.model；原 FP 资源始终保持只读。"""
    with torch.inference_mode():
        if hasattr(backend, "forward_model"):
            return backend.forward_model(model, inputs)
        return (
            model.forward_features(inputs)
            if backend.spec["family"].startswith(("DINOv2", "DINOv3"))
            else model(inputs)
        )


class QuantizedLinear(torch.nn.Module):
    def __init__(self, original, layer, execution, weight_record, activation_params):
        super().__init__()
        # Attention/MLP 可能读取 Linear 维度元数据；替换后须保留其模块接口。
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.layer, self.execution = deepcopy(layer), deepcopy(execution)
        self.weight_record = weight_record
        method = get_method(layer["method"])
        weight = (
            method.decode_weight(weight_record, original.weight.device)
            if weight_record
            else original.weight.detach().clone()
        )
        # 缓存是 reference 运行的真实内存成本；导出仅保存紧密编码和尺度。
        self.register_buffer("decoded_weight", weight, persistent=False)
        self.register_buffer("bias", original.bias.detach().clone() if original.bias is not None else None)
        self.register_buffer(
            "bias_mask", original.bias_mask.detach().clone() if hasattr(original, "bias_mask") else None
        )
        self.activation_parameter_names = list(activation_params)
        for key, value in activation_params.items():
            self.register_buffer("activation_" + key, value.to(weight.device))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        original_dtype = x.dtype
        spec = self.layer["activation"]
        if spec:
            params = {k: getattr(self, "activation_" + k) for k in self.activation_parameter_names}
            x = get_method(self.layer["method"]).quantize_activation(
                x, spec, params if self.execution["activation_calibration"] == "static" else None
            )
        dtype = DTYPES[self.execution["compute_dtype"]]
        bias = self.bias
        if bias is not None and self.bias_mask is not None:
            bias = bias * self.bias_mask
        return torch.nn.functional.linear(
            x.to(dtype), self.decoded_weight.to(dtype), bias.to(dtype) if bias is not None else None
        ).to(original_dtype)


def calibrate(backend, plan, data):
    """一次 FP 数据遍历累积静态范围，不缓存全量 token 激活。"""
    selected = {
        l["module"]: l["activation"]
        for l in plan["layers"]
        if l["activation"]
        and l["activation"]["format"] not in RAW_FLOATS
        and plan["execution"]["activation_calibration"] == "static"
    }
    bounds, shapes, handles = {}, {}, []
    activations = {l["module"]: l["activation"] for l in plan["layers"] if l["activation"]}

    def hook(name, spec):
        def observe(_module, args):
            x = args[0].detach()
            require(bool(torch.isfinite(x).all()), "校准激活非有限", code="numerical_error")
            shapes[name] = list(x.shape)
            if name not in selected:
                return
            from .contracts import validate_spec
            validate_spec(spec, "activation", shapes[name], plan["execution"]["activation_calibration"])
            lo, hi = ranges(view(x.to(torch.float64), spec), spec)
            lo, hi = lo.cpu(), hi.cpu()
            if name in bounds:
                lo, hi = torch.minimum(lo, bounds[name][0]), torch.maximum(hi, bounds[name][1])
            bounds[name] = (lo, hi)

        return observe

    try:
        for name, spec in activations.items():
            handles.append(backend.model.get_submodule(name).register_forward_pre_hook(hook(name, spec)))
        if selected:
            for x, _ in data.batches("calibration"):
                forward(backend, backend.model, x)
        elif any(spec["granularity"] != "per_tensor" for spec in activations.values()):
            # Dynamic grids need actual shape, not fitted calibration state.
            sample = next(iter(data.batches("search")), None)
            require(sample is not None, "形状准备需要已有 search 数据样本", code="invalid_data")
            forward(backend, backend.model, sample[0])
    finally:
        for handle in handles:
            handle.remove()
        backend.reset()
    require(set(bounds) == set(selected), "校准未覆盖全部目标位点", code="coverage_mismatch")
    params = {}
    for name, spec in selected.items():
        lo, hi = bounds[name]
        sample = torch.stack((lo, hi)) if spec["granularity"] == "per_tensor" else torch.cat((lo, hi), dim=0)
        method = next(l["method"] for l in plan["layers"] if l["module"] == name)
        params[name] = get_method(method).fit_activation(sample, spec)
    return params, shapes


def build(backend, plan, records, activation_params):
    model = deepcopy(backend.model).eval().requires_grad_(False)
    for layer in plan["layers"]:
        name = layer["module"]
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(name)
        replacement = QuantizedLinear(
            module, layer, plan["execution"], records.get(name), activation_params.get(name, {})
        )
        setattr(model.get_submodule(parent) if parent else model, leaf, replacement)
    return model


class RTNExecutor:
    """方法注册扩展点；每种方法须同时提供 prepare 与能力/校验实现。"""

    def __init__(self, backend, config, data):
        self.backend, self.config, self.data = backend, config, data
        self.input_shapes = {}

    def prepare(self, plan, shapes):
        plan = normalize_plan(plan)
        if plan.get("schema_version") == 3:
            from .pipeline import QuantizationExecutor
            self.plugin_executor = QuantizationExecutor(self.backend, self.config, self.data)
            result = self.plugin_executor.prepare(plan, shapes)
            self.input_shapes = deepcopy(self.plugin_executor.input_shapes)
            return result
        validate_plan(self.backend, plan, self.config, detect(self.backend), shapes, allow_unknown_shapes=True)
        for layer in plan["layers"]:
            get_method(layer["method"])
        params, observed = calibrate(self.backend, plan, self.data)
        self.input_shapes = {**shapes, **observed}
        validate_plan(self.backend, plan, self.config, detect(self.backend), self.input_shapes)
        records = {
            l["module"]: get_method(l["method"]).encode_weight(
                self.backend.model.get_submodule(l["module"]).weight, l["weight"]
            )
            for l in plan["layers"]
            if l["weight"]
        }
        model = build(self.backend, plan, records, params)
        audit = self.audit(model, plan)
        self.backend.reset()
        return model, records, params, audit

    def audit(self, model, plan):
        plan = normalize_plan(plan)
        if plan.get("schema_version") == 3:
            from .pipeline import QuantizationExecutor
            delegate = QuantizationExecutor(self.backend, self.config, self.data)
            delegate.input_shapes = deepcopy(self.input_shapes)
            return delegate.audit(model, plan)
        wrappers = {n: m for n, m in model.named_modules() if isinstance(m, QuantizedLinear)}
        require(
            set(wrappers) == {l["module"] for l in plan["layers"]},
            "实际位点与计划不一致",
            code="coverage_mismatch",
        )
        for layer in plan["layers"]:
            m = wrappers[layer["module"]]
            require(
                m.layer == layer and m.execution == plan["execution"],
                "实际执行规格与配置不一致",
                code="coverage_mismatch",
            )
            source = self.backend.model.get_submodule(layer["module"])
            if layer["weight"]:
                require(m.weight_record is not None, "缺少目标权重编码", code="coverage_mismatch")
                validate_record(m.weight_record, layer["weight"], list(source.weight.shape))
                expected = get_method(layer["method"]).decode_weight(m.weight_record, m.decoded_weight.device)
            else:
                require(m.weight_record is None, "未要求量化的权重被替换", code="coverage_mismatch")
                expected = source.weight
            require(
                torch.equal(m.decoded_weight, expected), "执行权重与配置编码不同", code="coverage_mismatch"
            )
        original = self.backend.model.state_dict()
        current = model.state_dict()
        omitted = {l["module"] + ".weight" for l in plan["layers"]}
        for key, tensor in original.items():
            if key not in omitted:
                require(
                    key in current and tensor_digest(current[key]) == tensor_digest(tensor),
                    f"非目标参数/buffer 改变：{key}",
                    code="protection_failed",
                )
        require(not model.training and all(not p.requires_grad for p in model.parameters()), "候选模型未冻结")
        return {
            "configuration_hash": fingerprint(plan),
            "coverage_exact": True,
            "unselected_state_unchanged": True,
            "protected_head_neck_unchanged": True,
            "all_parameters_frozen": True,
            "modules": list(wrappers),
            "execution": deepcopy(plan["execution"]),
        }

    def export(self, model, plan, records, params, directory, audit):
        plan = normalize_plan(plan)
        if plan.get("schema_version") == 3:
            return self.plugin_executor.export(model, plan, records, params, directory, audit)
        require(audit == self.audit(model, plan), "导出前模型审计与准备阶段不同")
        directory = Path(directory)
        require(not directory.exists(), "产物目录已存在，拒绝覆盖", code="run_exists")
        directory.mkdir(parents=True)
        removed = {l["module"] + ".weight" for l in plan["layers"] if l["weight"]}
        # 保存全部未量化状态，使权重包完整；加载模型架构仍使用固定的宿主适配器。
        state = {
            k: v.detach().cpu().clone()
            for k, v in self.backend.model.state_dict().items()
            if k not in removed
        }
        payload = {"state": state, "weights": records, "activations": params}
        handle, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        os.close(handle)
        try:
            torch.save(payload, tmp)
            with open(tmp, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(tmp, directory / "weights.pt")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        storage_bytes = sum(v.numel() * v.element_size() for v in state.values())
        storage_bytes += sum(
            r["data"].numel() * r["data"].element_size()
            + sum(v.numel() * v.element_size() for v in r["params"].values())
            for r in records.values()
        )
        storage_bytes += sum(
            v.numel() * v.element_size() for values in params.values() for v in values.values()
        )
        manifest = seal(
            {
                "schema_version": 1,
                "status": "exported_pending_validation",
                "identity": self.backend.identity(),
                "plan": deepcopy(plan),
                "configuration_hash": fingerprint(plan),
                "implementation": versions(),
                "input_shapes": self.input_shapes,
                "runtime_config": {k: deepcopy(self.config[k]) for k in ("backend", "constraints", "loop")},
                "weights_sha256": file_sha256(directory / "weights.pt"),
                "tensor_payload_bytes": storage_bytes,
                "file_bytes": (directory / "weights.pt").stat().st_size,
                "calibration_ids": self.data.ids("calibration") if params else [],
                "audit": audit,
                "representation": "packed_weights_with_scales; runtime behavior given by plan.execution",
            }
        )
        atomic_json(directory / "manifest.json", manifest)
        return manifest

    def load(self, directory):
        directory = Path(directory)
        manifest = read_json(directory / "manifest.json")
        plan_version = manifest.get("plan", {}).get("schema_version")
        require((plan_version, manifest.get("schema_version")) in ((2, 1), (3, 2)),
                "通用加载器仅支持标准 v2/v3 模型包；自定义模型包须由对应批准的 make_executor.load 加载",
                code="unsupported_artifact")
        if manifest.get("plan", {}).get("schema_version") == 3:
            from .pipeline import QuantizationExecutor
            self.plugin_executor = QuantizationExecutor(self.backend, self.config, self.data)
            result = self.plugin_executor.load(directory)
            self.input_shapes = deepcopy(self.plugin_executor.input_shapes)
            return result
        require(
            manifest.get("content_sha256")
            == fingerprint({k: v for k, v in manifest.items() if k != "content_sha256"}),
            "产物 manifest 损坏",
            code="invalid_artifact",
        )
        expected_identity = deepcopy(manifest["identity"])
        actual_identity = self.backend.identity()
        if "config" in expected_identity:
            # 仅设备可显式迁移；源码、checkpoint、预处理及数据身份仍须一致。
            # 下面重新实测目标设备能力，native 不会因迁移而回退成 reference。
            expected_identity["config"]["device"] = actual_identity["config"]["device"]
        require(
            expected_identity == actual_identity, "产物模型/数据/源码身份不匹配", code="identity_mismatch"
        )
        require(
            manifest["weights_sha256"] == file_sha256(directory / "weights.pt"),
            "产物权重文件 hash 不匹配",
            code="invalid_artifact",
        )
        plan = manifest["plan"]
        self.input_shapes = deepcopy(manifest["input_shapes"])
        require(fingerprint(plan) == manifest["configuration_hash"], "产物配置不匹配")
        require(
            manifest.get("schema_version") == 1 and manifest.get("implementation") == versions(),
            "产物格式/数值实现版本不匹配",
            code="invalid_artifact",
        )
        validate_plan(self.backend, plan, self.config, detect(self.backend), manifest["input_shapes"])
        # weights_only 避免反序列化任意 Python 模型对象。
        payload = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        expected = {l["module"] for l in plan["layers"] if l["weight"]}
        require(set(payload["weights"]) == expected, "导出权重覆盖不符", code="coverage_mismatch")
        for layer in plan["layers"]:
            if layer["weight"]:
                validate_record(
                    payload["weights"][layer["module"]],
                    layer["weight"],
                    list(self.backend.model.get_submodule(layer["module"]).weight.shape),
                )
        static = {
            l["module"]: l["activation"]
            for l in plan["layers"]
            if l["activation"]
            and l["activation"]["format"] not in RAW_FLOATS
            and plan["execution"]["activation_calibration"] == "static"
        }
        require(set(payload["activations"]) == set(static), "静态激活参数覆盖不同", code="coverage_mismatch")
        for name, spec in static.items():
            validate_params(payload["activations"][name], spec, manifest["input_shapes"][name])
        base = self.backend.model.state_dict()
        removed = {name + ".weight" for name in expected}
        require(set(payload["state"]) == set(base) - removed, "未量化状态覆盖不同")
        for name, value in payload["state"].items():
            require(
                tensor_digest(value) == tensor_digest(base[name]),
                f"导出非目标状态不同：{name}",
                code="protection_failed",
            )
        model = build(self.backend, plan, payload["weights"], payload["activations"])
        self.audit(model, plan)
        return model, manifest


class LoadedQuantizedModel:
    """无需 LLM 的推理入口；输入使用 manifest 指定的数据预处理。"""

    def __init__(self, directory, *, device=None):
        from ..backends import create_backend

        manifest = read_json(Path(directory) / "manifest.json")
        config = deepcopy(manifest["runtime_config"])
        if device is not None:
            config["backend"]["device"] = device
        self.backend = create_backend(config["backend"])
        self.model, self.manifest = RTNExecutor(self.backend, config, None).load(directory)
        self.plan = deepcopy(self.manifest["plan"])

    def __call__(self, inputs):
        return forward(self.backend, self.model, inputs.to(self.backend.device))
