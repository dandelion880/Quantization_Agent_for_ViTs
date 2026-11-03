"""experiment_v1 的真实 Data/Backend 接口及宿主观测，不接受插件自报执行范围。"""
from copy import deepcopy
from core.cards import ContractError, fingerprint


class ObservedData:
    def __init__(self, data):
        self._data = data
        self.labels, self.identity = data.labels, deepcopy(data.identity)
        self.reads = {}

    def ids(self, split):
        if not isinstance(split, str) or not split.strip():
            raise ContractError("data.ids/batches(split) 要求非空字符串，不能传入选择字典")
        if split not in self.identity["split_ids"]:
            raise ContractError("数据 split 未注册: " + split)
        return self._data.ids(split)

    def batches(self, split):
        self.ids(split)
        for inputs, sid in self._data.batches(split):
            self.reads.setdefault(split, []).append(sid)
            yield inputs, sid


def execute_custom(plugin, payload, backend, cfg, data, directory):
    import torch
    from ptq_agent.execution.evaluation import model_digest
    spec = payload["execution"]
    observed = ObservedData(data)
    candidate = deepcopy(backend.model)
    sites = set()
    selected_site = spec.get("site", spec["resolved_choices"].get("target_site"))
    handles = []
    if selected_site:
        # 只记录实际前向访问的明确模块；无法映射的特殊位点须使用已有量化服务。
        for model in (backend.model, candidate):
            if selected_site in dict(model.named_modules()):
                handles.append(model.get_submodule(selected_site).register_forward_pre_hook(
                    lambda _module, _args: sites.add(selected_site)))
    calls = []

    class BackendView:
        def __getattr__(self, name):
            return getattr(backend, name)

        def forward_model(self, model, inputs):
            if not torch.is_tensor(inputs):
                raise ContractError("backend.forward_model(model, image_tensor) 不能接收整个 batch 二元组")
            calls.append("reference" if model is backend.model else "candidate")
            return backend.forward_model(model, inputs)

    parameters = deepcopy(spec.get("parameters", {}))
    original_parameters = fingerprint(parameters)
    original = model_digest(backend.model)
    try:
        value = plugin.execute(deepcopy(payload), {"candidate_model": candidate, "reference_model": backend.model,
            "data": observed, "backend": BackendView(), "configuration": deepcopy(cfg),
            "parameters": parameters, "directory": str(directory)})
    finally:
        for handle in handles:
            handle.remove()
    if model_digest(backend.model) != original or fingerprint(parameters) != original_parameters:
        raise ContractError("自定义执行改写了冻结基准或参数")
    if not isinstance(value, dict) or not isinstance(value.get("observations"), dict):
        raise ContractError("自定义实验缺少规范 observations")
    value.pop("execution_facts", None)
    if isinstance(value.get("observations", {}).get("reload"), dict):
        value["observations"]["reload"].pop("host_verified", None)
    measured = {}
    for split, ids in observed.reads.items():
        unique_ids = list(dict.fromkeys(ids))
        measured[split] = {"split": split, "sample_ids": unique_ids, "sample_count": len(unique_ids),
                           "full_dataset": split == "final" and unique_ids == data.ids(split)
                                           and data.identity.get("full_validation") is True}
    for key in ("calibration", "evaluation"):
        selection = spec.get(key)
        if selection:
            split = selection["split"]
            offset, count = selection.get("offset", 0), selection["count"]
            if offset < 0 or offset >= len(data.ids(split)) or (count is not None and count > len(data.ids(split)) - offset):
                raise ContractError("自定义数据视图/接口案例样本池不足，不能静默缩小冻结样本数")
            expected = data.ids(split)[offset:None if count is None else offset + count]
            # 超额读取也是真实范围，不能以插件自报 sample_selection 隐藏它。
            if measured.get(split, {}).get("sample_ids") != expected:
                raise ContractError("自定义执行实际读取范围与冻结 " + key + " 不一致")
    for row in value["observations"].get("evaluations", {}).values():
        if row.get("sample_ids") != measured.get(row.get("split"), {}).get("sample_ids"):
            raise ContractError("自定义评测 sample_ids 没有实际数据读取支持")
        if row.get("full_dataset") is True and measured[row["split"]]["full_dataset"] is not True:
            raise ContractError("自定义评测 full_dataset 没有实际完整数据范围支持")
    value["execution_facts"] = {"measurement_kind": "custom_local_statistic" if sites else "custom_computation",
        "sites": sorted(sites), "data": measured, "parameters": deepcopy(parameters), "forward_calls": len(calls)}
    value["sample_selection"] = {split: row["sample_ids"] for split, row in measured.items()}
    return value


def interface_fixture(execution, case):
    """显式 CPU 接口案例；使用生产 VerifiedManifestData，绝不把此数据作为科研结果。"""
    import torch
    from types import SimpleNamespace
    from ptq_agent.execution.evaluation import VerifiedManifestData
    from ptq_agent.backends.server import ServerBackend
    shape, sites = case["shape"], case["sites"]
    torch.manual_seed(case["seed"])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in sites:
                parent = self
                for part in name.split(".")[:-1]:
                    if part not in parent._modules:
                        parent.add_module(part, torch.nn.Module())
                    parent = parent._modules[part]
                parent.add_module(name.split(".")[-1], torch.nn.Linear(shape[-1], shape[-1]))

        def forward(self, x):
            for name in sites:
                x = self.get_submodule(name)(x)
            return x

    class Backend:
        device = "cpu"
        spec = {"family": "interface_fixture", "classification_head": None}
        task_head = None
        modules = ServerBackend.modules

        def forward_model(self, model, inputs):
            with torch.inference_mode():
                return model(inputs)

    backend = Backend()
    if len(shape) == 4 and 'patch_embed.proj' in sites:
        # 复用已有 tiny ViT 的构造：保留 Conv2d patch embedding 和三维 Linear 输入。
        # 这里只验证接口；随机小模型不会成为 DeiT/ImageNet 科研 Evidence。
        import collections.abc
        import re
        import sys
        import types
        shim = types.ModuleType('torch._six')
        shim.container_abcs = collections.abc
        sys.modules.setdefault('torch._six', shim)
        from timm.models.vision_transformer import VisionTransformer
        blocks = [int(match[1]) for name in sites if (match := re.match(r'^blocks\.(\d+)\.', name))]
        if any(size % 16 for size in shape[2:]):
            raise ContractError('ViT CPU 接口夹具的图像边长须是 patch_size=16 的整数倍')
        model = VisionTransformer(img_size=tuple(shape[2:]), patch_size=16, in_chans=shape[1],
                                  embed_dim=24, depth=max(blocks, default=0) + 1,
                                  num_heads=3, mlp_ratio=2, num_classes=1000)
        if any(name not in dict(model.named_modules()) for name in sites):
            raise ContractError('ViT CPU 接口夹具不含声明的位点，不能替换为 Linear 链')
        backend.model = model.eval().requires_grad_(False)
    else:
        backend.model = Model().eval().requires_grad_(False)
    # Native site_catalog consumes the same readonly module inventory as the real backend.
    # Derive it from the constructed fixture, never invent unconstructed scientific targets.
    backend._modules = []
    for name, module in backend.model.named_modules():
        if not name or not list(module.parameters(recurse=False)):
            continue
        parts, weight = name.split('.'), getattr(module, 'weight', None)
        backend._modules.append({'name': name, 'type': type(module).__name__,
            'block': int(parts[1]) if len(parts)>1 and parts[0]=='blocks' and parts[1].isdigit() else None,
            'protected': not isinstance(module, torch.nn.Linear) or name=='head'
                         or name.startswith(('head.', 'neck.', 'decode_head.')),
            'weight_shape': list(weight.shape) if weight is not None else [],
            'activation_shape': None, 'activation_layout': 'captured at runtime; no assumed feature axis'})
    counts = case["counts"]
    splits = {split: [{"sample_id": f"interface-{split}-{i}", "path": f"interface-{split}-{i}"}
                      for i in range(counts[split])] for split in ("calibration", "search", "final")}
    images = {row["sample_id"]: torch.randn(shape) for rows in splits.values() for row in rows}
    backend.rows = [row for rows in splits.values() for row in rows]
    backend.data_adapter = SimpleNamespace(splits=splits, PREPROCESS="CPU interface fixture only", full_validation=False,
                                           load_image=lambda row: images[row["sample_id"]].clone())
    cfg = {"backend": {**{s + "_samples": n for s, n in counts.items()},
                        "data_seeds": {s: case["seed"] for s in counts}}, "loop": {"evaluation": "feature_fidelity"}}
    return backend, cfg, VerifiedManifestData(backend, cfg)


def validate_interface_cases(payload):
    if payload["execution"].get("interface") != "experiment_v1":
        return
    cases = payload["validation"].get("interface_cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("experiment_v1 需要 validation.interface_cases；数学 context 不能替代真实 Data/Backend 接口案例")
    for index, case in enumerate(cases):
        if (not isinstance(case, dict) or set(case) != {"seed", "shape", "sites", "counts"}
                or type(case["seed"]) is not int or not isinstance(case["shape"], list) or len(case["shape"]) < 2
                or any(type(n) is not int or n <= 0 for n in case["shape"]) or case["shape"][0] != 1
                or not isinstance(case["sites"], list) or not case["sites"]
                or any(not isinstance(s, str) or not s.strip() for s in case["sites"])
                or not isinstance(case["counts"], dict) or set(case["counts"]) != {"calibration", "search", "final"}
                or any(type(n) is not int or n <= 0 for n in case["counts"].values())):
            raise ContractError("interface_cases 需要 seed、batch=1 的正整数 shape、sites 和三个 split 的正整数 counts")
        for key in ('calibration', 'evaluation'):
            selection = payload['execution'].get(key)
            if selection is None:
                continue
            if (not isinstance(selection, dict) or selection.get('split') not in case['counts']
                    or 'count' not in selection or (selection['count'] is not None
                        and (type(selection['count']) is not int or selection['count'] <= 0))
                    or type(selection.get('offset', 0)) is not int or selection.get('offset', 0) < 0):
                raise ContractError(f'validation.interface_cases[{index}] 无法提供冻结 {key} 数据视图 {selection!r}')
            capacity = case['counts'][selection['split']]
            count, offset = selection['count'], selection.get('offset', 0)
            if offset >= capacity or (count is not None and count + offset > capacity):
                raise ContractError(f'validation.interface_cases[{index}] {key} 冻结 count={count}, offset={offset} '
                                    f'超过 {selection["split"]} 夹具容量 {capacity}；须由草案作者明确修订，不能缩小科学样本数')
