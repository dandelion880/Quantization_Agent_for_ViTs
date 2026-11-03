"""执行 Profiling 工具的完整注册；闭包都绑定同一纯工具 Host/Registry。

保留执行许可、特殊位点、计划展开、RTN敏感度、任务程序和证据索引。
数值方法与原初始化顺序不变；不注册或构造科研 Agent。
"""
from copy import deepcopy
from dataclasses import replace
import torch
from ..profiling.programs import ProfilePrograms, error_metrics
from ..schemas import STR, arr, obj, require, fingerprint
from ..registry import ToolDefinition
from .contracts import PLUGIN_PLAN_SCHEMA, OPTIONAL_SPEC
from .capabilities import detect, validate_plan
from .codecs import DTYPES, fit, qdq

def register_execution_tools(self):
    registry = self.registry
    from .plugin_tools import register_plugin_tools
    register_plugin_tools(self)
    # 旧 probe 只支持 toy；真实会话使用已验证的 profile_sensitivity。
    # 不向 LLM 宣传一个在 server 后端必然失败的重复入口。
    registry.definitions.pop("probe", None)
    cap = detect(self.backend)
    old = registry.definitions["query_structure"]
    def structure():
        # 普通 backend 以只读 Linear 为边界；执行会话必须展示实际 Conv/head 许可。
        permissions = self.profile_module_permissions()
        return {"modules": [{**row, "protected": permissions[row["name"]]}
                            for row in self.backend.modules()], "capabilities": cap}
    registry.definitions["query_structure"] = replace(
        old, handler=structure
    )
    free = {"type": "object", "additionalProperties": True}
    # v2 trial 在外层 loop 管理，不能继续返回旧 toy schema 或空历史。
    old = registry.definitions["read_trials"]

    def read_trials(ids):
        rows = self.state.get("execution_feedback", [])
        require(set(ids) <= {r["trial"] for r in rows}, "未知执行 trial")
        return {"trials": [deepcopy(r) for r in rows if not ids or r["trial"] in ids]}

    registry.definitions["read_trials"] = replace(
        old, returns=obj({"trials": arr(free)}), handler=read_trials
    )
    registry.register(
        ToolDefinition(
            "query_execution_capabilities",
            "实际设备、格式/method/原生路径和资源情况；不能从格式名称推断 kernel",
            obj({}),
            free,
            ("profiling",),
            lambda: detect(self.backend),
        )
    )

    registry.register(
        ToolDefinition(
            "validate_experiment",
            "核验 v3 配置的位点、组合、格式、粒度、硬件和资源；plan 与本会话 draft 引用二选一。compose 已执行相同预检。",
            obj({"plan": PLUGIN_PLAN_SCHEMA, "draft": STR}, []),
            free,
            ("profiling",),
            self.validate_plan_or_draft,
        )
    )
    from .planning import register_planning_tool
    register_planning_tool(self)
    from .task_programs import register_task_programs
    register_task_programs(self)
    programs = ProfilePrograms(self)

    def sensitivity(modules, variants, count, seed, offset):
        keys = [fingerprint({k: v for k, v in item.items() if k != 'name'}) for item in variants]
        require(len(keys) == len(set(keys)),
                '变体数值配置重复；name 只是标签，不能用改名选择算法。profile_sensitivity 仅执行 RTN；'
                '其它方法请用 profile_ptq_candidates 并为不同方法分别提供独立 plan',
                code='duplicate_variant', path='$.variants')
        required = self.config.get('loop', {}).get('required_execution', {})
        # 局部诊断遵守任务协议；不能用硬编码 dynamic 拒绝合法 static 请求。
        default_mode = 'dynamic' if any(
            v['activation'] and v['activation']['granularity'] in ('per_token', 'per_group')
            for v in variants) else 'static'
        execution = {'mode': 'reference_qdq', 'compute_dtype': required.get('compute_dtype', 'FP32'),
                     'activation_calibration': required.get('activation_calibration', default_mode),
                     'order': 'joint'}
        args = {
            "modules": modules,
            "variants": variants,
            "count": count,
            "seed": seed,
            "offset": offset,
            "definition": "reference_rtn_v2",
            "execution": execution,
        }

        def compute():
            from .plugin_tools import require_regular_modules
            from .sites import site_catalog
            from .contracts import validate_spec
            require_regular_modules(self.backend, modules, "profile_sensitivity")
            infos = programs.module_info(modules)
            catalog = site_catalog(self.backend, self.config)
            require(len({v['name'] for v in variants}) == len(variants), 'variant 名重复')
            local_config = deepcopy(self.config)
            local_config['constraints']['precision_mode'] = 'mixed'
            plans, local_configs = {}, {}
            for name, info in infos.items():
                require(name in catalog and catalog[name]['kind'] in ('linear', 'conv2d'),
                        f'{name} 不是已验证的 Linear/Conv；特殊位点用 profile_ptq_candidates', code='unsupported_site')
                local_configs[name] = deepcopy(local_config)
                local_configs[name]['constraints']['fixed_layers'] = {n: v for n, v in self.config['constraints']['fixed_layers'].items() if n == name}
                for variant in variants:
                    for target in ('weight', 'activation'):
                        spec = variant[target]
                        if spec and (target == 'weight' or spec['granularity'] == 'per_tensor'):
                            # 形状无关错误先拒绝，避免无效 axis 导致一次昂贵采集。
                            validate_spec(spec, target, info['weight_shape'] if target == 'weight' else [1, 1],
                                          execution['activation_calibration'])
                    layer = {'module': name, 'kind': catalog[name]['kind'], 'weight': variant['weight'],
                             'activation': variant['activation'], 'weight_method': 'rtn' if variant['weight'] else None,
                             'activation_method': 'rtn' if variant['activation'] else None,
                             'reason': 'local diagnostic', 'evidence': ['local']}
                    plan = {'schema_version': 3, 'unlisted_modules': 'keep_fp', 'preprocess': [],
                            'execution': deepcopy(execution),
                            'layers': [layer], 'unresolved': []}
                    validate_plan(self.backend, plan, local_configs[name], cap, diagnostic=True)
                    plans[name, variant['name']] = plan
            rows, shapes, ids = [], {}, None
            dtype = DTYPES[execution['compute_dtype']]
            device = 'cpu' if dtype == torch.float32 else self.backend.device

            def errors(before, after):
                return error_metrics(before.detach().float().cpu().numpy(), after.detach().float().cpu().numpy())

            for name in modules:
                # 保留完整 count 和固定样本；逐模块采集避免多层缓存相加超过限额。
                values, sample_ids, _ = programs.capture([name], count, seed, offset)
                require(ids is None or sample_ids == ids, '局部诊断样本身份不一致')
                ids = sample_ids
                source = self.backend.model.get_submodule(name)
                x = torch.from_numpy(values[name]).to(device=device, dtype=dtype)
                shapes[name] = list(x.shape)
                w = source.weight.detach().to(device=device, dtype=dtype)
                bias = source.bias.detach().to(device=device, dtype=dtype) if source.bias is not None else None
                if bias is not None and hasattr(source, "bias_mask"):
                    bias = bias * source.bias_mask.to(device=device, dtype=dtype)
                operator = source._conv_forward if catalog[name]['kind'] == 'conv2d' else torch.nn.functional.linear
                reference = operator(x, w, bias)
                for variant in variants:
                    # 局部诊断无需包含所有固定层，但所探测层的规格和保护仍须满足约束。
                    validate_plan(self.backend, plans[name, variant['name']], local_configs[name], cap, {name: list(x.shape)}, diagnostic=True)
                    qw = qdq(w, variant["weight"]) if variant["weight"] else w
                    spec = variant['activation']
                    if not spec:
                        qx = x
                    elif execution['activation_calibration'] == 'static':
                        # 在固定采样池拟合一次并冻结；不在每张图像上重新估计尺度。
                        qx = qdq(x, spec, fit(x, spec))
                    else:
                        # capture 把 count 个输入组成一个 batch，dynamic 在此 batch 拟合。
                        qx = qdq(x, spec)
                    result = operator(qx, qw, bias)
                    rows.append(
                        {
                            "module": name,
                            "variant": variant,
                            "weight": errors(w, qw),
                            "activation": errors(x, qx),
                            "weight_only_output": errors(reference, operator(x, qw, bias)),
                            "activation_only_output": errors(reference, operator(qx, w, bias)),
                            "joint_output": errors(reference, result),
                        }
                    )
            self.backend.reset()
            return {
                "rows": rows,
                "shapes": shapes,
                "sample_ids": ids,
                "execution": execution,
                "operator_device": str(device),
                "scope": "isolated Linear/Conv QDQ on FP captured inputs, local minmax; static fits this sampled pool once, dynamic fits this captured batch of count images; not whole-model calibration or task accuracy",
                "task_accuracy": None,
                "speedup": None,
            }

        detail, ref, cached = registry._cached_measurement(
            "execution_sensitivity",
            args,
            len(modules) * len(variants) * count,
            compute,
            adjustable_request=True,
        )
        return {
            "summary": detail["result"],
            "shapes": detail["result"]["shapes"],
            "evidence": ref,
            "cached": cached,
            "source": detail["source"],
        }

    old = registry.definitions["profile_sensitivity"]
    params = deepcopy(old.parameters)
    params["properties"]["variants"] = arr(
        obj(
            {
                "name": STR,
                "method": {"enum": ["rtn"]},
                "weight": OPTIONAL_SPEC,
                "activation": OPTIONAL_SPEC,
            }
        ),
        minItems=1,
        maxItems=8,
    )
    registry.definitions["profile_sensitivity"] = replace(
        old,
        parameters=params,
        handler=sensitivity,
        description="实际 Linear/Conv 输入/权重的 RTN 局部 QDQ，遵守任务 static/dynamic 和计算精度；static 在本次采样池拟合一次，dynamic 在本次 count 张图像的 batch 拟合；输入来自FP模型，非全模型精度。per_tensor 的 axis/group_size 必须 null；特殊网格用 profile_ptq_candidates；消耗 count×模块数×变体数",
    )
    # 会话已知实际校准池容量；schema 不应继续声称可请求固定池之外的样本数。
    for name, tool in list(registry.definitions.items()):
        parameters = deepcopy(tool.parameters)
        changed = False
        for branch in parameters.get("anyOf", [parameters]):
            fields = branch.get("properties", {})
            if "count" in fields and "offset" in fields:
                fields["count"]["maximum"] = min(fields["count"].get("maximum", 256),
                                                  self.config["backend"]["calibration_samples"])
                changed = True
        if changed:
            registry.definitions[name] = replace(tool, parameters=parameters)
    from .search_context import register_search_context
    register_search_context(self)

