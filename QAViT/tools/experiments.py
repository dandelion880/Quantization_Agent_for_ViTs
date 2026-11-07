"""ExperimentCard/EvidenceCard 边界、精确缓存及已有工具适配入口。

科研理由与执行指纹分开：不同问题可以引用同一次有效测量；明确的重复验证
必须产生新的 execution_id。事实只从经过校验的工作进程回执构建。
"""
from copy import deepcopy
from pathlib import Path
import platform
from importlib.metadata import version

from runtime.paths import ROOT
from runtime.identity import measurement_implementation
from core.cards import ContractError, check_constraints, fingerprint, make_card, validate_payload, validate_schema, TEXT, OBJECT, PAYLOAD_SCHEMAS
from core.journal import file_sha256, read_json
from core.execution_contract import freeze_contract, declared_contract, declared_rules, research_contracts, merge, uses_quantization_contract
from tools.result_contract import check_results, result_requirements
from core.evidence_scope import canonical_scope


SERVICE_VERSION = "qavit.experiments/2.1"
SERVICES = {
    "profile": {"levels": ["L1"], "description": "已有15个真实 Profiling 接口；v2独立会话和全局预算，无measurement次数上限"},
    "tensor_probe": {"levels": ["L1", "L2"], "description": "真实ViT位点输入、局部输出及小样本整网回放；可挂载经批准的quantizer_v1实现"},
    "evaluate_artifact": {"levels": ["L2", "L3"], "description": "加载标准 plan v2/package v1 或 plan v3/package v2 冻结模型包，使用显式数据视图进行分类/分割；不继承历史指标。自定义包须由对应批准的 executor_v1 加载"},
    "quantize_evaluate": {"levels": ["L2", "L3"], "description": "已有数值算法直接提供 execution.plan；新增算法才经批准 executor_v1 生成计划/执行器。两路共用冻结合同、覆盖审计、导出重载及评测"},
    "custom_experiment": {"levels": ["L1", "L2", "L3"], "description": "experiment_v1实现新增算法/统计；按原始需求审查源码及实际执行条件，宿主记录真实结果"},
}

# 同一 ExperimentCard 结构的提交视图：只省略宿主可确定的机械字段。
# 冻结卡片仍使用原完整合同，不增加另一套服务/卡片接口。
EXPERIMENT_DRAFT_SCHEMA = deepcopy(PAYLOAD_SCHEMAS['ExperimentCard'])
EXPERIMENT_DRAFT_SCHEMA['required'] = ['question', 'measurements']
EXPERIMENT_DRAFT_SCHEMA['properties']['resources']['required'] = []
EXPERIMENT_DRAFT_SCHEMA['properties']['execution'] = deepcopy(EXPERIMENT_DRAFT_SCHEMA['properties']['execution'])
EXPERIMENT_DRAFT_SCHEMA['properties']['execution']['required'] = []
EXPERIMENT_DRAFT_SCHEMA['properties']['execution']['properties'] = {
    **{key: deepcopy(OBJECT) for key in ('backend', 'resolved_choices', 'parameters',
        'quantization_contract', 'method_options', 'arguments')},
    'requires_code': {'type': 'boolean'}, 'training': {'type': 'boolean'},
    'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2 ** 63 - 1},
    **{key: TEXT for key in ('service', 'interface', 'site', 'method', 'model_dir')},
    'calibration': {'type': 'object', 'additionalProperties': True, 'properties': {
        'split': TEXT, 'count': {'anyOf': [{'type': 'integer', 'minimum': 1}, {'type': 'null'}]},
        'offset': {'type': 'integer', 'minimum': 0}}},
    'evaluations': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'properties': {
        'kind': TEXT, 'split': TEXT, 'count': {'anyOf': [{'type': 'integer', 'minimum': 1}, {'type': 'null'}]},
        'offset': {'type': 'integer', 'enum': [0]}, 'full_dataset': {'type': 'boolean'},
        'metrics': {'type': 'array', 'items': TEXT}}}},
}
EXPERIMENT_DRAFT_SCHEMA['properties']['resources']['properties']['gpu'] = {
    'anyOf': [{'type': 'integer', 'minimum': 0}, {'type': 'null'}]}
EXPERIMENT_DRAFT_SCHEMA['properties']['resources']['properties']['gpu_count'] = {
    'type': 'integer', 'minimum': 1}


def normalize_probe_choices(execution):
    """Mirror supplied activation choices using the existing tensor-probe semantics."""
    if execution['service'] != 'tensor_probe':
        return
    choices, parameters = execution.setdefault('resolved_choices', {}), execution.get('parameters', {})
    if 'bits' in parameters:
        choices.setdefault('activation_bits', parameters['bits'])
        if parameters['bits'] != choices['activation_bits']:
            raise ContractError('execution.parameters.bits 与 execution.resolved_choices.activation_bits 不一致')
    granularity = choices.get('granularity', {})
    wanted = granularity.get('activation') if isinstance(granularity, dict) else granularity
    if wanted and 'granularity' in parameters and parameters['granularity'] != wanted:
        raise ContractError('execution.parameters.granularity 与 execution.resolved_choices.granularity 不一致')
    if isinstance(granularity, str):
        choices['granularity'] = {'activation': granularity}


def check_resources(resources, runtime):
    if runtime is None:
        return
    limit = runtime.get('cpu_threads')
    if limit is not None and resources.get('cpu_threads', limit) > limit:
        raise ContractError('请求 CPU 线程超出本次资源上限')
    if resources.get('gpu') is not None and resources['gpu'] not in runtime.get('allowed_gpus', []):
        raise ContractError('请求 GPU 不在本次授权范围')
    if type(resources.get('gpu_count', 1)) is not int or resources.get('gpu_count', 1) < 1:
        raise ContractError('gpu_count 必须为正整数；实际按能力与空闲授权卡分配')


def check_backend(adapter, brief):
    if not isinstance(adapter, dict):
        raise ContractError('execution.backend 必须是对象')
    approved = [brief['model']['adapter']] if brief['model'].get('adapter') else []
    approved += brief['model'].get('allowed_adapters', [])
    if approved and adapter not in approved:
        raise ContractError('execution.backend 与 ResearchBrief.model 的授权适配器冲突')


def prepare_request(payload, brief, runtime=None):
    """Preserve scientific intent while filling only host defaults; no execution freeze."""
    validate_schema(payload, EXPERIMENT_DRAFT_SCHEMA, 'ExperimentCard draft')
    value = deepcopy(payload)
    value.setdefault('level', 'L1')
    value.setdefault('controls', {})
    value.setdefault('scope', {})
    value.setdefault('validation', {})
    value.setdefault('replication', {'requested': False, 'purpose': '', 'configuration_difference': {}})
    resources = value.setdefault('resources', {})
    resources.setdefault('estimated_seconds', (runtime or {}).get('task_timeout_seconds', 300))
    resources.setdefault('cpu_threads', (runtime or {}).get('cpu_threads', 1))
    check_resources(resources, runtime)
    execution = value.setdefault('execution', {})
    execution.setdefault('service', 'custom_experiment')
    service = execution['service']
    if service not in SERVICES:
        raise ContractError('execution.service 未知: ' + str(service))
    if value['level'] not in SERVICES[service]['levels']:
        raise ContractError('level=' + value['level'] + ' 与 execution.service=' + service + ' 支持的实验层级不一致')
    if 'backend' in execution:
        check_backend(execution['backend'], brief)
    if any(k.startswith('required_') for k in value):
        raise ContractError('ExperimentCard 必需执行约束须放入 execution.quantization_contract')
    execution.setdefault('requires_code', service == 'custom_experiment' or
                         (service == 'quantize_evaluate' and 'plan' not in execution))
    if service == 'quantize_evaluate' and 'plan' in execution and (execution['requires_code'] or 'interface' in execution):
        raise ContractError('execution.plan 使用现有执行器，不能同时请求新增代码/interface；明确选择实现来源')
    if execution['requires_code']:
        interfaces = {'tensor_probe': 'quantizer_v1', 'quantize_evaluate': 'executor_v1',
                      'custom_experiment': 'experiment_v1'}
        if service not in interfaces:
            raise ContractError('此已有服务不接收新增代码')
        execution.setdefault('interface', interfaces[service])
        if execution['interface'] != interfaces[service]:
            raise ContractError('execution.interface 与 execution.service 不匹配')
    execution.setdefault('resolved_choices', {})
    choices = execution['resolved_choices']
    normalize_probe_choices(execution)
    own = declared_contract(execution, require_targets=False) if uses_quantization_contract(execution) else {}
    if 'training' in choices and type(choices['training']) is not bool:
        raise ContractError('execution.resolved_choices.training 必须为 boolean')
    if 'training' in execution:
        # Check the mirror without rewriting the original request.
        merge(deepcopy(choices), {'training': execution['training']}, 'resolved_choices')
    training = execution.get('training', choices.get('training'))
    if training is True and (service in ('profile', 'tensor_probe', 'evaluate_artifact')
                            or brief.get('hard_constraints', {}).get('training_allowed') is False
                            or brief.get('hard_constraints', {}).get('training') is False):
        raise ContractError('execution.training 与此次执行不训练的约束冲突')
    if 'calibration_samples' in choices:
        count = choices['calibration_samples']
        if count is not None and (type(count) is not int or count <= 0):
            raise ContractError('execution.resolved_choices.calibration_samples 必须是正整数或 null')
    calibration = execution.get('calibration')
    if 'calibration' in execution and not isinstance(calibration, dict):
        raise ContractError('execution.calibration 必须是对象')
    if calibration is not None and 'count' in calibration and 'calibration_samples' in choices:
        if calibration['count'] != choices['calibration_samples']:
            raise ContractError(f"execution.calibration.count={calibration['count']!r} 与 "
                                f"execution.resolved_choices.calibration_samples={choices['calibration_samples']!r} "
                                "显式冲突；请按原科学规格修复，尚未读取实际数据")
    if service in ('tensor_probe', 'quantize_evaluate') and isinstance(calibration, dict):
        if 'split' in calibration and calibration['split'] != 'calibration':
            raise ContractError('execution.calibration.split=' + repr(calibration['split'])
                                + ' 与 service=' + service + " 固定读取的 'calibration' split 冲突")
    # Training permission applies throughout the run. Final quantization goals
    # are enforced only for L3, not for local diagnostics or ablations.
    applicable = {'training'}
    if value['level'] == 'L3':
        applicable.update(brief.get('hard_constraints', {}))
    actual = deepcopy(choices)
    if 'training' in execution:
        actual['training'] = execution['training']
    if isinstance(calibration, dict) and 'count' in calibration:
        actual['calibration_samples'] = calibration['count']
    fixed = {field: rule for field, rule in brief.get('hard_constraints', {}).items() if field in applicable}
    check_constraints({'hard_constraints': declared_rules(fixed, actual)}, actual)
    if value['level'] == 'L3':
        hard, final = research_contracts(brief)
        merge(final, own)
        scalar_fields = ('specs', 'methods', 'execution', 'parameters')
        rules = declared_rules({k: v for k, v in hard.items() if k in scalar_fields}, own)
        check_constraints({'hard_constraints': rules}, own)
        merge({k: v for k, v in hard.items() if k not in scalar_fields}, own)
    value['stage'] = 'request'
    return value


def expand_plan(plan):
    """相同设置的显式 modules 数组展开成原生逐位点计划；不推断模型结构。"""
    if not isinstance(plan, dict) or not isinstance(plan.get('layers'), list):
        raise ContractError('execution.plan 需要 layers 数组')
    value = deepcopy(plan)
    layers = []
    for row in value['layers']:
        if not isinstance(row, dict):
            raise ContractError('plan.layers 每项必须是对象')
        if 'modules' not in row:
            layers.append(row)
            continue
        names = row.pop('modules')
        if ('module' in row or not isinstance(names, list) or not names
                or any(not isinstance(n, str) or not n.strip() for n in names) or len(names) != len(set(names))):
            raise ContractError('plan.layers.modules 需要非空无重复名称，且不能同时指定 module')
        layers.extend({'module': name, **deepcopy(row)} for name in names)
    value['layers'] = layers
    from core.execution_contract import validate_plan_schema
    return validate_plan_schema(value)


def asset(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ContractError("实验资产不存在: " + str(path))
    if ROOT.parent not in path.parents:
        raise ContractError("实验资产必须位于用户工作区内")
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def runtime_identity():
    return {"python": platform.python_version(), "platform": platform.system(), "host": platform.node(),
            "architecture": platform.machine(), "torch": version("torch"), "numpy": version("numpy")}


def normalize_calibration(value, brief=None):
    """execution.calibration 是执行视图；旧的样本数选择只作为一致性镜像。"""
    choices = value["resolved_choices"]
    fixed_split = value["service"] in ("tensor_probe", "quantize_evaluate")
    from ptq_agent.execution.pipeline import plan_uses_calibration
    required = value["service"] == "tensor_probe" or (
        value["service"] == "quantize_evaluate" and not value.get("requires_code")
        and plan_uses_calibration(value.get("plan", {})))
    if not required and "calibration" not in value and "calibration_samples" not in choices:
        return
    calibration = value.setdefault("calibration", {})
    if not isinstance(calibration, dict):
        raise ContractError("execution.calibration 必须是对象；不能从 null 或其它类型推断数据视图")
    if "calibration_samples" in choices:
        count = choices["calibration_samples"]
        if count is not None and (type(count) is not int or count <= 0):
            raise ContractError("execution.resolved_choices.calibration_samples 必须是正整数或 null（完整视图）")
    if "count" not in calibration and "calibration_samples" not in choices and brief:
        # 已明确的协议是执行条件，不要求模型重复拷贝；数据池大小不是默认选择。
        fixed = brief.get("hard_constraints", {}).get("calibration_samples",
                    brief.get("evaluation_protocol", {}).get("calibration_samples"))
        if type(fixed) is int:
            calibration["count"] = fixed
    if "count" not in calibration:
        if choices.get("calibration_samples") is None:
            raise ContractError("execution.calibration.count 缺失，execution.resolved_choices.calibration_samples "
                                "也未给出明确样本数；请声明 count（完整视图须显式 null），不能从后端样本池或全局默认值猜测")
        calibration["count"] = choices["calibration_samples"]
    count = calibration["count"]
    if count is not None and (type(count) is not int or count <= 0):
        raise ContractError("execution.calibration.count 必须是正整数或 null（完整视图）")
    if "calibration_samples" in choices and choices["calibration_samples"] != count:
        raise ContractError(f"execution.calibration.count={count!r} 与 "
                            f"execution.resolved_choices.calibration_samples={choices['calibration_samples']!r} "
                            "显式冲突；请按原科学规格修复，尚未读取实际数据")
    choices["calibration_samples"] = count
    if "split" not in calibration:
        if not fixed_split:
            raise ContractError("execution.calibration.split 缺失；service=" + value["service"]
                                + " 未固定校准 split，仅凭样本数无法唯一确定，请明确声明")
        # 两个内置 worker 均以 data_view(..., 'calibration') 读取，不是科学默认值。
        calibration["split"] = "calibration"
    if not isinstance(calibration["split"], str) or not calibration["split"].strip():
        raise ContractError("execution.calibration.split 必须是非空字符串")
    if fixed_split and calibration["split"] != "calibration":
        raise ContractError("execution.calibration.split=" + repr(calibration["split"])
                            + " 与 service=" + value["service"] + " 固定读取的 'calibration' split 冲突")
    if "offset" in calibration and (type(calibration["offset"]) is not int or calibration["offset"] < 0):
        raise ContractError("execution.calibration.offset 必须是非负整数")


def normalize_execution(execution, brief, *, level=None):
    """纯参数检查和唯一值补全；不读取资产、不创建卡片、不执行实验。"""
    if not isinstance(execution, dict):
        raise ContractError("execution 必须是对象")
    value = deepcopy(execution)
    service = value.get("service")
    if not isinstance(service, str) or service not in SERVICES:
        # 新能力通过custom_experiment扩展，不能把未知名称当作已经实现。
        raise ContractError("execution.service 未知；可用 custom_experiment 请求 Coding 开发: " + str(service))
    if "backend" not in value:
        if not brief["model"].get("adapter"):
            raise ContractError("execution.backend 缺失且 ResearchBrief.model.adapter 未指定")
        value["backend"] = deepcopy(brief["model"]["adapter"])
    check_backend(value['backend'], brief)
    choices = value.setdefault("resolved_choices", {})
    if not isinstance(choices, dict):
        raise ContractError("execution.resolved_choices 必须是对象；不能推断科学参数")
    from core.execution_contract import merge
    if 'training' in value:
        merge(choices, {'training': value.pop('training')}, 'resolved_choices')
    fixed_training = brief.get('hard_constraints', {}).get('training')
    if 'training' not in choices and type(fixed_training) is bool:
        choices['training'] = fixed_training
    if service in ('profile', 'tensor_probe', 'evaluate_artifact'):
        merge(choices, {'training': False}, 'resolved_choices')
    if type(choices.get("training")) is not bool:
        raise ContractError("execution.resolved_choices.training 必须明确此次执行是否训练；Agent依据用户输入与搜索空间选择")
    value.setdefault("seed", brief.get("execution_guidance", {}).get("seed", 0))
    if type(value.get("seed")) is not int or not 0 <= value["seed"] < 2 ** 63:
        raise ContractError("execution.seed 必须是明确的非负64位整数")
    if "requires_code" in value and type(value["requires_code"]) is not bool:
        raise ContractError("execution.requires_code 必须为 boolean")
    builtin_plan = service == 'quantize_evaluate' and 'plan' in value
    value.setdefault('requires_code', service == 'custom_experiment' or (service == 'quantize_evaluate' and not builtin_plan))
    if service == 'custom_experiment' and not value['requires_code']:
        raise ContractError('custom_experiment.requires_code 必须为 true；新增实现先经 Coding 和 Safeguard')
    if service == 'quantize_evaluate':
        if builtin_plan:
            if value['requires_code'] or 'interface' in value:
                raise ContractError('execution.plan 使用现有执行器，不能同时请求新增代码/interface；明确选择实现来源')
            value['plan'] = expand_plan(value['plan'])
            # 共同值由实际计划派生；有不同位点设置时不猜全局值。
            derived = {'execution': deepcopy(value['plan']['execution'])}
            for target in ('weight', 'activation'):
                specs = [r[target] for r in value['plan']['layers'] if r[target] is not None]
                if specs and all(s['bits'] == specs[0]['bits'] for s in specs):
                    derived[target + '_bits'] = specs[0]['bits']
            merge(choices, derived, 'resolved_choices')
        elif not value['requires_code']:
            raise ContractError('quantize_evaluate 缺少 execution.plan；新增实现需要 Coding 和 Safeguard')
    interface = {"tensor_probe": "quantizer_v1", "quantize_evaluate": "executor_v1", "custom_experiment": "experiment_v1"}
    if value['requires_code'] and service in interface:
        value.setdefault('interface', interface[service])
    if value.get("requires_code") and value.get("interface") != interface.get(service):
        raise ContractError("execution.interface 与 execution.service 不匹配")
    if service in ('profile', 'evaluate_artifact') and value['requires_code']:
        raise ContractError('已有服务不接收新插件；新增算法须明确使用 custom_experiment 或相应扩展接口')
    if service == 'profile':
        from tools.profiling import PROFILE_NAMES
        if value.get('tool') not in PROFILE_NAMES or not isinstance(value.get('arguments'), dict):
            raise ContractError('profile 需要已注册 tool 及 arguments 对象；可用工具: ' + ', '.join(PROFILE_NAMES))
    from ptq_agent.execution.pipeline import plan_uses_calibration
    consumes_calibration = (service == "tensor_probe" or "calibration" in value
        or service == "quantize_evaluate" and not value['requires_code']
        and plan_uses_calibration(value.get('plan', {})))
    if service == 'profile':
        from tools.profiling import uses_calibration
        consumes_calibration = uses_calibration(value)
        if consumes_calibration:
            args = value['arguments']
            fixed = brief.get('hard_constraints', {}).get('calibration_samples',
                        brief.get('evaluation_protocol', {}).get('calibration_samples'))
            if 'count' not in args and type(fixed) is int:
                args['count'] = fixed
            args.setdefault('seed', value['seed'])
            args.setdefault('offset', 0)
            if 'count' in args:
                merge(value.setdefault('calibration', {}),
                      {k: args[k] for k in ('count', 'offset')}, 'execution.calibration')
                merge(value['calibration'], {'split': 'calibration'}, 'execution.calibration')
        else:
            # 已有草案可能携带全局参数镜像；不适用的字段不进入执行事实。
            value.pop('calibration', None)
            choices.pop('calibration_samples', None)
            if value['tool'] == 'profile_distribution':
                # 权重读取不消费图像；旧底层签名的无效采样参数由宿主提供。
                for key, default in (('count', 1), ('seed', value['seed']), ('offset', 0)):
                    value['arguments'].setdefault(key, default)
    normalize_calibration(value, brief if consumes_calibration else None)
    if service == "tensor_probe":
        parameters = value.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            raise ContractError("execution.parameters 必须是对象")
        fixed_activation = brief.get('quantization_contract', {}).get('specs', {}).get('activation', {})
        for field, key in (('bits', 'activation_bits'), ('granularity', 'granularity')):
            known = choices.get(key, brief.get('hard_constraints', {}).get(key, fixed_activation.get(field)))
            if field == 'granularity' and isinstance(known, dict):
                known = known.get('activation')
            if known is not None and not isinstance(known, (dict, list)):
                parameters.setdefault(field, known)
        for target, source in (('signed', 'symmetric'), ('axis', 'axis'), ('group_size', 'group_size')):
            if source in fixed_activation:
                parameters.setdefault(target, deepcopy(fixed_activation[source]))
        if parameters.get('granularity') == 'per_tensor':
            parameters.setdefault('axis', None)
            parameters.setdefault('group_size', None)
        missing = [field for field in ('bits', 'signed', 'granularity', 'axis', 'group_size') if field not in parameters]
        if missing:
            raise ContractError('缺失字段: ' + ', '.join('execution.parameters.' + field for field in missing)
                                + '；在 execution_choices.parameters 一次补全原研究所需选择，不猜测科学参数。'
                                  ' per_tensor 的 axis/group_size 可省略，由程序补 null。')
        validate_schema(parameters, {"type": "object", "required": ["bits", "signed", "granularity", "axis", "group_size"],
            "properties": {"bits": {"type": "integer", "minimum": 1}, "signed": {"type": "boolean"},
                           "granularity": TEXT, "axis": {"type": ["integer", "null"]},
                           "group_size": {"type": ["integer", "null"], "minimum": 1}}}, "execution.parameters")
        normalize_probe_choices(value)
    # 量化字段通过唯一结构化合同核验，避免平铺/嵌套别名被当作不同选择。
    scalar_constraints = {k: v for k, v in brief.get("hard_constraints", {}).items()
                          if k == "training" or (k == "calibration_samples" and consumes_calibration and level == "L3")}
    check_constraints({"hard_constraints": scalar_constraints}, choices)
    if brief.get("hard_constraints", {}).get("training_allowed") is False and choices["training"]:
        raise ContractError("execution.resolved_choices.training 与用户禁止训练的约束冲突")
    if "protocol_hash" in value and value["protocol_hash"] != fingerprint(brief["evaluation_protocol"]):
        raise ContractError("execution.protocol_hash 与 ResearchBrief.evaluation_protocol 不一致")
    if "result_protocol" in value and value["result_protocol"] != brief["evaluation_protocol"]:
        raise ContractError("execution.result_protocol 与 ResearchBrief.evaluation_protocol 不一致")
    if value["backend"] == brief["model"].get("adapter"):
        value["model_scope"] = deepcopy(brief["model"].get("scope", {"model": brief["model"].get("name")}))
    else:
        value["model_scope"] = {"model": "adapter:" + fingerprint(value["backend"])}
    value = freeze_contract(value, brief, level=level)
    if builtin_plan:
        bindings = {row['parameter'] for row in value['quantization_contract'].get('parameter_bindings', [])}
        unconsumed = set(value.get('parameters', {})) - bindings
        if unconsumed:
            raise ContractError('内置 execution.plan 不消费 execution.parameters: ' + ', '.join(sorted(unconsumed))
                + '；将选择写入实际 plan 字段，或用源码实现新增算法参数')
    return value


def check_l3_evaluations(payload, brief):
    """只检查 Brief 已结构化声明的 L3 评测项，不推断自然语言任务或补默认双任务。"""
    required = brief["evaluation_protocol"].get("evaluations")
    if payload["level"] != "L3" or required is None:
        return
    if not isinstance(required, list) or any(not isinstance(r, dict) or not r.get("kind") for r in required):
        raise ContractError("ResearchBrief.evaluation_protocol.evaluations 必须明确列出带 kind 的评测项；不能猜测任务")
    requested = payload["execution"].get("evaluations", [])
    if not isinstance(requested, list) or len(requested) != len(required):
        raise ContractError("execution.evaluations 与 ResearchBrief.evaluation_protocol.evaluations 的任务数量不一致")
    remaining = list(requested)
    for index, row in enumerate(required):
        fields = dict(row)
        match = next((i for i, actual in enumerate(remaining) if isinstance(actual, dict)
                      and all(k in actual and fingerprint(actual[k]) == fingerprint(v) for k, v in fields.items())), None)
        if match is None:
            raise ContractError(f"execution.evaluations 缺少与 ResearchBrief.evaluation_protocol.evaluations[{index}] "
                                f"一致的任务/数据范围：{fields!r}；实际为 {requested!r}")
        remaining.pop(match)


def preflight_experiment(payload, brief, runtime=None, *, executable=False):
    """New implementations are finalized after Coding; built-ins retain direct execution."""
    value = prepare_request(payload, brief, runtime)
    if value['execution']['requires_code'] and not executable:
        return value
    value["execution"] = normalize_execution(value["execution"], brief, level=value["level"])
    if not value['scope']:
        value['scope'] = deepcopy(value['execution']['model_scope'])
    if 'plan' in value['execution']:
        from core.execution_contract import check_plan
        execution = value['execution']
        check_plan(execution['plan'], execution['resolved_choices'], contract=execution['quantization_contract'])
    value['stage'] = 'execution'
    validate_payload('ExperimentCard', value)
    check_l3_evaluations(value, brief)
    result_requirements(value, brief)
    if value["level"] == "L3":
        value["execution"]["result_protocol"] = deepcopy(brief["evaluation_protocol"])
    return value


def resolve_execution(execution, brief, *, level=None):
    value = normalize_execution(execution, brief, level=level)
    adapter = value["backend"]
    if adapter.get("config_path"):
        path = Path(adapter["config_path"])
        if not path.is_absolute():
            path = ROOT / path
        adapter["config_asset"] = asset(path)
        from ptq_agent.configuration import load_config
        cfg = load_config(path)
        backend = cfg.get("backend", cfg)
        overrides = adapter.get("backend_overrides", {})
        for key in ("model_path", "data_path", "asset_manifest"):
            if overrides.get(key):
                selected = Path(overrides[key])
                overrides[key] = str((selected if selected.is_absolute() else path.parent / selected).resolve())
        backend.update(deepcopy(overrides))
        adapter["checkpoint_assets"] = []
        keys = () if adapter.get("kind") == "fixture" else ("model_path", "data_path", "asset_manifest")
        for key in keys:
            selected = backend.get(key)
            if selected and Path(selected).is_file():
                adapter["checkpoint_assets"].append(asset(selected))
        if adapter.get("kind") != "fixture" and backend.get("task_head"):
            adapter["checkpoint_assets"].append(asset(backend["task_head"]["path"]))
    if adapter.get("factory"):
        filename, _, function = adapter["factory"].partition(":")
        path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        adapter["factory_asset"] = asset(path)
        adapter["factory_function"] = function
    if value.get("model_dir"):
        model = Path(value["model_dir"]).resolve()
        value["model_assets"] = [asset(model / n) for n in ("manifest.json", "weights.pt")]
    value["protocol_hash"] = fingerprint(brief["evaluation_protocol"])
    value["service_version"] = SERVICE_VERSION
    value["runtime"] = runtime_identity()
    value["measurement_implementation"] = measurement_implementation()
    return value


def verify_assets(execution):
    if execution.get("runtime") and execution["runtime"] != runtime_identity():
        raise ContractError("冻结实验的运行时/主机环境已变化，需要刷新规格并复核")
    if execution.get("measurement_implementation") and execution["measurement_implementation"] != measurement_implementation():
        raise ContractError("冻结实验的测量实现已变化，需要新的实验版本和检查")
    def walk(value):
        if isinstance(value, dict):
            if set(("path", "sha256", "bytes")) <= set(value):
                path = Path(value["path"])
                if not path.is_file() or path.stat().st_size != value["bytes"] or file_sha256(path) != value["sha256"]:
                    raise ContractError("冻结实验资产已变化: " + str(path))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(execution)


def execution_fingerprint(card, *, code_artifact=None):
    payload = card["payload"]
    # 内置服务只消费冻结execution、层级和实际计算资源。科学措辞、scope标签
    # 不会改变数值，不能阻止相同配置复用；L1与L2执行路径不同，必须隔离。
    # 插件接口可读取完整payload，保守地将其实际全部输入也纳入身份。
    plugin_input = {k: v for k, v in payload.items() if k != "replication"} if payload["execution"].get("requires_code") else None
    return fingerprint({"execution": payload["execution"], "level": payload["level"], "plugin_input": plugin_input,
                        "execution_resources": {k: payload["resources"].get(k) for k in ("cpu_threads", "gpu")},
                        "code_artifact": ({k: code_artifact[k] for k in ("source_sha256", "interface")} if code_artifact else None),
                        "service_version": SERVICE_VERSION})


class ExperimentBroker:
    def __init__(self, journal, brief_ref, ledger, runtime=None):
        self.journal, self.brief_ref, self.ledger = journal, brief_ref, ledger
        self.runtime = runtime

    def preflight(self, payload):
        return preflight_experiment(payload, self.journal.card(self.brief_ref)["payload"], self.runtime)

    def card_references(self, payload, refs=()):
        """Preserve explicitly cited evidence in every host-created envelope.

        Role visibility and alias resolution remain at the authored handoff;
        this broker validates kind/identity and never grants visibility itself.
        """
        cited = payload.get('evidence_refs', [])
        if not isinstance(cited, list) or any(not isinstance(ref, str) for ref in cited):
            raise ContractError('ExperimentCard.evidence_refs 必须为 EvidenceCard 引用列表')
        for ref in cited:
            if self.journal.card(ref)['kind'] != 'EvidenceCard':
                raise ContractError('ExperimentCard.evidence_refs 只能引用 EvidenceCard')
        return list(dict.fromkeys([self.brief_ref, *refs, *cited]))

    def request(self, payload, *, requester, refs=(), original_request=None):
        original = deepcopy(payload if original_request is None else original_request)
        value = prepare_request(payload, self.journal.card(self.brief_ref)['payload'], self.runtime)
        value['original_request'] = original
        return make_card('ExperimentCard', self.journal.run_id, value, author=requester,
                         refs=self.card_references(value, refs))

    def bind_execution(self, request, artifact):
        from core.execution_contract import merge
        from tools.safeguards import verify_artifact
        verify_artifact(artifact, request)
        payload = deepcopy(request['payload'])
        choices = artifact.get('execution_choices', {})
        if not isinstance(choices, dict):
            raise ContractError('Coding.execution_choices 必须为对象')
        def complete(left, right, path):
            for key, value in right.items():
                if key in left and isinstance(left[key], dict) and isinstance(value, dict):
                    complete(left[key], value, path + '.' + key)
                elif (key in left and isinstance(left[key], list) and isinstance(value, list)
                      and len(left[key]) == len(value)
                      and all(isinstance(a, dict) and isinstance(b, dict) for a, b in zip(left[key], value))):
                    for index, (a, b) in enumerate(zip(left[key], value)):
                        complete(a, b, path + '.' + key + '[' + str(index) + ']')
                else:
                    merge(left, {key: value}, path)
        complete(payload['execution'], choices, 'execution')
        payload.update(request_ref=request['id'], request_sha256=request['content_sha256'])
        return self.freeze(payload, requester=request['author'], refs=[request['id']])

    def freeze(self, payload, *, requester, refs=()):
        payload = preflight_experiment(payload, self.journal.card(self.brief_ref)['payload'], self.runtime, executable=True)
        brief = self.journal.card(self.brief_ref)["payload"]
        payload["execution"] = resolve_execution(payload["execution"], brief, level=payload["level"])
        execution = payload["execution"]
        # GPU 是宿主资源选择；CUDA 后端不能因草案省略编号而在隐藏全部 GPU 后启动。
        if self.runtime is not None:
            from tools.experiment_worker import backend_device
            device = backend_device(execution)
            if str(device).startswith('cuda') and not self.runtime.get('allowed_gpus', []):
                raise ContractError('CUDA 后端没有已授权 GPU；在研究输入 runtime.allowed_gpus 集中配置')
            # All adapters (including factories) use the same host placement.
            # Physical ordinals never select a worker-local CUDA device.
        # 跨作业读取权限只由本次 Journal 派生，不能接受 Agent 自填的文件绑定。
        execution.pop("profile_evidence_bindings", None)
        profile_refs = []
        if execution["service"] == "profile" and execution.get("tool") == "read_profile_evidence":
            from tools.profiling import bind_profile_evidence
            execution["profile_evidence_bindings"] = bind_profile_evidence(self.journal, execution["arguments"]["evidence"])
            profile_refs = [row["evidence_ref"] for row in execution["profile_evidence_bindings"]]
        trial_refs = execution.get("trial_evidence_refs", [])
        bindings = []
        for ref in trial_refs:
            evidence = self.journal.card(ref)
            facts = evidence["payload"]
            if evidence["kind"] != "EvidenceCard" or facts["status"] != "completed" or facts["measurement_source"] == "test_double":
                raise ContractError("冻结候选诊断必须引用本次运行已经完成的实测证据")
            artifacts = facts["artifacts"]
            evaluation = next((a for a in artifacts if Path(a["path"]).name == "classification.json"), None)
            model_assets = [a for a in artifacts if Path(a["path"]).name in ("manifest.json", "weights.pt")]
            if not model_assets:
                model_assets = facts["provenance"]["resolved_execution"].get("model_assets", [])
            if evaluation is None or len(model_assets) != 2 or read_json(evaluation["path"]).get("split") != "search":
                raise ContractError("候选诊断需要冻结模型包及真实 search/pilot 回执，不能读 final 冒充搜索反馈")
            manifest = next(a for a in model_assets if Path(a["path"]).name == "manifest.json")
            trial_id = "ev-" + fingerprint(ref)[:20]
            bindings.append({"trial_id": trial_id, "evidence_ref": ref, "evidence_asset": asset(self.journal.card_path(ref)),
                             "model_assets": model_assets, "model_dir": str(Path(manifest["path"]).parent),
                             "evaluation_asset": evaluation})
            if execution.get("arguments", {}).get("trial_id") == ref:
                execution["arguments"]["trial_id"] = trial_id
        if bindings:
            execution["trial_bindings"] = bindings
        # L3只有预算门禁；不在此引入L2性能阈值。
        self.ledger.check(estimated_seconds=payload["resources"]["estimated_seconds"], kind='experiment')
        return make_card("ExperimentCard", self.journal.run_id, payload, author=requester,
                         refs=self.card_references(payload, [*refs, *trial_refs, *profile_refs]))

    def cached(self, card, *, code_artifact=None, visible=None):
        verify_assets(card["payload"]["execution"])
        if card["payload"]["replication"]["requested"]:
            return None
        key = execution_fingerprint(card, code_artifact=code_artifact)
        for value in self.journal.find_cards("EvidenceCard"):
            if visible is not None and not visible(value):
                continue  # Exact numerical reuse must not disclose a private debate branch.
            payload = value["payload"]
            previous = self.journal.card(payload["experiment_ref"])
            if payload["status"] == "completed" and payload["execution_fingerprint"] == key and execution_fingerprint(previous, code_artifact=code_artifact) == key:
                if payload["measurement_source"] == "test_double":
                    continue
                valid = all(Path(a["path"]).is_file() and file_sha256(a["path"]) == a["sha256"]
                            for a in payload["artifacts"])
                if valid:
                    try:
                        self.validate_evidence_for(card, value, code_artifact=code_artifact)
                    except ContractError:
                        continue
                    return value
        return None

    def validate_evidence_for(self, experiment, evidence, *, code_artifact=None):
        """报告用有效性检查：规格相同的已完成实测，不能用任务终态代替证据。"""
        brief = self.journal.card(self.brief_ref)["payload"]
        check_l3_evaluations(experiment["payload"], brief)
        payload = evidence["payload"]
        if (evidence["kind"] != "EvidenceCard" or evidence["run_id"] != self.journal.run_id
                or payload["status"] != "completed" or payload["measurement_source"] == "test_double"):
            raise ContractError("缺少本次运行已完成的实测证据")
        original = self.journal.card(payload["experiment_ref"])
        key = execution_fingerprint(experiment, code_artifact=code_artifact)
        provenance = payload["provenance"]
        if (payload["execution_fingerprint"] != key
                or execution_fingerprint(original, code_artifact=provenance.get("code_artifact")) != key
                or provenance.get("experiment_sha256") != original["content_sha256"]
                or provenance.get("resolved_execution") != original["payload"]["execution"]):
            raise ContractError("测量证据与冻结实验规格/代码不一致")
        spec = experiment["payload"]["execution"]
        if provenance.get("protocol_hash") != spec.get("protocol_hash"):
            raise ContractError("测量证据与评测协议不一致")
        if payload["measurement_source"] == "synthetic_computation" and spec["backend"].get("kind") != "fixture":
            raise ContractError("合成测量不能替代真实模型实验")
        if not payload["observations"] or not payload["artifacts"]:
            raise ContractError("测量证据缺少观测或原始资产")
        # 只核验已保存证据，不重新执行实验，也不把当前环境变化当作历史证据损坏。
        verify_assets({"artifacts": payload["artifacts"]})
        check_results(experiment["payload"], brief, payload)
        if payload.get("scope") != canonical_scope(experiment, payload):
            raise ContractError("Evidence 范围与冻结输入/核验执行事实不一致")

    def evidence(self, card, receipt, receipt_path, *, code_artifact=None):
        if receipt.get("experiment_ref") != card["id"] or receipt.get("experiment_sha256") != card["content_sha256"]:
            raise ContractError("工作进程回执没有绑定冻结 ExperimentCard")
        if receipt.get("status") != "completed":
            raise ContractError("失败/中断不能封装成完成的测量")
        data = receipt["data"]
        provenance = deepcopy(data["provenance"])
        provenance.update(experiment_sha256=card["content_sha256"],
                          resolved_execution=card["payload"]["execution"], code_artifact=code_artifact)
        provenance["research_target"] = {"scope": deepcopy(card["payload"]["scope"]),
                                         "resolved_choices": deepcopy(card["payload"]["execution"]["resolved_choices"])}
        check_results(card["payload"], self.journal.card(self.brief_ref)["payload"], data)
        artifacts = [asset(receipt_path), *data.get("artifacts", [])]
        for value in artifacts:
            if file_sha256(value["path"]) != value["sha256"]:
                raise ContractError("实验原始资产哈希不匹配")
        return make_card("EvidenceCard", self.journal.run_id, {
            "experiment_ref": card["id"], "execution_fingerprint": execution_fingerprint(card, code_artifact=code_artifact),
            "status": "completed", "observations": data["observations"], "provenance": provenance,
            "resources": receipt["resources"], "artifacts": artifacts,
            "measurement_source": data["measurement_source"], "execution_id": receipt["execution_id"],
            "scope": canonical_scope(card, {**data, "provenance": provenance}),
        }, refs=[card["id"]], author="program")

