"""运行输入校验与角色模型配置；模型默认值唯一来自 configs/models.json。"""
import math

from core.cards import ContractError, ROLES, validate_payload
from core.journal import read_json
from runtime.paths import QAVIT


NUMERIC_TASK_TIMEOUT_SECONDS = {"L1": 3600, "L2": 10800, "L3": 86400}


def numeric_task_timeout(runtime, level="L1"):
    """Numerical deadlines are independent of the LLM task timeout and active budget."""
    return (runtime or {}).get("numeric_task_timeout_seconds", {}).get(
        level, NUMERIC_TASK_TIMEOUT_SECONDS[level])


def model_defaults():
    return read_json(QAVIT / "configs/models.json")


def config_check(config, *, resolve_resources=False):
    if config.get("schema_version") != "qavit.research/2":
        raise ContractError("不是 v2 研究配置")
    validate_payload("ResearchBrief", config["research_brief"])
    if config["research_brief"].get("unresolved_inputs"):
        raise ContractError("研究模板仍有待填写输入: " + str(config["research_brief"]["unresolved_inputs"]))
    if config.get("purpose") not in ("research", "validation", "infrastructure_test"):
        raise ContractError("必须明确本次运行用途")
    settings = config.setdefault("llm", {})
    settings.setdefault("provider", "codex_subscription")
    settings.setdefault("agent_models", model_defaults())
    if set(settings["agent_models"]) != set(ROLES):
        raise ContractError("需配置全部七角色模型路由")
    for value in settings["agent_models"].values():
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(v, str) and v for v in value):
            raise ContractError("模型路由须为 [model, effort]")
    runtime = config.setdefault("runtime", {})
    if resolve_resources:
        from runtime.gpu_resources import resolve_runtime_resources
        resolve_runtime_resources(runtime)
    runtime.setdefault("cpu_threads", 4)
    runtime.setdefault("llm_concurrency", 4)
    runtime.setdefault("experiment_concurrency", 1)
    runtime.setdefault("task_timeout_seconds", 600)
    timeouts = runtime.setdefault("numeric_task_timeout_seconds", {})
    if not isinstance(timeouts, dict) or set(timeouts) - set(NUMERIC_TASK_TIMEOUT_SECONDS):
        raise ContractError("数值任务超时须为 L1/L2/L3 秒数映射")
    for level, default in NUMERIC_TASK_TIMEOUT_SECONDS.items():
        timeout = timeouts.setdefault(level, default)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ContractError("数值任务超时必须为有限正数: " + level)
    for key in ("cpu_threads", "experiment_concurrency"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ContractError("运行资源需要正整数: " + key)
    if runtime["task_timeout_seconds"] <= 0:
        raise ContractError("任务超时必须为正数")
    if any(type(gpu) is not int or gpu < 0 for gpu in runtime.get("allowed_gpus", [])):
        raise ContractError("GPU编号必须为非负整数")
    if type(runtime["llm_concurrency"]) is not int or not 1 <= runtime["llm_concurrency"] <= 4:
        raise ContractError("LLM 总并发须为1到4：最多3个Judger及1个其它角色")
    if config.get("enable_l3") is not True and config.get("enable_l3") is not False:
        raise ContractError("请明确本次运行是否启用 L3；验证配置必须关闭")
    if config["purpose"] == "validation" and config["enable_l3"]:
        raise ContractError("本版小规模验证禁止启动完整 L3")
    target = config.get("scope_target")
    if target is not None:
        if (set(target) != {"round", "stage"} or target["stage"] != "generation"
                or type(target["round"]) is not int or target["round"] < 1
                or target["round"] > config.get("budget", {}).get("max_rounds", 10)):
            raise ContractError("scope_target 须明确指定预算轮次内的 generation 终点")
    return config

