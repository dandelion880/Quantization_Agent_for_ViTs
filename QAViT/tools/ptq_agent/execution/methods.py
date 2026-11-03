"""执行方法扩展点；工具负责调度，plugin 提供实际数值实现。

新增方法时需实现下面四个函数并扩展 PLAN_SCHEMA 与能力声明。不能只登记
一个方法名却继续执行 RTN；导出包记录版本，加载时严格核对。
"""

from dataclasses import dataclass
from typing import Callable
from functools import partial, lru_cache

from ..schemas import require
from . import codecs


@dataclass(frozen=True)
class Method:
    version: str
    encode_weight: Callable
    decode_weight: Callable
    fit_activation: Callable
    quantize_activation: Callable


METHODS = {
    "rtn": Method("reference_rtn_v1", codecs.encode_weight, codecs.decode_weight, codecs.fit, codecs.qdq)
}


def get_method(name):
    require(name in METHODS, f"执行方法未注册：{name}", code="unsupported_algorithm")
    return METHODS[name]


def versions():
    return {name: method.version for name, method in METHODS.items()}


@dataclass(frozen=True)
class TargetMethod:
    """v3 的 W/A/预处理独立接口；新增插件还须扩展契约、冲突表和数值验证。"""
    version: str
    fit: Callable
    apply: Callable | None = None


@dataclass(frozen=True)
class OperatorMethod:
    """消耗已冻结尺度的算子；不拥有 observer，也不强制绑定某个量化器。"""
    version: str
    apply: Callable
    during_calibration: bool = False


@lru_cache(maxsize=1)
def target_methods():
    # 延迟导入：旧 v2 RTN 与新插件共用 codec，但不改变历史执行语义。
    from . import plugin_math as p
    from . import grids
    from . import operators
    from . import adalog, adaround, gptaq
    return {
        "weight": {"rtn": TargetMethod("rtn_scale_v1", p.encode_rtn, p.decode_weight),
                   "gptq": TargetMethod("gptq_static_groups_v1", p.gptq, p.decode_weight),
                   "gptaq": TargetMethod("gptaq_d6bd2e4_v1", gptaq.fit, p.decode_weight),
                   "adaround": TargetMethod("adaround_7b60b525_v1", adaround.fit, adaround.decode),
                   **{name: TargetMethod(name + "_v1", partial(grids.encode, method=name), grids.decode) for name in grids.METHODS}},
        "activation": {"adalog": TargetMethod("adalog_75a57d2_v1", adalog.fit, adalog.apply),
                       **{name: TargetMethod(name + "_v1", partial(p.fit_activation, method=name),
                                           partial(p.activation_qdq, method=name))
                       for name in ("rtn", "uniform_full", "uniform_affine", "uniform_minmax", "twin_softmax", "log2", "log_sqrt2", "twin_gelu", "asymmetric_gelu", "ptf")}},
        "preprocess": {"smoothquant": TargetMethod("smoothquant_edge_v1", p.smoothquant),
                       "repq_ln": TargetMethod("repq_minmax_edge_v1", p.repq_ln),
                       "adalog_bias": TargetMethod("adalog_decoded_bias_v1", adalog.bias_transform)},
        "operator": {"integer_layernorm": OperatorMethod("fq_integer_layernorm_v1", operators.integer_layernorm),
                     "integer_softmax": OperatorMethod("fq_integer_softmax_v1", operators.integer_softmax, True)},
    }


def get_target_method(target, name):
    methods = target_methods().get(target, {})
    require(name in methods, f"未注册的 {target} 方法：{name}", code="unsupported_algorithm")
    return methods[name]


def plugin_versions():
    return {target: {name: method.version for name, method in methods.items()}
            for target, methods in target_methods().items()}
