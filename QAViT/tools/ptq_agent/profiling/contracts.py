"""Profiling 的张量规格与实际模块语义；不包含 Agent 建议协议。"""

from copy import deepcopy

from ..quantization.config import SPEC_SCHEMA, validate_spec
from ..schemas import JSON, NULL, STR, arr, fingerprint, obj, require, validate

QUANT_SPEC = deepcopy(SPEC_SCHEMA)
QUANT_SPEC["properties"]["format"] = {"enum": ["INT4", "INT8"]}
QUANT_SPEC["properties"]["bits"] = {"type": "integer", "enum": [4, 8]}
QUANT_SPEC["required"].append("bits")
OPTIONAL_QUANT_SPEC = {"anyOf": [NULL, QUANT_SPEC]}


def profile_spec_schema(target, *, with_bits=False):
    """把已存在的粒度/轴规则暴露给 LLM；真实形状整除与位宽关联仍由程序校验。"""
    base = QUANT_SPEC if with_bits else SPEC_SCHEMA
    options = []
    modes = ["per_tensor", "per_channel"] + (["per_group"] if target == "weight" else [])
    for mode in modes:
        spec = deepcopy(base)
        spec["properties"].update({
            "format": {"enum": ["INT4", "INT8"]}, "granularity": {"enum": [mode]},
            "axis": NULL if mode == "per_tensor" else {"type": "integer", "enum": [0, 1] if target == "weight" else [1, 2]},
            "group_size": {"type": "integer", "minimum": 1} if mode == "per_group" else NULL,
        })
        options.append(spec)
    return {"anyOf": [NULL, *options]}





def is_linear(backend, info):
    if hasattr(backend, "model"):
        import torch
        return isinstance(backend.model.get_submodule(info["name"]), torch.nn.Linear)
    return info["type"] == "Linear"


def linear_forward_supported(module):
    """只把已核对 forward 语义的模块作为普通 Linear 做局部数值诊断。"""
    import torch
    return (type(module).forward is torch.nn.Linear.forward or
            (type(module).__name__ == "LinearKMaskedBias" and hasattr(module, "bias_mask")))


def validate_profile_spec(spec: JSON, target: str, shape: list[int]) -> None:
    validate(spec, SPEC_SCHEMA)
    require(spec["format"] in ("INT4", "INT8"), "局部数值探测仅支持现有 INT4/INT8")
    if target == "weight":
        validate_spec(spec, target, shape, "$.weight")
        return
    require(shape and len(shape) in (2, 3), "当前 Linear 输入要求 [B,C] 或 [B,N,C]")
    require(spec["granularity"] in ("per_tensor", "per_channel"), "激活不支持分组拟合")
    # ViT 输入的 feature 轴为 2；不能照搬 toy 的 axis=1（token 轴）。
    require(spec["group_size"] is None, "激活 group_size 必须为 null")
    require(spec["axis"] == (len(shape) - 1 if spec["granularity"] == "per_channel" else None),
            "激活量化必须沿真实 feature 轴，不能按 batch/token 位置拟合")
