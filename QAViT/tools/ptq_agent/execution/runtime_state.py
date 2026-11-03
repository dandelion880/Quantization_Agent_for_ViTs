"""量化运行状态的只读指纹；包含非持久 decoded 权重和目标专属 options。"""

import torch
from ..schemas import fingerprint
from .model import tensor_digest


def record_signature(value):
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_digest(value)}
    if isinstance(value, dict):
        return {k: record_signature(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [record_signature(v) for v in value]
    return value


def target_options(layer, target):
    weight_keys = {"blocksize", "damping", "actorder", "gptaq_clip"}
    return {k: v for k, v in layer.get("options", {}).items()
            if ((k in weight_keys or k.startswith("rounding_")) == (target == "weight"))}


def target_signature(q, target):
    fields = {k: q.layer.get(k) for k in (target, target + "_method", target + "_calibration")}
    fields["options"] = target_options(q.layer, target)
    if target == "weight":
        fields.update({k: getattr(q, k, None) for k in ("reference_weight", "decoded_weight", "weight_record", "weight_enabled")})
    else:
        fields.update(params=q.params(), enabled=q.enabled)
    fields.update({k: getattr(q, k, None) for k in ("input_scale", "input_shift", "bias_uses_decoded_weight")})
    return fingerprint(record_signature(fields))
