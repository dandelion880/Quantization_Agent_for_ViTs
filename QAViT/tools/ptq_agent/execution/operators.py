"""FQ-ViT 独立整数参考算子与显式张量边。

数学来自 MEGVII/FQ-ViT models/ptq/layers.py (Apache-2.0)，固定版本见共享索引。
这里只用浮点张量模拟原整数表达式，无 CUDA 整数 kernel/加速声明。
算子读取已冻结的输入/输出量化尺度，不决定使用 PTF、位宽或其它插件。
"""

from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

from ..schemas import require

FLOW_KINDS = ("attention_residual", "block_output", "embedding_tokens", "position_embedding", "embedding_sum")
SOFTMAX_KINDS = ("attention_logits",)


def integer_layernorm(x, weight, bias, input_scale, output_scale):
    si, so = input_scale.to(x.device).reshape(1, 1, -1), output_scale.to(x.device).reshape(1, 1, -1)
    q = (x / si).round()
    base = si.min()
    q = q * (si / base).round()
    channels = x.shape[-1]
    mean = q.mean(-1) * base
    std = (base / channels) * torch.sqrt(channels * (q ** 2).sum(-1) - q.sum(-1) ** 2)
    # 忠实保留原文无 eps 的整数分支；退化输入明确失败，不静默改变分母。
    require(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "整数 LN 方差退化；原式不可执行")
    a = (base / std).unsqueeze(-1) * weight.reshape(1, 1, -1) / so
    shift = torch.clamp(7 - torch.floor(torch.log2(a.abs())), 0, 31)
    multiplier = torch.clamp(torch.floor(a.abs() * 2 ** shift), 0, 255)
    b = ((bias.reshape(1, 1, -1) - (mean / std).unsqueeze(-1) * weight.reshape(1, 1, -1)) / so * 2 ** shift).round()
    return ((a.sign() * multiplier * q + b) / 2 ** shift).round() * so


def integer_softmax(x, scale, bits):
    """原 int_exp 多项式及 1.5×2^k 的 log_round 阈值，不用普通 log2 舍入替代。"""
    scale = scale.to(x.device)
    q = x / scale
    q = q - q.max(dim=-1, keepdim=True).values
    x0 = torch.floor(-.6931 / scale)
    require(bool((x0 < 0).all()), "整数 softmax scale 超出原式有效范围")
    q = torch.maximum(q, 30 * x0)
    exponent = torch.floor(q / x0)
    remainder = q - x0 * exponent
    a, b, c = .35815147, .96963238, 1.
    b_int, c_int = torch.floor((b / a) / scale), torch.floor((c / a) / scale ** 2)
    # 原多项式在最大 logit 的 remainder=0 处靠常数项保持正值。
    # 很粗的 logit 尺度会使该项为0并产生0/0；不能用eps偷偷改变原式。
    require(bool((c_int > 0).all()),
            "整数 softmax 的 logit scale 过粗，原多项式常数项为0；请诊断尺度或选择其它概率量化器",
            code="unsupported_operator_scale")
    exp_int = torch.clamp(torch.floor((remainder * (remainder + b_int) + c_int) * 2 ** (30 - exponent)), min=0)
    ratio = torch.round(exp_int.sum(dim=-1, keepdim=True) / exp_int)
    power = ratio.log2().floor()
    power = power + ((ratio - 2 ** power) >= 2 ** (power - 1))
    result = 2 ** (-power.clamp(0, 2 ** bits - 1))
    result = result.masked_fill(power >= 2 ** bits, 0)
    require(bool(torch.isfinite(result).all()), "整数 softmax 原式产生非有限值")
    return result


class TensorSite(nn.Module):
    def forward(self, x):
        return self.quantizer(x) if hasattr(self, "quantizer") else x


class SoftmaxSite(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = TensorSite()
        self.integer_enabled = False

    def forward(self, x):
        x = self.logits(x)
        if self.integer_enabled:
            from .methods import get_target_method
            return get_target_method("operator", "integer_softmax").apply(x, self.input_scale, self.bits)
        return x.softmax(-1)


def _block_forward(self, x):
    x = self._ptq_attention_residual(x + self.drop_path(self.attn(self.norm1(x))))
    return self._ptq_block_output(x + self.drop_path(self.mlp(self.norm2(x))))


def _features(self, x):
    batch = x.shape[0]
    x = self.patch_embed(x)
    x = self.pos_drop._ptq_embedding_tokens(torch.cat((self.cls_token.expand(batch, -1, -1), x), 1))
    x = self.pos_drop._ptq_embedding_sum(x + self.pos_drop._ptq_position_embedding(self.pos_embed))
    x = self.pos_drop(x)
    for block in self.blocks:
        x = block(x)
    return self.norm(x)[:, 0]


def legacy_vit(model):
    return (type(model).__module__ == "timm.models.vision_transformer" and type(model).__name__ == "VisionTransformer"
            and all(hasattr(model, k) for k in ("cls_token", "pos_embed", "pos_drop", "blocks", "norm"))
            and not any(hasattr(model, k) for k in ("dist_token", "norm_pre", "global_pool")))


def legacy_block(module):
    return (type(module).__module__ == "timm.models.vision_transformer" and type(module).__name__ == "Block"
            and hasattr(module, "drop_path") and not hasattr(module, "ls1"))


def install_flows(model, layers):
    """只暴露已核对的边；完整 FP 等价门槛仍在任何量化之前执行。"""
    for layer in layers:
        kind = layer["kind"]
        if kind in ("attention_residual", "block_output"):
            block = model.get_submodule(layer["module"])
            require(legacy_block(block), "残差边仅适配已核对的 timm Block；不能猜未知 forward")
            if not hasattr(block, "_ptq_attention_residual"):
                block.add_module("_ptq_attention_residual", TensorSite())
                block.add_module("_ptq_block_output", TensorSite())
                block.forward = MethodType(_block_forward, block)
        elif kind in ("embedding_tokens", "position_embedding", "embedding_sum"):
            require(legacy_vit(model) and layer["module"] == "pos_drop", "embedding 边需要已核对的单 CLS timm ViT")
            if not hasattr(model.pos_drop, "_ptq_embedding_tokens"):
                for key in ("embedding_tokens", "position_embedding", "embedding_sum"):
                    model.pos_drop.add_module("_ptq_" + key, TensorSite())
                model.forward_features = MethodType(_features, model)


def _norm_forward(self, x):
    if self._ptq_integer_enabled:
        from .methods import get_target_method
        return get_target_method("operator", "integer_layernorm").apply(x, self.weight, self.bias, self._ptq_input_scale, self._ptq_output_scale)
    return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def install_operators(model, plan):
    for operator in plan.get("operators", []):
        module = model.get_submodule(operator["module"])
        if operator["method"] == "integer_layernorm":
            module._ptq_integer_enabled = False
            module.forward = MethodType(_norm_forward, module)
        else:
            module._ptq_softmax.integer_enabled = False


def bind_operators(model, plan, qmap):
    for operator in plan.get("operators", []):
        module = model.get_submodule(operator["module"])
        source, destination = qmap[operator["input_site"]], qmap[operator["output_site"]]
        input_scale, output_scale = source.params()["scale"], destination.params()["scale"]
        if operator["method"] == "integer_layernorm":
            module.register_buffer("_ptq_input_scale", input_scale.detach().clone())
            module.register_buffer("_ptq_output_scale", output_scale.detach().clone())
            module._ptq_integer_enabled = True
        else:
            op = module._ptq_softmax
            op.register_buffer("input_scale", input_scale.detach().clone())
            op.bits = destination.layer["activation"]["bits"]
            require(output_scale.numel() == 1 and output_scale.item() == 1, "LIS 输出 log2 网格必须使用原始 scale=1")
            op.integer_enabled = True


def bind_calibration_operators(model, plan, params):
    """部分参考算子在输入 observer 冻结后即可参与后续 observer 测量。

FQ 源码的最后校准 forward 即如此启用 LIS；LN 要等正式推理才切到整数。
本调度只改变算子，普通 W/A QDQ 仍遵循 parallel/sequential 的明确开关。
"""
    if plan["execution"].get("operator_calibration", "float") != "progressive":
        return
    from .methods import get_target_method
    layers = {getattr(l, "ptq_site_id", None): l for l in model.modules() if hasattr(l, "ptq_site_id")}
    for op in plan.get("operators", []):
        if get_target_method("operator", op["method"]).during_calibration and op["input_site"] in params:
            module = model.get_submodule(op["module"])._ptq_softmax
            module.register_buffer("input_scale", params[op["input_site"]]["scale"].detach().clone())
            module.bits = layers[op["output_site"]].layer["activation"]["bits"]
            module.integer_enabled = True


def audit_operators(model, plan, qmap):
    for operator in plan.get("operators", []):
        module = model.get_submodule(operator["module"])
        source, destination = qmap[operator["input_site"]], qmap[operator["output_site"]]
        if operator["method"] == "integer_layernorm":
            require(module._ptq_integer_enabled and torch.equal(module._ptq_input_scale, source.params()["scale"])
                    and torch.equal(module._ptq_output_scale, destination.params()["scale"]), "整数 LN 的实际尺度与方案不符")
        else:
            op = module._ptq_softmax
            require(op.integer_enabled and torch.equal(op.input_scale, source.params()["scale"])
                    and op.bits == destination.layer["activation"]["bits"], "整数 softmax 规格不符")
