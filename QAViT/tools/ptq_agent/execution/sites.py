"""显式激活位点与 attention 适配。只在候选副本上安装，不改上游仓库。"""

from types import MethodType
from contextlib import contextmanager
import torch
from torch import nn

from ..schemas import require
from ..profiling.contracts import linear_forward_supported
from .plugin_contracts import site_id, ATTENTION_KINDS, WEIGHT_KINDS
from .operators import SoftmaxSite, FLOW_KINDS, legacy_vit, legacy_block


class ProbabilityMatmul(nn.Module):
    """post-softmax 量化点，同时暴露 AV 输出供 Hessian-guided 评价。"""
    def forward(self, probability, value):
        if hasattr(self, "quant_a"):
            probability = self.quant_a(probability)
        if hasattr(self, "quant_b"):
            value = self.quant_b(value)
        return probability @ value


def attention_kind(module):
    source, name = type(module).__module__, type(module).__name__
    if source.startswith("dinov3.") and name == "SelfAttention":
        return "dinov3"
    if source.startswith("dinov2.") and name in ("Attention", "MemEffAttention"):
        return "dinov2"
    if source == "timm.models.vision_transformer" and name == "Attention":
        return "timm"
    if source in ("timm.models.swin_transformer", "asset_swin") and name == "WindowAttention":
        return "swin"
    return None


def _dinov3_compute(self, qkv, attn_bias=None, rope=None):
    require(attn_bias is None, "当前 DINOv3 attention 不支持 attn_bias")
    b, n, _ = qkv.shape
    c = self.qkv.in_features
    q, k, v = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)
    p = self._ptq_softmax(self._ptq_qk(q, k.transpose(-2, -1)) * self.scale)
    return self._ptq_probability(p, v).transpose(1, 2).reshape(b, n, c)


def _vit_forward(self, x, attn_bias=None, is_causal=False):
    require(attn_bias is None and not is_causal, "显式视觉 attention 仅接入无 mask 的非因果推理路径")
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
    if hasattr(self, "q_norm"):
        q, k = self.q_norm(q), self.k_norm(k)
    # 老版 timm 在 QK 后缩放；新版 timm 与 DINOv2 在 Q 上缩放。
    logits = (self._ptq_qk(q, k.transpose(-2, -1)) * self.scale if attention_kind(self) == "timm" and not hasattr(self, "q_norm")
              else self._ptq_qk(q * self.scale, k.transpose(-2, -1)))
    p = self._ptq_softmax(logits)
    # DINOv2 的 SDPA 版本把 attn_drop 保存为 float，eval 时不执行 dropout。
    if callable(self.attn_drop):
        p = self.attn_drop(p)
    x = self._ptq_probability(p, v).transpose(1, 2).reshape(b, n, c)
    return self.proj_drop(self.proj(x))


def _swin_forward(self, x, mask=None):
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    attn = self._ptq_qk(q * self.scale, k.transpose(-2, -1))
    if hasattr(self, "_get_rel_pos_bias"):
        bias = self._get_rel_pos_bias()
    else:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
    attn = attn + bias
    if mask is not None:
        nw = mask.shape[0]
        attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, n, n)
    p = self.attn_drop(self._ptq_softmax(attn))
    x = self._ptq_probability(p, v).transpose(1, 2).reshape(b, n, -1)
    return self.proj_drop(self.proj(x))


def install_attention(model, names):
    for name in names:
        m = model.get_submodule(name)
        kind = attention_kind(m)
        require(kind is not None and not m.training, "attention 类型/推理模式尚未适配")
        if hasattr(m, "_ptq_probability"):
            continue
        m.add_module("_ptq_probability", ProbabilityMatmul())
        m.add_module("_ptq_qk", ProbabilityMatmul())
        m.add_module("_ptq_softmax", SoftmaxSite())
        if kind == "dinov3":
            m.compute_attention = MethodType(_dinov3_compute, m)
        else:
            m.forward = MethodType(_swin_forward if kind == "swin" else _vit_forward, m)


def site_catalog(backend, config=None):
    """结构仅提供候选边；预处理还必须以真实校准数据验证 LN→Linear 连通关系。"""
    modules = dict(backend.model.named_modules())
    metadata = {info["name"]: info for info in backend.modules()}
    rows = {}
    # backend.modules() 仅列有直属参数的模块；GELU/attention 容器必须从真实模块树补全。
    for name, m in modules.items():
        if not name:
            continue
        info = metadata.get(name, {"name": name, "type": type(m).__name__, "protected": True})
        if linear_forward_supported(m):
            semantic, edge = "linear_input", name + ":input"
            if name.endswith(".mlp.fc2"):
                act = name.rsplit(".", 1)[0] + ".act"
                if isinstance(modules.get(act), nn.GELU):
                    semantic, edge = "post_gelu", act + ":output"
            for suffix, norm_suffix in ((".attn.qkv", ".norm1"), (".mlp.fc1", ".norm2")):
                if name.endswith(suffix):
                    norm = name[:-len(suffix)] + norm_suffix
                    if isinstance(modules.get(norm), nn.LayerNorm):
                        semantic, edge = "post_layernorm", norm + ":output"
            rows[name] = {**info, "kind": "linear", "semantic": semantic, "edge": edge}
            if (config and config["loop"].get("quantize_classifier", False) and name == "head"
                    and backend.spec.get("classification_head") == "official pretrained ImageNet-1k head retained"):
                # 仅用户显式选择的原生分类器；neck/下游 head 与用户保护名单仍不可放行。
                rows[name]["protected"] = False
        elif type(m) is nn.Conv2d and m.padding_mode == "zeros":
            # 仅开放标准 Conv2d 的已知前向；不凭继承关系替换自定义算子。
            rows[name] = {**info, "kind": "conv2d", "semantic": "conv_input", "edge": name + ":input",
                          "protected": any(p in name.split(".") for p in ("head", "neck", "decode_head"))}
        # 非 Linear 原先被统称 protected；这里仅放行已明确适配的 backbone 激活位点。
        protected_path = any(p in name.split(".") for p in ("head", "neck", "decode_head"))
        if name == "head" and rows.get(name, {}).get("protected") is False:
            protected_path = False
        kinds = []
        if isinstance(m, nn.LayerNorm):
            kinds = ["pre_layernorm", "post_layernorm"]
        elif isinstance(m, nn.GELU):
            kinds = ["post_gelu"]
        elif attention_kind(m):
            kinds = [*ATTENTION_KINDS, "attention_logits"]
        elif legacy_block(m):
            kinds = ["attention_residual", "block_output"]
        elif name == "pos_drop" and legacy_vit(backend.model):
            kinds = ["embedding_tokens", "position_embedding", "embedding_sum"]
        elif linear_forward_supported(m) or type(m) is nn.Conv2d:
            kinds = ["module_output"]
        for kind in kinds:
            edge = name + (":" + kind if kind in (*ATTENTION_KINDS, *FLOW_KINDS, "attention_logits") else ":input" if kind == "pre_layernorm" else ":output")
            rows[name + "::" + kind] = {"name": name, "kind": kind, "semantic": kind,
                "protected": protected_path, "type": type(m).__name__, "weight_shape": [],
                "activation_shape": None, "edge": edge}
            if kind in ("attention_residual", "block_output", "embedding_sum"):
                rows[name + "::" + kind]["semantic"] = "pre_layernorm"
    for row in rows.values():
        # K 在 QK 的第二操作数处已转置为 [B,H,C,N]，不能把 token 轴当通道。
        row["activation_channel_axis"] = (
            -2 if row["kind"] == "attention_key" else
            1 if row["kind"] == "conv2d" or
            (row["kind"] == "module_output" and type(modules[row["name"]]) is nn.Conv2d)
            else -1)
    return rows


def observation_module(model, layer):
    module = model.get_submodule(layer["module"])
    if layer["kind"] in FLOW_KINDS:
        return getattr(module, "_ptq_" + layer["kind"])
    if layer["kind"] == "attention_logits":
        return module._ptq_softmax.logits
    if layer["kind"] in ("attention_query", "attention_key"):
        return module._ptq_qk
    return module._ptq_probability if layer["kind"] in ("post_softmax", "attention_value") else module


@contextmanager
def semantic_edge_checks(backend, model, plan):
    """模块命名仅作候选；特殊分布方法必须验证实际数据边，不能凭 fc2/qkv 名称推断。"""
    catalog, handles, values, calls = site_catalog(backend), [], {}, {}
    try:
        for layer in plan["layers"]:
            if layer["kind"] != "linear" or layer["activation_method"] not in ("twin_gelu", "asymmetric_gelu", "ptf"):
                continue
            name = layer["module"]
            source = catalog[name]["edge"].removesuffix(":output")
            calls[name] = 0
            def save(module, args, output, key=name):
                values[key] = output
            def check(module, args, key=name):
                reference = values.pop(key, None)
                require(reference is not None and torch.equal(args[0], reference),
                        f"特殊激活方法所需的 GELU/LN 直接边未经验证：{key}", code="unsupported_site")
                calls[key] += 1
            handles.append(model.get_submodule(source).register_forward_hook(save))
            handles.append(model.get_submodule(name).register_forward_pre_hook(check))
        for operator in plan.get("operators", []):
            if operator["method"] != "integer_layernorm":
                continue
            key = operator["input_site"] + "->" + operator["module"]
            source = next(m for m in model.modules() if getattr(m, "ptq_site_id", None) == operator["input_site"])
            norm = model.get_submodule(operator["module"])
            require(norm.weight is not None and norm.bias is not None and len(norm.normalized_shape) == 1, "整数 LN 需要单 feature 轴和 affine 参数")
            calls[key] = 0
            def save_input(m, a, y, key=key):
                values[key] = y
            def verify_input(m, args, key=key):
                value = values.pop(key, None)
                require(value is not None and torch.equal(value, args[0]), "整数 LN 的尺度来源不是实际输入边")
                calls[key] += 1
            handles.append(source.register_forward_hook(save_input))
            handles.append(norm.register_forward_pre_hook(verify_input))
            if operator["module"] == "norm" and operator["output_site"] == "head":
                require(legacy_vit(model), "最终 CLS→head 边仅支持已核对单 CLS ViT")
                key = "norm:cls->head"
                calls[key] = 0
                def save_cls(m, a, y, key=key):
                    values[key] = y[:, 0]
                def verify_cls(m, args, key=key):
                    value = values.pop(key, None)
                    require(value is not None and torch.equal(value, args[0]), "最终 head 输入不是 norm 的 CLS 输出")
                    calls[key] += 1
                handles.append(norm.register_forward_hook(save_cls))
                handles.append(model.get_submodule("head").register_forward_pre_hook(verify_cls))
        yield calls
        require(all(n > 0 for n in calls.values()), "特殊激活输入边未实际执行")
    finally:
        for handle in handles:
            handle.remove()
        values.clear()


def observe(model, layer, callback):
    """callback(args, output)；input/概率保留原张量，供等价检查和局部输出误差使用。"""
    m = observation_module(model, layer)
    return m.register_forward_hook(lambda module, args, output: callback(args, output))


def target_tensor(layer, args, output):
    if layer["kind"] in (*FLOW_KINDS, "attention_logits"):
        return args[0]
    if layer["kind"] in ("attention_key", "attention_value"):
        return args[1]
    return args[0] if layer["kind"] in (*WEIGHT_KINDS, "pre_layernorm", "post_softmax", "attention_query") else output


def coverage(model):
    result = {}
    for name, m in model.named_modules():
        if hasattr(m, "ptq_site_id"):
            result[m.ptq_site_id] = m.calls
        elif type(m).__name__ == "QuantizedLinear":
            result[name] = m.calls
    return result
