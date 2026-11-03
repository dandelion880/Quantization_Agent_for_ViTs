"""论文配置是可展开的数据配方，不是把多个方法绑死的新数值插件。

原始位点/粒度/搜索预算来自固定版本源码。返回普通 v3 计划，仍必须通过
任务白名单、保护、真实分布证据、内存和执行审计；不会自动降低用户要求。
"""

from copy import deepcopy

from ..schemas import require
from .sites import site_catalog, attention_kind
from .operators import legacy_vit

SOURCES = {
    "ptq4vit": "https://github.com/hahnyuan/PTQ4ViT/tree/b42c790c2811e7835d435a9f9a6c64a079a7483a",
    "repq": "https://github.com/zkkli/RepQ-ViT/tree/a160561b9233945e6680132f129d13041b787318",
    "fq_vit": "https://github.com/megvii-research/FQ-ViT/tree/5daf5915c715dc21d6afcd64e661652030e640ab",
}


def spec(bits, signed, granularity="per_tensor", axis=None, group_size=None):
    return {"format": ("INT" if signed else "UINT") + str(bits), "bits": bits, "granularity": granularity,
            "axis": axis, "group_size": group_size, "symmetric": signed, "scale": "minmax"}


def build_recipe(backend, name, bits, evidence):
    require(name in SOURCES and bits in (4, 6, 8) and evidence, "配方名称/位宽/证据非法")
    catalog, layers, preprocess = site_catalog(backend), [], []
    plan = {"schema_version": 3, "unlisted_modules": "keep_fp", "execution": {
        "mode": "reference_qdq", "compute_dtype": "FP32", "activation_calibration": "static", "order": "joint",
        "within_stage": "sequential" if name == "repq" else "parallel"},
        "preprocess": preprocess, "layers": layers, "unresolved": []}
    coordinate = {"strategy": "coordinate", "metric": "hessian", "loss": "teacher_kl", "rounds": 3,
                  "factor_grid": {"start": .01, "stop": 1.2, "count": 100}}
    def add(module, kind, weight, activation, wm, am, wc=None, ac=None):
        row = {"module": module, "kind": kind, "weight": weight, "activation": activation,
               "weight_method": wm, "activation_method": am,
               "reason": f"{name} 源码配方；需结合本任务 profile/评价决定是否采用", "evidence": list(evidence)}
        if wc is not None:
            row["weight_calibration"] = deepcopy(wc)
        if ac is not None:
            row["activation_calibration"] = deepcopy(ac)
        layers.append(row)

    require(any(info["kind"] == "conv2d" for info in catalog.values()), "原配方需要 patch projection Conv2d")
    require("head" in catalog and backend.spec.get("classification_head") == "official pretrained ImageNet-1k head retained",
            "原分类配方需要官方分类 head；不能冒充 backbone-only 方案")
    # 配方只处理已核对结构；未知分支需显式扩展配方数据，不能静默遗漏。
    for module_name, module in backend.model.named_modules():
        if attention_kind(module):
            require(attention_kind(module) == "timm", "当前完整分类配方只核对 timm ViT/DeiT；其它模型仍可独立选插件")
    if name == "fq_vit":
        require(legacy_vit(backend.model), "FQ 全图配方需要已核对的单 CLS timm ViT；其它结构不可冒用")
        operators = plan["operators"] = []
        plan["execution"]["operator_calibration"] = "progressive"
        plan["execution"]["observer_batch_size"] = 100  # 原仓库 CLI 默认；样本池和数量仍由任务配置决定。
        for sid, info in catalog.items():
            module, kind = info["name"], info["kind"]
            if kind not in ("linear", "conv2d"):
                continue
            use_input = module == "head" or module.endswith((".attn.proj", ".mlp.fc2")) or (kind == "conv2d" and backend.spec["family"].startswith("DeiT"))
            add(module, kind, spec(bits, True, "per_channel", 0), spec(bits, False) if use_input else None,
                "uniform_minmax", "uniform_minmax" if use_input else None)
            if module == "head" or module.endswith((".attn.qkv", ".attn.proj", ".mlp.fc2")):
                add(module, "module_output", None, spec(bits, False), None, "uniform_minmax")
        for kind in ("embedding_tokens", "position_embedding", "embedding_sum"):
            add("pos_drop", kind, None, spec(bits, False, "per_channel", 2) if kind == "embedding_sum" else spec(bits, False),
                None, "ptf" if kind == "embedding_sum" else "uniform_minmax")
        previous = "pos_drop::embedding_sum"
        for index, block in enumerate(backend.model.blocks):
            prefix = f"blocks.{index}"
            for norm, source in (("norm1", previous), ("norm2", prefix + "::attention_residual")):
                path = prefix + "." + norm
                add(path, "post_layernorm", None, spec(bits, False), None, "uniform_minmax")
                operators.append({"method": "integer_layernorm", "module": path, "input_site": source, "output_site": path + "::post_layernorm"})
            for kind in ("attention_residual", "block_output"):
                add(prefix, kind, None, spec(bits, False, "per_channel", 2), None, "ptf")
            attn = prefix + ".attn"
            add(attn, "attention_logits", None, spec(bits, False), None, "uniform_minmax")
            add(attn, "post_softmax", None, spec(4, False), None, "log2")
            operators.append({"method": "integer_softmax", "module": attn,
                              "input_site": attn + "::attention_logits", "output_site": attn + "::post_softmax"})
            previous = prefix + "::block_output"
        operators.append({"method": "integer_layernorm", "module": "norm", "input_site": previous, "output_site": "head"})
        return plan
    for sid, info in catalog.items():
        module, kind = info["name"], info["kind"]
        if kind in ("linear", "conv2d"):
            if any(p in module.split(".") for p in ("neck", "decode_head")):
                continue
            if name == "ptq4vit":
                w = spec(bits, True, "per_channel", 0) if kind == "conv2d" else spec(bits, True)
                if module.endswith(".attn.qkv"):
                    w = spec(bits, True, "per_block", 0, info["weight_shape"][0] // 3)
                a = None if kind == "conv2d" else spec(bits, True)
                am = None if a is None else "twin_gelu" if info["semantic"] == "post_gelu" else "uniform_full"
                add(module, kind, w, a, "uniform_full", am, coordinate, coordinate if a else None)
            else:
                add(module, kind, spec(bits, False, "per_channel", 0), spec(8 if kind == "conv2d" else bits, False),
                    "uniform_affine", "uniform_affine", {"strategy": "percentile"}, {"strategy": "percentile"})
                if info["semantic"] == "post_layernorm":
                    preprocess.append({"method": "repq_ln", "module": module, "norm": info["edge"].removesuffix(":output"),
                                       "observation": "quantized_prefix", "calibration": {"strategy": "percentile"}, "evidence": list(evidence)})
        elif kind in ("attention_query", "attention_key", "post_softmax", "attention_value"):
            softmax = kind == "post_softmax"
            if name == "ptq4vit":
                c = deepcopy(coordinate)
                if softmax:
                    c.pop("factor_grid")
                    c["peer_precision"] = "float"
                add(module, kind, None, spec(bits, False) if softmax else spec(bits, True, "per_channel", 1),
                    None, "twin_softmax" if softmax else "uniform_full", ac=c)
            else:
                add(module, kind, None, spec(bits, False), None, "log_sqrt2" if softmax else "uniform_affine", ac={"strategy": "percentile"})
    return plan
