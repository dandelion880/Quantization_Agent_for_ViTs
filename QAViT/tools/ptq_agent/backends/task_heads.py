"""显式绑定已有冻结分类头；不训练、不改变backbone命名或token语义。"""

from pathlib import Path
import torch
from torch import nn

from ..schemas import fingerprint, require


VERIFIED_HEAD = "verified frozen normalized-CLS ImageNet-1k linear head"


def attach_head(model, spec, options, checkpoint_hash):
    from .server import file_sha256, state_sha256

    path = Path(options["path"])
    require(file_sha256(path) == options["sha256"], "分类头SHA256不匹配", code="asset_mismatch")
    require(options["backbone_sha256"] == checkpoint_hash, "分类头绑定的backbone不匹配", code="asset_mismatch")
    require(spec["family"].startswith(("DINOv2", "DINOv3")), "normalized CLS head需要已验证DINO特征接口")
    previous = getattr(model, "head", None)
    require(previous is None or isinstance(previous, nn.Identity), "不能覆盖模型已有分类头")
    payload = torch.load(path, map_location="cpu")
    require(payload.get("input_dim") == spec["hidden_dim"] and payload.get("num_classes") == 1000,
            "分类头维度与backbone不符", code="asset_mismatch")
    mapping = payload.get("class_to_idx")
    require(isinstance(mapping, dict) and len(mapping) == 1000
            and sorted(mapping.values()) == list(range(1000)), "分类头缺少有效标签映射")
    state = payload["classifier_state_dict"]
    require(set(state) == {"classifier.weight", "classifier.bias"}, "不支持的分类头结构")
    # 复用原helper的Linear(features.float())，仅去除外层classifier名称前缀。
    head = nn.Linear(payload["input_dim"], 1000)
    head.load_state_dict({k.removeprefix("classifier."): v for k, v in state.items()}, strict=True)
    head = head.to(next(model.parameters()).device).eval().requires_grad_(False)
    require(all(torch.equal(head.state_dict()[k].cpu(), state["classifier." + k])
                for k in head.state_dict()), "分类头strict-load后参数不一致")
    model.add_module("head", head)
    return {"kind": VERIFIED_HEAD, "path": str(path.resolve()), "sha256": options["sha256"],
            "backbone_sha256": checkpoint_hash, "feature": "x_norm_clstoken", "head_state_sha256": state_sha256(head),
            "class_mapping_sha256": fingerprint(mapping), "protected_module": "head",
            "training": False, "provenance": {k: payload.get(k) for k in
                ("dino_checkpoint", "train_split", "training_config", "seed", "best_val_top1", "best_val_top5")}}, mapping


def forward_head(model, inputs):
    features = model.forward_features(inputs)
    require(isinstance(features, dict) and "x_norm_clstoken" in features,
            "分类必须使用最终normalized CLS，不能用未归一化CLS或patch均值替代")
    cls = features["x_norm_clstoken"]
    require(cls.ndim == 2 and cls.shape[1] == model.head.in_features, "CLS特征维度错误")
    return model.head(cls.float())
