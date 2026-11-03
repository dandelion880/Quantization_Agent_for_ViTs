"""只读 ImageFolder 协议：固定清单、逐图 hash 与互斥分区，不改写数据集。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from ..schemas import require
from ..storage import atomic_json
from .server import file_sha256

PREPROCESS = {"resize_shorter": 256, "crop_size": 224, "interpolation": "bicubic",
              "antialias": True, "mean": [.485, .456, .406], "std": [.229, .224, .225]}


class ImageManifest:
    def __init__(self, path, expected_sha256):
        require(expected_sha256 and file_sha256(path) == expected_sha256,
                "图像协议清单 SHA256 不符", code="data_protocol_mismatch")
        payload = json.loads(path.read_text())
        schema = payload.get("schema")
        require(schema in ("imagefolder_manifest_v1", "imagefolder_manifest_v2"), "未知图像协议")
        require(payload["preprocessing"] == PREPROCESS, "尚未核对该预处理；不能静默替换")
        self.PREPROCESS = payload["preprocessing"]
        self.class_to_idx = payload["class_to_idx"]
        require(len(self.class_to_idx) == 1000 and list(self.class_to_idx.values()) == list(range(1000)),
                "分类标签必须为已排序的 ImageNet 1000 类")
        require(list(self.class_to_idx) == sorted(self.class_to_idx), "ImageFolder 类顺序无效")
        self.splits = payload["splits"]
        require(set(self.splits) == {"calibration", "search", "final"}, "缺少数据分区")
        rows = self.records()
        for key in ("sample_id", "path"):
            require(len({r[key] for r in rows}) == len(rows), f"分区存在重复图像 {key}")
        if schema == "imagefolder_manifest_v1":
            require(len({r["file_sha256"] for r in rows}) == len(rows), "分区存在重复图像 file_sha256")
            sources = {"calibration": "train", "search": "val", "final": "val"}
        else:
            sources = payload["split_sources"]
            require(sources == {"calibration": "train", "search": "train", "final": "val"},
                    "v2 必须为 train 校准/搜索、完整 val 终评")
            # 原数据集内可能有重复内容；仍保留全部原图，但禁止跨分区内容泄漏。
            hashes = {s: {r["file_sha256"] for r in items} for s, items in self.splits.items()}
            require(not (hashes["calibration"] & hashes["search"] or hashes["calibration"] & hashes["final"]
                         or hashes["search"] & hashes["final"]), "校准/搜索/终评存在相同图像内容")
        for split, items in self.splits.items():
            require(items, "数据分区为空")
            for r in items:
                p = Path(r["path"])
                require(r["label"] == self.class_to_idx.get(p.parent.name), "标签与类目录不符")
                require(r["dataset_split"] == sources[split], "图像来源违反显式数据协议")
                require(p.parent.parent.name == r["dataset_split"], "原始数据目录与 split 不符")
        self.full_validation = schema == "imagefolder_manifest_v2"
        if self.full_validation:
            root = Path(payload["selection"]["validation_root"])
            expected = {str(p.resolve()) for name in self.class_to_idx for p in image_files(root / name)}
            require(len(expected) == 50000 and {r["path"] for r in self.splits["final"]} == expected,
                    "终评必须恰好覆盖原始 ImageNet validation 50k，不能只声明数量")
            require(Counter(r["label"] for r in self.splits["final"]) == {i: 50 for i in range(1000)},
                    "validation 必须每类50张")
        self._transform = None

    def records(self):
        # Backend.samples 只读开头的 calibration 池；不可把终评交给 profile。
        return [r for split in ("calibration", "search", "final") for r in self.splits[split]]

    def load_image(self, row):
        from PIL import Image
        from torchvision import transforms as T
        require(file_sha256(Path(row["path"])) == row["file_sha256"], "图像内容 hash 变化")
        if self._transform is None:
            self._transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(224), T.ToTensor(), T.Normalize(PREPROCESS["mean"], PREPROCESS["std"])])
        with Image.open(row["path"]) as im:
            x = self._transform(im.convert("RGB")).unsqueeze(0)
        require(hashlib.sha256(x.numpy().tobytes()).hexdigest() == row["input_sha256"],
                "预处理 tensor hash 变化", code="data_protocol_mismatch")
        return x


def image_files(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png"))


def prepare(argv=None):
    """生成固定元数据，不复制图像或缓存张量；复用现有 train32 的确切样本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--search-train-root", type=Path,
                        help="显式启用v2：从train选搜索图，全部50k validation只作终评")
    parser.add_argument("--search-per-class", type=int, default=1)
    parser.add_argument("--final-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "清单已存在，禁止覆盖")
    require(args.search_per_class > 0 and args.final_per_class > 0, "分区大小必须为正")
    source = json.loads(args.calibration_metadata.read_text())
    require(source["task"] == "imagenet" and source["split"] == "training"
            and source["preprocessing"] == "imagenet_resize256_center224_bicubic", "未核对的校准数据协议")
    root = args.validation_root.resolve()
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    mapping = {n: i for i, n in enumerate(classes)}
    require(len(classes) == 1000 and mapping == source["class_to_idx"], "train/val 类映射不一致")
    from PIL import Image
    from torchvision import transforms as T
    transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                           T.CenterCrop(224), T.ToTensor(), T.Normalize(PREPROCESS["mean"], PREPROCESS["std"])])
    def row(path, label, split, expected=None):
        path = Path(path).resolve()
        digest = file_sha256(path)
        require(expected is None or digest == expected, "已有校准图 hash 不符")
        with Image.open(path) as im:
            x = transform(im.convert("RGB")).unsqueeze(0)
        return {"sample_id": f"{split}/{path.parent.name}/{path.name}", "path": str(path),
                "label": label, "dataset_split": split, "file_sha256": digest,
                "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()}
    splits = {"calibration": [row(r["path"], r["label"], "train", r["sha256"]) for r in source["manifest"]],
              "search": [], "final": []}
    rng = random.Random(args.seed)
    total = 0
    for i, name in enumerate(classes):
        files = image_files(root / name)
        require(len(files) == 50, "需要原始 ImageNet validation 每类 50 张图")
        total += len(files)
        chosen = files if args.search_train_root else rng.sample(files, args.search_per_class + args.final_per_class)
        for j, p in enumerate(chosen):
            split = "search" if not args.search_train_root and j < args.search_per_class else "final"
            splits[split].append(row(p, i, "val"))
        if (i + 1) % 100 == 0:
            print(f"manifest: {i + 1}/1000 classes", flush=True)
    payload = {"schema": "imagefolder_manifest_v1", "preprocessing": PREPROCESS, "class_to_idx": mapping,
               "splits": splits, "selection": {"seed": args.seed, "search_per_class": args.search_per_class,
               "final_per_class": args.final_per_class, "source_validation_count": total,
               "calibration_metadata": str(args.calibration_metadata.resolve()),
               "calibration_metadata_sha256": file_sha256(args.calibration_metadata),
               "algorithm": "Python Random.sample per sorted class; search first, final next; mutually disjoint"}}
    if args.search_train_root:
        train = args.search_train_root.resolve()
        require(train.name == "train" and sorted(p.name for p in train.iterdir() if p.is_dir()) == classes,
                "搜索训练目录与ImageNet类映射不一致")
        require(not ({r["file_sha256"] for r in splits["calibration"]}
                     & {r["file_sha256"] for r in splits["final"]}),
                "已有校准池与validation内容重叠，不能宣称独立终评")
        used = {r["file_sha256"] for items in splits.values() for r in items}
        for i, name in enumerate(classes):
            files = image_files(train / name)
            rng.shuffle(files)
            selected = []
            for p in files:
                digest = file_sha256(p)
                if digest in used:
                    continue
                selected.append(row(p, i, "train", digest))
                used.add(digest)
                if len(selected) == args.search_per_class:
                    break
            require(len(selected) == args.search_per_class, "缺少与校准/终评互斥的训练搜索图")
            splits["search"].extend(selected)
        payload.update(schema="imagefolder_manifest_v2",
                       split_sources={"calibration": "train", "search": "train", "final": "val"})
        payload["selection"].update(validation_root=str(root), search_train_root=str(train), final_per_class=50,
            algorithm="all sorted val files; seeded shuffle of sorted train files per class, excluding content in calibration/final/search")
    # 先验证内存中的跨分区身份；没有完成的清单不发布。
    rows = [r for values in splits.values() for r in values]
    if not args.search_train_root:
        require(len({r["file_sha256"] for r in rows}) == len(rows), "存在内容重复的图像")
    atomic_json(args.output, payload)
    ImageManifest(args.output, file_sha256(args.output))
    print(json.dumps({"manifest": str(args.output), "sha256": file_sha256(args.output),
                      "counts": {s: len(v) for s, v in splits.items()}}), flush=True)
