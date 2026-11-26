"""Finite calibration views joining actual classifier inputs and annotated dense training windows."""

from copy import copy, deepcopy
from pathlib import Path
import torch
from ..schemas import require, fingerprint
from .model import tensor_digest


class TaskView:
    def __init__(self, backend, data, spec):
        from tools.experiment_worker import data_view
        from tools.segmentation import DenseTask
        from tools.vision_models import segmentation_assets

        source = spec.get("classification")
        self.classification = data_view(data, source, "calibration") if source else None
        self._data = data
        self._rows = []
        self.task = DenseTask(backend)
        _, settings, _, _ = segmentation_assets(backend.config)
        training_root = Path(settings["data_root"]).resolve()
        self.base = backend
        self.backend = copy(backend)
        self.backend.forward_model = self.forward_model
        self.labels = deepcopy(data.labels)
        self.geometry = {}
        image_ids = []
        for row in spec["dense"]:
            require(
                Path(row["path"])
                .resolve()
                .is_relative_to(training_root / "images/training")
                and Path(row["seg_map_path"])
                .resolve()
                .is_relative_to(training_root / "annotations/training"),
                "dense calibration must bind a training image and annotation",
            )
            require(
                len(row["windows"]) == len(set(row["windows"]))
                and min(row["windows"]) >= 0,
                "invalid selected dense window indices",
            )
            found = set()
            for i, (x, meta, geometry) in enumerate(self.task.windows(row)):
                if i not in row["windows"]:
                    continue
                found.add(i)
                sid = "dense/" + fingerprint(
                    {
                        "image": row["file_sha256"],
                        "annotation": row["annotation_sha256"],
                        "window": geometry,
                        "head": self.task.identity,
                    }
                )
                self.geometry[sid] = {
                    **geometry,
                    "source_image": row["file_sha256"],
                    "annotation": row["annotation_sha256"],
                    "window_index": i,
                }
                image_ids.append(sid)
                self._rows.append(
                    (sid, x.detach().cpu(), meta.pop("_qavit_labels").cpu(), meta)
                )
            require(
                found == set(row["windows"]),
                "dense manifest names a nonexistent window",
            )
        mode = spec["mode"]
        require(
            image_ids
            and len(set(image_ids)) == len(image_ids)
            and (
                mode == "joint_calibration_same_backbone"
                and self.classification
                or mode
                in (
                    "single_source_same_backbone_transfer",
                    "separate_task_packages_with_separate_claims",
                )
                and not self.classification
            ),
            "source set conflicts with research task mode",
        )
        self._ids = (
            self.classification.ids("calibration") if self.classification else []
        ) + image_ids
        self.descriptor = {
            "mode": mode,
            "head": self.task.identity,
            "classification": source,
            "dense_windows": self.geometry,
            "sample_ids": self._ids,
            "dense_original_image_count": len(
                {g["source_image"] for g in self.geometry.values()}
            ),
        }
        self.identity = deepcopy(data.identity)
        self.identity.update(task_calibration=fingerprint(self.descriptor))
        self.identity["split_ids"]["calibration"] = self._ids
        self.identity["counts"]["calibration"] = len(self._ids)
        self.shape_observation_splits = ("calibration",)

    def forward_model(self, model, x):
        meta = getattr(x, "_qavit_dense_metadata", None)
        if meta is not None:
            return self.task.observe(model, x, meta, logits=True)["logits"]
        return self.base.forward_model(model, x)

    def ids(self, split):
        return list(self._ids) if split == "calibration" else self._data.ids(split)

    def batches(self, split):
        if split != "calibration":
            yield from self._data.batches(split)
            return
        if self.classification:
            yield from self.classification.batches(split)
        for sid, cpu, labels, metadata in self._rows:
            x = cpu.to(self.base.device)
            x._qavit_dense_metadata = metadata
            x._qavit_dense_labels = labels.to(x.device)
            x._qavit_ignore_index = self.task.segmentor.decode_head.ignore_index
            yield x, sid

    def source_rows(self, split):
        if split != "calibration":
            yield from self._data.source_rows(split)
            return
        if self.classification:
            yield from self.classification.source_rows(split)
        for sid, _, _, _ in self._rows:
            yield {
                "sample_id": sid,
                **self.geometry[sid],
                "dataset_split": "dense_calibration",
            }
