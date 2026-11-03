"""服务器适配契约。数据 split 和模块输入捕获位置必须保持明确。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ..schemas import JSON, PTQError


class Backend:
    def missing(self, operation: str) -> None:
        raise PTQError("not_implemented", f"后端未实现 {operation}", status="unsupported")

    def identity(self) -> JSON:
        """返回模型/数据/预处理/实现版本/来源的可序列化身份。"""
        self.missing("identity")

    def modules(self) -> list[JSON]:
        """返回 name/type/block/protected/weight_shape/activation_shape。"""
        self.missing("modules")

    def samples(self, split: str, count: int | None = None, seed: int = 0, offset: int = 0) -> tuple[Any, list[str]]:
        """返回预处理输入和稳定样本 ID；split=calibration/search/final。"""
        self.missing("samples")

    def tensor(self, module: str, target: str, count: int, seed: int, offset: int = 0) -> tuple[Any, list[str]]:
        """返回原始权重或 calibration 上模块算子输入激活，绝不读 final。"""
        self.missing("tensor")

    def prepare(self, expanded: JSON) -> Any:
        """从不可变源模型构建新的候选；只读取 calibration。"""
        self.missing("prepare")

    def evaluate(self, prepared: Any, split: str) -> JSON:
        """返回实测命名指标、协议和样本 ID；不可测指标为 null + reason。"""
        self.missing("evaluate")

    def baseline(self) -> JSON:
        self.missing("baseline")

    def local_probe(self, module: str, weight: JSON | None, activation: JSON | None,
                    count: int, seed: int, offset: int = 0) -> JSON:
        """在相同 calibration 输入下比较局部算子和张量误差。"""
        self.missing("local_probe")

    def reset(self) -> None:
        self.missing("reset")

    def export(self, prepared: Any, directory: Path) -> JSON:
        """写入完整配置和必要产物，返回 manifest 引用。"""
        self.missing("export")

    def load(self, directory: Path) -> Any:
        self.missing("load")

    def hardware_metrics(self) -> JSON:
        self.missing("hardware_metrics")
