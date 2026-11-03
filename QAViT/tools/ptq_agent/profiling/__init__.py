"""按需测量与建议；导入统计 API 不加载 torch、模型或会话。"""

from .statistics import DEFINITION_VERSION, STATISTICS, tensor_statistics

__all__ = ["DEFINITION_VERSION", "STATISTICS", "tensor_statistics"]
