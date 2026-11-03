"""CPU NumPy 整数量化：供局部 Profiling 使用，不是低比特 kernel。"""
from .integer import QuantizedTensor, quantize
__all__ = ["QuantizedTensor", "quantize"]
