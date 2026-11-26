# 部署相关的 PTQ：整数非线性、PoT scale 与特定架构

部署能力不只是一种 W/A 位宽。一个方法可能固定 W4A4，却仍以浮点计算 Softmax、LayerNorm 或重定标；也可能整数推理成立，却依靠专用硬件才能得到所报加速。本文只展开已有独立 PTQ 证据的方法，以及一项无需模型训练的算子旁证。依赖 QAT 的路线不录入；非 PTQ 的剪枝/结构训练路线与来源不足的条目仅在覆盖账本说明去向。

<a id="practicaledge"></a>

- [PracticalEdge：受范围约束的整数非线性核](../../docs/methods/recipes/practicaledge.md)

<a id="iptq-vit"></a>

- [IPTQ-ViT](../../docs/methods/recipes/iptq-vit.md)

<a id="p2-vit"></a>

- [P2-ViT](../../docs/methods/recipes/p2-vit.md)

<a id="h-vit"></a>

- [H-ViT](../../docs/methods/recipes/h-vit.md)

<a id="trio-vit"></a>

- [Trio-ViT](../../docs/methods/recipes/trio-vit.md)

<a id="approx-softmax-layernorm-accelerator"></a>

- [无训练算子旁证：Approximation-based Softmax / LayerNorm Accelerator](sources/approx-softmax-layernorm-accelerator.md)

## 共同的落地顺序

QAViT 的 `quantization/integer.py`、`execution/profile_tools.py`、`profiling/torch_profiler.py`、backend 与导出路径是可复用基础。上述方法应让候选同时声明：数值表示、算子实现、scale/shift 状态、运行时路由、实际后端覆盖。先取得算法级结果，再确认 export/reload 后一致性，最后在指定设备上测延迟与资源。这个顺序使研究证据逐步增加，而不把硬件不具备的能力伪装成软件已经实现。
