# 精度与格式分配：固定精度组件、混精度和输入相关路线

本文区分三种决策：离线为各层分配不同位宽、推理时为不同 token/window 选择位宽、在相同总位数下改变数值格式。三者都可丰富研究库，但不能仅按“平均 4 bit”与固定 W4A4 混排。新协议允许保留完整原法，并在明确的资源预算与任务目标下比较。本文仅吸收 PTQ 配方；双分支论文中的 QAT 内容不进入方法、组件或 workflow。

<a id="amp-vit"></a>

- [AMP-ViT](../../docs/methods/recipes/amp-vit.md)

<a id="mptq-vit"></a>

- [MPTQ-ViT](../../docs/methods/recipes/mptq-vit.md)

<a id="lrp-qvit"></a>

- [LRP-QViT](../../docs/methods/recipes/lrp-qvit.md)

<a id="mix-qvit"></a>

- [Mix-QViT（仅 PTQ 分支）](../../docs/methods/recipes/mix-qvit.md)

<a id="lampq"></a>

- [LampQ](../../docs/methods/recipes/lampq.md)

<a id="pomq-vit"></a>

- [PoMQ-ViT](../../docs/methods/recipes/pomq-vit.md)

<a id="mixa-q"></a>

- [MixA-Q（仅无需训练的 PTQ 分支）](../../docs/methods/recipes/mixa-q.md)

<a id="patchwise-mpq-vit"></a>

- [Patch-wise Mixed-Precision Quantization of Vision Transformer](../../docs/methods/recipes/patchwise-mpq-vit.md)

<a id="token-dynamic-bit-width"></a>

- [Token-based Dynamic Bit-width Assignment](sources/token-dynamic-bit-width.md)

<a id="optimal-numerical-format"></a>

- [Finding Optimal Numerical Format](sources/optimal-numerical-format.md)

## 这些方法如何进入 QAViT 的研究流程

现有 `tools/ptq_agent/quantization/config.py`、`integer.py`、`registry.py` 的位宽/格式容器、`execution/frozen_trials.py` 的逐层试验以及 `profiling/torch_profiler.py` 可复用。应补统一的候选精度向量、cost model、选择数据角色与运行时路由状态。研究选题可以由三类证据触发：层误差不均→静态 mixed；输入/token 范围波动→动态；同位数下分布难以表示→格式搜索。

比较始终落在同任务目标、同预算口径、同允许精度集合中。平均 bit、参数字节、BOPs、算子 latency、端到端 latency 各有含义，应并列保存，不能互相代替。候选在精度上获益但导出格式不可用时，研究结果仍可保留为算法证据，部署状态则单独记录。
