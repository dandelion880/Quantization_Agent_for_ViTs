---
name: validate-ptq-deployment
description: >-
  新码本、补偿、重参数化或运行时路由需要导出；整数化导致退化；理论成本下降没有转化为实际速度。
---

## 区分数值表示、算子路径与真实速度

**触发。** 新码本、补偿、重参数化或运行时路由需要导出；整数化导致退化；理论成本下降没有转化为实际速度。

**前提与范围。** 明确目标是数学实现、可重载状态还是指定设备性能。没有目标硬件时可以完成前两者，不虚构 kernel 支持或部署数字。

**步骤。**

1. 从实际执行图列出量化边与状态：码本、scale/zero、分组、补偿、路由、bias、accumulator 及浮点例外。先关注本次新增和依赖部分，不重复全库审计。
2. 用有辨别力的小输入检查边界、零、饱和、总码态、整数范围和变换等价性。H-ViT 式上下界若产生多一个码态，应保留原文疑点与适配记录，不能静默放宽位数。
3. 检查构造后冻结、导出/重载与真实 apply 路径一致；输入 shape/token 布局改变时，明确哪些状态合法复用。位置相关补偿不能未经处理跨分辨率套用。
4. 有真实 backend 时辨认 Q/DQ、pack/unpack、布局、重定标、非线性和 fallback，再测匹配 batch/分辨率下的端到端时间；记录构造成本与运行成本各自范围。
5. 若性能未兑现，定位是表示成本、算子缺失还是调度/搬运；回传具体可简化对象，而非仅增加算法复杂度。

**结果解释与下一步。** 重载后不同先修 artifact；单算子准确但整网下降，查范围与跨算子 scale；BOPs 下降而延迟不降，研究布局/融合；只在某后端有效，保留该条件，不扩为通用加速。

**案例草案。** IPTQ 非线性仿真结果良好，但当前配置的某个 log-family kernel 不支持。可以保留算法研究结果，明确该 route 的 fallback；不能把浮点执行时间解释为 integer-only 速度，也不因部署未完成否认数学近似的已有证据。

**复用边界。** 参考 QDQ 的耗时不是 native 4 bit kernel 的性能；kernel 加速不是整网加速；Approximation Softmax/LN 的 BERT/GPT-2 结果仅为算子旁证；Trio 的 EfficientViT 拓扑不能直接替代标准 ViT。

**来源。** `04_RESEARCH_WORKFLOWS.md` §9–11；[PracticalEdge](../../methods/recipes/practicaledge.md#practicaledge)、[IPTQ](../../methods/recipes/iptq-vit.md#iptq-vit)、[H-ViT](../../methods/recipes/h-vit.md#h-vit)、[Trio](../../methods/recipes/trio-vit.md#trio-vit)；`metrics_registry.md` §10。
