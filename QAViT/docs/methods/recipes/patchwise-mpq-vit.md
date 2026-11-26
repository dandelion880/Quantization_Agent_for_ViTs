## Patch-wise Mixed-Precision Quantization of Vision Transformer

<a id="patchwise-mpq-vit"></a>

**证据。** 综述 [119]；[arXiv:2305.06559](https://arxiv.org/abs/2305.06559)，§III-B–D、Algorithm 1、§IV-A。已读原文；作者实现未核实。原文明示本研究采用 PTQ，QAT 只是后续可能性，不能因题目混精度而误删。

**第一层：原配方。** 先用 Global Sensitivity Metric 估计各模块对模型输出的影响，结合不同位宽的权重量化扰动与模型大小构建 Pareto frontier，得到层级位宽；推理时 Adaptive Attention Shrink 按跨 head 的 patch attention 重要度及其统计量，进一步为 patch 重分配精度。权重对称、激活非对称，默认 Percentile 校准；**32 张 ImageNet 图**，W/A 候选范围 2–8 bit。模型无任务训练。它同时包含离线层分配和输入相关 patch 精度，不能只记录最终平均 bit。

**第二层：primitive。** 模块敏感度评分、按资源分桶的 Pareto 搜索、patch 重要度聚合、推理位宽路由四者独立。patch 精度改变可能影响矩阵乘法布局与实际 packing；quantization metadata 必须能重放当次路由。

**第三层：workflow。** 原文有 GSM 与 AAS 消融。QAViT 可先保留层分配、关闭 patch 动态，建立可解释基线；再启用 AAS，分别统计动态策略的精度收益和真实调度代价。跨分辨率、增加 register token 或 dense 输出时，应专门验证 patch 重要度规则，而非把分类 CLS 关系直接视为 ADE 空间质量。

**协议与差距。** 完整原法进入 token-dynamic/mixed profile；只保留离线层分配时必须登记为消融。现有动态尺度能力需要补 bit routing、打包及成本记录。


## 接入依据与来源边界

**基线组件** percentile、只读敏感度与任务评价。

**配方阶段与依赖** 补 GSM 层分配/Pareto、attention 驱动 patch 位宽路由及路由打包元数据。

**部署条件** 输入相关 token-dynamic/mixed，不能借固定部署算法名称归 static。

**尚需核对** 原文 PTQ 身份已核；官方代码、实际 packing 和运行路由成本待闭合。
