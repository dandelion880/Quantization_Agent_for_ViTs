## Trio-ViT

<a id="trio-vit"></a>

**证据。** 综述 [46]；[Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer](https://arxiv.org/abs/2405.03882)，§IV、§VI-A/B；[作者仓库](https://github.com/shihuihong214/Trio-ViT)。已读算法及实验，核对 `main_imagenet.py` 使用入口；未在本环境运行。

### 第一层：原配方

目标是 **EfficientViT 的卷积—线性 attention 混合架构**，不是把标准 ViT 的 Softmax 任意删除。先针对 MBConv 的分布做 channel-wise migration 和 filter-wise shifting；后者用校准均值平移输入，并对后接 pointwise convolution 的 bias 作补偿。对轻量 attention 的分母使用 log2 量化，避免均匀量化把小分母变成零而造成 NaN。再按 BRECQ 类框架逐块重构，以对角 Fisher 目标优化量化权重舍入和激活 scale。这里是有限校准 PTQ，不能因使用可学习 scale 就误判成 QAT。

原设置 **1024 张 ImageNet 图**；主要为 W8A8，attention 分母有 4 bit log2 的配置。原文未在已核实设置中提供成熟的全 W4A4 路线。逐块迭代数等应按作者版本配置补齐；不从其他 BRECQ 实现继承默认值。硬件为专门处理卷积与轻量 attention 的 FPGA 方案。

### 第二层：primitive

保留特定拓扑的等价迁移、filter bias 补偿、分母非零保护及局部重构目标。触发条件应是“存在 MBConv/DWConv/PWConv 或归一化分母的线性 attention”，而不是简单 `model contains transformer`。QAViT 已有 AdaRound 和梯度误差基础，需要补的是图拓扑适配及对应分母的编码语义。

### 第三层：workflow

原文将 migration、shifting、分母 log2 分别消融，并解释 NaN 来源。QAViT 首先在符合拓扑的模型上复现数值故障和各组件的修复，再比较任务表现与预算。对 DINOv3 的 Softmax attention，Trio 整套配方不直接成立；只有在另开高效 backbone profile 时才有完整原法入口。算法与专用加速器分开评价，不能借用 FPGA 数字证明普通设备已提速。

**差距与协议。** 需要结构匹配的 adapter、分母稳定性 metric 和模型 artifact；无需复制一套研究工作流。当前作为 architecture-specific PTQ 扩展，优先级取决于用户是否要研究该类 backbone。


## 接入依据与来源边界

**基线组件** AdaRound、局部误差/梯度、图结构观测。

**配方阶段与依赖** 补 MBConv 迁移、filter-bias 修正、线性 attention 分母 log 编码及有限块重构。

**部署条件** EfficientViT 专用结构；原 W8A8 加四位分母，不是普通 DINO Softmax 全 W4A4。

**尚需核对** 完整块迭代配置和专用 backend 尚需锁定；不能把 FP 拓扑差异或 FPGA 数字当通用 PTQ 收益。
