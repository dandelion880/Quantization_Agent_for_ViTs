## Mix-QViT（仅 PTQ 分支）

<a id="mix-qvit"></a>

**证据。** 综述 [41]；[arXiv:2501.06357](https://arxiv.org/abs/2501.06357)，§II-B–II-F、§III-A、PTQ 结果。已读 PTQ 算法与设置；QAT 分支不录入。没有核实到作者可复现仓库。

**第一层：原配方。** 使用 clipped post-LN 重参数化增强固定精度量化；以 LRP 重要度和逐层量化敏感度共同决定 mixed 分配。敏感度通过“其余层固定于基准位宽，仅改变一层位宽”的 loss 差得到，再平移归一化；用整数二次规划在模型大小和 BitOps 约束下选层级 W/A。PTQ 使用 32 张 ImageNet 训练图校准，COCO 用 1 张；敏感度另用 **256 张 ImageNet validation 图**估计。固定与 mixed PTQ 实验覆盖 3、4、6 bit。原文还有独立全数据训练结果，本包不吸收该分支，也不以其 2 bit 成绩支持 PTQ。

**第二层：primitive。** 层重要度与量化敏感度应是两个字段，不能只保留一个合并分数。增加单层扰动矩阵、资源约束求解器和评分 provenance；选定位宽后完整重校准，避免沿用不相容的旧 scale。IQP 解法及约束应和量化 backend 解耦。

**第三层：workflow。** 原文展示 fixed backend 与分配的增益，以及敏感度随位宽变化。QAViT 可以固定 backend，比较 importance-only、sensitivity-only、两者组合。原文用于敏感度的 validation 子集，在 QAViT 应明确归为 search/development，最终评价另留独立数据；不能复制数据泄漏路径。跨任务转移时分别报告分类评分分配与 dense 评分分配，禁止混用选择目标。

**协议与差距。** mixed 配方进 layer-mixed，clipped reparameterization 可进固定 4 bit 主线。当前缺少带真实资源约束的分配器与选择数据收据；现有逐层 trial 可作为扰动矩阵的执行底座。


## 接入依据与来源边界

**基线组件** RepQ/逐层试验/梯度基础；LRP 实现需先补。

**配方阶段与依赖** 仅接独立 PTQ：单层 bit 扰动损失矩阵、relevance 评分、IQP 大小/BitOps 约束。

**部署条件** 完整 layer-mixed；固定 clipping 可主线，全部比较按实际资源与覆盖。

**尚需核对** 256 张 validation 敏感度集须按 search 用途记录并独立最终评价；仅保留已核 PTQ 分支。
