## 1. PTQ-ViT / Ranking-ViT（NeurIPS 2021）

<a id="ptq-vit-2021"></a>

**身份与来源。** Liu 等，*Post-Training Quantization for Vision Transformer*，综述 [22]，不同于 PTQ4ViT（ECCV 2022）的 twin quantization。核查 [NeurIPS 原文](https://proceedings.neurips.cc/paper_files/paper/2021/file/ec8956637a99787bd197eacd77acce5e-Paper.pdf) §3.2–3.3、§4.1。本次没有确认作者专属、可运行的公开仓库。

**paper-recipe。** 在各线性层 W/A 的 scale 上交替搜索，以输出 Pearson 相关系数为相似度；注意力额外保留 pre-Softmax score 的次序。设某个 head 的 score 为 \(A\in\mathbb R^{T\times T}\)，候选结果为 \(\hat A\)，作者排序项为
\[
L_{\rm rank}=\sum_{k=1}^T\sum_{p<q}
[\theta-(\hat A_{kp}-\hat A_{kq})\operatorname{sign}(A_{kp}-A_{kq})]_+ .
\]
整体目标为 \(\gamma L_{\rm rank}-\operatorname{Pearson}(O,\hat O)\)。每轮再由校准输出的均值误差修正 bias。原文取 scale 搜索倍数 [0.5,1.2]、最多 20 轮，ImageNet/COCO 各 1000 张，CIFAR 各 100 张；本文未核实 \(\gamma,\theta\) 与候选数 C 的最终配置，接入时定位原文补充材料或作者实现。

完整 recipe 还包括混合精度：对 MSA attention map、MLP output 求核范数 \(\|M\|_*=\sum_i\sigma_i(M)\)，以其乘输出平方误差估计配置代价，在目标模型大小下选 Pareto 配置。目标 6bit 的候选位宽为 4–8，目标 8bit 为 6–10；MSA/MLP 单元内部共用位宽。核范数是作者的敏感度代理，不是计算出的 Hessian。Softmax/LN 运算保持浮点。

**primitive。** 可独立登记 attention_pairwise_hinge、centered_correlation、output_mean_bias 与 nuclear_norm_precision_proxy。排序项的输入是同一 head、同一 query 的 key scores；若改成概率、抽样 key 对或 top-k 次序须另命名，记录采样与 tie 处理。核范数随 token 数、特征尺度而变，跨分辨率归一化属于新设计。

**workflow 与接入。** 先固定候选池，比较 Pearson、现有 cosine、排序项、AV 输出误差对最终任务的排序与 selected regret，再决定是否用于搜索。固定 W4A4 可迁移相似度/排序/偏置部分；混合精度完整原配方属扩展 profile，不能把原文 6MP/8MP 结果计作 W4A4 证据。作者有 ImageNet 与 DETR 检测实验；DINOv3/ADE20K 需要独立检验。状态：原文方法、配方和精度表已读；实现未复现。


## 接入依据与来源边界

**基线组件** 位点输出捕获、尺度候选、梯度/任务评价基础。

**配方阶段与依赖** 补 Pearson、pre-Softmax pairwise ranking、均值 bias、交替 scale 搜索；完整法另含核范数混精分配。

**部署条件** 完整 mixed 6/8 bit 原法归混精；排序/相关性等可用于固定四位适配。

**尚需核对** γ、θ、候选数等最终设置及官方代码未核；核范数不是 Hessian，不能与 PTQ4ViT 混名。
