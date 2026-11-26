## 18. CPT-V（2022 预印本；Evol-Q 的前序版本）

<a id="cpt-v"></a>

**身份与来源。** 综述 [113]，[原文](https://arxiv.org/pdf/2211.09643) §3–4、Algorithm 1。作者为 Frumkin、Gope、Marculescu。它与下一节 Evol-Q 属于同一研究路线，不作为两个独立现代强基线重复计数。单独 CPT-V 版本的可执行官方 commit 本次未确认；后续 [Evol-Q 作者仓库](https://github.com/enyac-group/evol-q) 仍保留 CPT-V 名称。

**paper-recipe。** 从现成 PTQ 模型开始，冻结浮点权重，只对一个 attention block 的量化 scale 向量 \(\Delta_b\) 做小幅黑盒扰动；每个候选跑完整量化模型。某 batch 大小 \(B\)，量化 logits \(q_i\) 与同图浮点 logits \(k_i\) 是正对，其余 \(k_j\) 为负对，以对比目标评估：
\[
L_{\mathrm{NCE}}=-\frac1B\sum_i\log
\frac{\exp(\langle\tilde q_i,\tilde k_i\rangle/\tau)}
{\sum_{j=1}^{B}\exp(\langle\tilde q_i,\tilde k_j\rangle/\tau)}.
\]
\(\tilde q,\tilde k\) 是否归一化按版本记录；后续代码明确 L2 归一化。搜索维护15个候选，抽10个做 tournament，选最优父代后均匀加性扰动，更新种群；每块3个循环、全网10遍、1000张无标签 ImageNet。论文扰动半径为 W8A8 的 \(10^{-3}\)、W4A8/W3A8 的 \(10^{-4}\)。它不反向传播；搜索代价包含每个候选的完整网络推理，不能只计“每块3次”。

**primitive。** 新增 fixed-negative-batch contrastive scorer 与 scale-vector mutation/search。必须缓存 teacher logits、batch组成及顺序；若负样本随候选变化，fitness 不再可直接比较。候选改变 scale 后应明确是否重估 zero-point；原文 scale-only 搜索不能偷偷改成重新校准全部参数。对 scale 取 abs、乘性扰动、log-scale 搜索是不同策略。

**workflow 与接入。** 在相同候选轨迹上比较 KL/MSE/对比 loss 的 selected regret，再比较固定搜索预算的搜索器，以分开“目标更好”和“候选找得更多”。QAViT 现有 frozen 同候选评分可复用，但对比目标必须整 batch 运行，不能先算逐图 loss 后假装负样本不依赖 batch。CPT-V 的 ViT 主实验为 A8；迁入 W4A4 是新 profile，最终 scale 静态不构成动态推理。状态：原论文方法/配方已读；历史版本代码未闭合，优先以 Evol-Q 作为实现入口。


## 接入依据与来源边界

**基线组件** 冻结候选评价、teacher logits、FQ 量化基础。

**配方阶段与依赖** 补固定负样本 batch 的对比 fitness、block scale 向量变异、种群/tournament 状态。

**部署条件** 最终 scale 固定；ViT 原 A8，A4 为新适配，完整模型候选 forward 计预算。

**尚需核对** 历史 CPT-V 官方 commit 未闭合，优先共享 Evol-Q 实现；保留独立版本，不能算两份独立算法证据。
