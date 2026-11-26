## 4. DopQ-ViT：TanQ 与稳健尺度汇总是不同作用点

<a id="dopq_vit"></a>

**证据。** 综述 [65]；本文以 [arXiv 2408.03291 v4](https://arxiv.org/html/2408.03291v4) 为基准，2026-07-28 更新、摘要页记录被 Machine Intelligence Research 接收；读 §3、§4.1–4.2、§5.1 与参数附表。当前未核实作者官方完整代码。旧版本的名称和无优化印象不能覆盖 v4。

### 原配方与版本细节

TanQ 对 post-Softmax 值先做 \(u=\tan(a(x-c))\)，在 u 域均匀编码，再作反变换。a/c 控制两端分辨率与中部压缩，并要求极点在 `[0,1]` 之外。原文通过校准搜索参数，而不是逐输入学习。MOSF 则处理 post-LN 通道尺度合并：使 \(\sum_c |s_c-\tilde s|\) 较小的汇总量是中位数，以此替代易被异常通道影响的均值，再完成 LN 与相邻 Linear 的尺度重参数化。[原文 §4.1–4.2](https://arxiv.org/html/2408.03291v4)

v4 的完整实验包括 I&S-ViT 式块重构：1,024 校准图，分类 batch 64、通常 1,000 次优化，6bit 为 200 次；COCO backbone batch 1、1,000 次。MOSF 本身不需要梯度，不代表整套 DopQ v4 是无优化算法。原文只写 Adam 初始学习率而未在该段给出数值，执行配方应从正式补充或作者实现确认，不能从其他 I&S 配置随意填一个数。

### primitive 与精确实现边界

`rep.tangent_companding` 负责固定 a/c/scale/zp 的编码，`aggregate.robust_channel_scale` 负责 median/mean 等汇总策略；两者可独立插入现有 RepQ 流程。必须先完成 LN/Linear 等价变换，再确定最终 W 网格与重构目标。若放在 GPTQ 之后又修改 W，原重构目标即失效。

这里有一个需要保留的原式歧义：v4 式 (7) 的解码在 arctan 之前仍写了 round。普通的 companding 推导通常不需要这一舍入，但本包不擅自把它删除。Codex 应将原式和去除该舍入的数学适配作为两个显式候选，先对照正式 PDF/作者代码，再确定哪一个可以叫原配方。a/c 的完整网格也要来自附表，不能仅以“grid search”代替参数范围。Tan/atan 的 CORDIC 实验不是 QAViT 现成低位宽内核的证据。

### 研究工作流与当前差距

QAViT 需要 TanQ 表示与 RepQ shared-scale 的可替换汇总器；无需新建与现有双视图指标重复的诊断。拟议 `study.attention_tail_budget` 在固定码字数与搜索预算下，分别测近零概率质量、头部概率质量和输出误差，检验“更多保护近 1 值”在哪些层、哪些任务成立。`study.reparam_weight_tradeoff` 记录 median 汇总后激活误差、W 条件数及后续量化误差，防止只降低 A 误差却损害 W。固定参数的 TanQ 可以满足静态配置；原文中的部分 3bit 结果还把 post-Softmax 设为 2bit，这种混合设置须单独标明，不能宣称已经证明整网均为 2bit。


## 接入依据与来源边界

**基线组件** RepQ 等价变换、均匀编码、候选评分。

**配方阶段与依赖** 补 TanQ 编解码与参数搜索、MOSF 中位尺度汇总；完整 v4 还依赖 I&S 类块重构顺序。

**部署条件** 固定参数可主线适配；原部分结果的 Softmax 更低位宽另列 mixed 配置。

**尚需核对** v4 解码 round 的位置有歧义，完整网格与学习率需锁定；MOSF 无梯度不代表整法无优化。
