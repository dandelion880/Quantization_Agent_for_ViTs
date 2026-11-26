## 10. MGRQ（ICIP 2024）

<a id="mgrq"></a>

**来源。** 综述[74]，[原文](https://arxiv.org/html/2406.09229v1) §3.1–3.4、Algorithm1、Table1、§4.1；[会议条目](https://cmsworkshops.com/ICIP2024/view_paper.php?PaperNum=1680)。官方算法代码未确认。

**paper-recipe。** MGRQ 同时融合三个尺度的监督。对第l个块，OBWR 将 FP前缀输出 \(x_l^{fp}\) 同时送给 FP/Q块，\(L_B=\|B_l^{fp}(x_l^{fp})-B_l^q(x_l^{fp})\|^2\)；IBLS 对该块内部n个 Linear/Conv输出取平均平方差，\(L_I=n^{-1}\sum_j\|g_j^{fp}-g_j^q\|^2\)；EBGS 则用同一原图分别经过完整FP/Q网络，比最终logits，\(L_G=\|F(x)-Q(x)\|^2\)。总loss为 \(L_B+\alpha_lL_G+\beta_lL_I\)，只更新当前块参数，按深度顺序遍历。尤其 EBGS 的后缀是Q网络，不能换成PD-Quant的FP后缀而仍使用原名。

作者用RepQ路线处理LN与Softmax等ViT难点，然后混合监督重构；ImageNet1024张、batch32、每块3000次、Adam lr1e-5。原文将α、β按块调整到相近loss尺度，但没有给出可直接复刻的完整数值表或代码。本次不杜撰“统一α=β=1”。Table1明确 patch embedding与head是 W8A8，其余标W4A4/W6A6，这个例外必须进入作者profile。

**primitive。** 对输入/reference/suffix显式建图后，mixed_granularity_loss只是组合器：block-local-on-FP-input、intrablock-feature-matching、full-quantized-suffix-logit-MSE三个独立项。权重需要冻结前声明或按一段固定 warmup 的量级确定，后一种是QAViT实现选择，应记录归一化基数及每块最终值。

**workflow 与接入。** 现有 QAViT 多观察点与多目标权重已经具备表达三项的部分能力；差量为可微分的块重构调度、层内hook、完整Q后缀梯度路径和权重策略。要比较只优化各项和组合，判断改善是否来自全局目标、局部正则或训练开销。原作者只有ImageNet证据；ADE分割logits/head_features的版本是迁移实验。主协议全覆盖W4A4需另跑，不可把作者W8首尾直接算同覆盖。状态：原文已核，α/β确定规则和更新参数明细待官方实现佐证。


## 接入依据与来源边界

**基线组件** 多 tap/多目标观测、RepQ、任务 logits。

**配方阶段与依赖** 补当前块优化器、FP-input 块损失、块内特征损失、完整 Q-suffix 全局损失的精确图。

**部署条件** 原 patch/head W8A8 保留；无例外 W4A4 另为迁移。全局后缀不能换成 FP。

**尚需核对** 逐块 α/β 规则和更新参数明细未闭合；现有多 tap 打分不等于可微块重构。
