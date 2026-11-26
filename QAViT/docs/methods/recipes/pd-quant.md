## 17. PD-Quant（CVPR 2023）

<a id="pd-quant"></a>

**身份与来源。** 综述 [36]。[原文](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.pdf) §3、§4；[官方代码](https://github.com/hustvl/PD-Quant) main_imagenet.py、quant/block_recon.py、quant/quant_layer.py。原验证主体为 CNN，是有限无标签校准优化 PTQ。

**paper-recipe。** 对当前块使用已量化前缀输入 \(x_q\)，把块输出送入后续浮点网络 \(F_{>l}\)，以完整浮点模型的预测 \(p_f\) 为 teacher，优化块内舍入变量 \(\theta\) 和激活 scale \(s_a\)：
\[
L=\operatorname{KL}\!\left(p_f\;\|\;
\operatorname{softmax}(F_{>l}(B_l^q(x_q;\theta,s_a))/T)\right)
+\lambda_r\|Y_{\rm ref}-Y_q\|_F^2+L_{\rm round}.
\]
分类概率两端须使用同一 temperature；这里写的是概念分解，官方代码实际采用 MSE + KL/\(\lambda_r\) + rounding，且 KL 不乘 \(T^2\)，原文形式与实现的总梯度尺度不可混同。

DC 是另一必要组成。利用预训练 BN 的 running mean/variance，优化块输入 \(A^{DC}\)，使块内 BN 激活的均值/标准差匹配训练统计，同时不偏离原浮点校准输入：
\[
\min_{A^{DC}}\;\|A^{DC}-A^{FP}\|_F^2+
\lambda_c\sum_i\big(\|\hat\mu_i-\mu_i^{BN}\|_2^2+
\|\hat\sigma_i-\sigma_i^{BN}\|_2^2\big).
\]
经 DC 的浮点输出成为重构参考。1024 张 ImageNet，batch32，每块 20000 次；舍入 Adam lr .003，scale lr \(4\times10^{-5}\) cosine，DC lr .001。当前入口默认 \(\lambda_r=.1,\lambda_c=.02,T=4\)，QDrop prob/input_prob=.5，rounding weight=.01、温度20→2、warmup=.2；首尾 W/A 默认8bit，可由参数关闭。

代码将两份输入拼 batch：一份混合 FP/Q 输入用于局部正则，另一份量化前缀输入用于预测差异；但 block 内 act_quantizer 的随机 drop 对整个 batch 生效。因此不能把当前 official-code PD 路径称为“完全确定的纯量化前向”。导出最终模型时 drop 关闭、舍入硬化。

**primitive。** 可拆成 task_prediction_loss、differentiable_FP_suffix、BN_distribution_correction 和 qdrop_rounding_reconstruction。QAViT 已有 teacher KL；新增的是可微 suffix 优化执行器与 DC，而非再造 KL。BN-DC 依赖真实预训练 BN 统计；DINO 的 LayerNorm 无该状态，不能生搬 DC 或虚造“训练分布”。

**workflow 与接入。** 作者完整 CNN 配方独立保存。ViT 先做 PD loss without DC，比较 FP suffix 与 Q suffix、单任务与双任务，并标为迁移变体；设置同一校准和优化预算才能区分 loss 的贡献。固定部署 W4A4 可行，但原 recipe 的8bit首尾例外须单列。状态：原文、关键实现与默认值已核；未复现。


## 接入依据与来源边界

**基线组件** teacher KL、AdaRound、配对块输入和任务 head。

**配方阶段与依赖** 补可微 FP-suffix 预测目标、BN 分布修正、两份输入及 QDrop 重构；明确局部/全局 loss 路径。

**部署条件** 作者 CNN/BN 原配方与首尾 8bit 单列；无 BN 的 ViT 只用 PD 部分须标迁移。

**尚需核对** DINO LN 不能伪造 BN 训练统计；官方 PD 支路仍有 activation-drop，不应记为全确定量化前向。
