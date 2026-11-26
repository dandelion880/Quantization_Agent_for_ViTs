## 20. Bias Compensation / BiasComp（2024 预印本；2025 期刊卷期）

<a id="biascomp"></a>

**身份与来源。** *Minimize Quantization Output Error with Bias Compensation*，综述 [110]。[原文](https://arxiv.org/pdf/2404.01892) §3、§5.1.3；[正式 DOI](https://doi.org/10.26599/AIR.2024.9150036)；[官方代码](https://github.com/GongCheng1919/bias-compensation)。这是闭式 PTQ 输出修正，不更新全数据训练权重。

**paper-recipe。** 收集量化模块与浮点参考输出误差 \(E=Y_f-Y_q\)，拟合常数偏移。对每个保留坐标 \(j\)，\(N\) 个样本的最小二乘解为
\[
b_j^*=\frac1N\sum_{n=1}^N E_{nj},\qquad Y_q'=Y_q+b^*.
\]
“常数”是相对被平均的样本轴，不等于全张量一个 scalar。作者 ViT 实验对 FC 与 BMM 加补偿，32张校准图，基于 PTQ4ViT；输出 \(B\times T\times D\) 时**只平均 batch，偏置保留 \(T\times D\)**，BMM 也可保留对应 head/query/key 坐标。不能把该 recipe 简化成普通每通道 bias 却沿用作者结果。

零补偿在可行域内，所以对同一拟合样本、同一配对参考的输出 SSE 不会变差；这不推出 heldout、分类或分割必然改善。是否逐模块更新后重采下游输入、保留哪些 BMM 轴，以官方配置为准，不能从公式自行补成未经核对的全网流程。

**primitive。** 输入为配对 output、reduce_axes 和 preserve_axes，流式维护误差 sum/count，输出固定 correction tensor 与 shape contract。可形成 per-channel、per-token-channel、per-head-query-key 三种独立变体；按通道平均是更便宜的 QAViT 对照，不是原始 ViT 配方的同义实现。上游量化状态或当前量化器变化后重新估计。

**workflow 与接入。** 先比较训练集/heldout 的误差均值与方差，再与 QwT-v2 的 affine、QwT 的 full matrix 对比误差收益、任务增益和真实字节成本。DINO 的 register/CLS/patch、不同窗口尺寸及位置编码需与补偿轴匹配；保留位置偏置无法自然泛化到任意 token 数，需另定义插值/共享策略。可融合进原 Linear 的 per-channel bias 通常只需更新常规高精度 bias；位置相关或 BMM 输出补偿需额外常驻张量与加法，归入声明补偿的 profile。状态：原文算法、ViT shape 与作者实现入口已核；未复现。


## 接入依据与来源边界

**基线组件** 配对输出、流式误差与 bias/常量状态基础。

**配方阶段与依赖** 补显式 reduce/preserve axes 的均值残差拟合及 shape-bound 修正；原 ViT 保留 token 坐标。

**部署条件** 通道 bias 可融合主线；位置/BMM 常量需声明额外状态、形状与算子，按终图判断配置。

**尚需核对** 不能把 batch-only 平均换成全 token 平均冒充原法；逐模块顺序与 BMM 轴按有效配方核定。
