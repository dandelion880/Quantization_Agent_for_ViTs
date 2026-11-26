## PTQ4ViT：twin 表示与 Hessian-guided 搜索是一套配方的不同部分

<a id="ptq4vit"></a>

**来源。** [原论文 arXiv:2111.12293v3](https://arxiv.org/abs/2111.12293v3)，ECCV 2022，读取 §3.1–3.4、§4.1 和附录 6.1–6.3；[官方实现](https://github.com/hahnyuan/PTQ4ViT)，本次读取 main 的 `configs/PTQ4ViT.py` 和 `utils/quant_calib.py`。源码阅读日期 2026-11-25，未将浮动分支冒充固定 commit。注意它与 NeurIPS 2021 的 PTQ-ViT [22] 是两篇不同论文。

### 1. 完整原配方

原方法研究 post-Softmax 分布不均衡、post-GELU 正负范围不对称，以及普通局部误差不足以选择量化参数三个问题。除特殊激活外采用均匀量化；twin quantization 用一个范围标志位与其余 b−1 位的整数，选择两个均匀网格。它不是两个 b 位张量相加。

对 post-Softmax，小值区间使用可搜索的细尺度；大值区间尺度固定为 \(2^{-(b-1)}\)，覆盖概率范围。对 post-GELU，负半轴网格固定以覆盖 GELU 的负值范围，正半轴尺度搜索。原法约束尺度关系以便用移位对齐；复现时保留范围判定、舍入和端点规则，不能只用“两段 piecewise”名称替代编码。

候选以矩阵乘输出误差评分。令 \(e=\hat O-O\)、\(g=\partial L/\partial O\)，其 Hessian-guided 近似目标为

\[
J=\mathbb E[e^\top\operatorname{diag}(g^2)e]
  =\mathbb E\!\left[\sum_i g_i^2 e_i^2\right].
\]

这是对角敏感度近似，不是完整 Hessian。先在校准数据上取得 FP 输入、输出与反向信号，再交替搜索一个矩阵乘的两个输入或权重/激活尺度。论文用 FP 预测提供替代监督信号，没有更新预训练模型的优化器训练流程。

论文默认 ImageNet 训练划分随机 32 张图像，ViT/DeiT/Swin 分类；普通尺度网格 α=0、β=1.2、100 个候选、3 轮交替搜索；post-Softmax 细尺度搜索 \(\{2^{-b},2^{-(b+1)},\ldots,2^{-(b+10)}\}\)。附录明确使用来自 FP 前缀的 parallel calibration。论文报告量化首个投影与最后分类层的权重/输入，以及 attention 两个矩阵乘的两侧输入；不同 attention head 和 Q/K/V 可以使用不同尺度。LN/Softmax 内部保持浮点。

需要保留以下作者代码配置差异：

- 当前 `configs/PTQ4ViT.py` 的 α=0.01，其余上述普通搜索数量和轮数相同。
- 该配置构造 `qconv` 时写入 `a_bit=32`，即关闭其输入激活量化。这个代码配置与论文的首层覆盖叙述分别保存；不能默认为主配置完整 W4A4。
- 代码包括 `bias_correction` 选项；论文表中的 PTQ4ViT 与 PTQ4ViT+BC 分开。仅看到配置选项不能证明具体运行实际执行了补偿。
- Hessian 校准代码用 FP softmax 输出作 teacher，以 KL 反向取得信号。若当前比较的是同一状态、同一输入上的完全相同分布，理论一阶梯度可以为零；数值残差也不等于有可靠的敏感度信息。记录实际参考、量化状态、损失和梯度范数。换成 hard pseudo-label、真实标签或别的 Fisher 估计时另命名，不静默声称与作者信号相同。
- 当前校准脚本在 dataloader 循环中覆盖 `raw_pred_softmax`，后续按子批切片读取，依赖校准数据装在同一个大批次中的用法。扩展为多个 dataloader batch 时要按样本身份保存和匹配 teacher，不能让各批复用最后一批的预测。

### 2. 可组合机制

可分别复用 twin_softmax、twin_gelu、候选网格、交替搜索、对角敏感度评分和 FP-prefix 缓存。现有 QAViT 已有 twin 与 hessian 路径；本轮主要补配方语义、精度例外和 metric 身份。

组合时明确实际算子输出是 Linear、QK 还是 AV，保持梯度与同一输出轴对应。更换候选池不应同时悄悄换 metric；改用量化前缀输入后需要重新定义参考和收集信号。候选批处理仅是数值等价的执行优化，不能改变损失归约方式。

原法的 FP-prefix 选择与一些后续方法的顺序重构不同。两者都可研究，不能把“顺序总是更真实”写成普遍原则，也不能用旧配方的校准成本代表改后的重构算法。

### 3. 研究工作流

论文用尺度与任务损失的关系指出普通局部 metric 的选择失配，再引入更贴近任务敏感度的目标；用分布形状说明为什么要改变表示。QAViT 应吸收这个论证顺序：先确认失败来自表示覆盖还是排序失配，再选择机制。

建议固定相同候选池比较 MSE、cosine、对角敏感度、FIMA 型近似及 AV 输出目标，评估 selector regret；再固定有效 selector 对比 uniform/twin/log 类型。分别报告搜索预算和准备梯度的成本。当前 dense 观测不等于已经实现 dense task gradient，不能直接把分类梯度公式套到未定义的空间归约上。


## 接入依据与来源边界

**基线组件** twin 网格、位点捕获、梯度平方输出评分、尺度候选搜索。

**配方阶段与依赖** 补完整 FP-prefix 校准、双输入交替搜索、特殊网格及 teacher 按样本配对；把原法与 +BC 分开。

**部署条件** 原代码 qconv 输入 A32 是覆盖例外；完整 A4 另建适配。LN/Softmax 内部 FP 与 A4 消费边分列。

**尚需核对** 论文/代码搜索下界不同；同分布 KL 的零梯度和多 batch teacher 对齐要解决。已有 twin 不等于完整原法。
