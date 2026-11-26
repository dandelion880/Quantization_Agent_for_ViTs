# 重构、敏感度、搜索与补偿方法：原配方、可组合机制和 QAViT 接入说明

核验日期：2026-11-25。本文提供文字规格，不代表已经接入、复现或在 GPU 上验证。仅收录原文明确属于 PTQ 的方法；有限校准集上的反向传播、舍入优化、权重修正和架构局部重构不自动等同于全数据训练。本文不录入 QAT 方法。

## 阅读方式与公共约定

<a id="reading-conventions"></a>

每个方法分成三层：**paper-recipe** 保留作者实际方法、数据和精度设定；**primitive** 定义可拆出的数学操作和依赖；**workflow** 是 QAViT 拟进行的实验，不能作为作者已经证明的结论。论文与当前官方代码不一致时分别登记，两者都不覆盖另一者。

除特别说明，线性层采用列样本记法：\(X\in\mathbb R^{d_{\rm in}\times M}\)，\(W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}\)，\(Y=WX+b\mathbf1^\top\)。\(M\) 可以合并图像、token 或窗口轴，但所有实验必须保留原图 ID，统计预算不得把这些 token 算成独立图像。块输出原始形状通常为 \(B\times T\times D\)；写成向量 \(z\in\mathbb R^a\) 时，\(a=T D\)，不默认把 CLS、register 和 patch 混为一类。

参考必须明确区分：**fp_model** 是完整浮点前缀产生的参考；当前 frozen_trials.py 的 **local_fp_parent** 是对固定 parent 输入运行浮点块，candidate 输出仍来自 candidate 全网；若候选同时改变上游，二者不再是严格同输入的局部误差。**local_fp_candidate** 则对 candidate 的当前输入分别运行 FP/Q 块，是 QwT 拟合与局部分解需要的语义。单站点且上游不变时二者一致。现有 frozen_trials.py 和 propagation.py 已有双任务视图、多层 head_features、NMSE/cosine/KL、传播误差及交叉项，新增工作应复用这些实现。现有 frozen 候选只接受 scale 编辑，重构与补偿需要扩为携带完整 checkpoint/adapter 的候选。

主协议允许校准优化，最终推理固定 W4A4。优化过程中临时浮点权重、QDrop 混合输入、渐降 bit 不构成动态推理；推理保留高精度权重矩阵、输入驱动分组或不同位宽则须登记扩展 profile。LayerNorm、Softmax 的运算精度与其输出量化位宽是两个字段，不能用“全量化”一词替代。

<a id="ptq-vit-2021"></a>

- [1. PTQ-ViT / Ranking-ViT（NeurIPS 2021）](../../docs/methods/recipes/ptq-vit-2021.md)

<a id="apq-vit"></a>

- [2. APQ-ViT（ACM MM 2022）](../../docs/methods/recipes/apq-vit.md)

<a id="aphq-vit"></a>

- [3. APHQ-ViT（CVPR 2025）](../../docs/methods/recipes/aphq-vit.md)

<a id="fima-q"></a>

- [4. FIMA-Q（CVPR 2025）](../../docs/methods/recipes/fima-q.md)

<a id="erq-icml2024"></a>

- [5. ERQ（ICML 2024）](../../docs/methods/recipes/erq-icml2024.md)

<a id="erq-tpami2025-duq"></a>

- [6. ERQ 扩展版与 DUQ（TPAMI 2025）](../../docs/methods/recipes/erq-tpami2025-duq.md)

<a id="uniq-vit"></a>

- [7. UniQ-ViT（Neurocomputing 2026；2025 在线）](sources/uniq-vit.md)

<a id="is-vit"></a>

- [8. I&S-ViT（2023 预印本；TPAMI 2025 在线/2026 卷期）](../../docs/methods/recipes/is-vit.md)

<a id="pfcr-vit"></a>

- [9. PFCR-ViT（Neural Networks 2025）](../../docs/methods/recipes/pfcr-vit.md)

<a id="mgrq"></a>

- [10. MGRQ（ICIP 2024）](../../docs/methods/recipes/mgrq.md)

<a id="qdrop"></a>

- [11. QDrop（ICLR 2022）](../../docs/methods/recipes/qdrop.md)

<a id="bit-shrinking"></a>

- [12. Bit-Shrinking（CVPR 2023）](../../docs/methods/recipes/bit-shrinking.md)

<a id="outlier-aware-slicing"></a>

- [13. Outlier-aware Slicing / OAS-ViT（ICML 2024）](../../docs/methods/recipes/outlier-aware-slicing.md)

<a id="mrospu"></a>

- [14. MROSPU（PRCV 2025 会议；2026 正式出版）](sources/mrospu.md)

<a id="q-hyvit"></a>

- [15. Q-HyViT（IEEE IoT Journal 2024）](../../docs/methods/recipes/q-hyvit.md)

<a id="qep"></a>

- [16. QEP（NeurIPS 2025）](../../docs/methods/recipes/qep.md)

<a id="pd-quant"></a>

- [17. PD-Quant（CVPR 2023）](../../docs/methods/recipes/pd-quant.md)

<a id="cpt-v"></a>

- [18. CPT-V（2022 预印本；Evol-Q 的前序版本）](../../docs/methods/recipes/cpt-v.md)

<a id="evol-q"></a>

- [19. Evol-Q / Jumping Through Local Minima（ICCV 2023）](../../docs/methods/recipes/evol-q.md)

<a id="biascomp"></a>

- [20. Bias Compensation / BiasComp（2024 预印本；2025 期刊卷期）](../../docs/methods/recipes/biascomp.md)

<a id="qwt"></a>

- [21. QwT（CVPR 2025）](../../docs/methods/recipes/qwt.md)

<a id="qwt-v2"></a>

- [22. QwT-v2（2025 预印本）](../../docs/methods/recipes/qwt-v2.md)

## 23. 与现有 GPTQ / GPTAQ / AdaRound 的联动

<a id="existing-method-linkage"></a>

这三项已经有 QAViT 执行实现，不计为新增方法。原始出处和上游 revision 复用 docs/references/ptq_methods.json，避免另建来源冲突。

- **GPTQ。** [原文](https://arxiv.org/abs/2210.17323)、[作者仓库](https://github.com/IST-DASLab/gptq)。当前 execution/plugin_math.py 与 calibration.py 已有输入二阶矩、阻尼、逐列量化及余列补偿；QEP 需在其前加入受控连续权重修正，ERQ 则是另一套分段量化/舍入翻转/剩余权重修正。二者不能简单叠加所有补偿项。比较应共享量化网格、grouping和输入来源，并保留真实计算量。
- **GPTAQ v1。** [2504.02692v1](https://arxiv.org/abs/2504.02692v1)，上游固定 commit d6bd2e41e9abc8f22a76d6ea789c450b6c143ced。当前 execution/gptaq.py 已累计 \(H=2X_qX_q^\top/N\)、\(C=2(X_f-X_q)X_q^\top/N\)，用 Cholesky 逆因子构造上三角输入补偿，与列舍入一同更新。它不是“打开量化前缀的GPTQ”。QEP 的全矩阵预修正与此三角逐步补偿应做明确公式和网格对照；不能宣称输入传播补偿是QAViT全新能力。后续论文/仓库改名版本不能覆盖现有v1。
- **AdaRound。** [原文](https://proceedings.mlr.press/v119/nagel20a.html)。现有舍入变量、硬化与正则可作为QDrop、APHQ/FIMA重构底座；新增执行器应显式控制优化参数、激活尺度、块范围和随机量化路径。若只用既有单层AdaRound，不能宣称已实现完整QDrop或FIMA。

PTQ重构并不意味着以上权重算法都能连续叠加。一个最终权重张量的离散解应归属明确执行链；若后一步重构覆盖前一步，应记录前者是初始化还是最终独立贡献。

## 24. 需要保留的原文/代码校正记录

<a id="corrections"></a>

1. PTQ-ViT（2021，[22]）与PTQ4ViT（2022，[16]）不是同一方法；前者包括attention ranking和核范数混精代理，后者的twin quantization由其他增补卡负责。
2. APQ的MPQ是偏向保留最大值的uniform Softmax量化；不是log quantizer。其bottom-elimination Eq.(8)与文字存在方向歧义，不能隐去。
3. APHQ“全元素同时加减扰动”一般测得 \(H\mathbf1\)，不能无假设叫exact diagonal；源码取abs、p=.98，与论文表述须分开。
4. FIMA论文的secant/DPLR与当前源码的abs、两项归一化损失不相同；论文rank15与当前4bit rank5也分别保存。
5. ERQ2024 T=100，2025扩展版T=20；DUQ是后者增量。DUQ按d_out决定选中输入列数，不能静默改成d_in。
6. Outlier-aware Slicing主要划分重构子图，不是把异常通道分出高精度旁路；I&S的校准后quantizer参数固定，不等于联合学习所有scale。
7. MGRQ的全局loss走量化后缀；PD-Quant的概念路径使用FP后缀，当前PD源码内部仍含随机activation-drop。两者需精确记录路径。
8. CPT-V/Evol-Q的ViT主要实证为A8；Evol-Q当前代码的扰动幅度、scale覆盖、初始化种群及validation选模与论文描述有差异。
9. Q-HyViT的bridge reconstruction核心是离散候选搜索，不能从reconstruction一词推断为全权重梯度优化。
10. BiasComp作者ViT补偿保留token坐标；QwT实际拟合local_fp而非直接完整FP前缀target；QwT-v2作用于每个Linear/Conv，并未提供可直接继承的实测“零开销”端到端部署结果。
11. QEP正式2025版本仍以LLM weight-only验证，且修正damping默认1.0倍H对角均值，不能沿用GPTQ .01而标原配方。
12. current frozen_trials 的local_fp参考绑定parent输入。跨多层候选需显式区分parent/current-candidate输入，不能据旧名称假设总是同输入局部误差。

## 25. 本文件的完成范围与交接

正文有22个方法/版本条目，对应21个综述主参考项；ERQ2025是[37]的附属版本，CPT-V和Evol-Q按综述分别保留但标注家族关联。另有3个已实现方法的联动说明，不当作新增。

其中20个方法/版本实际读到原论文方法和实验设置；UniQ与MROSPU只到作者/出版社可访问的摘要和来源层，完整recipe待获取。APHQ、FIMA、ERQ2025、I&S、QDrop、QEP、PD-Quant、Evol-Q、QwT等进一步核对关键官方源码；BiasComp确认作者实现入口与原文shape定义。已读全文也不代表每项超参都闭合：APQ mask、MGRQ损失权重、PFCR优化变量细节、Bit-Shrinking总数据预算、部分方法部署例外仍逐条注明。

所有新增方法在本次交付中的状态均为**文字材料及明确的来源状态，QAViT接入与数值复现未执行**；来源未闭合条目仍保持待补齐状态。后续Codex应先保留作者recipe与适配variant两个配置，再实现共享primitive和workflow；不能因为文稿列出一个方法名字就把 executable 或 reproduced 设为true。统一覆盖清单为 source_coverage.json，指标身份/数学定义/成本与选择评估见 metrics_registry.md。
