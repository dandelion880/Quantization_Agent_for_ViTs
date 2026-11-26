# 原始来源核验、身份勘误与覆盖边界

核验日期：2026-11-25。本文负责数据来源、精度分配、部署与邻域研究共 **46 个综述条目**。逐条可追溯记录已并入 `source_coverage.json`；不纳入的方法在 `research_screening_data_precision_deployment.md`。表示分布路线的补充勘误另见 `research_corrections_distribution.md`。

## 1. 先校正“83 篇”的含义

用户提供的综述为 [A survey on post-training quantization for Vision Transformers: Failure modes and deployment constraints](https://doi.org/10.1016/j.sysarc.2026.104001)，Journal of Systems Architecture 181 (2026), 104001。PDF 首页载明 2026-09-11 online；本文按用户提供版本核查。**Table B.1 是分类判定规则，83 项逐论文说明实际在 Table B.2。**

83 个 corpus 条目不是 83 个相互独立、可直接执行且已经成熟的 PTQ 配方。其中有同一方法的版本、QAT、需要结构训练的加速方案、算子硬件旁证，以及尚不能从原始来源完整核验的条目。综述适合建立问题地图，方法身份、训练边界、预算和精度声明仍须回到原论文。

本组 46 个条目的去向如下。这里统计的是文献条目，不是独立算法数，也不是已在 QAViT 实现的数量。

| 去向 | 条目数 | 实际含义 |
|---|---:|---|
| 纳入 PTQ 材料 | 16 | 核心 PTQ 方法已核定；不同条目的预算、代码和任务证据完整度仍有差异 |
| 只纳入独立 PTQ 分支 | 2 | Mix-QViT、MixA-Q；其 QAT 分支不进入材料 |
| 同家族正式出版记录 | 1 | E2H/AC 与 DFQ/QuaSC 共用家族，保留版本差异 |
| 无训练算子旁证 | 1 | Approximation Softmax/LN；不是 ViT PTQ baseline |
| PTQ 身份已知但原配方来源不全 | 2 | TokenDynamic、OptimalFormat；仅来源待办，不能启用执行 |
| 排除 QAT | 13 | 只留排除原因与原始来源，无方法/组件/workflow |
| 排除非独立 PTQ 的训练路线 | 6 | 结构、稀疏或浮点微调后量化；不笼统贴 QAT 标签 |
| 训练边界或原文不足，暂不纳入 | 5 | Eff-DFQT、BinaryFormer、HEAT、IQGVSAA、PF²A-ViT |

AMP/MPTQ 也属于同一研究家族，虽在前表保留两条版本材料，不能作为两份独立成功证据计数。

## 2. 文献身份、作者与版本校正

| 综述条目 | 核验结果 | 库内处理与原始来源 |
|---|---|---|
| PoMQ-ViT [69] | 出版社署名为 **Zhiqiang Wu、Zhong Zhao**；综述写为 H. Lee、S. Yang、S. Yoo | 使用出版社身份；[Applied Sciences 正文](https://www.mdpi.com/2076-3417/15/18/9856)，题头与作者栏 |
| HEAT [83] | IEEE 登记作者为 **Pan Zhao、Donghui Xue、Licheng Wu、Liang Chang、Haining Tan、Yinhe Han、Jun Zhou**，与综述作者串不同 | 身份已核定，但未读到足以核定完整训练边界的方法正文；不因综述写 retraining-free 就录入。[DOI](https://doi.org/10.1109/TCSII.2025.3547340)、[作者目录](https://panzhao08.github.io/publications/) |
| PF²A-ViT [93] | IEEE/Crossref 登记作者为 **Zihan Zou、Xinming Yan、Chen Zhang、Shikuang Chen、Guang Yang、Han Yan、Hao Cai、Bo Liu**；综述作者串部分不符 | DOI 身份已核定；parameter-free token pruning 不等于整条量化路线无需 QAT，训练边界维持未决。[DOI](https://doi.org/10.1109/TCAD.2025.3565468)、[出版社登记](https://api.crossref.org/works/10.1109/TCAD.2025.3565468) |
| HC-PTQ [39] | CVF 正式页码为 **4053–4060**，综述写 3994–4001 | 保留正确正式来源；[ICCVW 原文](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/papers/Mineo_HC-PTQ_Poincare-Based_Hyperbolic_Clustering_for_Data-Free_Quantization_of_Vision_Transformers_ICCVW_2025_paper.pdf) |
| DFQ-ViT [78] 与 E2H+AC [80] | 作者与 E2H synthesis/activation correction 核心对应；ACM 最终摘要把方法称为 **QuaSC** | 共用家族，不制造重复 baseline；已读配方来自 [arXiv 2507.14481](https://arxiv.org/abs/2507.14481)，最终题名/摘要来自 [ACM DOI](https://doi.org/10.1145/3771936)。最终全文未完整取得，不能把预印本全部设置自动归到定稿 |
| AMP-ViT [68] 与 MPTQ-ViT [40] | 同作者 Yu-Shan Tai、An-Yeu Wu；核心模块与结果高度重合，SQ-b/OPT-m 后续对应 SymAlign/AutoScale | 同家族、独立版本记录；[WACV AMP 原文](https://openaccess.thecvf.com/content/WACV2025/html/Tai_AMP-ViT_Optimizing_Vision_Transformer_Efficiency_with_Adaptive_Mixed-Precision_Post-Training_Quantization_WACV_2025_paper.html)、[MPTQ 预印本](https://arxiv.org/abs/2401.14895) |
| SARDFQ [85] | 早期 arXiv:2412.16553 名为 SPDFQ；正式 ICCV 2025 名为 SARDFQ | 优先引用正式 §3.4–4.1 的生成与重构设置；[ICCV 论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Semantic_Alignment_and_Reinforcement_for_Data-Free_Quantization_of_Vision_Transformers_ICCV_2025_paper.html)、[早期版本](https://arxiv.org/abs/2412.16553) |
| Variation-aware [124] | 后续正式标题为 **Quantization Variation: A New Perspective on Training Transformers with Low-Bit Precision**，TMLR 2024；arXiv v2 作者增加 Pingcheng Dong | 保留综述旧名 alias 与正式身份；方法为 QAT，正文不纳入。[arXiv](https://arxiv.org/abs/2307.00331)、[OpenReview](https://openreview.net/forum?id=MHfoA0Qf6g) |
| LampQ [115] | 官方仓库现标 AAAI 2026，并已公开代码 | 不再仅记作 2025 未实现预印本；仍不能等同完整复现。[作者仓库](https://github.com/snudatalab/LampQ)、[论文](https://arxiv.org/abs/2511.10004) |
| IPTQ-ViT [21] | 官方仓库现标 WACV 2026，并含 IPTQ++ 更新 | 原论文、后续版本和代码运行模式分别锁定；不将新版 README 设置无条件倒填旧论文。[作者仓库](https://github.com/gihwan-kim/IPTQ-ViT)、[论文](https://arxiv.org/abs/2511.15369) |

**PracticalEdge [26] 的 MLSys hash 是真实匹配来源，不是错误指向 I-ViT。** 正确题名为 *Practical Edge Kernels for Integer-Only Vision Transformers Under Post-training Quantization*；以[原始 MLSys PDF](https://proceedings.mlsys.org/paper_files/paper/2023/file/12f429641e5c63a9ca7fd1c5c4804d32-Paper-mlsys2023.pdf)的标题、作者和方法核对。不能因为缩写不是正式标题或检索不顺而宣称该方法不存在。

## 3. 防止同名论文、仓库串线

这些错误会使后续 Codex 抄到另一方法，即使标题里都有“量化”也不能替代。

| 条目 | 正确身份 | 不应误用 |
|---|---|---|
| BinaryFormer [111] | IEEE TII，DOI [10.1109/TII.2024.3396520](https://doi.org/10.1109/TII.2024.3396520)，Miaohui Wang、Zhuowei Xu、Bin Zheng、Wuyuan Xie | MIDL 2025 的同名 BinaryFormer 仓库不能证明这篇的代码和训练边界 |
| BinaryViT [112] | [arXiv:2306.16678](https://arxiv.org/abs/2306.16678)，*Pushing Binary Vision Transformers Towards Convolutional Models*，作者 Phuoc Hoan Charles Le、Xinlin Li | 不使用 arXiv:2305.14730 的另一篇二值 ViT 论文替代 |
| Q-ViT [121] | [NeurIPS 2022 正式论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/deb921bff461a7b0a5c344a4871e7101-Abstract.html)，*Accurate and Fully Quantized Low-bit Vision Transformer* | 不使用 arXiv:2201.07703 的另一篇同缩写论文 |
| Quantformer [122] | [TPAMI DOI](https://doi.org/10.1109/TPAMI.2022.3229313)，*Learning Extremely Low-precision Vision Transformers*，官方 [ZiweiWangTHU/Quantformer](https://github.com/ZiweiWangTHU/Quantformer) | 不使用金融时序等领域的同名 QuantFormer |
| HC-PTQ [39] | Poincaré 双曲聚类权重码本 | 不等同 AHCPTQ 的 attention-head 方法 |

前四项只用于核验和筛查，不由此引入其训练方法。

## 4. 训练边界要按原流程判断

**PSAQ-ViT 与 PSAQ-ViT v2 不可共用一个“纯 PTQ”许可。** 原 PSAQ 的官方生成器只优化图片；v2 的全网量化学生学习超出本包 PTQ 边界。MimiQ 的量化学生训练亦按 QAT 排除。I-ViT、PackQViT 的整数/移动部署属性也不能覆盖其 QAT 依赖。具体来源和排除段落已逐项列在筛查记录；这里不展开这些配方。

**SARDFQ、Trio-ViT 的有限局部重构可以保留。** 前者正式 §3.4.2/§4.1 用有限合成集做当前块重构；后者在有限校准集上使用局部 PTQ 重构。不能将“出现 Adam、backward、权重优化”作为排除条件，也不能把这类方法写成完全无参数优化。应记录优化对象、样本集合、迭代数与停止边界。来源分别为 [SARDFQ 正式论文](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Semantic_Alignment_and_Reinforcement_for_Data-Free_Quantization_of_Vision_Transformers_ICCV_2025_paper.html)和 [Trio-ViT 原文](https://arxiv.org/abs/2405.03882)。

**LRA-QViT、QuantTune 与稀疏/结构训练加速器的排除不应一律叫 QAT。** 它们缺少能与前置全数据或结构训练解耦的独立 PTQ 原配方，故本包只留筛查记录。这样既落实纯 PTQ 范围，也不歪曲论文分类。

**Mix-QViT 与 MixA-Q 只截取已经独立成立的 PTQ 分支。** 后者对应正式论文 **§4.3 PTQ Result on COCO**；其 PTQ 的 W4A8 基础、A4 低精度窗口和 1 张校准图并不能由另一个训练分支的成绩背书。[Mix-Q 原文](https://arxiv.org/abs/2501.06357)、[MixA-Q 原文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_MixA-Q_Revisiting_Activation_Sparsity_for_Vision_Transformers_from_a_Mixed-Precision_ICCV_2025_paper.pdf)。

## 5. 参数、成本和效果口径中不能省略的差异

| 风险 | 原始证据 | 对接入的影响 |
|---|---|---|
| 校准图数不等于全流程数据预算 | LRP-QViT §3.3.2 用 50,000 训练图统计 relevance，校准本身才是 32 图；Mix-Q 另用 256 张 validation 图估计敏感度 | 数据收据分别记录生成/评分/校准/选择/最终验证，不宣称全流程只用 32 图。[LRP](https://arxiv.org/abs/2401.11243)、[Mix-Q](https://arxiv.org/abs/2501.06357) |
| 参数量加权平均 bit 不等于层平均 | LampQ Step 1/Appendix E 用 `Σ c_i B_i ≤ b_t Σ c_i`；Step 2 的一层加 1、一层减 1 在 `c_up≠c_down` 时使成本改变 | 每次更新核对真实预算；作者代码是否做同规模配对/约束修复尚未核实。QAViT 新增投影必须登记为适配变体。[LampQ Algorithm 1 与 Appendix E](https://arxiv.org/abs/2511.10004) |
| 原文 clip 区间可能多一个码态 | H-ViT 正式 PDF p.6 Eq.(18) 为 `[-2^(b−1)−1,2^(b−1)−1]`，整数闭区间含 `2^b+1` 个值 | 疑似边界笔误；作者代码/勘误尚未核实，不能照抄为 b bit 编码。自行修正须标适配。[H-ViT 原文](https://link.springer.com/article/10.1007/s43684-025-00121-0) |
| TV 先验并非总是直接最小化 TV | PSAQ 官方 `generate_data.py` 用 `norm(TV(img)−var_pred)`，`var_pred∼U(2500,3000)`，注释对应 batch 32 | 改变 batch、归一化或分辨率要检查目标尺度；论文与作者代码 profile 分开保存。[官方脚本](https://github.com/zkkli/PSAQ-ViT/blob/main/generate_data.py) |
| 向量索引 bit 不等于 scalar WbAb | HC-PTQ 的总存储含 FP32 码本，公式 `K×d×32 + N×b`；原文尚未结合激活量化 | 作为 weight-codebook PTQ 扩展，不冒充 W4A4 主线；按真实存储预算比较。[HC-PTQ §3–4](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/papers/Mineo_HC-PTQ_Poincare-Based_Hyperbolic_Clustering_for_Data-Free_Quantization_of_Vision_Transformers_ICCVW_2025_paper.pdf) |
| BOPs、kernel 速度与端到端延迟不同 | MixA-Q PTQ 的 1.35× 是计算量比；PracticalEdge 提供具体算子 kernel 速度；不同论文硬件也不同 | 不把 BOPs 比或单算子比直接写为 QAViT 端到端加速。[MixA §4.3](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_MixA-Q_Revisiting_Activation_Sparsity_for_Vision_Transformers_from_a_Mixed-Precision_ICCV_2025_paper.pdf)、[PracticalEdge](https://proceedings.mlsys.org/paper_files/paper/2023/file/12f429641e5c63a9ca7fd1c5c4804d32-Paper-mlsys2023.pdf) |
| integer-only 标题不保证每个仓库模式全整数 | IPTQ 官方 README 区分仿真、CUDA kernel 及尚未支持的 log-mode kernel，并含实现精度说明 | 分别记录数学配方、仿真路径、实际部署路径及 fallback，不能用标题覆盖运行事实。[IPTQ 官方 README](https://github.com/gihwan-kim/IPTQ-ViT) |
| 原文 SQNR 记号与通常定义可能不一致 | IPTQ Eq.18 对功率比使用 `20 log`；通常功率比使用 `10 log` | 忠实复现模式保留原式，通用 metric 另设标准定义；不得静默合并。这一差异也可能影响后续 Softplus 归一化的数值。[IPTQ 原文 Eq.18](https://arxiv.org/abs/2511.15369) |
| Softmax-free 架构不能直接替换标准 ViT 路径 | Trio-ViT 针对 EfficientViT 的 MBConv 与线性 attention 分母 | 架构前提进入 applicability；DINOv3 标准 attention 的适配是新实验。[Trio-ViT §III](https://arxiv.org/abs/2405.03882) |
| 模块近似的 NLP 实验不是 ViT PTQ 证据 | Approximation Softmax/LN 的系统验证对象为 BERT、GPT-2 | 仅作无训练算子旁证，不列成 ViT 分类/ADE baseline。[Electronics 正文](https://www.mdpi.com/2079-9292/14/12/2337) |

CLAMP 文字版 scale 扰动区间存在上下标可疑处：正文语义为小半径扰动，而抽取形式可能显示 `10^3`。目前不指定未经代码证实的精确区间；应以论文 PDF 与作者实现共同锁定。来源：[CLAMP §3.4](https://arxiv.org/abs/2407.05266)。

## 6. 尚未补齐的原始来源

缺口是具体的信息缺失，不是默认认定方法无效。以下项目在补齐之前不会产生可执行原法候选。

| 条目 | 已核验 | 仍需补齐 |
|---|---|---|
| Eff-DFQT [86] | [IEEE DOI 元数据](https://doi.org/10.1109/ICME59968.2025.11209107) | 全文生成目标、优化对象/预算、下游量化器及位宽；据此确认 PTQ 边界 |
| BinaryFormer [111] | [TII DOI 身份](https://doi.org/10.1109/TII.2024.3396520) | 正确论文全文的训练设置；不能用同名开源项目替代 |
| HEAT [83] | [DOI](https://doi.org/10.1109/TCSII.2025.3547340)、作者目录 | retraining-free 的原方法正文、校准预算、混精度方案及模型生成过程 |
| IQGVSAA [82] | [IEEE 摘要](https://ieeexplore.ieee.org/abstract/document/10288182/) | 模型量化与恢复过程是否独立 PTQ、原精度与校准设置 |
| PF²A-ViT [93] | [IEEE 摘要及身份](https://ieeexplore.ieee.org/abstract/document/10979990/authors) | 量化/编码整个流程的训练依赖；parameter-free pruning 不足以回答此问题 |
| TokenDynamic [42] | [出版社摘要/引言](https://www.sciencedirect.com/science/article/pii/S0031320325009306)、PTQ 身份 | §3 完整分配公式、候选 bit、阈值、§4.1 校准预算、实现 |
| OptimalFormat [87] | [IEEE DOI](https://doi.org/10.1109/ICASSP49357.2023.10096798)、[SPS 摘要](https://rc.signalprocessingsociety.org/conferences/icassp-2023/spsicassp23vid1432) | 数值格式候选与编码规则、scale 求解、数据预算、精确 W/A 覆盖 |
| E2H/QuaSC 正式版 [80] | [ACM DOI 最终题名/摘要](https://doi.org/10.1145/3771936)，早期版本已读 | 正式全文与预印本逐式/逐设置差异；并非整个家族未核验 |

还有已纳入核心配方但不能宣称端到端复现的缺口：CLAMP 公开入口是否覆盖完整生成搜索链，PoMQ 完整校准与计时配置，AMP/MPTQ/LRP/Mix-Q/Patchwise 的作者实现，HC-PTQ 码本实现，IPTQ 全部附录预算与 deployment kernel 覆盖等。每张方法材料和 coverage 的 `read_scope`、`author_code_evidence` 已保留这些边界。后续优先补能改变接入决策的缺口，不以“有论文链接”替代可复现性证据。
