# 综述 83 条目的覆盖、纳入与排除账本

本表按用户提供综述的编号逐条核对，共 83 条；每条都有去向、阅读范围和原始来源。综述中的归类不自动构成 PTQ 身份证据，QAT 严禁录入。

**计数单位是综述条目，不是独立算法数，也不是已复现的方法数。** ERQ 的后续 DUQ 版本在 [37] 下另有正文；DFQ/QuaSC、AMP/MPTQ、CPT-V/Evol-Q 等同源或演进关系在方法卡与机器可读账本中说明。

## 覆盖统计

| 去向 | 条目数 | 含义 |
| --- | ---: | --- |
| PTQ 方法拆解 | 51 | 已有原文方法或足以核定行为的作者源码，形成三层材料；包括既有组件补全、跨领域 PTQ 与扩展配置，仍逐项保留未闭合参数。 |
| PTQ 来源待补齐 | 5 | 原始来源已确认 PTQ，但正文或关键配方证据不足，仅保留有依据的已知部分与待办，不提供可执行原法 preset。 |
| 同家族版本记录 | 1 | 合并方法家族、保留文献版本；最终出版配方尚未独立核完。 |
| 无训练算子旁证 | 1 | 无训练算子/硬件证据；原实验为 BERT/GPT-2，不作为 ViT PTQ baseline。 |
| QAT 排除 | 14 | 仅保留题名、排除理由与来源；无配方、primitive、metric 或 workflow。 |
| 非独立 PTQ 训练路线排除 | 6 | 完整流程依赖其它训练或结构恢复，未确认可独立吸收的纯 PTQ 配方；不把它们全部误称为 QAT。 |
| 训练边界未决，暂不录入 | 5 | 尚无足够原始材料确认训练边界；不录入方法内容。 |

各行的完整来源列表、精确阅读范围、代码状态和已知限制见 [source_coverage.json](source_coverage.json)。该 JSON 是文稿索引，不是当前 QAViT 可执行 registry。

## 尚未补齐的来源

配方来源待补齐：FGPTQ-ViT [98]、UniQ-ViT [25]、MROSPU [75]、Token-Dynamic Bit-Width [42]、Optimal Numerical Format [87]。

训练边界尚未确认：Eff-DFQT [86]、BinaryFormer [111]、HEAT [83]、IQGVSAA [82]、PF2A-ViT [93]。这些缺口没有用综述推断或相近论文填补。

## 逐条去向

### 量化表示

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [62] AdaLog | PTQ 方法拆解 | [对应条目](../../docs/methods/adalog.md#adalog) | [原始来源](https://arxiv.org/html/2407.12951v1) |
| [71] ADFQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/adfq_vit.md#adfq_vit) | [原始来源](https://arxiv.org/html/2407.02763v1) |
| [63] AIQViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/aiqvit.md#aiqvit) | [原始来源](https://arxiv.org/html/2502.04628v1) |
| [17] APQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/apq-vit.md#apq-vit) | [原始来源](https://arxiv.org/pdf/2303.14341) |
| [65] DopQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/dopq_vit.md#dopq_vit) | [原始来源](https://arxiv.org/abs/2408.03291) |
| [98] FGPTQ-ViT | PTQ 来源待补齐 | [对应条目](sources/fgptq_vit.md#fgptq_vit) | [原始来源](https://link.springer.com/chapter/10.1007/978-981-99-8546-3_7) |
| [18] FQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/fq_vit.md#fq_vit) | [原始来源](https://arxiv.org/html/2111.13824v3) |
| [117] MNQ-ViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/html/2407.18437v2) |
| [20] TSPTQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/tsptq_vit.md#tsptq_vit) | [原始来源](https://arxiv.org/abs/2305.12901) |

### 分布与重参数化

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [28] NoisyQuant | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/noisyquant.md#noisyquant) | [原始来源](https://arxiv.org/html/2211.16056v2) |
| [29] ORQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/orq_vit.md#orq_vit) | [原始来源](https://doi.org/10.1016/j.sysarc.2025.103530) |
| [19] RepQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/repq_vit.md#repq_vit) | [原始来源](https://arxiv.org/abs/2212.08254) |
| [23] RepQuant | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/repquant.md#repquant) | [原始来源](https://arxiv.org/html/2402.05628v1) |

### 分组

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [70] IGQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/igq_vit.md#igq_vit) | [原始来源](https://arxiv.org/html/2404.00928v1) |

### W/A 耦合与联合处理

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [37] ERQ | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/erq-icml2024.md#erq-icml2024) | [原始来源](https://proceedings.mlr.press/v235/zhong24a.html) |
| [25] UniQ-ViT | PTQ 来源待补齐 | [对应条目](sources/uniq-vit.md#uniq-vit) | [原始来源](https://doi.org/10.1016/j.neucom.2025.132072) |

### 重构粒度与调度

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [33] Bit-Shrinking | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/bit-shrinking.md#bit-shrinking) | [原始来源](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf) |
| [24] I&S-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/is-vit.md#is-vit) | [原始来源](https://arxiv.org/html/2311.10126v3) |
| [27] Outlier-aware Slicing | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/outlier-aware-slicing.md#outlier-aware-slicing) | [原始来源](https://proceedings.mlr.press/v235/ma24f.html) |
| [76] PFCR-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/pfcr-vit.md#pfcr-vit) | [原始来源](https://arxiv.org/html/2412.14633v1) |
| [75] MROSPU | PTQ 来源待补齐 | [对应条目](sources/mrospu.md#mrospu) | [原始来源](https://link.springer.com/chapter/10.1007/978-981-95-5693-9_9) |
| [74] MGRQ | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/mgrq.md#mgrq) | [原始来源](https://arxiv.org/html/2406.09229v1) |
| [120] Q-HyViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/q-hyvit.md#q-hyvit) | [原始来源](https://arxiv.org/pdf/2303.12557v3) |
| [54] QDrop | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/qdrop.md#qdrop) | [原始来源](https://arxiv.org/pdf/2203.05740) |
| [99] QEP | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/qep.md#qep) | [原始来源](https://proceedings.neurips.cc/paper_files/paper/2025/file/df2034a516cbd617a96492cc476276c9-Paper-Conference.pdf) |

### 敏感度、目标与搜索

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [16] PTQ4ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/ptq4vit.md#ptq4vit) | [原始来源](https://arxiv.org/abs/2111.12293) |
| [22] PTQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/ptq-vit-2021.md#ptq-vit-2021) | [原始来源](https://proceedings.neurips.cc/paper_files/paper/2021/file/ec8956637a99787bd197eacd77acce5e-Paper.pdf) |
| [34] FIMA-Q | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/fima-q.md#fima-q) | [原始来源](https://arxiv.org/html/2506.11543v1) |
| [35] APHQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/aphq-vit.md#aphq-vit) | [原始来源](https://arxiv.org/html/2504.02508v1) |
| [113] CPT-V | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/cpt-v.md#cpt-v) | [原始来源](https://arxiv.org/pdf/2211.09643) |
| [114] Evol-Q / Jumping through Local Minima | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/evol-q.md#evol-q) | [原始来源](https://openaccess.thecvf.com/content/ICCV2023/html/Frumkin_Jumping_Through_Local_Minima_Quantization_in_the_Loss_Landscape_of_ICCV_2023_paper.html) |
| [36] PD-Quant | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/pd-quant.md#pd-quant) | [原始来源](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.pdf) |

### 误差补偿

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [110] Bias Compensation | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/biascomp.md#biascomp) | [原始来源](https://arxiv.org/pdf/2404.01892) |
| [15] QwT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/qwt.md#qwt) | [原始来源](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Quantization_without_Tears_CVPR_2025_paper.pdf) |
| [44] QwT-v2 | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/qwt-v2.md#qwt-v2) | [原始来源](https://arxiv.org/html/2505.20932v1) |

### 域适配

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [30] DAQuant / Domain-Aware PTQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/daquant.md#daquant) | [原始来源](https://doi.org/10.1016/j.patcog.2025.112182) |

### 无真实校准数据

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [78] DFQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/dfq-vit.md#dfq-vit) | [原始来源](https://arxiv.org/abs/2507.14481) |
| [80] E2H + Activation Correction / QuaSC | 同家族版本记录 | [对应条目](../../docs/methods/recipes/dfq-vit.md#e2h-activation-correction) | [原始来源](https://doi.org/10.1145/3771936) |
| [86] Eff-DFQT | 训练边界未决，暂不录入 | [对应条目](source_coverage.json) | [原始来源](https://doi.org/10.1109/ICME59968.2025.11209107) |
| [39] HC-PTQ | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/hc-ptq.md#hc-ptq) | [原始来源](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/papers/Mineo_HC-PTQ_Poincare-Based_Hyperbolic_Clustering_for_Data-Free_Quantization_of_Vision_Transformers_ICCVW_2025_paper.pdf) |
| [84] MimiQ | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2407.20021) |
| [79] PSAQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/psaq-vit.md#psaq-vit) | [原始来源](https://arxiv.org/abs/2203.02250) |
| [57] PSAQ-ViT v2 | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2209.05687) |
| [85] SARDFQ | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/sardfq.md#sardfq) | [原始来源](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Semantic_Alignment_and_Reinforcement_for_Data-Free_Quantization_of_Vision_Transformers_ICCV_2025_paper.html) |
| [38] CLAMP-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/clamp-vit.md#clamp-vit) | [原始来源](https://arxiv.org/abs/2407.05266) |

### 精度与格式分配

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [68] AMP-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/amp-vit.md#amp-vit) | [原始来源](https://openaccess.thecvf.com/content/WACV2025/html/Tai_AMP-ViT_Optimizing_Vision_Transformer_Efficiency_with_Adaptive_Mixed-Precision_Post-Training_Quantization_WACV_2025_paper.html) |
| [115] LampQ | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/lampq.md#lampq) | [原始来源](https://arxiv.org/abs/2511.10004) |
| [67] LRP-QViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/lrp-qvit.md#lrp-qvit) | [原始来源](https://arxiv.org/abs/2401.11243) |
| [41] Mix-QViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/mix-qvit.md#mix-qvit) | [原始来源](https://arxiv.org/abs/2501.06357) |
| [116] MixA-Q | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/mixa-q.md#mixa-q) | [原始来源](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_MixA-Q_Revisiting_Activation_Sparsity_for_Vision_Transformers_from_a_Mixed-Precision_ICCV_2025_paper.pdf) |
| [40] MPTQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/mptq-vit.md#mptq-vit) | [原始来源](https://arxiv.org/abs/2401.14895) |
| [69] PoMQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/pomq-vit.md#pomq-vit) | [原始来源](https://www.mdpi.com/2076-3417/15/18/9856) |
| [119] Patch-wise MPQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/patchwise-mpq-vit.md#patchwise-mpq-vit) | [原始来源](https://arxiv.org/abs/2305.06559) |
| [42] Token-Dynamic Bit-Width | PTQ 来源待补齐 | [对应条目](sources/token-dynamic-bit-width.md#token-dynamic-bit-width) | [原始来源](https://www.sciencedirect.com/science/article/pii/S0031320325009306) |
| [87] Optimal Numerical Format | PTQ 来源待补齐 | [对应条目](sources/optimal-numerical-format.md#optimal-numerical-format) | [原始来源](https://doi.org/10.1109/ICASSP49357.2023.10096798) |

### 结构条目筛查

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [32] RegCache | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/regcache.md#regcache) | [原始来源](https://arxiv.org/html/2510.04547v5) |
| [90] LRA-QViT | 非独立 PTQ 训练路线排除 | [对应条目](source_coverage.json) | [原始来源](https://proceedings.mlr.press/v267/kang25d.html) |
| [111] BinaryFormer | 训练边界未决，暂不录入 | [对应条目](source_coverage.json) | [原始来源](https://doi.org/10.1109/TII.2024.3396520) |
| [112] BinaryViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2306.16678) |
| [89] BiViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://openaccess.thecvf.com/content/ICCV2023/papers/He_BiViT_Extremely_Compressed_Binary_Vision_Transformers_ICCV_2023_paper.pdf) |
| [118] OFQ | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://proceedings.mlr.press/v202/liu23w.html) |
| [121] Q-ViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://proceedings.neurips.cc/paper_files/paper/2022/hash/deb921bff461a7b0a5c344a4871e7101-Abstract.html) |
| [122] Quantformer | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://doi.org/10.1109/TPAMI.2022.3229313) |
| [31] QuantTune | 非独立 PTQ 训练路线排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2403.06497) |
| [123] TerViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2201.08050) |
| [124] Variation-aware ViT Quantization | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2307.00331) |

### 部署与算子

| 综述编号与条目 | 去向 | 文稿位置 | 原始来源入口 |
| --- | --- | --- | --- |
| [95] Approx. Softmax/LayerNorm Accelerator | 无训练算子旁证 | [对应条目](sources/approx-softmax-layernorm-accelerator.md#approx-softmax-layernorm-accelerator) | [原始来源](https://www.mdpi.com/2079-9292/14/12/2337) |
| [91] Auto-ViT-Acc | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2208.05163) |
| [97] ELSA | 非独立 PTQ 训练路线排除 | [对应条目](source_coverage.json) | [原始来源](https://openaccess.thecvf.com/content/CVPR2024W/ECV24/papers/Huang_ELSA_Exploiting_Layer-wise_NM_Sparsity_for_Vision_Transformer_Acceleration_CVPRW_2024_paper.pdf) |
| [45] H-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/h-vit.md#h-vit) | [原始来源](https://link.springer.com/article/10.1007/s43684-025-00121-0) |
| [83] HEAT | 训练边界未决，暂不录入 | [对应条目](source_coverage.json) | [原始来源](https://doi.org/10.1109/TCSII.2025.3547340) |
| [58] I-ViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_I-ViT_Integer-only_Quantization_for_Efficient_Vision_Transformer_Inference_ICCV_2023_paper.pdf) |
| [21] IPTQ-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/iptq-vit.md#iptq-vit) | [原始来源](https://arxiv.org/abs/2511.15369) |
| [82] IQGVSAA | 训练边界未决，暂不录入 | [对应条目](source_coverage.json) | [原始来源](https://ieeexplore.ieee.org/abstract/document/10288182/) |
| [96] N:M Sparse Transformer Accelerator | 非独立 PTQ 训练路线排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2208.06118) |
| [43] P2-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/p2-vit.md#p2-vit) | [原始来源](https://arxiv.org/abs/2405.19915) |
| [88] PackQViT | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c92edb990a05f2269f0cc3afbb4c952-Abstract-Conference.html) |
| [93] PF2A-ViT | 训练边界未决，暂不录入 | [对应条目](source_coverage.json) | [原始来源](https://ieeexplore.ieee.org/abstract/document/10979990/authors) |
| [26] PracticalEdge | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/practicaledge.md#practicaledge) | [原始来源](https://proceedings.mlsys.org/paper_files/paper/2023/hash/12f429641e5c63a9ca7fd1c5c4804d32-Abstract-mlsys2023.html) |
| [46] Trio-ViT | PTQ 方法拆解 | [对应条目](../../docs/methods/recipes/trio-vit.md#trio-vit) | [原始来源](https://arxiv.org/abs/2405.03882) |
| [81] VAQF | QAT 排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2201.06618) |
| [94] ViTaLiTy | 非独立 PTQ 训练路线排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2211.05109) |
| [92] ViTCoD | 非独立 PTQ 训练路线排除 | [对应条目](source_coverage.json) | [原始来源](https://arxiv.org/abs/2210.09573) |

## 来源更正入口

- [表示与分布的具体更正](research_corrections_distribution.md)：AIQ、MNQ、DopQ、ADFQ、NoisyQuant、DAQuant、RegCache 等。
- [数据、精度与部署的具体更正](research_corrections_coverage.md)：文献身份、同源版本、数据成本、LampQ 预算、H-ViT 码态等。
- [重构与 metric 文稿](methods_reconstruction_metrics.md)：APHQ/FIMA 论文和代码版本、ERQ-DUQ、QEP 任务范围、Evol-Q 搜索与选择数据、BiasComp/QwT 参考和轴语义。
- [既有方法补全](existing_methods_delta.md)：PTQ4ViT 首层精度/缓存行为、RepQ 编码容量与原配方。

以上来源状态截止 2026-11-25。本包没有执行算法精度复现或原生设备测速，也没有把公开论文的数字写成 QAViT 实测。
