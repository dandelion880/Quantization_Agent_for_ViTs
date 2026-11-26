## 8. I&S-ViT（2023 预印本；TPAMI 2025 在线/2026 卷期）

<a id="is-vit"></a>

**来源。** 综述 [24]，[原文 v3](https://arxiv.org/html/2311.10126v3) §4–5，[DOI](https://doi.org/10.1109/TPAMI.2025.3610466)、[官方仓库](https://github.com/zysxmu/IaS-ViT)。本次已读论文、README 和 reconstruction/quantizer 文件入口。

**paper-recipe。** SULQ 与 SOS 是两个独立贡献。对 post-Softmax \(x\ge0\)，先作 \(v=-\log_2(x+\eta)\)，再均匀量化 v；反量化为 \(\hat x=2^{-\operatorname{round}(\hat v)}-\eta\)。\(\eta\) 在优化前从候选网格按量化误差选择；不能用固定1e-6替代全部配置。SOS 为：①保持 W 浮点，post-LN A channelwise，其他 A layerwise，进行块输出重构；②把 post-LN channel scale/zp 等价迁移到 LN affine 和下游 Linear；③量化 W，继续块重构。作者明确量化参数校准后固定，主要优化块内权重，不是联合学习全部 scale 的配方。

为说明融合接口，设 \(y=(x-c)\oslash r\)，原线性 \(Wx+b\) 可写 \((W\operatorname{diag}r)y+(b+Wc)\)；这是 QAViT 自述的统一等价式。取 \(r=s/\bar s,\ c=-s\odot(z-\bar z)\) 即可对应原文 Eq.(15–16) 的 LN、W、bias 转换，需注意原文权重以输入行存放，与本文列样本记法转置对应。融合前 W 仍为FP，不能先硬量化再声称等价。

原文 ImageNet/COCO 各1024张，Adam 权重 lr4e-5、weight_decay0、cosine；ImageNet batch64，6bit 200次，其余1000次，COCO batch1/1000次，仅backbone优化，其余结构按校准策略量化。LN/Softmax运算浮点；矩阵乘权重/输入目标低位宽。

**primitive。** sulq_quantizer、activation_first_reconstruct、scale_reparameterize、weight_activation_reconstruct。SULQ有整数指数舍入和减η，必须保存实际码表、合法零点/负值处理；不能只登记“log2”。MR/ReLU、AdaLog post-GELU等改变激活时，需要重新选择适合的量化器，不能不经对照叠加。

**workflow 与接入。** 单独比较SULQ、SOS及其组合；记录阶段1收益能否在融合与最终W4A4后保留。QAViT已有repq_ln，缺的是分阶段参数可变范围、缓存刷新与终态验证。SOS不会造成动态推理，可进入staticW4A4；全recipe需保留SULQ的实际算子成本。作者有ViT分类和COCO证据；状态为原文/关键官方入口已核、未复现。


## 接入依据与来源边界

**基线组件** repq_ln、uniform/log 表示、捕获和量化执行骨架。

**配方阶段与依赖** 补 SULQ 与 η 搜索；SOS 的 A-first 权重重构→尺度融合→W/A 重构，分阶段冻结 quantizer。

**部署条件** 最终静态可主线；不能把 SOS 写成推理动态或联合学习全部尺度。

**尚需核对** SULQ 的整数指数、减 η、边界码表须实作；公开代码可核，但未执行，完整链不能仅用 RepQ 代替。
