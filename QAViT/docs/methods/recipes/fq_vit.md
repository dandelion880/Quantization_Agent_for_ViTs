## 7. FQ-ViT：PTF 的位置与 LIS 的边界不能缩成一个 log2 插件

<a id="fq_vit"></a>

**证据。** 综述 [18]；[原文 v3](https://arxiv.org/html/2111.13824v3)，IJCAI 2022，§3.2–3.3、§4.1、附录 A–B；[官方仓库](https://github.com/megvii-research/FQ-ViT) main 已核读 quantizer、PTF observer 与量化层。整数内核的系统部署计划由本包部署文档承接。

### 原配方

PTF 量化的是 **LayerNorm 输入**，与 RepQ 处理 post-LN 输出不同。每通道使用 \(s_c=2^{\alpha_c}s\)，共享基础 scale 与 zero-point，\(\alpha_c\in\{0,\ldots,K\}\)，逐通道以量化误差选指数。执行 LN 前先按指数对齐整数值，再在共同尺度下处理统计量。原文默认 K=3，校准数据为 1,000 张。[原文 §3.2、§4.1](https://arxiv.org/html/2111.13824v3)

LIS 把 i-exp 近似和 log2 attention 编码连接起来，目标是减少 Softmax 两侧的浮点往返，后续 attention×V 可采用移位。原论文的代表性低位设置是 W8A8 加 4bit attention map，不能登记为全 W4A4 成熟基线。推理时保留怎样的 scale 计算、整数位宽和截断顺序，属于完整 LIS 配方，而不是调用浮点 Softmax 后再套 `log2`。[原文 §3.3、附录 Algorithm 1–2](https://arxiv.org/html/2111.13824v3)

### 可组合 primitive 与现有实现复用

QAViT 已有 PTF 和 log2 的 reference；新增 `pipeline.ptf_integer_ln` 把指数对齐、整数统计与输出 requant 串成可追踪步骤，`pipeline.lis` 明确 logits 的输入尺度、减最大值、i-exp、整数和/除法、I-Log2 和输出表示。不要改动已有 PTF 数学来迁就新的部署 API；通过显式适配层连接。

官方 [`Log2Quantizer`](https://github.com/megvii-research/FQ-ViT/blob/main/models/ptq/quantizer/log2.py) 使用输入相关 mask，将超出指数范围的值解码为零。因此“16 个指数加下溢零”的状态必须纳入实际编码审计。原论文模拟、真 4bit 的零码适配和完整整数 LIS 应分别有 profile，不能把三者的精度或速度相互转移。对于可变 token 数，Softmax 行长度也影响累加与指数范围，边界测试需要真实序列长度。

### 研究工作流

拟议 `study.nonlinear_error_split` 分别测表示误差、i-exp 近似误差、整数除法/截断误差和整层输出误差。这样才能回答性能损失来自 4bit 码本还是整数 Softmax 近似。PTF 则比较共享尺度、固定两组尺度和多指数尺度，在相同 metadata 成本下检验通道异质性是否足以支持更细分辨率。这里的价值是把整数近似与量化误差分解，而非再新增一个与现有 PTF 同名的工具。W4A4 是需要重新标定的扩展，不沿用 W8A8+attention4 的精度结论。

建议保留逐操作中间整数及其 scale 的诊断模式。出现整层偏差时，可以定位到指数近似、求和溢出、位移或最终 requant；仅比较最后一个浮点张量不足以证明整数链正确。


## 接入依据与来源边界

**基线组件** PTF、log2、integer_layernorm/integer_softmax 参考算子。

**配方阶段与依赖** 补 pre-LN PTF、LIS、观察器 finalize 与矩阵输入覆盖的完整连接；核对整数尺度传递。

**部署条件** 原 W8A8/attention4 不能作全 A4；主线低位适配、integer-only 和原生内核分别验证。

**尚需核对** log2 额外零的容量问题须处理；现有参考算子不构成完整原法或端到端加速证据。
