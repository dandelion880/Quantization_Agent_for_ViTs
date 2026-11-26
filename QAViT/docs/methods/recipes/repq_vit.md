## RepQ-ViT：先建立准确的细粒度网格，再转换部署表示

<a id="repq_vit"></a>

**来源。** [ICCV 2023 正式论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.pdf)，读取 Algorithm 1、§3.1–3.3、§4.1 和消融；[官方实现](https://github.com/zkkli/RepQ-ViT)，本次读取 main 的 `classification/quant/quantizer.py` 和 `classification/test_quant.py`。读取日期 2026-11-25，未执行源码。

### 1. 完整原配方

post-LN 先拟合逐通道尺度 \(s\) 与零点 \(z\)，取共享目标 \(\bar s=\operatorname{mean}(s)\)、\(\bar z=\operatorname{mean}(z)\)，通过 LN 和下游 Linear 的等价变换消除激活通道间网格差异。以下按行向量输入、\(Y=XW+b\) 约定书写。令

\[
r=s/\bar s,\qquad c=s\odot(z-\bar z),\qquad
X'=(X+c)/r.
\]

相应变换为

\[
\gamma'=\gamma/r,\quad \beta'=(\beta+c)/r,\quad
W'=\operatorname{diag}(r)W,\quad b'=b-cW.
\]

于是 \(X'W'+b'=XW+b\) 在未量化实数运算下成立。PyTorch 存储 W 的转置约定与此不同，接入时沿真实输入通道轴操作。变换后重新校准 W，不能复用旧 W 的量化参数。原法将其应用于 block 的 LN，并涵盖 Swin PatchMerging 的相关 LN。

post-Softmax 使用 log√2 网格，提高 log2 相比之下较稀疏的大概率区间的分辨率。指数编码 \(q=\operatorname{round}(-2\log_2(x/s))\)，解码 \(s2^{-q/2}\)；通过 q 的奇偶性表示成整数移位与 1/√2 相关常数因子。这个固定码本分支由 code 决定，不是每输入重新拟合的动态 scale。

论文对权重采用 per-channel、部署激活采用 per-layer；量化大矩阵乘的权重与输入，LN/Softmax 内部浮点。分类校准 32 张 ImageNet 图像，COCO 检测/实例分割校准 1 张图像。校准使用 Percentile，没有重构优化器。作者代码对分位点 {0.999, 0.9999, 0.99999} 按误差选取；LN 重参数化后重新开启相关量化状态并重校准 W。将它替换成单纯 minmax 是适配变体，不能默认为完整论文配方。

四位严格编码需要处理两处原实现语义：

- 官方 log√2 代码保留 16 个非零解码值，并对更小值额外输出零；这在所有可输入值上可能有 17 个输出状态。论文理论网格、作者参考实现和符合 16 码字的 QAViT 变体应分别记录。
- `target_zero_point=mean(z)` 可能不是整数，而代码先 `round(x/scale)+zero_point` 再 clamp。非整数网格加饱和端点可能增加状态数。要按完整编码规则检查，不能仅凭校准样本中出现的 unique 值宣称合规。把目标零点取整或重新定义 code 路径会改变数值，需显式建立适配变体与误差对照。

### 2. 可组合机制

QAViT 已有 `repq_ln`、逐通道参数输入与 log√2 参考量化。该基础可供 RepQuant 的双边 clipping、DopQ 的共享尺度汇总等机制复用。完整流程应明确“通道网格拟合 → LN/Linear 等价变换 → W 与后续统计重估 → 冻结量化状态”。

同一 LN 输出有多个消费者时，每条实际消费者边都需要相应的等价补偿。无法补偿的分支不应忽略；可选择边上独立变换等明确实现。跨 window、特殊 token、PatchMerging 等语义由模型适配器决定，不能只靠模块名字字符串断言成立。

先运行 GPTQ 再用 RepQ 修改 W，会使之前的 W 网格和重构目标失效。合理组合通常先完成表示变换，再重构改变后的 W；若研究反向顺序，必须重建相关状态，并将其标为新组合。

### 3. 研究工作流

原论文的核心论证是把校准阶段需要的细粒度表达与部署阶段希望使用的简单表示分开，再用可验证的变换连接。这个思想支持新研究，但不能预设所有变换都等价，也不能把参考 QDQ 可执行性等同为内核效率证据。

建议把对照分成逐通道直接量化、共享网格直接量化、等价变换后共享网格、变换后再做 W 重构四项。先验证 FP 等价，再看 A 与 W 误差的交换，最后看任务指标。尺度汇总器、clipping 或目标改变时同时记录 W 分布和下游敏感度，从而判断收益是由通道均衡、搜索质量还是重构补偿产生。


## 接入依据与来源边界

**基线组件** repq_ln、channel_params、percentile observer、log_sqrt2。

**配方阶段与依赖** 通道网格→LN/所有消费者等价改写→重校准 W→冻结；补原版 percentile 配方和扇出图处理。

**部署条件** 可做静态主线适配；原 log 额外零、非整数 mean-z 的码态须按完整编码审计。

**尚需核对** 16 个非零值加零及浮点零点可能超出四位容量；修订码本/零点必须保存适配身份。
