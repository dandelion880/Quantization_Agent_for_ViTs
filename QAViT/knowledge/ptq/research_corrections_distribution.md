# 表示与分布方法：来源更正和实现边界

核验日期：2026-11-25。这里区分综述归类、此前计划的简化说法、论文与公开代码之间的差异。更正用于建立可追溯配方，不把尚待核对的一处公式扩大为整篇方法不可用。

| 条目 | 综述或旧计划容易形成的说法 | 原始来源显示的实际情况 | 本包处理 |
|---|---|---|---|
| AIQViT [63] | DFQ 是对数域量化；dynamic 必然是输入驱动的运行时范围；低秩补偿必有 FP 推理旁路 | [原文 Method、式 (9)、DFQ 小节](https://arxiv.org/html/2502.04628v1)明确学习关注区间后做均匀量化；权重计算是先形成 W+BA 再量化。不能从命名推断推理期 observer | 纳入均匀区间学习、合并后量化修正和有限校准优化 PTQ；与 FP 并行补偿配置区分 |
| <a id="mnq_vit"></a>MNQ-ViT [117] | 综述把它纳入 PTQ 方法覆盖，可作为可吸收的方法或旁支 | [原文 v2 §4.1](https://arxiv.org/html/2407.18437v2)明确采用 quantization-aware training | `excluded_qat`。只留标题、排除理由与证据；不保留方法机制、primitive、配方或研究路线 |
| ADFQ-ViT [71] | 所有激活都为指定低位宽；per-patch 必然动态 scale | [原文 §3.2](https://arxiv.org/html/2407.02763v1)保留异常值的高精度路径；[官方 TokenQuantizer](https://github.com/llwx593/adfq-vit/blob/main/quant/quant_layer.py)每次按 abs(x) 取异常值，但只在未初始化时生成并缓存 scale/zp | 原配方归入带异常值旁路的 PTQ；静态范围与动态异常 mask 分别登记；去掉旁路的版本另命名 |
| ADFQ-ViT [71] | 原文 AS 可直接当成 QK logits 或通用 attention KL；一个 outlier 参数支配全网 | 原文 §3.4 的 AS 定义涉及 Softmax 与 V 的乘积；[QuantAttention](https://github.com/llwx593/adfq-vit/blob/main/quant/quant_block.py)实际返回乘 V 前 attn；[重构代码](https://github.com/llwx593/adfq-vit/blob/main/quant/block_recon.py)沿 head 轴 dim=1 归一化，调用顺序为 log(FP)、prob(quant)，并除以 lamb_r。qkv 分支还会设阈值 5，而 README 示例传入 10 | 保存 paper/code 损失定义与有效阈值；改方向、轴或权重的版本独立命名，不以通用 KL 模板代替 |
| DopQ-ViT [65] | MOSF 的 training-free 等于整篇方法无优化；旧预印本即最终配方 | [v4 §4–5](https://arxiv.org/html/2408.03291v4)采用 MAD-guided 名称并导出中位数，完整设置包含块重构；v4 是 2026-07-28 版本 | 单列无梯度 MOSF primitive 与带有限校准重构的完整 PTQ recipe，按 v4 记录 |
| DopQ-ViT [65] | 可以按常见 companding 推导直接省略反变换中的 round | [v4 式 (7)](https://arxiv.org/html/2408.03291v4)在 arctan 前仍写 round | 保留明确公式核对项；原式与数学适配不得无标记混用 |
| FGPTQ-ViT [98] | 作者为 Z. Hu；2024 必然是错误年份；摘要足够实现完整方法 | [Springer](https://link.springer.com/chapter/10.1007/978-981-99-8546-3_7)和[Crossmark](https://crossmark.crossref.org/dialog/?doi=10.1007%2F978-981-99-8546-3_7)列 Caihua Liu、Hongyang Shi、Xinyu He；会议 2023、在线 2023-12-26、纸本 2024。此次只有正式摘要/元数据 | 更正作者，保留双年份；已确认 PTQ 与机制归属，但完整配方状态为全文待补齐，不编码假想阈值 |
| AdaLog [62] | 现有 AdaLog 等于全 FPCS；论文参数与当前仓库默认相同；4bit 名字保证恰好 16 个值 | [论文 §3–4](https://arxiv.org/html/2407.12951v1)与[配置](https://github.com/GoatWu/AdaLog/blob/master/configs/4bit.py)的细化轮数等不同，代码首卷积 A 为 8bit；[log quantizer](https://github.com/GoatWu/AdaLog/blob/master/quantizers/logarithm.py)有独立下溢 mask | 拆分 paper/code/真 4bit 适配 profile，补 FPCS 与真实码字审计；现有 shift 和 base×scale 复用 |
| FQ-ViT [18] | 4bit attention map 等于整网 W4A4；PTF 与 RepQ-LN 在同一位置；浮点 Softmax 后 log2 即完整 LIS | [原文 §3.2–3.3、Table 3、附录 A](https://arxiv.org/html/2111.13824v3)区分 pre-LN PTF、i-exp/LIS 与 W8A8+attention4；[代码](https://github.com/megvii-research/FQ-ViT/blob/main/models/ptq/quantizer/log2.py)另有零 mask | 补全已有 PTF/LIS 的连接和原始 profile；不能把现有参考 QDQ 当已部署全整数图 |
| TSPTQ-ViT [20] | 文本提取后的指数可直接作实现阈值；尺度符号无歧义 | 已渲染核读[原 PDF 图 2](https://arxiv.org/pdf/2305.12901)：6bit Softmax/GELU 阈值为 2^5/2^4，8bit 为 2^7/2^6；文字的尺度倍率方向与保留高位后解码的步长含义有疑点 | 写入已确认 bit layout，逐位核对码字与移位恢复；将倍率符号列为局部核对项 |
| RepQuant [23] | 有 RepQ 与 GPTQ 就自动等于 RepQuant；使用 GPTAQ 也仍算原配方 | [原文 §III-D–E、Algorithm 1](https://arxiv.org/html/2402.05628v1)增加 sigmoid 双边 clipping，并在激活重参数化后做 GPTQ | 新增 clipping 与正确顺序；GPTAQ 作为有独立名称的组合实验 |
| ORQ-ViT [29] | outlier decomposition 必是高精度稀疏旁路；逐 token 必然在线重估范围 | [原文 §3、Algorithm 1–2、§4.2](https://doi.org/10.1016/j.sysarc.2025.103530)是按行定义量化器、初始化后优化固定 scale；异常值通过范围/饱和处理。检测场景因行数变化只在 MatMul 使用相应机制 | 扩展固定 token 位置尺度协议；显式绑定分辨率与布局；不引入论文没有的 FP 残差支路 |
| NoisyQuant [28] | 论文与官方代码完全一致，噪声在推理时重复抽取 | [原文 §3.3](https://arxiv.org/html/2211.16056v2)是一次抽样的 Uniform 固定偏移；[fast_quant.py](https://github.com/kriskrisliu/NoisyQuant/blob/main/fast_quant.py)用 randn_like 并按层输出 MSE 搜 mean/range | 原文/代码两份配置；保留一次采样的 seed；补偿始终依赖最终解码 W；明确 INT16 元数据 |
| IGQ-ViT [70] | 动态的是量化器参数；已有连续 group quantization 就覆盖它 | [原文 §3.2](https://arxiv.org/html/2404.00928v1)测试期固定量化器、按输入重分配通道/attention 行；不同组输出缩放后求和 | 动态分配、固定中心、组数预算三项分别建模；固定分组版本独立命名 |
| IGQ-ViT [70] | 式 (4) 字母可直接照抄为无歧义实现；所有任务的扰动指标均无标签 | 原文 min/max 与 u/l 的文字定义存在方向不一致；附录 A.3 检测扰动中部分项用 ground-truth 类别索引 | 列局部公式核对项；通用双任务适配重新定义标签依赖，不冒称原版无标签 |
| DAQuant [30] | 它已给出校准集挑选算法、语义分割验证及原生 W4A4 加速；完全无参数优化 | [官方源码](https://github.com/iLearn-Lab/PR26-DAQuant)含随机 loader、block 级 SmoothL1 与跨域特征 KL、DGA/LWC/LAC/WRC 参数更新；real-quant 示例为 W4A16；此次没有语义分割原始证据 | 纳入优化 PTQ 和领域适配机制；选样是 QAViT 拟议实验，分割是新验证，不能冒用原论文结论 |
| DAQuant [30] | CLI 写 per_token 就证明所有激活均按 token 求动态范围 | [quantizer.py](https://github.com/iLearn-Lab/PR26-DAQuant/blob/main/quantize/quantizer.py)在 LAC 开启时 reduce_shape=[1]，B×N×C 情况对应跨 token 归约 | manifest 记录实际张量形状和归约轴；静态版必须另加冻结语义并重新评估 |
| RegCache [32] | 100 个候选意味着只需要 100 张图；可以直接复用 DINOv3 已验证结果；整套静态 | [v5 §4、附录 E/H/L、引言脚注](https://arxiv.org/html/2510.04547v5)从 50,000 张选 top-100 token；基线动态量化，在线删除；DINOv3 是已有 register 的例外且无相应实验 | 分开候选池、搜索与底层校准预算；保留有标签/无标签两种选择；DINOv3 先验证适用性；ADE 的 W8A8 结果不转称 W4A4 |
| QAViT 现有能力 | 需要从零加 dynamic/token/group、PTF、log、AdaRound 或双视图传播诊断 | 实际 `docs/METHODS.md`、`execution/contracts.py`、`plugin_math.py` 已有这些基础；AdaLog 非全 FPCS、native kernel 覆盖另受限 | 文档集中在缺失配方、接口差异与编码约束，避免按论文名重复造工具 |

## 对后续实现的约束

有梯度不自动等于 QAT。有限校准数据上的 scale、舍入、clipping、合并权重修正和块重构可以构成 PTQ，必须通过数据规模、参数更新范围和推理产物说明。明确采用 QAT 的论文则依用户要求排除，不因机制看似有用而重新包装进 PTQ 库。

本次原文与代码核验没有执行模型训练、精度评估或设备测速。`paper_read`/`code_read` 表示来源已经核读，不能改写为 `reproduced`。所有新适配须通过其独立配置的任务验证才能标为成熟能力。
