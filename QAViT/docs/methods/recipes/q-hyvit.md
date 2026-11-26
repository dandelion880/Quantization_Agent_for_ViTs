## 15. Q-HyViT（IEEE IoT Journal 2024）

<a id="q-hyvit"></a>

**来源。** 综述[120]引用2023预印本，本文使用[2024 v3原文](https://arxiv.org/pdf/2303.12557v3) §III–V、Algorithm1，[DOI](https://doi.org/10.1109/JIOT.2024.3403844)。作者给出的[代码地址](https://gitlab.com/ones-ai/q-hyvit)本次未核实可运行release。

**paper-recipe。** 方法针对MobileViT、Mobile-Former、EfficientFormer等CNN/Transformer混合网络。桥接block连接局部与全局表征，量化问题包括动态幅度范围、零点溢出和不同归一化结构。其核心Alg1是**候选搜索**：桥接区观察整个bridge输出，其他区观察层输出；前向缓存FP结果，反向取得输出梯度，然后搜索weight/activation scale、channel/layer粒度、对称/非对称scheme，使平方梯度加权的输出重构误差最小。不能因为标题有reconstruction，就自动认定是QDrop式梯度更新W。

设观察点o对应输出 \(z_o\)，候选参数 \((s_w,s_a,g,\mathrm{scheme})\) 产生 \(\hat z_o\)，目标为 \(\sum_i(\partial L/\partial z_{o,i})^2(\hat z_{o,i}-z_{o,i})^2\)。Algorithm1执行三轮选择，论文把scale空间写为100候选；bridge边界依各模型的实际转换算子确定，TableI给出具体模块名。校准32张，覆盖首projection与末head、Linear及attention两侧matmul输入；另分别报告保留浮点LN/Softmax与fully-quantized实验，不能混表。

**primitive。** bridge_observation_scope、scheme_granularity_search、zero_point_overflow_audit。最后一项值得独立入metric：记录初始zp超出可表示整数域比例，以及将zp clamp后损失了哪些输入区间；不能只看scale MSE。分组和对称性最终固定，不是实例动态量化。

**workflow 与接入。** QAViT已有搜索与site梯度，可扩为scheme/granularity候选；现有固定scale候选表示需要结构化quantizer specification。对标准DINOv3没有原文同类bridge，首先将其作为混合架构支持与诊断能力，不强行在每个ViT block套bridge名。作者主要W8A8/W6A6证据；W4A4是独立迁移profile。状态：原文算法及模型覆盖已核；全部scheme超参、整数算子实现及官方代码待核。



## 接入依据与来源边界

**基线组件** 输出梯度、离散尺度搜索、quantizer spec 容器。

**配方阶段与依赖** 补 hybrid bridge 边界、scheme/granularity 联合候选、整桥输出评分、zero-point 溢出审计。

**部署条件** 分组固定可静态；原 W6A6/W8A8 与整数/FP 非线性两套配置分开，W4A4 需迁移。

**尚需核对** 完整 scheme 超参、整数实现和作者 release 未核；普通 DINO block 无原 bridge，不能硬套。
