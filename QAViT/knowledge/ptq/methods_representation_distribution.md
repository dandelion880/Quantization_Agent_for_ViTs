# QAViT：表示、分布与结构处理的 PTQ 方法规格

核验日期：2026-11-25。本文是交给后续 Codex 的研究与实现规格，**没有修改 QAViT，也没有执行原论文的精度复现**。论文原配方、官方代码的当前行为、针对 QAViT 的拟议改造分别标明；建议的 primitive、metric、workflow ID 都是待实现的稳定命名，不代表已有可调用工具。

## 阅读约定与现有能力

本包的 static W4A4 按协议 `static_w4a4/2` 解释，`/2` 是协议版本号：推理时量化参数及分组固定，指定张量的实际编码不超过 16 种码字。允许有限校准集上的离线搜索、重构和参数优化。动态分组、动态范围、浮点异常值支路仍属于广义 PTQ 研究配置，但不能使用同一个 static W4A4 标签。高精度 bias、scale、整数累加器与高精度激活旁路应分开登记；前者不是自动的 W/A 位宽例外，后者会改变实际算子输入精度。每个 profile 必须列出张量覆盖、元数据和边界算子。

对照基线为 QAViT(20261124-101904)。仓库相对路径 `docs/METHODS.md` 与 `tools/ptq_agent/execution/plugin_math.py` 已有 PTQ4ViT twin、FQ-ViT PTF/log2、RepQ-LN/log√2、SmoothQuant、AdaLog base×scale、AdaLog bias、GPTQ/AdaRound/GPTAQ。`tools/ptq_agent/execution/contracts.py` 已支持 per_tensor/per_channel/per_group/per_token 与 static/dynamic；但是当前协议把 activation 的 per_token/per_group 限定为 dynamic，尚不能保存与固定 token 位置绑定的静态尺度。现有浮点 QDQ、码本和 bit packing 不等于已有对应的原生整数矩阵乘内核。真实 CLS/special/patch 布局、双视图多 tap 的 NMSE/cos/KL 与传播诊断也已存在。

因此新增工作的重点是：把已存在的数学零件组织成正确配方，补齐搜索、参数更新、轴语义与依赖顺序；不重复建立一个同名但空洞的 quantizer。本文 13 个正文条目中，FGPTQ 的完整配方仍等待全文；其余均读到原文方法部分，DAQuant 以已取得的官方源码补足可访问论文摘要之外的实现细节。MNQ 已按用户要求排除，正文没有其方法机制或实验路线。

| 方法 ID | 综述编号 | 本次新增或补全的主要对象 | 与 static W4A4 的关系 |
|---|---:|---|---|
| `adalog` | 62 | 完整 FPCS、原配方、码本与 LUT 一致性 | 可适配；原代码的下溢零和首层 8bit 要单列 |
| `adfq_vit` | 71 | 异常值支路、ShiftLog2、模块重构 | 原版保留 FP 异常值；另设扩展 profile |
| `aiqvit` | 63 | 合并后量化的低秩修正、区间学习、rank 搜索 | 冻结区间并合并修正后可以是静态 PTQ |
| `dopq_vit` | 65 | TanQ、稳健尺度汇总、优化完整配方 | 可冻结为静态；需核对原式的舍入细节 |
| `fgptq_vit` | 98 | 幂次分段搜索的证据登记 | 已确认 PTQ；未取得全文，不签发复现配方 |
| `tsptq_vit` | 20 | 数值双尺度、异常通道双尺度与真实码字预算 | 固定两尺度可静态；原验证主要为 6/8bit |
| `fq_vit` | 18 | PTF 与 LIS 完整连接及表示约束 | 已有部分；原论文的 4bit 是 attention map 配置 |
| `repquant` | 23 | 双边可学习 clipping 与先 A 后 W 顺序 | 有限校准优化后静态 |
| `orq_vit` | 29 | 固定 token 位置尺度、逐行范围选择、scale-only 重构 | 原算法可静态，但绑定分辨率/token 布局 |
| `noisyquant` | 28 | 固定噪声偏移与量化 W 的 bias 折叠 | 主 matmul 可静态；明确 INT16 元数据及加法 |
| `igq_vit` | 70 | 固定量化器、按输入重新分组、group 数预算分配 | 原版分组动态；固定分组是新适配 |
| `daquant` | 30 | 分布初始化、可学习 clipping/权重修正、域对齐 | 官方实现有动态范围；静态版须单独验证 |
| `regcache` | 32 | 中层 KV 前缀、候选搜索、异常 token 删除 | 原版动态量化且有在线结构选择 |

<a id="adalog"></a>

- [1. AdaLog：把表示能力与搜索能力分开补齐](../../docs/methods/adalog.md)

<a id="adfq_vit"></a>

- [2. ADFQ-ViT：异常值保护与模块目标必须一起记录](../../docs/methods/recipes/adfq_vit.md)

<a id="aiqvit"></a>

- [3. AIQViT：低秩修正合入量化权重，区间由校准学习](../../docs/methods/recipes/aiqvit.md)

<a id="dopq_vit"></a>

- [4. DopQ-ViT：TanQ 与稳健尺度汇总是不同作用点](../../docs/methods/recipes/dopq_vit.md)

<a id="fgptq_vit"></a>

- [5. FGPTQ-ViT：保留确定的机制归属，补齐全文后签发配方](sources/fgptq_vit.md)

<a id="tsptq_vit"></a>

- [6. TSPTQ-ViT：两个“双尺度”分别解决数值分布与通道异常](../../docs/methods/recipes/tsptq_vit.md)

<a id="fq_vit"></a>

- [7. FQ-ViT：PTF 的位置与 LIS 的边界不能缩成一个 log2 插件](../../docs/methods/recipes/fq_vit.md)

<a id="repquant"></a>

- [8. RepQuant：在量化空间学边界，再重参数化与重构 W](../../docs/methods/recipes/repquant.md)

<a id="orq_vit"></a>

- [9. ORQ-ViT：逐 token 位置的量化轴改变，不是保留 FP 异常值](../../docs/methods/recipes/orq_vit.md)

<a id="noisyquant"></a>

- [10. NoisyQuant：固定偏移改变网格相位，补偿依赖最终 W](../../docs/methods/recipes/noisyquant.md)

<a id="igq_vit"></a>

- [11. IGQ-ViT：量化器固定，但分配规则在运行时执行](../../docs/methods/recipes/igq_vit.md)

<a id="daquant"></a>

- [12. DAQuant：域对齐是校准优化目标，不是校准图像选择器](../../docs/methods/recipes/daquant.md)

<a id="regcache"></a>

- [13. RegCache：先重塑 token 结构，再给底层 PTQ 校准](../../docs/methods/recipes/regcache.md)

## 跨方法的实施依赖与可审稿产物

本文件建议一次按依赖完成以下内容；顺序用于避免重复返工，不是将成熟方法无期限推迟。

1. **统一表示与布局契约。** 扩展固定 token 位置尺度、任意分组索引与稀疏旁路的独立类型，明确参数是否随输入变化；为 log/twin/piecewise 编码导出真实 codebook，包括零、符号、region 和附加 mask。现有 quantizer/QDQ/packing 接口继续使用。
2. **建立可替换的校准与更新器。** 统一候选预算、FPCS、双边 clipping、scale-only 重构、低秩/全尺寸合并修正的输入输出记录。离线梯度仅改变声明参数，持久化后转换为固定推理图；原配方的例外单列。
3. **将依赖顺序写入 recipe。** 结构变换先于受影响统计；通道 shift/scale 先于最终 W 量化；任何 W 更新后重算依赖解码 W 的 bias 补偿；变换后使旧缓存、Hessian、observer 与搜索轨迹失效。已有 GPTAQ 作为独立组合选项，不替换原论文中的 GPTQ 而不改名。
4. **双层验收。** 第一层核验小张量上的数学与编码、原论文配置和官方实现差异；第二层在相同模型、任务、数据与预算上复现可比较结果。已有指标体系承接任务与传播评估，本文新增指标只填真实缺口。
5. **保留原配方和适配关系。** 每个结果必须包含 `recipe_id`、`source_version`、`profile_id`、`implementation_variant`、校准/验证样本哈希、码本清单、运行时动态字段、所有高精度支路及实际元数据成本。只有改名后的适配结果通过任务验证，才能将其标为 QAViT 的成熟能力。

最先适合接入并比较的是 AdaLog/FPCS、RepQuant 的双边 clipping 与顺序、ORQ 的固定位置 scale，以及可分离的 MOSF。AIQ、ADFQ、IGQ、DAQuant 与 RegCache 同样进入完整计划，以各自所需的优化器、动态/旁路或结构接口为依赖，不沿用旧 static 限制而排除。FGPTQ 的机制归属已经覆盖，完整复现任务明确依赖正式全文；不以编造实现消除这一个来源缺口。
