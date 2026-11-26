# 08 方法与共享组件接入对应表

> 上传材料的待接入事项和原文预算按 2026-11-25 出处保留。实际数值入口、适配差异与证据见 [实现账目](implementation_status.json)。

本文件面向人和后续 Codex，补足“论文材料已经写清”到“哪些已有数值可以复用、还要接通哪一段执行链”的对应关系。事实基线为 **QAViT(20261124-101904)**；来源身份以 [source_coverage.json](source_coverage.json) 为准，原公式、步骤、超参数、官方代码差异以各行链接的方法正文为准。这里只新增文字索引，没有修改 QAViT、实现插件或运行方法复现。

覆盖范围是 **51 条 `ptq_material` 主记录，加 ERQ 2025 的一个版本，合计 52 行**。它不是 52 个新算法、52 个独立强基线或 52 项已完成实现。已有 PTQ4ViT/RepQ 等组件、同家族版本、非 ViT 原生证据均保留各自身份；5 条 `ptq_source_gap` 只列来源待办。QAT 条目只保留账本排除路由，不进入方法、组件、metric 或 Skill。

## 1. 本表在五个交付包中的作用

| 包 | 本表提供的接口 | 完成边界 |
|---|---|---|
| A：协议、知识分类、来源账本 | 准确方法 ID、版本关系、配置条件与来源缺口 | 文献可发现、可定位且状态准确；不因有文档就开放执行。 |
| B：共享数值与完整 PTQ 方法 | 下列 52 行的新增对象、依赖和共享求解器 | 共享组件数值正确，各完整 recipe 有独立顺序、参数范围、数据与产物；不只登记名字。 |
| C：metric、任务与公平比较 | 准备统计/参考图、任务轴与失效规则 | 复用已有观测与评分，补目标定义、准备阶段和选择器评估；不把论文分数写成本地结果。 |
| D：研究流程、Skill 与提示词 | 从问题到组件和实验的依赖信息 | 研究角色能选择有前提的可执行链；未实现/未闭合条目不能被规划成已可调用工具。 |
| E：集成与低成本科研能力验证 | 第 6 节的分层验收 | 完成数值、重载、状态传播及小规模科研闭环验证；大规模调优、完整昂贵复现另作运行任务。 |

本表中的“主线”只指候选终态可以满足 [static_w4a4/2](01_STATIC_W4A4_V2.md) 的条件；`/2` 是协议版本号。是否实际满足，仍由全部矩阵消费边、码态容量、参数/分组是否输入相关、结构及保留分支决定。动态、混精、额外 FP 计算可进入 PTQ 研究空间，但不能沿用相同的全静态 W4A4 成绩标签。[扩展维度](02_PROTOCOL_PROFILES.json) 是说明性设计，尚不是运行时配置注册。

## 2. “复用”具体指哪些已有源码

下表中的称呼只为本文件阅读方便，不是新增 API 名称。路径均相对基线仓库。已有数值能力并不自动包含某篇论文的完整配方。

| 本表称呼 | 已读到的实际基础 | 可复用的范围与边界 |
|---|---|---|
| 网格/编码 | `tools/ptq_agent/execution/methods.py`、`plugin_math.py`、`grids.py`、`codecs.py`、`adalog.py` | 已有 RTN、uniform、twin、log2/log√2、PTF、AdaLog，以及编码/打包参考；新增码本仍需自己的边界和容量验证。 |
| 等价变换 | `plugin_math.py` 的 `smoothquant`、`repq_ln`；`adalog.py` 的 bias 变换 | 可复用 scale/shift 与最终解码 W 的折叠基础；原论文 observer、扇出补偿和顺序须另补。 |
| 单位点拟合/捕获 | `calibration.py`、`calibration_search.py`、`adaround.py`、`gptaq.py` | 已有局部捕获、二阶矩、RTN/GPTQ/GPTAQ/AdaRound、尺度选择；现梯度 capture 限二维分类 logits，不代表已有 dense 重构梯度。 |
| 冻结候选评价 | `frozen_trials.py`、`local_comparison.py`、`metrics.py` | 已有多视图、多 tap、NMSE/cos/KL；当前 frozen 候选主要是 scale 编辑。重构权重/结构/补偿需要可重载完整候选，而非伪装成 scale。 |
| 布局/传播 | `profiling/adapters.py`、`execution/propagation.py` | 已有 CLS/special/patch 与误差传播；仍需新结构的边、window、恢复映射。 |
| 任务/数据 | `backends/image_data.py`、`backends/task_heads.py`、`tools/data_receipt.py`、`tools/segmentation.py` | 已有来源凭据和任务视图基础；合成数据、源/目标域配对、任务相关优化目标需显式扩展。 |
| 契约/状态/成本 | `execution/contracts.py`、`plugin_contracts.py`、`pipeline.py`、`runtime_state.py`、`profiling/torch_profiler.py` | 可承接量化规格、状态和测量；底层容器存在不证明新算法、原生 kernel、真实部署加速已存在。 |

两项源码限制直接影响本表：**当前契约将激活 per_token/per_group 限为 dynamic，尚不能直接承载 ORQ 的固定位置尺度；现 `local_fp_parent` 在多层候选中也不一定等于当前候选同输入的局部 FP 参考。** 两者都需有明确的新语义和加载验证，不能只改名字。[方法正文公共约定](methods_reconstruction_metrics.md#reading-conventions)、[ORQ 说明](../../docs/methods/recipes/orq_vit.md#orq_vit)

`method_library.py` 当前会验证 `plugins` 是否真实注册，并限制单条文档长度。此文件是接入索引，不应整体塞成一个方法卡，也不应将下表 ID 填进 `plugins` 制造执行入口。保留长文/来源材料，方法短目录按实现状态链接；可调用能力必须有真正数值、契约、版本及加载路径。

## 3. 52 条接入对应

每行最后一列是该条目的**特定未闭合项或容易冒称的范围**。所有行共有的状态是“资料和源码核验，不等于已在 QAViT 执行复现”，故不逐行重复。没有核实作者代码不阻止依据充分原文独立实现；但不可补造作者未给出的默认设置，或把自己的数值修订称为原法。论文与代码不一致时保存各自版本，所选 recipe 要明确取哪一版。

### 3.1 表示、分布与结构入口（14 行）

| 准确 ID 与原文材料入口 | 已有可复用基础 | 需要新增的组件/完整顺序 | 部署配置关键条件 | 未闭合项与声明边界 |
|---|---|---|---|---|
| [ptq4vit](../../docs/methods/recipes/ptq4vit.md#ptq4vit) [综述 16] | twin 网格、位点捕获、梯度平方输出评分、尺度候选搜索。 | 补完整 FP-prefix 校准、双输入交替搜索、特殊网格及 teacher 按样本配对；把原法与 +BC 分开。 | 原代码 qconv 输入 A32 是覆盖例外；完整 A4 另建适配。LN/Softmax 内部 FP 与 A4 消费边分列。 | 论文/代码搜索下界不同；同分布 KL 的零梯度和多 batch teacher 对齐要解决。已有 twin 不等于完整原法。 |
| [repq_vit](../../docs/methods/recipes/repq_vit.md#repq_vit) [综述 19] | repq_ln、channel_params、percentile observer、log_sqrt2。 | 通道网格→LN/所有消费者等价改写→重校准 W→冻结；补原版 percentile 配方和扇出图处理。 | 可做静态主线适配；原 log 额外零、非整数 mean-z 的码态须按完整编码审计。 | 16 个非零值加零及浮点零点可能超出四位容量；修订码本/零点必须保存适配身份。 |
| [adalog](../../docs/methods/adalog.md#adalog) [综述 62] | AdaLog base×scale、adalog_bias、log/twin 数值与候选评分。 | 补多区域保留的完整 FPCS、搜索轨迹、LUT 解码；补偿绑定最终解码 W，纯校准与重构配方分开。 | 固定码本可主线；原代码首层 A8 和额外下溢零需保留作者配置或另建合规版本。 | 论文与代码细化次数、入口默认值不同；不能把现有 base×scale 搜索称 FPCS，LUT 与参考解码仍需对齐。 |
| [adfq_vit](../../docs/methods/recipes/adfq_vit.md#adfq_vit) [综述 71] | log/shift、AdaRound 单位点基础、布局与输出观测。 | 补异常值载荷/稀疏 FP 计算、ShiftLog2、位置尺度、模块重构；attention 目标保留 paper/code 两版。 | 原法保留输入相关 FP 异常值，属额外计算/动态旁路；缓存位置尺度另需静态布局契约。 | AS 观察点、KL 方向/轴/权重和有效异常阈值存在已记录差异；去旁路版本不叫原法。 |
| [aiqvit](../../docs/methods/recipes/aiqvit.md#aiqvit) [综述 63] | 均匀网格、RepQ 类变换、捕获与误差评价。 | 补 Q(W0+BA) 的可合并低秩修正、均匀关注区间学习、rank 搜索、easy-to-hard 样本调度。 | 区间冻结、BA 合入且重新量化 W 后可主线；保留 FP 低秩分支则是另一配置。 | 官方完整代码未核；初始化、NAS 数据分工、边界/零码处理须按原文补齐。不能写成 log quantizer。 |
| [dopq_vit](../../docs/methods/recipes/dopq_vit.md#dopq_vit) [综述 65] | RepQ 等价变换、均匀编码、候选评分。 | 补 TanQ 编解码与参数搜索、MOSF 中位尺度汇总；完整 v4 还依赖 I&S 类块重构顺序。 | 固定参数可主线适配；原部分结果的 Softmax 更低位宽另列 mixed 配置。 | v4 解码 round 的位置有歧义，完整网格与学习率需锁定；MOSF 无梯度不代表整法无优化。 |
| [fq_vit](../../docs/methods/recipes/fq_vit.md#fq_vit) [综述 18] | PTF、log2、integer_layernorm/integer_softmax 参考算子。 | 补 pre-LN PTF、LIS、观察器 finalize 与矩阵输入覆盖的完整连接；核对整数尺度传递。 | 原 W8A8/attention4 不能作全 A4；主线低位适配、integer-only 和原生内核分别验证。 | log2 额外零的容量问题须处理；现有参考算子不构成完整原法或端到端加速证据。 |
| [tsptq_vit](../../docs/methods/recipes/tsptq_vit.md#tsptq_vit) [综述 20] | twin/PTF 基础、通道统计、Hessian 代理与尺度搜索。 | 补 V-2SF 的区域/符号码字、O-2SF 的通道 K-means 与 PoT 尺度比；保持交替搜索原顺序。 | 固定两组/两尺度可静态；原 6/8 bit 的字段常量不能直接搬到四位。 | 尺度符号方向存在局部歧义；官方代码未核。必须从总 bit 内分配 region/sign，不能加免费标签。 |
| [repquant](../../docs/methods/recipes/repquant.md#repquant) [综述 23] | repq_ln 的 channel_params 接口、GPTQ、log_sqrt2。 | 学习双边 clipping→A 网格→等价重参数化→重新收集 W 统计→GPTQ；显式限定优化变量。 | 有限校准后冻结可主线；继承的 zero/log 码态仍须检查，GPTAQ 替换 GPTQ 另记变体。 | 独立官方完整代码未核；clipping 初始化/STE 与零点语义需按所选来源闭合，不能只套现有 RepQ。 |
| [orq_vit](../../docs/methods/recipes/orq_vit.md#orq_vit) [综述 29] | percentile、捕获、token 布局、动态 per_token 表示基础。 | 补固定位置逐行尺度、first-batch 初始化、scale-only 重构及 V 的正确轴；扩展静态位置契约。 | 原尺度可静态；必须绑定分辨率/window/token 布局，检测原版只在适用 MatMul 用逐行机制。 | 现 contracts 禁止静态 per_token，不能直接加载原法；官方代码未核，跨布局共享是新实验。 |
| [noisyquant](../../docs/methods/recipes/noisyquant.md#noisyquant) [综述 28] | 固定偏移与 decoded-W bias 折叠基础、局部输出 MSE。 | 补一次采样的通道 N、均值/幅度搜索及种子；更新 W 后重算补偿，保留 paper/code 噪声与目标差异。 | 原主要 W6A6/W8A8；真四位适配可静态，INT16 N/bias 与输入加法须计成本。 | 论文 Uniform/激活误差与代码 Gaussian/Linear 输出误差不同；四位和目标模型尚无本地验证。 |
| [igq_vit](../../docs/methods/recipes/igq_vit.md#igq_vit) [综述 70] | 范围统计、分组容器、任务输出误差与成本记录。 | 补实例 min/max 分组、固定量化器交替拟合、组数 ILP、非连续 gather 与分组缩放求和。 | 原法按输入重分组，属于动态；冻结组是新静态适配，不能用固定相邻 group_size 代替。 | upper/lower 式号歧义及作者完整代码未闭合；检测扰动部分用标签索引，运行分组/求和成本不能省。 |
| [daquant](../../docs/methods/recipes/daquant.md#daquant) [综述 30] | SmoothQuant 类变换、捕获、均匀网格、任务数据入口。 | 补分布 scale/shift 初始化、DGA/LWC/LAC/WRC 参数组、源/目标域特征对齐及 block 调度；WRC 合并。 | 所核代码 forward 重估范围，属动态；冻结观察器的版本需独立验证。W4A16 部署不是 W4A4。 | 正文未取得、已有官方源码级配方；不冒称论文全文已核。随机 loader 不是选样算法，未见分割验证。 |
| [regcache](../../docs/methods/recipes/regcache.md#regcache) [综述 32] | CLS/special/patch 布局、逐层误差与传播诊断。 | 补候选 token/KV bank、缓存前缀插入、当前输入 token 删除与 dense 恢复；结构定后重校准底座。 | 原动态 PTQ 加在线删除，属动态/结构扩展；只前缀或固定删除是独立变体。 | 候选来自 5 万图、top-100 是 token；DINOv3 自带 register 且无原文实证，先验不能替代适用性验证。 |

### 3.2 敏感度、重构、搜索与补偿（20 行）

| 准确 ID 与原文材料入口 | 已有可复用基础 | 需要新增的组件/完整顺序 | 部署配置关键条件 | 未闭合项与声明边界 |
|---|---|---|---|---|
| [ptq-vit-2021](../../docs/methods/recipes/ptq-vit-2021.md#ptq-vit-2021) [综述 22] | 位点输出捕获、尺度候选、梯度/任务评价基础。 | 补 Pearson、pre-Softmax pairwise ranking、均值 bias、交替 scale 搜索；完整法另含核范数混精分配。 | 完整 mixed 6/8 bit 原法归混精；排序/相关性等可用于固定四位适配。 | γ、θ、候选数等最终设置及官方代码未核；核范数不是 Hessian，不能与 PTQ4ViT 混名。 |
| [apq-vit](../../docs/methods/recipes/apq-vit.md#apq-vit) [综述 17] | 梯度平方评分、位点/块观测、尺度搜索。 | 补 BBC 块级响应、逆层序交替搜索、候选相关 bottom mask、MPQ 的均匀 Softmax 网格。 | 静态四位可接入；MPQ 的 max 归约轴及参数冻结行为要明确。 | Eq.(8) 与文字的 mask 方向不一致，max 轴/零 scale 候选处理待闭合；不得静默修正成唯一原法。 |
| [aphq-vit](../../docs/methods/recipes/aphq-vit.md#aphq-vit) [综述 35] | 只读梯度捕获、AdaRound、teacher KL、结构观测。 | 补 APH 双扰动准备、加权损失、GELU→ReLU MLP 恢复、QDrop 块重构；MR 后刷新量化与敏感度。 | 离线恢复后可静态；原代码 patch 输入 A8 单列。结构变化另记成本，不自动排除主线。 | 全方向差分一般是 H1 非精确对角；paper 与 abs-code、p 值不同。只加 APH metric 不等于整法。 |
| [fima-q](../../docs/methods/recipes/fima-q.md#fima-q) [综述 34] | 梯度捕获、AdaRound、冻结多观察点评分。 | 补渐进扰动/梯度缓存、DPLR 小矩阵求解、QDrop 重构；论文 secant 与官方 abs-normalized 分开。 | 有限校准可主线；作者 A8 例外和 token/head 分组另存实际配置。 | rank、归一化及损失公式 paper/code 不同；直接 inv 的退化回退若修改需标变体，不能称精确 PSD Fisher。 |
| [erq-icml2024](../../docs/methods/recipes/erq-icml2024.md#erq-icml2024) [综述 37] | 已有 ERQ 研究卡、二阶矩/GPTQ/GPTAQ 基础、RepQ 变换。 | 补 Aqer ridge→二分 Wqer→舍入翻转→剩余 W ridge；统一误差符号、矩阵轴与统计归一化。 | 临时 FP 待量化列不留到推理，最终可主线；保持 2024 配方身份。 | 没有原生 ERQ 执行注册；2024 的 T=100、λ 和符号不能混入 2025。闭式下降不证明任务提升。 |
| [erq-tpami2025-duq](../../docs/methods/recipes/erq-tpami2025-duq.md#erq-tpami2025-duq) [综述 37；2025 变体] | 上一行 ERQ 底座、固定分组/量化存储基础。 | 先完成重参数化与 Aqer，再在该 W 状态上确定 DUQ 尾部列分区，接 Wqer 舍入与剩余权重修正；独立记录 T=20 和代码求解尺度。 | 两个分区均真四位且成员固定时可主线；另计索引、scale 与分组算子。 | 选列数原按 d_out 计算；空/单符号行边界、sum/mean λ、log√2 容量要核。是 [37] 变体，不是第 84 条文献。 |
| [is-vit](../../docs/methods/recipes/is-vit.md#is-vit) [综述 24] | repq_ln、uniform/log 表示、捕获和量化执行骨架。 | 补 SULQ 与 η 搜索；SOS 的 A-first 权重重构→尺度融合→W/A 重构，分阶段冻结 quantizer。 | 最终静态可主线；不能把 SOS 写成推理动态或联合学习全部尺度。 | SULQ 的整数指数、减 η、边界码表须实作；公开代码可核，但未执行，完整链不能仅用 RepQ 代替。 |
| [pfcr-vit](../../docs/methods/recipes/pfcr-vit.md#pfcr-vit) [综述 76] | 块观测、实际 shortcut/stage 元数据、原子量化基础。 | 补重构二叉分区树、先细后粗继承、POS A-first/W+A 两阶段、层级预算调度。 | 终态可静态；非 2 幂 block 数按原规则截层，不能擅自扩大到整网。 | 作者代码及精确 optimizer 参数集合未核；log 零码和首尾覆盖待审计，不擅自加入可学习 scale/zp。 |
| [mgrq](../../docs/methods/recipes/mgrq.md#mgrq) [综述 74] | 多 tap/多目标观测、RepQ、任务 logits。 | 补当前块优化器、FP-input 块损失、块内特征损失、完整 Q-suffix 全局损失的精确图。 | 原 patch/head W8A8 保留；无例外 W4A4 另为迁移。全局后缀不能换成 FP。 | 逐块 α/β 规则和更新参数明细未闭合；现有多 tap 打分不等于可微块重构。 |
| [qdrop](../../docs/methods/recipes/qdrop.md#qdrop) [综述 54] | AdaRound 舍入/正则/硬化、FP/Q 输入捕获。 | 补块输入和块内激活的随机 FP/Q 混合、A scale 优化、块调度；终态关闭随机 drop。 | 优化中混合合法，推理全量化后可静态；作者首尾 8bit 配置另存。 | mask 概率含义、README 覆盖默认值和首尾例外须锁定；原证据主为 CNN/NLP，ViT 采用后续明确 recipe。 |
| [bit-shrinking](../../docs/methods/recipes/bit-shrinking.md#bit-shrinking) [综述 33] | 局部误差、权重访问、最终网格表示。 | 补量化 sharpness、连续分辨率、二分选下一 bit、逐阶段局部权重重构及最终目标 bit 重构。 | 仅构造阶段变 bit，终态固定可主线；不是运行时动态位宽。 | 整套数据量、首尾例外和官方代码未闭合；ViT 原主要 6/8 bit，A4 是适配而非已验证结论。 |
| [outlier-aware-slicing](../../docs/methods/recipes/outlier-aware-slicing.md#outlier-aware-slicing) [综述 27] | 布局/传播诊断、AdaRound 与块捕获基础。 | 补 stage/downsample 子图解析、细单元/连续多 block 分区、QDrop 式子图重构。 | 分的是重构子图，最终静态；不是高精度离群通道旁路。 | supplementary 代码具体实现/全部覆盖例外未核；不能用模型名字代替真实边界或声称越粗必越好。 |
| [q-hyvit](../../docs/methods/recipes/q-hyvit.md#q-hyvit) [综述 120] | 输出梯度、离散尺度搜索、quantizer spec 容器。 | 补 hybrid bridge 边界、scheme/granularity 联合候选、整桥输出评分、zero-point 溢出审计。 | 分组固定可静态；原 W6A6/W8A8 与整数/FP 非线性两套配置分开，W4A4 需迁移。 | 完整 scheme 超参、整数实现和作者 release 未核；普通 DINO block 无原 bridge，不能硬套。 |
| [qep](../../docs/methods/recipes/qep.md#qep) [综述 99] | GPTQ/GPTAQ、配对输入、二阶矩和传播诊断。 | 补全矩阵连续 W 修正、独立 α/damping、然后调用选定 W 量化器；逐层刷新 Q-prefix。 | 原为 LLM weight-only；把输入含 A4 误差用于 ViT 后重新量化 W4，是主线迁移变体。 | 不能声称已验证 ViT W4A4；与 GPTAQ 三角逐列补偿不等价，QEP 原 damping 不能用 GPTQ 默认替代。 |
| [pd-quant](../../docs/methods/recipes/pd-quant.md#pd-quant) [综述 36] | teacher KL、AdaRound、配对块输入和任务 head。 | 补可微 FP-suffix 预测目标、BN 分布修正、两份输入及 QDrop 重构；明确局部/全局 loss 路径。 | 作者 CNN/BN 原配方与首尾 8bit 单列；无 BN 的 ViT 只用 PD 部分须标迁移。 | DINO LN 不能伪造 BN 训练统计；官方 PD 支路仍有 activation-drop，不应记为全确定量化前向。 |
| [cpt-v](../../docs/methods/recipes/cpt-v.md#cpt-v) [综述 113] | 冻结候选评价、teacher logits、FQ 量化基础。 | 补固定负样本 batch 的对比 fitness、block scale 向量变异、种群/tournament 状态。 | 最终 scale 固定；ViT 原 A8，A4 为新适配，完整模型候选 forward 计预算。 | 历史 CPT-V 官方 commit 未闭合，优先共享 Evol-Q 实现；保留独立版本，不能算两份独立算法证据。 |
| [evol-q](../../docs/methods/recipes/evol-q.md#evol-q) [综述 114] | FQ 组件、冻结评分、teacher 缓存，与 CPT-V 共享搜索骨架。 | 补有版本的进化 scale 配方、真实搜索覆盖/顺序/种群；选择集选模与最终评测分离。 | 最终静态；原 ViT 主要 A8，A4 适配需独立验证。 | 当前代码只改子 QLinear scale；扰动半径、种群初始化、验证集选模与论文不同，不能统一写全 W/A 搜索。 |
| [biascomp](../../docs/methods/recipes/biascomp.md#biascomp) [综述 110] | 配对输出、流式误差与 bias/常量状态基础。 | 补显式 reduce/preserve axes 的均值残差拟合及 shape-bound 修正；原 ViT 保留 token 坐标。 | 通道 bias 可融合主线；位置/BMM 常量需声明额外状态、形状与算子，按终图判断配置。 | 不能把 batch-only 平均换成全 token 平均冒充原法；逐模块顺序与 BMM 轴按有效配方核定。 |
| [qwt](../../docs/methods/recipes/qwt.md#qwt) [综述 15] | paired hooks、块边界、传播诊断。 | 补同一当前 Q-prefix 输入上的 local FP/Q 残差矩阵最小二乘、R² 接受规则、逐块补偿与刷新。 | 原保留 FP16 矩阵补偿分支，属于额外计算；计矩阵/输入存活/带宽，非纯 W4A4 同成本。 | 现 local_fp_parent 在改上游候选时不是所需 local_fp_candidate；ridge/低秩/full-FP target 都是另名变体。 |
| [qwt-v2](../../docs/methods/recipes/qwt-v2.md#qwt-v2) [综述 44] | Linear/Conv 输出观测、每通道统计、bias/scale 状态。 | 补每个 Linear/Conv 输出的 affine 拟合；再实现 multiplier 与 accumulator bias 的真实整数融合。 | 浮点 affine 路径先作补偿配置；确认融合、四位覆盖及终图后才可主线。 | 独立 v2 代码/后端融合未核；零方差、零/负 α、整数舍入须定义，不能引用“零开销”作实测结论。 |

### 3.3 无真实校准数据（5 行）

| 准确 ID 与原文材料入口 | 已有可复用基础 | 需要新增的组件/完整顺序 | 部署配置关键条件 | 未闭合项与声明边界 |
|---|---|---|---|---|
| [psaq-vit](../../docs/methods/recipes/psaq-vit.md#psaq-vit) [综述 79] | 图片加载/数据凭据、patch 布局、MinMax 数值基础。 | 补只优化图片的 patch-similarity KDE 熵、类别/TV 目标、增强与生成状态；生成后再校准。 | data-free 是来源属性；原 W4A8/W8A8，合成图可用于主线但 A4 效果另验。 | 代码 TV 是目标差且依赖 batch；教师分类头、模型/分辨率/seed 必须绑定，不继承 v2。 |
| [clamp-vit](../../docs/methods/recipes/clamp-vit.md#clamp-vit) [综述 38] | 候选误差评价、基础量化、图片入口。 | 补局部 patch 对比合成、周期刷新、进化 scale/bit 搜索和 fitness 缓存失效规则。 | 固定 bit 的 scale 搜索可主线适配；完整 2–8 bit 分配归 mixed。 | 公开入口主要是评估，完整生成/搜索代码未核；scale 扰动排版疑点未闭合，A4 不承接原 A8 结论。 |
| [dfq-vit](../../docs/methods/recipes/dfq-vit.md#dfq-vit) [综述 78] | 合成数据接口基础、配对激活误差、常量状态。 | 补 E2H 裁剪课程、图片生成及激活差均值矩阵校正；与 e2h-activation-correction 共一家族。 | 原 W3/4/6/8A8；E2H 可作主线来源，ACM 的位置状态/加法/表示按实际终图归属。 | ACM 不能直接简化为通道 bias；ACM 出版社最终版全文/官方代码未全核，A4 与变形状支持为新验证。 |
| [sardfq](../../docs/methods/recipes/sardfq.md#sardfq) [综述 85] | attention 观测、图片入口、有限重构数值基础。 | 补 attention prior、局部语义图像更新、软目标；以生成数据驱动独立有限块重构器。 | 原含 W4A4，可按实际覆盖进入主线；有限校准更新当前块量化模型参数（含权重），不等于无参数更新。 | 按正式 SARDFQ 锁版本、模型 prior 系数和有效命令；不能将早期 SPDFQ 或库中同名入口混用。 |
| [hc-ptq](../../docs/methods/recipes/hc-ptq.md#hc-ptq) [综述 39] | 权重访问、任务评价与通用成本记录；非现有 scalar quantizer。 | 补向量码本、双曲/切空间映射与聚类、索引序列化和真实存储计算。 | b 位是权重行向量索引，另存 FP32 码本；归独立 weight-codebook 比较，不是 WbAb。 | 原未量化 A、低位联合覆盖明显退化、官方代码未核；不能称成熟全 W4A4 或用索引位数代替标量位宽。 |

### 3.4 精度与格式分配（8 行）

| 准确 ID 与原文材料入口 | 已有可复用基础 | 需要新增的组件/完整顺序 | 部署配置关键条件 | 未闭合项与声明边界 |
|---|---|---|---|---|
| [amp-vit](../../docs/methods/recipes/amp-vit.md#amp-vit) [综述 68] | SmoothQuant/bias、GELU 网格基础、输出误差。 | 补 SymAlign 精确折叠、三区域 AutoScale/尺度比搜索、SQNR×log(numel) 贪心分配。 | 固定组件可 W4A4；完整分配属 mixed，默认 LN/Softmax FP 与 fully-integer 分开。 | 作者代码/完整搜索网格未核；区域与符号必须在总 bit 内，与 MPTQ 共家族而非两个独立算法。 |
| [mptq-vit](../../docs/methods/recipes/mptq-vit.md#mptq-vit) [综述 40] | 同 AMP 底座，复用网格、变换与候选接口。 | 保留 SQ-b/OPT-m/Greedy MP 的原版本绑定，逐码字比对与 AMP 的对应关系。 | 固定 W4A4 组件与 mixed 分配分别记录；6bit 示例的移位常量不能直接移植四位。 | 官方代码未核；核心高度重合，不另建近似同义求解器或重复统计独立有效方法。 |
| [lrp-qvit](../../docs/methods/recipes/lrp-qvit.md#lrp-qvit) [综述 67] | 只读梯度、RepQ-LN、任务输出与单层试验。 | 补 relevance 传播、正贡献聚合、层重要度/保护规则和预算位宽分配。 | 完整法属于 layer-mixed；clipping/评分可单独用于固定四位研究。 | 重要度原用 5 万训练图，量化校准才是 32 图；dense relevance 为新适配，作者代码未核。 |
| [mix-qvit](../../docs/methods/recipes/mix-qvit.md#mix-qvit) [综述 41] | RepQ/逐层试验/梯度基础；LRP 实现需先补。 | 仅接独立 PTQ：单层 bit 扰动损失矩阵、relevance 评分、IQP 大小/BitOps 约束。 | 完整 layer-mixed；固定 clipping 可主线，全部比较按实际资源与覆盖。 | 256 张 validation 敏感度集须按 search 用途记录并独立最终评价；仅保留已核 PTQ 分支。 |
| [lampq](../../docs/methods/recipes/lampq.md#lampq) [综述 115] | 梯度/Fisher 代理捕获、量化后重构误差。 | 补 type-aware 系数、加权位宽 ILP、量化后误差反馈与成对重分配。 | 完整法属 layer-mixed；固定四位只能借用评分，不能声称执行原分配。 | ±1 bit 两层参数量不同会改变加权预算，需每步重算；官方入口存在，但约束实现是否修复未核。 |
| [pomq-vit](../../docs/methods/recipes/pomq-vit.md#pomq-vit) [综述 69] | 候选评估、局部误差、设备 profiling 接口。 | 补 encoder 4/8bit 网格、三误差向量、设备延迟与 Pareto 排序/配置导出。 | 离线选择的 layer-mixed；不是按输入动态，也不是平均四位等于全四位。 | 作者代码、总校准预算、实际打包和计时细节未闭合；代理 latency 不能代替目标设备测量。 |
| [mixa-q](../../docs/methods/recipes/mixa-q.md#mixa-q) [综述 116] | RepQ、window 布局/均匀量化基础。 | 仅接 §4.3 PTQ：window L2 路由、双精度 gather/scatter、共享 W 与分支 LN/bias、比例进化搜索。 | 原 W4A8，低重要窗 A4，属 window-dynamic/mixed；不是静态全 A4。 | 全局 attention 的 DINO 无原 window 条件；1.35× 是 BOP 比率，官方完整代码未核。 |
| [patchwise-mpq-vit](../../docs/methods/recipes/patchwise-mpq-vit.md#patchwise-mpq-vit) [综述 119] | percentile、只读敏感度与任务评价。 | 补 GSM 层分配/Pareto、attention 驱动 patch 位宽路由及路由打包元数据。 | 输入相关 token-dynamic/mixed，不能借固定部署算法名称归 static。 | 原文 PTQ 身份已核；官方代码、实际 packing 和运行路由成本待闭合。 |

### 3.5 整数算子与专用结构部署（5 行）

| 准确 ID 与原文材料入口 | 已有可复用基础 | 需要新增的组件/完整顺序 | 部署配置关键条件 | 未闭合项与声明边界 |
|---|---|---|---|---|
| [practicaledge](../../docs/methods/recipes/practicaledge.md#practicaledge) [综述 26] | integer operator 接口、冻结 scale 状态和 profiling。 | 补范围受控整数 Softmax/LN/GELU、shift/dtype/溢出语义及真实 ARM 内核适配。 | 原 Softmax/GELU uint8、LN uint16、内部 int32；它是整数算子证据，非全 A4。 | 公开 kernel release 未核；ARM NEON 算子收益不能写成任意平台端到端加速。 |
| [iptq-vit](../../docs/methods/recipes/iptq-vit.md#iptq-vit) [综述 21] | SQNR/L2、operator 接口、成本测量基础。 | 补视觉范围 Poly-GELU、Bit-Softmax、按层近似选择与统一成本代理。 | 原主 W8A8/W4A8；integer-only 与主线四位覆盖分别判定。 | 锁原版本与后续 IPTQ++ 差别；预算附录/实际 kernels 未全闭合，功率比 20log 需保留原指标身份。 |
| [p2-vit](../../docs/methods/recipes/p2-vit.md#p2-vit) [综述 43] | PTF/PoT、SmoothQuant、梯度与量化基础。 | 补误差驱动 PoT 候选、PoT smoothing/requantization；完整法还需 Hessian/Pareto 与遗传 4/8 W 分配。 | 固定 PoT 组件可主线适配；完整 mixed 与 FPGA 配置单列，W4 不等于 A4。 | 官方量化入口可核，完整搜索/RTL 是否公开未闭合；专用硬件收益不能继承到参考 QDQ。 |
| [h-vit](../../docs/methods/recipes/h-vit.md#h-vit) [综述 45] | PTF/PoT、integer LN、token 表示基础。 | 补相邻 PoT 误差比较、裁剪搜索、LN 通道尺度统一与 token 输出参数。 | 原 W8A8/Attn4；输出尺度是否输入相关须据实际实现判断，固定 A4 是适配。 | Eq.(18) 印刷上下界有 2^b+1 码态；作者代码/勘误未核，禁止照抄成“真四位”。 |
| [trio-vit](../../docs/methods/recipes/trio-vit.md#trio-vit) [综述 46] | AdaRound、局部误差/梯度、图结构观测。 | 补 MBConv 迁移、filter-bias 修正、线性 attention 分母 log 编码及有限块重构。 | EfficientViT 专用结构；原 W8A8 加四位分母，不是普通 DINO Softmax 全 W4A4。 | 完整块迭代配置和专用 backend 尚需锁定；不能把 FP 拓扑差异或 FPGA 数字当通用 PTQ 收益。 |

## 4. 共享 primitive 与求解器：实现一次，配方保留差异

以下是建议的**数学组件族和职责划分**，不是已注册工具或要照抄的模块命名。每个组件都应暴露输入对象、更新范围、输出状态、随机性和下游失效关系。相同求解器可以复用，目标、参考和顺序不能因复用而合并成一个算法。

| 共享组件族 | 输入、输出与核心依赖 | 主要使用者及不可合并的差异 |
|---|---|---|
| 标量码本与实际编码 | 输入域、总 bit、scale/zp、零/符号/region 约定 → 整数 code、解码表、饱和行为及容量证明；先于参数搜索与导出。 | AdaLog、SULQ、TanQ、twin、ShiftLog2、三区域/双尺度、PoT。不能用相似 QDQ 曲线代替同一码本；向量码本另设表示。 |
| 分组、位置与结构布局 | 实际图、token/head/window 轴、固定分区或输入路由规则 → 成员映射、shape contract 与元数据；量化前先确定语义。 | ORQ 固定位置、ERQ-DUQ 固定列、IGQ 实例分组、ADFQ 旁路、MixA/patch 路由、RegCache 前缀。固定索引与在线分组不可互换。 |
| 等价仿射变换与折叠 | 输入边 scale/shift、LN/Linear/所有消费者、选定 W → 改写图、W/bias 及等价误差；随后重新拟合受影响量化状态。 | RepQ/RepQuant、MOSF、SymAlign、I&S、NoisyQuant、AdaLog。实数等价、QDQ 等价和最终整数融合是三个检查，不能互相代替。 |
| 配对捕获与充分统计 | 数据/视图、FP/Q 前缀、观察点及参考身份 → 样本对、均值/交叉矩、梯度准备材料；保留归约与 sum/mean 约定。 | ERQ、QEP/GPTAQ、补偿、APH/FIMA、所有重构。`fp_model`、同当前输入 local FP、固定 parent local FP 是不同参考。 |
| 敏感度与目标准备 | 明确 task head/loss、teacher、输出轴、扰动/temperature → 只读权重或低秩统计；候选评价阶段只消费准备结果。 | 梯度平方、APH、FIMA、LRP、LampQ、APQ。APH H1 不等于精确对角，FIMA paper/code 不等价，候选相关 mask 不能预先永久冻结。 |
| 参数候选搜索 | 明确网格/分区/预算、父模型、评分器 → 冻结参数、候选轨迹与实际 forward 数。 | PTQ4/APQ 交替搜索、AdaLog FPCS、TanQ、PoT、Q-HyViT。FPCS 保留多个好区域，不等于一次 grid；不同目标使用同候选时才能比较 selector。 |
| 闭式线性求解与补偿 | 对齐的二阶矩/残差、阻尼、误差方向 → 权重修正或常量/矩阵补偿，附秩和条件诊断。 | ERQ ridge、QEP 连续预修正、QwT 矩阵回归、QwT-v2 每通道 affine、BiasComp 均值。共享 solve/统计，但输出类型、参考与融合方式不同；不叠加所有补偿假装必然更好。 |
| 有限校准参数优化 | 声明可变参数、参考图、固定校准集、loss/正则/预算 → 更新权重/rounding/scale/clip/补偿及硬化结果。 | QDrop、APHQ、FIMA、AIQ、RepQuant、ORQ、DAQuant、SARDFQ。AdaRound 底座不等于已实现 block 优化；参数白名单与每阶段冻结范围由 recipe 决定。 |
| 重构范围与阶段调度 | 真实子图边界、前后阶段状态、停止/继承规则 → 可重载阶段产物和下游缓存刷新。 | I&S SOS、PFCR 树、OAS slicing、Bit-Shrinking、MGRQ、PD-Quant。输入 FP/Q、后缀 FP/Q、先 A 后 W、逐级 bit 都必须显式，不能泛化为同一个“block loss”。 |
| 黑盒与架构候选搜索 | 固定 fitness 数据/负样本、候选状态、预算 → 搜索轨迹/选中配置；数据刷新时适应度失效。 | CPT-V/Evol-Q、CLAMP；AIQ rank NAS 共享候选身份与预算设施，但它的可微搜索不是 tournament 的同义实现。 |
| 资源约束分配 | 每层/组候选误差、实际 cost、允许精度集合 → 可验证的配置向量和资源账本。 | greedy AMP/MPTQ、ILP IGQ/LampQ、IQP Mix-Q、Pareto PoMQ/patchwise、P2/MixA 搜索。求解器不同，约束都要对最终配置重新计算，不能仅凭平均 bit 名称。 |
| 合成数据与课程 | 冻结 teacher、输入/patch 布局、像素目标、seed/更新预算 → 图像、软目标、区域与生成轨迹。 | PSAQ、CLAMP、DFQ/QuaSC、SARDFQ。数据生成与 PTQ 拟合分开计费、分别替换；生成模型/目标改变后旧图不是“同一校准源”。 |
| 运行时路由与额外载荷 | 当前输入、已冻结策略、数据布局 → 组/窗口/稀疏位置、额外值与恢复映射；独立计运行成本。 | IGQ、ADFQ、MixA、patchwise、DAQuant、RegCache。已有 dynamic scale 容器不能代替这些算法；FP 值、索引、重排与高精度求和都需记账。 |
| 整数算子与目标后端 | 冻结输入尺度、近似系数、dtype/累加/溢出约定 → 整数图、再量化结果和后端 artifact。 | FQ、PracticalEdge、IPTQ、P2、H-ViT；Trio 还依赖特定线性 attention。先验证参考数值，再核目标 kernel；BOP、QDQ 时间与真实低位延迟分别报告。 |
| 完整候选状态与导出 | 权重、量化参数、布局、结构、补偿、来源及 recipe 状态 → 可独立重载、无需再校准的模型包。 | 全部完整方法。扩展 frozen 候选承载完整包；不能用只记 scale 改动的 receipt 代表改过权重或结构的模型。 |

**按依赖完成，不按论文逐篇复制系统。** 先落实 A 包的协议/数据身份及真实图覆盖；B 包先补编码、布局、配对捕获、候选状态，再让 C 包的目标准备和 B 包的搜索/求解/重构共同使用。结构变换在受影响统计前，A 变换在最终 W 重构前，依赖 decoded-W 的偏置在最终 W 后。随后按表装配完整 recipe，并将可重载结果交给 D/E 的研究闭环。原配方明确采用 FP-prefix parallel calibration 时应保留；这不是要求所有算法强制改成顺序校准。

执行器应拒绝数学前提不成立的组合，例如缺 BN 却请求原 PD-DC、全局 attention 却请求原 MixA window 配方、token 布局变化却读取旧 ORQ 位置尺度。它不应把尚未测过但数学上成立的 PTQ 组合一概拒绝；这类组合使用适配身份和可比较实验。

## 5. 发生变更后，哪些状态必须重估

下面的“重估”以具体依赖为边界。**候选只改下游 Q 状态，并不自动使完全独立的 FP teacher 或上游原始输入失效。** 可保留的是具有相同数据、模型、轴和参考指纹的原始材料；已经依赖旧目标/旧前缀算出的拟合结果、排序或成本不能直接继承。

| 变更 | 必须重算/失效的状态 | 可以保留的条件与顺序 |
|---|---|---|
| 校准样本、数据域、crop/增强或抽样权重改变 | observer、clipping、统计矩、敏感度、补偿、依赖它们的拟合与搜索结果；更新数据用途/预算。 | 未变化样本的原始 teacher 输出可按确切输入哈希复用；不能把 token 数当新图像数。DAQuant 还需重建 source/target 身份和配对。 |
| batch 组成、顺序或第一批改变 | CPT/Evol/CLAMP 的对比 fitness、batch 归约、ORQ first-batch 初始化；代码中依赖 batch 的 TV/损失尺度。 | 纯逐样本充分统计可重聚合；对比负样本、BN 统计和 batch-dependent 操作不能用逐图缓存假装不变。 |
| 预训练 backbone 或 FP teacher 改变 | FP 目标、Q-prefix、梯度/曲率、权重修正、合成图教师身份及全部依赖拟合。 | 只有数据解码和与模型无关的几何记录可直接沿用；原校准结果不能跨模型冒称同源。 |
| task head、loss、temperature、双任务权重改变 | head 变化重算其 logits；loss/temperature 变化重算实际依赖它们的概率、梯度/敏感度、拟合与评分；组合权重变化重算组合目标和排名。 | 模型/head/输入不变时，原始未温度化 logits 可复用；独立单任务分量和任务无关的 backbone 激活/矩按真实依赖保留。数据视图也变则需重新捕获。 |
| 分辨率、window、padding、特殊 token 或 dense 几何改变 | 位置尺度、补偿张量、prefix/删除恢复、attention 轴、特征对齐和实际成本。 | 固定位置状态必须匹配 shape contract；不能按同一索引猜语义。共享尺度只在其原定义允许时复用。 |
| LN/Linear 的 scale/shift 等价改写 | 被改写 W 的网格、局部坐标统计和相应 Q 下游缓存；新 W 方法所用矩也需与新坐标对齐。 | 若验证 FP 等价且观察定义相同，可保留全模型 FP teacher；不能因 FP 等价就继承旧 W 量化结果。 |
| W、舍入或 W scale 变化 | decoded-W shift 补偿、该模块 Q 输出、下游 Q-prefix/补偿/输入矩和任务敏感度中依赖 Q 图的部分。 | Noisy/AdaLog 补偿必须用最终解码 W 重算；纯 FP 参考不随 Q 权重变化。后一个 W 求解器覆盖前者时记录初始化关系。 |
| A 码本、bit、范围、zero、分组或 clipping 变化 | 该边实际 Q 输入、输出误差、candidate mask、ERQ/QEP 等交叉矩、重构与下游拟合。 | 原始 FP 输入可重编码复用；固定 parent 与当前 candidate 的局部参考不可混用。 |
| GELU→ReLU、前缀插入、token 删除或其它结构变化 | 图覆盖、边界/布局、observer、重构缓存、敏感度、成本和导出结构。 | 原结构 FP teacher 可作为明确的恢复目标保存；它与改后结构的 local FP 不再同一参考。先结构后校准。 |
| 静态分区、组数或选择列集合改变 | 分组参数/索引、切片二阶矩、预算与算子数；受分区影响的搜索/重构解。 | 可由相同原始 X/W 重切片，无需无条件重跑全部 FP；ERQ-DUQ 分区需绑定重参数化后的 W。 |
| metric 公式、观察点、归约或参考路径改变 | 旧评分/排名/selector regret；新目标所需梯度/低秩准备；若用于拟合则相应最优参数需重求。 | 只改可由原始 paired tensors 计算的 forward metric 时重评分即可；不必无故重校准。APQ mask 仍随每个候选重算。 |
| 搜索空间、资源目标或成本模型改变 | 候选可行性、ILP/IQP/Pareto 解、排序/选中结果与总预算。 | 数值参数与原输入不变的已测候选可复用确切输出；新 budget 不能沿用旧赢家而声称已重新求解。 |
| 重构/补偿顺序或某阶段结果改变 | 受影响阶段后的 Q-prefix、阶段参考、补偿及优化状态；stage checkpoint 的父身份。 | 只保留不依赖被改阶段的状态。parallel FP 校准按原法保留，顺序 Q 校准按当前前缀刷新，不能混合两者。 |
| 合成图刷新、生成 seed/teacher/prior 改变 | 数据身份及所有在旧合成集上评估的 fitness/拟合；新的图像目标与预算记录。 | 可将旧图显式作为另一数据源对照，不能沿用原适应度缓存；固定种子的相同材料可重放。 |
| 后端、算子近似、dtype、融合或导出格式改变 | 真实执行误差、溢出/饱和、bit/元数据核算、kernel 与端到端成本；受新数值影响的下游统计。 | 纯无损 packing 的原数学参数可保留；QwT-v2 整数融合、LUT/PoT 近似不能只凭 FP 等价跳过数值检查。 |

缓存键至少绑定数据与视图、模型/阶段父状态、参考图、观察点/轴、实际 quantizer、任务/目标准备和版本。哪些字段确实影响某份缓存由宿主自动推导；不要求研究角色每轮手填全部信息。失效后可重算或明确拒绝继续，不能静默退回 MSE、RTN、FP 或较少样本。

## 6. 完成标准：组件完成、recipe 完成、论文复现分别验收

### 6.1 共享组件数值就绪

一个组件达到这一级，需要实际实现而不是空注册：输入/输出/更新参数明确；支持的轴、码本、形状与数值边界有代表性的小张量验证；已声明的编码可 pack/unpack/reload；随机性与费用可记录；失败时给出真实原因。闭式求解需检查残差与秩/阻尼行为，等价变换检查所有消费者，四位码本检查完整可表示集合而非只看校准样本 unique 数。

验证针对具体风险。例如，ERQ 测误差符号和 sum/mean λ，QwT 测同当前输入参考，ADFQ 测异常值真的进入 FP 计算且计费，Noisy/AdaLog 测 W 改变后补偿被刷新。不要为每个论文名字复制相同的浅测试，也不要用一个恒等/RTN 占位实现通过测试。

### 6.2 完整原配方接通

完整 recipe 另需同时具备：已锁定的论文或官方代码版本、有效超参/命令覆盖、数据/标签用途、真实 W/A/算子覆盖、全部阶段及顺序、每阶段可变参数、参考路径、最终硬化/冻结、例外配置和可独立重载的完整产物。所有必要阶段都必须运行；只实现 FIMA metric、MOSF、E2H 生成或 QDrop 输入混合，分别不等于完整 FIMA、DopQ、DFQ/QuaSC 或 QDrop。

依据充分原文独立实现也能达到“recipe 实现完成”，不强制依赖作者开源。但若某个具体原法字段仍未闭合，则该原法保留字段级阻塞；已经有充分证据的组件或明确适配可以完成，不能让整族永久停留在等待，也不能把自选默认填成作者原值。论文/代码差异若影响数值，分别给身份；选择只实现其中一个版本时明确写清覆盖，不能声称两者都复现。

### 6.3 低成本集成验证通过（五包交付的可执行验收）

在小模型/小校准子集或缩短迭代的实际任务切片上，验证真实执行链、只更新声明对象、父子状态与依赖刷新、完整导出重载、实际协议归属和至少一次“候选产生→冻结比较→结果解释→经验记录”的科研闭环。动态/混精/补偿/合成各取代表性路径验证基础设施，其余完整 recipes 仍需各自至少跑通支持配置的有界 smoke；不能用共享底座通过替代所有配方装配检查。

数据不具备、外部全文缺失、目标架构不存在等阻塞写成具体可定位缺口；不得下载任意同名替代、制造虚拟分数、静默缩覆盖或跳过关键阶段。小预算运行的输出标为集成验证，其任务分数只能说明该次小预算行为。完整昂贵 benchmark、全面调参和所有目标模型长跑不作为这一步的隐性前提。

### 6.4 进一步可以声明什么

| 声明 | 还需要的证据 |
|---|---|
| 完整原法已复现 | 在所声明作者配置、模型、数据、预算与覆盖下运行，核对实际有效参数及差异，并提供任务结果；有论文分数或 smoke 都不够。 |
| QAViT 主协议适配已验证 | 改动列表、真四位/静态覆盖与重载检查，加所声明任务上的实际结果；不能继承原 A8/mixed 的精度。 |
| 同一 backbone 支持双任务 | 两任务加载同一量化 backbone 状态，head/视图与 FP 对照固定；分别校准的两个包不能这样表述。 |
| 原生低位/整数部署有效 | 真实目标图、kernel、设备配置、算子/端到端测量及额外载荷；fake QDQ、BOP 或别人的 FPGA 测量不够。 |

接入报告应分别统计来源可用、组件完成、完整 recipe 接通、smoke 通过、原法复现、目标适配验证、后端实测。可以用宿主已有字段承接，不必把这些名词原样变成新的 capability 枚举。不得统一写一个“方法已支持”掩盖不同层级。

## 7. 其余账本条目的路由

### 7.1 五条来源未完备记录

这些记录已经有 PTQ 身份证据，但不足以签发完整算法配方。本表不继续推测其机制或默认值；后续只补原始材料，不借相似论文拼装“原法”。

| 准确 ID | 综述编号 | 已有入口与下一项来源工作 |
|---|---:|---|
| `fgptq_vit` | 98 | [来源卡](sources/fgptq_vit.md#fgptq_vit)：取得 Springer 正文方法、码字规则与实验设置，再核作者实现。 |
| `uniq-vit` | 25 | [来源卡](sources/uniq-vit.md#uniq-vit)：取得出版全文/补充材料；已见仓库 README 不是算法源码。 |
| `mrospu` | 75 | [来源卡](sources/mrospu.md#mrospu)：取得正式章节算法与设置，核对同题作者 release；不借其他方法补 loss/搜索参数。 |
| `token-dynamic-bit-width` | 42 | [来源卡](sources/token-dynamic-bit-width.md#token-dynamic-bit-width)：补 §3.2–3.3、§4.1、完整精度覆盖与实现；摘要不能生成执行 preset。 |
| `optimal-numerical-format` | 87 | [来源卡](sources/optimal-numerical-format.md#optimal-numerical-format)：取得原文正文、格式候选/选择规则和校准配置，才能确定实际 recipe。 |

### 7.2 版本、旁证与排除

| 账本类别 | 数量与 ID | 接入路由 |
|---|---|---|
| `family_version` | 1：`e2h-activation-correction` | 作为 `dfq-vit` / QuaSC 的文献版本，保留 [版本入口](../../docs/methods/recipes/dfq-vit.md#e2h-activation-correction)；最终版未读细节不反填预印本，不增一个独立实现。 |
| ERQ 附属版本 | `erq-tpami2025-duq` | 已在第 3 节 52 行中单列，仍隶属 [37]；不改变 83 条主文献计数。 |
| `operator_context` | 1：`approx-softmax-layernorm-accelerator` | 仅 [无训练算子旁证](sources/approx-softmax-layernorm-accelerator.md#approx-softmax-layernorm-accelerator)，不冒称已验证 ViT PTQ 原法或额外主方法。 |
| `excluded_qat` | 14：`mnq_vit`、`mimiq`、`psaq-vit-v2`、`binaryvit`、`bivit`、`ofq`、`q-vit`、`quantformer`、`tervit`、`variation-aware-vit`、`auto-vit-acc`、`i-vit`、`packqvit`、`vaqf` | 只保留 source_coverage 与筛查/勘误中的身份及排除依据；不录入算法、primitive、metric、案例或 Skill。 |
| `excluded_non_ptq` | 6：`lra-qvit`、`quanttune`、`elsa`、`n-m-sparse-transformer-accelerator`、`vitality`、`vitcod` | 只保留 [筛查路由](source_coverage.json) 与原始依据，不进入执行或研究内容。 |
| `unresolved_boundary` | 5：`eff-dfqt`、`binaryformer`、`heat`、`iqgvsaa`、`pf2a-vit` | 训练边界未决，暂不录入方法内容；只有独立 PTQ 证据闭合后才重新审定。 |

AMP/MPTQ、CPT-V/Evol-Q 在 51 条主记录中各保留两个来源 ID，是因为账本按综述文献记录计数；实现应共享其家族基础并锁版本，不把重复名称用于增加独立算法或有效性证据的数量。Mix-QViT/MixA-Q 只按已核实的独立 PTQ 分支接入。

**全账本验收口径**：83 条主记录各有唯一去向；其中 51 条进入本表的 PTQ 材料映射、5 条来源待补、1 条家族版本、1 条算子旁证、25 条排除或边界未决。ERQ2025 作为附属版本另有一行。此处验收的是覆盖与可追溯性；实现数量、运行成功数和研究有效性分别依第 6 节的事实报告。
