# QAViT G-only v1 两轮真实科研贯通实验

科研运行及独立 Eval-L0 已结束；原始轨迹和本报告均保留在本轮独立 EXP 目录。

两轮 H1/H2/H3、真实 L1/L2、反证后的假设修订与预选、完整双任务 L3、Recorder 和跨轮交接均已贯通，托管进程最终以 0 退出并清理资源。按 Top-1/mIoU 领先的已测候选 AdaLog 的 ImageNet Top-1 为 **72.482%**、ADE mIoU 为 **28.055648**，仍分别低于 FP **9.314 / 11.773935** 个百分点。此次支持 G-only 的自主研究决策与证据利用能力，但有一次外部工程修复，不能称为完全无外援自主；新算法 H3 未实测，不能声称已验证创新算法收益。没有配对 Full 对照，无法将结果或成本差异因果归于架构简化。

## 实验身份与口径

- Run：`deitb_static_w4a4_gonly_20261115_r1`，独立目录 `EXP/runs/deitb_static_w4a4_gonly_20261115_r1/`。
- 分支：`ablation/gonly`。初始 Commit `b6fd75e2689e15add23611005639bcbd5f250cc6`；一次外部工程修复后 `42cbdd293fd2c39e19ca9e7fcc2a157d9a4f935f`。未向 main 写入。
- 官方非蒸馏 DeiT-B；training-free、FP32 reference QDQ、Static W4A4；每个固定组至多 16 个码字，零计入。主体 W/A、Attention Q/K/V/P、post-GELU 均按冻结覆盖协议，分类 head 和 ADE neck/head 保持浮点并记录。
- ImageNet：calibration 32，固定 search pool 1,000，正式 val 50,000；ADE20K val 2,000。ADE 直接迁移分类校准状态，原冻结 Neck/UPerHead；无训练、任务头更新或 search/final/ADE 参数拟合。
- 两轮，每轮 H1/H2/H3；Generation `gpt-6-astra/high`，Coding/Safeguard `gpt-6.1-sol/medium`，Recorder `gpt-6-luna/high`。无 Judger/Evolution/H4 或对辩阶段。
- 活跃预算 10,800 秒，`active_segments`、`exclude_numeric_wait=true`。L1/L2/L3 独立超时 3,600/10,800/86,400 秒；暂停后同 run 恢复，无预算重置或延期。
- 初始可用 5 张 V100 16GB，自动资源池；独立任务按可用资源调度。实际并行度以回执为准，不把 5 张可用卡当作 5 路并行。
- 外部监控请求为 `gpt-6.1-sol-ultra`；本会话可见元数据为 `gpt-6.1-sol`、effort `max`，实际服务端身份未回显。科研 CLI 的 requested 与 actual 同样分列，actual 缺失不补写请求值。

固定配置 SHA256（宿主规范化 JSON）：`06676cdf3e11127510c6fc1d9ed627858034ef99e981816cdafdbb2fbc3dc29e`。ImageNet manifest：`7e560abfe98169fd72597ec87defd8bf12faadebdb3fb278bdbe9865ce400f4c`；DeiT 权重：`b5f2ef4d686982dcdab24fe285fd08fff40db01550d8d4833167a73dd85ca7a8`；ADE head：`408c1c49568dfa0b6e500e23d0a9185501642fccb85e934830ffa90e697730b5`。完整配置、来源和启动前源码身份见 `input.json`、`experiment_identity.json`、`PSEUDOCODE.md`。

## 轻量预检与证据边界

当前 G-only 的 12 项 CPU 回归通过；实跑一个分类 calibration 图像前向、一个 ADE 浮点图像，三个完整 GPTAQ/AdaLog/RepQ 方案通过 97 sites / 146 boundaries 覆盖检查。数值源码与此前已完成的 RepQ 小规模 PTQ 链路比对一致，没有重跑其耗时验证，也没有把历史测试算作本次测试。

默认 CLI 已升级到未审阅的 0.169.0，预检在调用模型前拒绝。配置显式固定已安装且原实现认可的 0.167.0，SHA256 `cdcb45aaa01d0f14e298985e09465aebccd47f64ecd298aea44814af7b62bbda`；未扩展白名单或降低模型配置。失败及通过的预检日志均保留。

这些验证仅证明所测接口/样本可用，不证明正式精度。当前完整研究结果只由有效 L3 回执给出。

## 科研轨迹：机制、实验与修订

| 轮次 / 候选 | 机制与实际行动 | 证据边界与决策 |
|---|---|---|
| R1 H1 | 完整原 GPTAQ v1 种子；双流输入差异补偿，提出固定网格、补偿置零对照 | 没有执行该机制消融；作为完整可执行储备被预选，不能据此证明补偿因果贡献 |
| R1 H2 | GPTAQ 配套概率尺度从 P-MSE 改为 PV-MSE；真实 L1，P-only、其余 FP、blocks 0/5/11 | 32 cal + 64 search，无重合；PV 目标改善局部 PV 误差，但排列 V 控制也有收益，因而收窄几何解释 |
| R1 H3 | 固定概率码本上的行内误差反馈，给出行和与 PV 均值/中心化分解、顺序敏感性 | 理论提案，未编码或实测；没有把未验证写成负结果，也没有首创或性能提升声明 |
| R2 H1 | 完整原 AdaLog reserve-zero 种子；局部输出重建搜索 base/scale，FC2 decoded-weight 补偿 | 提出固定 base 与可搜索 base 的配对消融，但未执行；最终按原完整方案规划 L3 |
| R2 H2 | 从已发布的确切 GPTAQ 包复制状态，在真实量化前缀、A4 V 上复测三个 P 位点 | 一次真实 L2 发现中后层总误差反向，候选缩到 block 0；随后新留出 256 张整网 L2 反证，未升级为 L3 候选 |
| R2 H3 | 从误差反馈改为显式静态行质量惩罚及两遍坐标下降，给出排列平均代理的适用假设 | 新算法仍未实现；有限扫描、非全局最优、真实 key 几何与迁移局限写明 |

R1 的 L1（`EvidenceCard-cecf07bdab8c47d48027bd02a932160c@1`）在同一次执行中得到：

| 位点 | P 目标的 search PV NMSE | PV 目标的 search PV NMSE |
|---|---:|---:|
| block 0 | 0.092553 | 0.062411 |
| block 5 | 0.029961 | 0.019564 |
| block 11 | 0.016184 | 0.004052 |

P 自身 NMSE 可变差；排列 V 控制也改善。该结果不支持“真实 key-value 对应结构是唯一机制”，更不等于完整 W4A4 或任务精度改善。Generation 预选 H1 时明确依据完整性和可执行性，而非已测优胜。

R1 L3 后 Recorder 完成交接，R2 H2 使用前轮建议，以同一个冻结 GPTAQ 包、实际量化前缀和 A4 V 执行 L2（`EvidenceCard-646077308373429fb07082f8d3e0e3d5@1`）：

| 位点 | 新增 PV NMSE：seed → PV 目标 | 相对原始 FP 的总 PV NMSE：seed → PV 目标 |
|---|---:|---:|
| block 0 | 0.136511 → 0.076147 | 0.291650 → 0.266457 |
| block 5 | 0.027430 → 0.015457 | 0.471752 → 0.491409 |
| block 11 | 0.024938 → 0.002769 | 0.844751 → 0.937468 |

这是对“局部新增误差下降即可改善 FP 对齐”的直接反证。模型利用交叉项解释不利误差抵消，只保留 block 0。32 cal 与 64 search 沿用前轮子集，不能称作独立留出复现。

Generation 的第二轮预选又主动请求整网 L2：新 search 区间 `[64,320)`，256 张，无 calibration 或前 64 张身份重合；固定既有尺度，不在新子集拟合。复用冻结参数，不计新的 calibration 拟合；迭代器读取前缀但未前向/计分的范围在原回执中披露。

| 冻结整网方案 | Top-1 / 正确数 | KL(FP∥Q) | logit NMSE |
|---|---:|---:|---:|
| FP | 96.09375% / 246 | — | — |
| 原 GPTAQ | 89.453125% / 229 | 0.338560 | 1.620021 |
| H2 首层 PV 尺度 | 87.5% / 224 | 0.348641 | 1.597608 |
| mass 尺度控制 | 91.015625% / 233 | 0.331876 | 1.613981 |

H2 的 logit NMSE 虽改善，KL 和分类正确数却退化；Generation 未以代理改善宣称任务收益，改选 H1 AdaLog。mass 是同次实验的控制臂，不是另一个已验证新假设；256 张及少量净正确数差异不支持正式性能结论。

两次 H3 均有可证伪定义、关键最近邻/阈值对照和代数边界，但没有真实新编码器实验；本轮不能据此证明创新方法的数值效力。H1 两轮也都没有完成其专属机制消融。当前支持的是自主构思、实验设计和利用反证的能力，不能把六张卡片本身当作六项实证成果。

## 正式双任务 L3

仅正式 50,000 / 2,000 张与宿主独立重载验证通过的结果计入下表。

| 方案 | ImageNet Top-1 | Top-5 | ADE mIoU | mAcc | aAcc |
|---|---:|---:|---:|---:|---:|
| 同协议 FP 对照 | 81.796 | 95.594 | 39.829582 | 50.606542 | 78.753181 |
| R1 完整 GPTAQ | 72.274 | 91.752 | 4.418857 | 7.779772 | 34.459588 |
| R2 完整 AdaLog | 72.482 | 90.294 | 28.055648 | 37.788110 | 71.391383 |

R1 ImageNet 下降 9.522 个百分点；ADE mIoU 大幅下降是有效的负结果。浮点 ADE 同条件结果存在于 `segmentation.json`，不是“无 FP 对照”。Evidence 摘要只展开 Q 指标，Recorder 未主动从已授权原始产物读取 FP，这是可用性/阅读缺口。

R1 Evidence：`EvidenceCard-15a015a9c39044759caf9a9edd7a0c22@1`；目录 `jobs/task-b82d0e9c4c38354f3e31a579-a0001/`，实际结果位于 `result.json#/data`。宿主记录 `host_verified=true`、`passed=true`、`evaluated_instance=reloaded`，一张 search 样本的 `torch.equal` parity 通过；此范围不等于全部样本逐图 parity。不可用导出 manifest 的不可变 `exported_pending_validation` 字段推翻后续有效回执。

R2 Evidence：`EvidenceCard-e643fc1b1e154c17a7553394f97281f1@1`；目录 `jobs/task-97cee009a3d280bd3aa744d2-a0001/`。同样由宿主核验独立重载的一张 search 图像后，以 reloaded 实例完成 50k/2k，分类及分割的原模型、冻结量化状态、任务头保护通过。相对 R1，Top-1 +0.208、Top-5 −1.458、ADE mIoU +23.636790 个百分点。分类与分割方向并不一致，两个方案的组件差异也未受控，不能以该差值证明单个机制。

reference QDQ 延迟包含浮点模拟与框架开销，不代表原生 INT4 加速。两轮完整方案改变多个组件，整网差值不能只归因于 AdaLog 底数搜索或 GPTAQ 单一补偿项。

## 工程故障、内外部恢复与有效自主性

1. R1 L1 初版 Coding 缺少显式 calibration split，内部修订后经过 Safeguard 和真实执行成功。
2. R2 新留出 L2 首次触发 `Model must remain eval`：冻结包中的量化 wrapper training flags 与根模块状态不一致。内部 Coding 在冻结数值状态和 signature 不变的前提下归一化 eval，并修正 FP 副本安装 attention 后的模式；重审、重跑成功。失败执行约 38.35 秒，不计 Evidence。
3. R2 原生 AdaLog 完整校准/导出后，宿主以 `AdaLog LUT 与 base/bit 不一致` 拒绝重载。此次失败耗时 1,159.337 秒，无正式任务指标。CPU/CUDA 的 table2 除法舍入存在最多 1 ULP 差异，原来严格逐值比较误拒；q=19/55 可复现，q=37/74 一致；q=1…256、bits=4 共 4,096 项的实测最大差为 1 ULP。
4. 唯一外部核心修复只调整 `QAViT/tools/ptq_agent/execution/adalog.py` 的 table2 参数验证：保留 table1 精确比较，table2 只接受同形状、同 dtype FP32/FP64 的精确相同或最多一个相邻浮点值；2 ULP、畸形/非有限/错误 base 仍拒绝。`parameters/fit/apply/bias_transform` AST 均与启动 Commit 一致；不改已保存 LUT、拟合、解码或推理。
5. 同一次干预先安全停止托管 segment，确认状态保存和清理，然后在同 run 恢复。恢复前累计活跃 5,087.348939 秒、剩余 5,712.651061 秒、已排除数值等待 7,163.278715 秒，`extensions=[]`；未重置或延长 10,800 秒。
6. Generation 在外部暂停前已发起自己的 canonical CPU LUT 方案。内部修订保留原 seed 和候选网格，以 CPU FP32 构造离散 LUT 后原值传到执行设备，显式记录新实现版本；没有篡改后端身份。其 revision1 重载后又因 custom audit 没有 `static_w4a4` 收据失败。内部再次规划→Coding→Safeguard，revision2 从真实冻结量化器构建完整收据，最终进入正式计分。不能把两种修复的重叠归为“完全无外援”或只归功于外部修复。

外部修复回归：新增 5 项测试通过，包含真实 CPU↔CUDA LUT 运输、保存的 CUDA LUT fixture、2 ULP 拒绝、完整 tiny 模型 CPU 导出→GPU 重载及输出；原有升级测试 15 项通过，覆盖三种完整种子 prepare/export/reload 和严格 16/legacy17 码本语义。新增 tiny fixture 的两次设置错误及原始失败日志保留；不把它们计入科研实验失败或独立数值证据。未声称重新执行历史全部 625 项测试。

只读 CPU 核对原始失败包与内部 revision1、revision2：展开 plan 相同，540 个保存张量全部逐值相同，payload 元数据无差异，前后文件 hash 不变。该核对支持冻结数值状态未变，不替代完整任务指标或所有图像 parity。

记录：`intervention_001.json/.patch`、`resume_001_launch.json/.log`、`adalog_lut_before/after.json`、`adalog_transport_final.log`、`adalog_upgrade_regression.log`、`adalog_package_comparison*.json`。初始 core/numerical SHA256 为 `8c76550850c9f3a29bb4898784b75c9048f0f4ea22abe304dfc72ad3885869da` / `f00b614ad7b4d408a2468113fd83d25ee31f7c5e73725a2ebf6ca05842d60ce4`；修复后为 `cb04d86178919244b9215ff4b2c9ef05100454f1e992c1315827c3d59aa220dd` / `466d4782742036f6c62992e33b9fdf3bd1489f013ac753ed80538e5ac34cb794`。

## 组件的实际使用和问题

| 组件 | 实际记录 | 贡献及局限 |
|---|---|---|
| Generation | 12 个任务、45 次调用：6 个 generate、2 个 select、4 个 plan_l3 | 自主发起 3 个探索性实验服务请求，利用中后层反例收窄 H2，再以新的整网 search 反证停止推进；两轮均选择完整 H1 储备，专属 H1 消融和 H3 实验未执行 |
| Skill | 目录交付给 Generation 45 次调用、Recorder 4 次调用；Skill 正文/参考读取 **0** | `design-discriminative-experiment` 等四项可发现，但没有实际调用证据。实际实验有判别性，不可归功于未读取的 Skill；不把可选 Skill 未调用自动当作质量零分 |
| 方法手册 | 6 篇、13 次成功读取：AdaLog 4、AdaLog bias 1、GPTAQ 3、log2 1、log√2 3、scale selection 1 | 支持完整种子公式、概率/FC2 机制与零码约束；未独立核验全文论文或首创性 |
| 方案库 | 方案说明/索引 4 篇 11 读；运行内 3 个冻结方案资产共 7 读；两轮有效 L3 均发布 | 最终 Top-3 为已测 AdaLog、已测 GPTAQ、待测 GPTAQ seed；同语义 GPTAQ 占两个位置，AdaLog/RepQ 待测 seed 曾被挤出，身份去重和多样性有改进空间 |
| Memory / Recorder | 5 次真实 retrieve_memory，28 次成功 read_card；两次 Metareview，3 条 L2 observed_once 知识，均无待归档科学批次 | R2 明确利用 R1 指导和证据；没有把同次执行当独立重复。两次 Recorder 均未补读已有 FP ADE 原件；R2 声明已解决的 PV 前缀问题仍出现在最终 L1 open_questions 中，开放问题投影未同步清理 |
| Coding / Safeguard | 9 / 8 个任务、42 / 28 次调用 | 保留科学需求并通过原生/自定义路径真实执行，内部修复多项接口问题。曾放过 eval flags 和缺少 custom 静态审计回执；这些由宿主保护捕获，不能以 Safeguard 自批准代替正确性证明 |

全部 150 次 `read_asset` 中 148 成功、2 拒绝；29 次 `read_card` 中 28 成功、1 字段错误；5 次 memory 查询均成功。两个资产拒绝分别是不存在于授权表的方案 hash 路径和未授权的 baseline 源码读取，随后使用正常接口恢复；没有证据将它们判作错误拒绝。没有实际文献检索/外部论文读取，原方法来源和未核验范围由候选自行注明。

Recorder 的 FP ADE 缺口并非原始实验缺少 FP，而是交付摘要未展开、角色未进一步读取。最终 Memory 中仍写“迁移损失未知”，而本报告从原始回执读得实际损失。`audit_run.py` 的 `pending_issues` 保留 8 个历史失败任务，并不表示终态还有运行任务；终态活动队列为空，pending scientific/knowledge 均为 0。结构审计的 `semantic_review_required=true` 也不等于科学质量自动通过。

## 效率与可靠性

31 个 LLM 角色任务共 119 次原子模型调用，全部启动并得到完整响应，无调用失败、缺失用量或孤立执行。2 个 select 任务无现成 rubric，仍纳入科研成本。

| 角色 | 任务 / 调用 | 输入 Token | 输出 Token | cached input | Router 累计秒 |
|---|---:|---:|---:|---:|---:|
| Generation | 12 / 45 | 2,447,293 | 47,507 | 79,744 | 2,005.045 |
| Coding | 9 / 42 | 3,977,112 | 55,211 | 86,784 | 1,673.863 |
| Safeguard | 8 / 28 | 2,525,572 | 7,412 | 61,824 | 452.054 |
| Recorder | 2 / 4 | 244,699 | 7,291 | 71,424 | 166.346 |
| 合计 | 31 / 119 | 9,194,676 | 117,421 | 299,776 | 4,297.309 |

输出中已包含 30,699 reasoning output Token，不再加一次；cached input 亦单列。以上来自 CLI 的已报告用量，不是订阅账单。实际模型和 effort 未回显，不能用 requested 值充当已核实 actual。Router 累计包含预检和订阅锁等待，记录的模型启动区间并集为 3,963.377 秒，均不能标为纯推理。Coding+Safeguard 占约 70.7% 输入 Token，说明移除科研角色并未消除实现和接口上下文成本；没有配对对照，不能声称净节省多少。

预算闭合：Agent 活跃 **6,135.015 秒 / 102.250 分钟**；符合原规则的数值等待 **14,041.374 秒 / 234.023 分钟**；两个托管 segment 墙钟并集 **20,176.389 秒 / 336.273 分钟**；中间安全暂停约 **857.932 秒 / 14.299 分钟**。从首 segment 启动到最终关闭约 **350.572 分钟**。预算上限始终 10,800 秒，剩余 4,664.985 秒，因两轮目标完成而结束；`extensions=[]`。统计采用已闭合 segment，不用事后 report 的 as_of 时间扩张墙钟。

| 数值执行 | 级别 | 结果 | worker 墙钟秒 |
|---|---|---|---:|
| R1 H2 FP 前缀 P/PV 对照 | L1 | 成功 | 77.792 |
| R1 完整 GPTAQ | L3 | 成功，50k+2k | 5,586.749 |
| R2 H2 冻结前缀诊断 | L2 | 成功 | 70.832 |
| R2 新 search 整网初次 | L2 | eval flags 失败 | 38.349 |
| R2 新 search 整网修订 | L2 | 成功 | 160.143 |
| R2 原生 AdaLog | L3 | LUT 运输验证误拒 | 1,159.337 |
| R2 canonical revision1 | L3 | custom audit 收据缺失 | 1,386.854 |
| R2 canonical revision2 | L3 | 成功，50k+2k | 5,447.387 |

7 个实验服务请求（3 个模型工具请求、4 个由 plan_l3 派发）产生 8 次数值执行；5 次完成对应 5 份真实 Evidence，无缓存复用。worker 墙钟合计 **13,927.443 秒 / 232.124 分钟**，其中失败 **2,584.540 秒 / 43.076 分钟**，占约 **18.6%**。预算排除数值等待比 worker 计算边界略宽，二者不能相加或冒充同一计时。

实际数值 worker 峰值并发 **1**；首个 L1 使用 GPU 2，其余使用 GPU 0。没有已发起的独立数值任务排队等待 GPU；混合双任务/自定义执行器回执说明未启用分类分片，不能把可用 5 GPU 宣称为并行加速。真实执行没有超时/OOM，失败由上述具体接口原因解释；保留的失败任务不擦除。分类同步前向计时（batch=1，warmup=2，5 次均值）R1 Q/FP 为 41.385/7.826 ms、R2 为 42.110/7.844 ms，排除磁盘/H2D，属于 reference QDQ。

## 独立 Eval-L0 与最终归档

独立目录：`EXP/evaluations/deitb_static_w4a4_gonly_20261115_r1_l0/`；冻结 Journal sequence 7,928，head `e92be184e5f62fa0a15e0742a0d007e474c210ab05c9d6193085a42afa8242e1`。原 15 文件评分包验证通过，fingerprint `476c85fd7f2dc11f6718b9189db11b997b31b40abedea0ca92155a1387e257de`。29 个现成可评分任务，独立 Judge `gpt-6.1-sol/medium`，每项最多两次尝试、300 秒、原默认 600,000 字符上限；2 个 select 无适用 rubric，保留未评分。不会为填满覆盖而截断原件或伪造分数。

评价结果为 **12 / 29 个任务有效、17 个 input_limit**，任务覆盖 **41.38%**。17 个上限事件未调用模型，没有截断、重试或合成分数；Judge 命令退出码 2 表示覆盖不完整，不是科研运行失败。

| role / purpose / slot | 有效 / 准备任务 | Q 均值 / 中位数 | 已评项 / 适用项 / 总项 |
|---|---:|---:|---:|
| Generation / generate / H1 | 2 / 2 | 100 / 100 | 10 / 10 / 10 |
| Generation / generate / H2 | 1 / 2 | 100 / 100 | 5 / 5 / 5 |
| Generation / generate / H3 | 2 / 2 | 100 / 100 | 10 / 10 / 10 |
| Generation / plan_l3 | 2 / 4 | 100 / 100 | 6 / 6 / 6 |
| Coding / code | 3 / 9 | 96.67 / 100 | 12 / 13 / 15 |
| Safeguard / safeguard | 2 / 8 | 100 / 100 | 8 / 8 / 8 |
| Recorder / record | 0 / 2 | 未评分 | — |

有效案例内部共 51 项 scored、52 项 applicable、54 项 total，1 unknown、2 N/A；**17 个整任务未评分不在这些分母中**。未评分包括第二轮 H2、两个 Recorder 和大部分复杂 Coding/Safeguard 修订，存在明显覆盖选择性。有效部分无 critical flag，不能据此断言全流程无严重问题。评分认可机制、对照设计和证据诚实性；唯一非满分案例的核心问题是手算码本检查没有直接覆盖实际 QDQ 输出。高分和明显天花板效应仍需用户今后人工校准；它们不证明新算法有效，也不推翻实际工程失败或 FP 性能损失。

两个 `select` 没有 v0.1 rubric，未硬套 Judger 的 Full 对辩标准；本报告仍根据原始选择与反馈分析其决策。独立调用只隔离上下文和科研工具，并非模型族统计独立或专家真值。

Judge 单独开销：**12 次调用、12 次成功、0 重试**；input **1,419,299**，output **13,912**，cached input **5,248**，reasoning output **1,142**（包含在 output 内）；Router 累计 **412.037 秒**，模型区间并集 **383.395 秒**。订阅货币成本未知。最终 [Eval-L0 报告](../../evaluations/deitb_static_w4a4_gonly_20261115_r1_l0/reports/report-0003.md) 保留逐条评分、来源指针、未评分和边界。

`evaluation verify` 通过 29 案例 / 9,897 源文件；另对原运行当前字节与冻结源 hash 全量比对，**0 变化**。Journal 停在同一 sequence/head。完整 LLM request/response、CLI 原始输出、代码、失败回执和数值原件均保留，不用后来的结果回填质量 Judge 当时的可见知识。

外部监控另有一份本地会话账本，截至 `2026-11-15T11:21:54.360Z`：累计 input **134,348,144**（其中 cached **133,800,576**）、output **248,956**（其中 reasoning **149,583**）。这是多次上下文重放的累计计数，不是唯一材料量，也不能与子字段重复相加；不含该时点之后的归档/最终回复。它包含预检、外部修复、监控和报告工作，与科研 119 次调用、Judge 12 次调用分开。虽然正常监视器本身每 30 秒采样且不调用 LLM，外部交互的总 Token 仍很大，不能称“整个过程低 Token”；下一轮应进一步减少无状态变化时的模型唤醒。实际账单、纯推理延迟和服务端实际模型身份均未知，原计数见 `external_monitor_usage.json`。

## 真实验证、工程测试、推断与未决

| 类型 | 本轮可以支持的结论 | 不能据此声称 |
|---|---|---|
| 真实科学数值 | 1 个 L1、2 个 L2、2 个有效全量双任务 L3；5 个不同 Evidence execution | 同次多臂是独立重复；局部/小 search 是正式精度 |
| 预检与工程测试 | 当前 12 个 CPU 回归；15 个原生适配 CPU 测试；5 个 CPU/CUDA LUT 与 tiny 真实模型重载测试；保存包 540 张量 CPU 等值核对 | tiny/smoke 提供 ImageNet/ADE 性能；参数等值证明全样本 parity |
| 理论与事后解释 | H3 代数与固定码本算法定义、PV 总误差的交叉项解释、对实际失败的源码诊断 | 首创性、跨任务机制因果效应、原生 INT4 加速 |
| 未决 | H1 专属消融、H3 数值效力、独立重复、FP 差距的主导来源、Full-vs-G-only 因果比较、17 个未评分任务及 select rubric | 未测=零精度、未评=零质量，或形式完成=科研质量已充分验证 |

## 下一轮以科研效果为导向的优化

1. **先定位任务损失，再增加方法复杂度。** 以匹配源码快照重载两个已测冻结包，在既定 calibration/search 内做 W、普通 A、P、FC2 的分组 FP 替换，配对比较 KL、CLS/patch 特征及累计误差。重点解释 GPTAQ 的 ADE 崩塌和 AdaLog 剩余的 11.77 点 mIoU 差距；不在 ADE/final 上拟合或选择参数，不绕过源身份验证。
2. **让未验证的新机制获得低成本实证机会。** 对 H3 固定实际码本，在同一个冻结前缀先做原编码、最近邻、λ=0、calibration λ 的局部对照，报告均值/中心化/交叉项和扫描顺序；用新留出 search 检验整网预测。H2 已显示代理失配，下一轮应优先检验这种失配而非重复正面局部统计。上述是 Generation 可选择的科学路径，不新增强制实验门禁。
3. **把工程浪费释放给科研。** 复用完整 baseline 的 prepare/export/load/audit 接口和真实冻结收据；将明确的实现合同错误送回 Coding，避免 Generation 重规划后反复整网校准。保留已有极小 CPU↔CUDA 运输回归，不新增耗时全量预检；只对身份一致且完整的已校准产物探索安全复用。此次两个失败 L3 已损失约 42.44 分钟。
4. **修正信息利用和重复上下文成本。** 在既有摘要中结构化交付 FP/Q 成对任务指标；正确关闭已解决 L1 问题；按语义 recipe 处理 seed/已测 wrapper 身份。将大型样本 ID 集和相同长 scope 保留在不可变原件，用 hash/count 与精确读取入口传递，避免 Recorder、Memory 和 Judge 重复展开。Skill 保持按需，改进决策点的入口价值，不能为提高调用率强制读表。
5. **改进评价和对照，而非追求满分。** 后续经独立审阅再给 Eval-L0 做可核验的重复前缀/大列表去重，不删关键反证、不改原评分包或回写科研历史；补充 G-only select 的适用标准，再做人类校准。对 Full/G-only 进行同源码、模型路由、数据、3 小时活跃预算与硬件条件的配对重复，分别报告科学结果、失败和全部三份成本账本，才讨论简化带来的收益。

## 产物与 Git 归档

| 内容 | 定位 |
|---|---|
| 本次冻结输入、身份、伪代码 | [input.json](input.json)、[experiment_identity.json](experiment_identity.json)、[PSEUDOCODE.md](PSEUDOCODE.md) |
| 机器可读结果与可重算分析 | [RESULTS.json](RESULTS.json)、[analyze_run.py](analyze_run.py)；完整派生账本 `ANALYSIS.json` 留在本地 |
| 原科研轨迹、Memory、代码和数值 | [Run](../../runs/deitb_static_w4a4_gonly_20261115_r1/research_run.json)、[终态](../../runs/deitb_static_w4a4_gonly_20261115_r1/summary.json)、同目录 `commits/`、`cards/`、`jobs/`、`code/`、`recording_submissions/` |
| 外部最小修复与恢复 | [intervention_001.json](intervention_001.json)、[源码补丁](intervention_001.patch)、本目录的恢复/回归/LUT/包等值原件 |
| 独立评分与只读核验 | [Eval manifest](../../evaluations/deitb_static_w4a4_gonly_20261115_r1_l0/manifest.json)、[最终评分报告](../../evaluations/deitb_static_w4a4_gonly_20261115_r1_l0/reports/report-0003.md)、`eval_source_isolation.json` |
| 完整文件清单和最终 Git 回执 | `ARTIFACTS.json`、`archive_receipt.json`；保留本地，按路径/长度/SHA256 核验，不将大模型和原始日志加入 Git |

托管退出观察另存于 [managed_terminal.json](managed_terminal.json)：原始 launcher stdout、Journal 的 processes_stopped 以及监控者对托管 summary 的直接观察分列。官方 `run.py report` 会重新生成派生 summary，不再包含 launcher 专有的清理字段；这不影响不可变 Journal 或已完成的评测。

Git 只接纳本次精简交付、冻结配置/身份、分析入口和干预记录；原始科学/评价目录保持完整并按原规则忽略。main 仍为 `8800006b687c56205ad624bae2ad7156c326d873`，未覆盖历史 EXP、数据或 checkpoint，也未发布/合并到 main。

须保留的边界：actual 模型未回显；订阅成本无法由 Token 直接换算；router 累计历时不是纯推理或墙钟；cached input/reasoning output 不重复加总；没有配对重复的 Full-vs-G-only 因果比较；无本次原生 INT4 速度证明；未执行实验、理论推导和事后推断分别标明。
