# DINOv3-B 四轮连续科研启动准备

准备阶段完成了一个全量双任务 B 的测量和独立重建，以及两段同主题、同灵感的真实角色研究。四轮正式研究尚未启动。下表是可比较的起点，后续方法增益与系统贡献需由正式运行及独立对照测量。

## 基线与协议

| 模型状态 | ImageNet Top-1 | ImageNet Top-5 | ADE mIoU | ADE mAcc | ADE aAcc |
|---|---:|---:|---:|---:|---:|
| 原 FP | 79.128% | 95.220% | 52.8578% | 64.8621% | 84.3351% |
| static W4A4 准备 B | 56.748% | 82.928% | 29.6848% | 37.7705% | 72.4708% |

正式范围为 50,000 张分类图及 2,000 张分割图。B 相对 FP 的 Top-1 和 mIoU 差距分别为 22.380 和 23.1730 个百分点。证据 ID 为 `EvidenceCard-05ba26ee5f474c0d8fe109239209e505@1`，来源标记为 `preparation`。[双任务原始回执](baseline_partition/baseline_receipt.json)

分类使用 DINOv3-B/16 LVD-1689M 原权重和已核验的归一化 CLS 冻结线性头，224 中心裁剪、短边 256、bicubic。校准与 search 分别取 ImageNet train 的 32 和 1,000 个固定 ID，final 取 val 的 50,000 个 ID，三者无交集。ADE20K 使用原始训练好的 MultiLevelNeck 和 UPerHead，读取第 2、5、8、11 个 block 的 patch 特征，窗口 512、步长 341，完整 validation 2,000 张。FP 适配器与独立分割骨干在首窗四组特征的相对 MSE 均为零。资产、检查点、头和预处理的 SHA 可单独核对。[绑定与测量记录](RESULTS.json)

一个骨干覆盖 49 组 W4 主权重及输入 A4，加上 12 层 Q、K、V、P 的 48 个边界，合计 146 项。权重、分组、scale 和 zero 在任务迁移前冻结。分类头和分割 Neck/Head 保持 FP。输入明确禁止此次训练式优化，包括研究参考 STE；有其他明确许可的运行仍可声明其参考优化范围。[完整起点](baseline_plan.json)

同协议下此前缺少已测 DINO B。准备阶段以 8 张 calibration、固定 24 张 search 及 ADE 前两张，实际检查了 GPTAQ 适配、RepQ 适配和 group-32 RTN / log2-P 对照。三条分类 Top-1 分别为 4.17%、0% 和 0%。在第三条已冻结状态上仅启用 W4 得到 18/24，同时启用 W4A4 得到 0/24，完全旁路得到与 FP 相同的 19/24。根据这项对照及历史 DINO 中前缀与 patch 分布不同的记录，准备了一个固定区间起点。小样本得到 11/24、ADE mIoU 19.58%，随后只为该方案进行了正式规模测量。历史案例使用过 ADE 校准、不同分区和更多优化预算，保留作方法来源。以上方案及选择属于工程准备。[起点构建代码](seed.py) · [历史来源](../../../QAViT/knowledge/research-skills/research-routes/ptq4dino.md)

最终起点对 Linear 权重使用 32 输入通道一组的静态 RTN，patch 卷积权重按输出通道做 minmax，P 使用留零的 log2。Linear 输入和 Q/K/V 对实际 CLS、storage、patch 三种固定区间分开拟合静态 affine 网格。每个特征坐标从预先声明的分位数中按校准 MSE 选择，迁移时只有 patch 尾段长度随图变化。完整 plan 记载其他位点、精度、网格和校准顺序。两次从原 FP 和同一 32 个校准 ID 独立构建的全部参数及 buffer 摘要相同，另经冷加载检查。两个 head 未参与拟合。计算走 FP32 reference QDQ；此次验证覆盖冻结精度和重建，未测原生 INT4 内核速度。[独立重建回执](independent_rebuild/rebuild_check.json)

## 组件改动与真实科研闭环

- Generation 的继续实验条件包含能改变判断、候选或实现的数值发展。H1/H2 引用灵感及登记吸收部分保留来源，普通父方案改良不会抢占 H3 的活跃记录。H3 与共同决策可继续、修复、调参或切换。主题、复议与原有综合修订、独立 B、三候选 L3 继续用原工作流。[提示词逐行差异](prompts.diff)
- 原生 v3 可以在 prepare 后通过 `refine_frozen` 在声明的校准视图中发展参数。导出读取实际装入模型的 W/A 和仿射状态，独立重载核验状态及同图输出。历史源绑定同时传递运行源码及完整归档身份。内置分组与目标的支持范围写入接口，其他科学机制仍可沿 Coding、Safeguard 和自定义执行器实现。[接口说明](../../../QAViT/docs/CODE_INTERFACES.md)
- 模型上下文直接给出实际 token、层和位点、资产、数据池和固定小样本清单入口。已完整交付的接口原文按 SHA 引用。尺度比较保留逐样本得分，分类保存每张图真值和 FP/Q top-5。分割只在源模型与头、源码/环境、预处理、按序图像及标注字节等身份一致时复用原 FP 混淆矩阵。一张真实 ADE 图命中后逐格相等，改为两张后重新实测。量化侧始终运行完整指定视图。[复用回执](fp_reference_check/receipt.json)
- 准备 B 从独立程序测量的原 Journal 导入。沿原 Evidence 验证器核对 L3、同协议、训练声明、冻结与重载、全量成绩、完整包及源码来源。重复导入幂等；后续任务可用 solution ID 走原读取接口。准备路径没有候选、综合记录或科研轮次。[接续检查](prepared_handoff/handoff_check.json)

第一次局部任务的研究方向、实验实现需求和反馈后的决定均由实际 Generation 作出。它选择检查 blocks.5 的 P 网格，对比 PV 重建与概率 MSE 目标。首份 Coding 有标量聚合类型错误，Safeguard 指出确切触发位置，修订后才执行。32 张校准和 64 张 search 的首批证据引发第二个实验，补上有限网格穷举及五个随机对照。在其余骨干 FP 的背景中，Generation 将原来的三类分组简化为 prefix/patch 两组，选中的逐键码器映射相同。Recorder 保留了原随机对照容量不完全匹配等限制。

重新打开同一 Journal 后，Generation 恢复主题的下一版本及同一活跃灵感，读取上一份确切源码和配置。新 Coding 继承了上一实验的代码身份。在冻结的完整 AdaLog 父状态下，Generation 改用了三个等容量随机双组和未用于前测的 64 张 search 图。语义双组的新增 PV NMSE 为 0.00826616，三个随机双组为 0.00807328–0.00809509。Generation 据此撤回总体语义优势解释，只保留可回退的单层候选。Recorder 区分了 128 个读取 ID 与 64 个实际前向评分 ID，记录了旧 FP 背景、不一致的代理指标和可变序列长度尚未验证。该角色候选没有正式精度，也没有进入 B。本次工程准备的完整起点、角色自主提出的局部研究和文献启发分别署明来源。[原始角色记录](roles_direct/) · [源码、配置和证据身份](source_index.json)

程序回归共 772 项，CPU 环境 770 项通过、2 项 CUDA 专用检查跳过；在 V100 上重跑相应模块的 5 项全部通过。29 项针对性回归覆盖非单位 W/A 倍率进入导出包与实评、冻结后的独立加载和禁止再校准、三条确切版本分别规划 L3、候选优先 GPU 调度和纯数值等待分账。真实 DeiT-B 也重走了模型/特征一致性、局部 probe、尺度搜索、冻结传播、分类与分割子集。没有为验收启动 DINO 或 DeiT 的三候选正式 L3。H1/H2 的来源和权限由程序回归及既有历史覆盖，本次真实局部回放重点是 H3。[CPU](cpu_verified.log) · [CUDA](cuda.log) · [专项](contracts.log) · [DeiT](deit/summary.json)

局部验证还留存了工程失败。分组草案曾因负轴不符合原生 schema 被拒绝，未产生数值结果。隔离源码目录被既有资产边界拒绝，真实角色改从本仓库运行。准备 B 的跨运行加载排查中，直接新建的后端遗漏协议身份，随后一次脱离运行段的规划也被宿主拒绝；按同一冻结 Brief 解析 execution 并建立程序运行段后，正常入口成功。检查没有放宽身份或预算校验。[工程脚本](handoff.py) · [验证回执](prepared_handoff/handoff_check.json)

## 成本与启动

两个组件共记录 49 次 LLM 执行尝试，48 次进入模型调用，47 次完成。一个进行中的调用因宿主合入已测的工程修改而中断，另一次官方账户预检返回 RPC 错误，未启动模型调用。原始原因保留于 Journal；重新进行官方只读预检成功后，沿原任务、同一订阅及模型路由恢复。已知输入 3,260,078 tokens（其中缓存输入 84,864），输出 76,469（其中推理 12,143）。订阅不折算成未核实的金额。三次数值实测 worker 墙钟分别为 48.2、55.0、1,215.1 秒。四个已闭合运行段合计 87.81 分钟，其中纯数值等待 22.98 分钟，扣除后 Agent 活跃预算约 64.83 分钟。离线 Eval-L0 冻结 14 个角色任务、核验 2,649 个来源文件，未另行调用评分 LLM。[角色成本](../../evaluations/dinov3b_preparation_20261118_r1/reports/report-0001.md)

全量 B 在两张 V100 上分开执行分类和分割。实测数值 worker 墙钟 79.58 分钟、CPU 7,054 秒。基线程序运行段 80.04 分钟，纯数值等待 79.97 分钟。另一次从原始状态重建及核对用时 17.67 分钟。四个小样本起点各自的耗时、失败及低分原件见 RESULTS 和原始目录。双卡租期包含分割任务先结束后的空闲时间。[基线成本](../../evaluations/dinov3b_prepared_baseline_20261118_r1/reports/report-0001.md)

正式输入给四轮预留 **12 小时 Agent 活跃预算**，每项 L3 沿用 24 小时数值上限。按此次约 80 分钟的基线和候选优先并行推算，可先安排 **18–24 小时日历窗口**；它包含三个候选各自的校准、任务内并行度、完整评测和可能的修订。新机制的校准成本和订阅窗口会改变用时。GPU 型号/设备不同会使 FP 身份缓存重新核验。保留预算边界和原始账本，未设置最低实验数、LLM 调用限额或小样本正式评测。

从本仓库执行以下命令。名称由入口生成时间戳，后续恢复沿同一运行目录。

```bash
PATH=/root/.local/bin:$PATH python -B QAViT/run.py start --config EXP/inputs/dinov3b_static_w4a4_gonly_next4/input.json
```

交付输入设四轮，三候选分别按综合后的确切版本进入既定双任务 L3。GPU 可用池与实验并发由宿主实际探测，优先让不同候选各获一张卡，Coding 串行。Generation 路由为 gpt-6-astra/high；Coding 和 Safeguard 为 gpt-6.1-sol/medium；Recorder 为 gpt-6-luna/high，Supervisor 为 gpt-6-luna/low。订阅 CLI 执行前逐次预检。此次 CLI 没有回显实际模型/effort 的调用在审计中保留为空。原始 LLM 请求、回答、工具反馈、修订、失败和外部工程干预保存在组件运行 Journal。Git 中另存四份原始 Coding 源码、实际执行参数及可回查的 SHA。启动文件引用本工作区保留的完整数值目录；迁移到只有 Git 文件的机器时，需提供原资产和该目录，或用 baseline.py 从 baseline_plan.json 重建，并绑定新产生的真实证据 ID。[启动输入](../../inputs/dinov3b_static_w4a4_gonly_next4/input.json) · [证据索引](source_index.json)
