# QAViT 科研导向简化报告

本轮将 Recorder 的全量来源覆盖、强制逐项审计、身份复述与执行镜像移交程序；LLM 集中提交科研判断。保留现有 V2 角色、两阶段选择、辩论结案、证据时序、预算及真实数值验证。没有启动新的真实 LLM/GPU 贯通实验，没有修改历史科研结果。

## 已确认的瓶颈

先阅读项目约定、实际工作流、角色提示词、工具、Memory、最新两次贯通报告及轨迹，再形成 [PLAN.md](PLAN.md)。

| 历史运行 | 已发生的结果 | 本轮处理范围 |
|---|---|---|
| [20261030](../../outputs/deitb_static_w4a4_closed_loop_20261030_r1/REPORT.md) | 40 次模型调用，Experiment/Evidence/Meta 均为 0；外部 account/read RPC 阻断。 | 不将外部故障误当科研失败，不改预算或重启该运行。 |
| [20261031](../../runs/deitb_static_w4a4_two_rounds_20261031_r1/REPORT.md) | 297 次模型调用，17,811,466 输入 / 420,585 输出 token；首轮四场比较及 H4 已执行，完整两轮为 0/2。 | 原始数值与选择完整保留；重构阻断跨轮知识交接的程序协议。 |

后一运行的 Recorder 共 8 次调用，消耗 2,692,167 输入 / 47,047 输出 token，仍没有 Meta/L2。155 项来源、90 项工程覆盖、必审判断清单、精确 ID 和局部修复要求导致有价值的科研分析无法发布。历史修复已解决数据接口、超时重复执行、错误定位及工程日志误识别等问题，本轮保留其数值与可信性边界。

实测经验进入新提示词：49 个独立算子的局部 NMSE 改善不等于整网改善；同一诊断中 H1 的 KL 下降但 logit NMSE 上升。校准集内拟合与诊断不能外推为分类性能或泛化收益。

## 实现变化

**Recorder 与 Memory。** `scientific_analysis`、`generation_guidance`、`next_questions` 是三个必填科研字段。知识、重点判断分析、指标适用性、失败模式和盲点可按需补充。删除新提交的 `reviewed_refs`、完整覆盖复述和强制 `judgment_audit` 穷举；保留可选的、有来源的重点分析。

程序按不可变版本自动生成批次和 `source_refs`、绑定 `recording_batch_id`、维护索引/去重/游标/原子发布。`analysis_scope` 明确“批次科研综合，不是逐条认可”。新知识 ID 可自动生成；仅在真实 Evidence 的范围完全相同时精确复制 scope。伪证据、未知引用、越权/未来证据、范围扩大、缓存冒充重复验证、无 supersedes 覆盖旧知识仍拒绝。

L0 完整事实、L1 当前状态、L2 有范围/版本的知识、L3 科研综合均保留。Recorder 默认收到假设、实测、冻结定义、裁决/选择/解释与工程失败摘要；成功工程日志和全部原件可按索引深入读取。标量统计和科学条件不作任意截断；必要长正文仍须完整分页交付。Generation 获得任务绑定的前轮指导，Evolution 获固定父版本及相关批评，Judger 可直接取得当前证据，不必等 Recorder。

**失败恢复。** 事实已归档与科研复盘完成分别标记。非关键 Recorder 失败保留 `scientific_review=pending`，不造 Meta、不推进综合游标；下一轮携带复盘缺口和原始事实，下一批自动合并未综合来源。收尾不对同一失败批次无限重试，显式重试/澄清回答仍可恢复原任务。报告分别给出 `workflow_completed`、`scientific_review_complete`、`formal_validation_success`；存储故障、硬截止和外部传输故障继续按原边界处理。

**调用与交接。** 程序补固定 round/slot/type/phase/side、父引用、预测编号和唯一绑定的反馈版本；显式冲突拒绝。短句柄 `C数字` 用于结构化卡片引用及读取，进入持久卡片前还原完整 ID，并执行原权限/快照检查。查询仍按权限过滤，不暴露私有记录。一次提交复用固定读取快照，避免每个来源都重读完整模型请求。

实验仍只有 `request_experiment` 一个入口。可省略确定性的 scope、resources.cpu_threads、接口/重复标记及 resolved_choices；程序从实际参数生成镜像，不能猜方法、位宽、符号、粒度、采样或科学范围。已有量化计划可用显式 `layers[].modules` 一次表达多个同配置位点，随后展开为原生计划并验证。已有方法直接执行；真正新增方法仍保留 Coding → 独立 Reference → 实际数值/接口验证 → Safeguard。

直接交付的 [EXPERIMENT_INPUTS.md](../../../QAViT/docs/EXPERIMENT_INPUTS.md) 和 [CODE_INTERFACES.md](../../../QAViT/docs/CODE_INTERFACES.md) 替代反复读取整本执行文档。修复了原始 result.json 被较早 TaskCard 归属遮蔽的问题：只有生产任务、实验身份、Evidence 输出与哈希相匹配的测量原件随 Evidence 开放；候选源码、独立参考和同回合私有来源继续隔离。

**科研讨论。** 提示词强调可证伪预测、控制变量、能改变判断的小实验、回应对方最强论据，以及局部代理指标/整网误差/最终任务性能的区别。证据足够时明确选择或提前结案，缺乏可比较依据时允许有理由的未决。没有增加评分体系、强制实验数量或审核层级。初始三场依次结案 → 固定 H* → H4 仅挑战 H* → 父子最终选择保持不变。

## 验证与实测比较

全部新增材料位于本目录，测试临时文件仍使用 EXP/cache/tmp。角色脚本夹具均明确标记，不冒充真实智能体科研。

| 验证 | 结果与证据 |
|---|---|
| 修改前完整回归 | 289 项通过，[baseline_tests.log](baseline_tests.log)。 |
| 最终完整回归 | 297 项通过，304.699 秒；[full_final.log](full_final.log)。新增 8 项针对性验证，其余沿既有功能和可信性边界回归。 |
| 四服务 CPU 集成 | 实际执行 profile、tensor_probe、quantize_evaluate、evaluate_artifact；4 张 synthetic_computation Evidence，4 份原始 result.json 可读；缓存无新 worker，量化包重载后评测，Recorder→Generation 连通。[结果](cpu_services_final/INTEGRATION_RESULT.json)。 |
| 历史 Recorder 回放 | 隔离复制原 Journal，原草稿错误引用仍拒绝；显式移除旧覆盖/强制审计字段后，保留的分析、知识和指导等字段原文不改，1 次脚本提交绑定 155 项来源、发布 2 条 L2 知识；下一轮 Generation 收到指导并检索到这 2 条知识。[结果](REPLAY_RESULT.json)、[转换清单](replay_migration.json)。 |
| 新边界 | 三字段提交、短引用、固定身份冲突、未来/私有来源隔离、范围不扩大、延后复盘、下一轮原始证据、后续补做、显式重试、同批不重复调用、分组计划。 |
| 数值与独立性 | 回归包含独立 FP32 参考、错误舍入/非有限数拒绝、实际数据接口、冻结范围与参数、覆盖审计、导出重载、L3 结果协议、同回合隔离和预算恢复。 |
| 结构与保护 | 130 个 Python 文件的语法/导入及必要资产路径检查通过；64 个已快照的数值源码/配置内容相同；42,015 个历史文件大小/纳秒 mtime 无变化、无缺失（未重算全部大型资产哈希）。详细结果见 [structure_final.json](structure_final.json)、[protection.json](protection.json)。 |

旧逐项审计块包含逐项 reasoning，本次没有将其作为新结论重新发布；原回执完整保留。提交体减少 71.5% 是改变职责后的体积差，不是整份旧响应的无损压缩。

历史回放是**协议转换与确定性交接验证**，没有重新运行原实验，也没有让新 LLM 重新推理；不能把它写成真实两轮科研成功。

| 同口径测量 | 修改前 | 修改后 |
|---|---:|---:|
| 同一历史 Recorder 任务重建请求（canonical 字符） | 593,924 | 396,025，减少 33.3% |
| 该任务强制文件读取项 | 2 | 0；原件仍可按需读 |
| Recorder 基础必填字段 | 10，另有覆盖/必审约束 | 3 个科研字段 |
| 保存草稿的提交体（canonical 字符） | 17,865 | 5,093，减少 71.5%，所保留科研字段不改 |
| Recorder 组装系统提示词字符 | 6,006 | 3,325，减少 44.6% |
| 14 类任务组装提示词字符总和 | 56,772 | 45,363，减少 20.1% |

[上下文基线](context_before.json)、[新上下文](context_after_final.json)、[复杂度记录](complexity.json) 可复核这些数字。没有以字符数冒充 token：本轮实际 LLM 调用为 0，未测得新版本真实 token、费用或科研成功率。历史 8 次 Recorder 调用失败与转换后 1 次脚本提交成功并非同类随机实验，不能据此声称模型调用节省 87.5%。已有 Judger 结果直返机制继续保留，未把此前已经删除的 interpret 调用算作本轮收益。

45 个文件发生源码/提示词/文档/测试变更。修改到的生产 Python 文件合计净增 231 行，提示词 Markdown 减少 74 行；详细口径见 complexity.json。程序源码行数没有整体下降：明确身份编译、范围补齐、延后恢复和原件权限需要宿主代码；无新增角色、卡片类型、调度阶段、数据库或外部框架。减少的是 LLM 协议职责与机械输出。输入构建仍须验证大量历史卡片：本次测得约 59 秒，基线约 52 秒，不能声称宿主性能变快。

## 结构、限制与后续实验

核心只保留代码、稳定配置、必要依赖、可复用测试和当前文档；历史说明从 README 收敛到 EXP 报告入口。没有发现可在不损害复现的前提下直接删除的数值后端/模型资产，因此保留实际依赖的计划 v2/v3、历史卡片读取和旧知识版本解析。完整变更见 [changes.patch](changes.patch)、[prompts.diff](prompts.diff)，主流程见 [PSEUDOCODE.md](PSEUDOCODE.md)。

剩余问题：真实模型是否能稳定提出更有价值的实验、短引用是否降低实际修复率、跨两轮结论质量及真实 token 节省尚未验证。约 39.6 万字符的历史 Recorder 输入仍较大，其中真实统计占主要部分；本轮没有通过删除不同数值或科学条件压缩它。直接交付工具速查也会增加部分单次输入，其减少文档读取回合的收益仍需真实轨迹测量。外部账户 RPC、订阅额度和实际 GPU 资源问题不由此重构解决。

后续建议在**新 EXP 运行**中沿现有协议开展授权后的真实两轮验证：先核对账户/资源及冻结资产；保留预算和收尾预留，不续写旧终态。第一轮以最可能区分 H1/H2/H3 的受控测量为先，记录改变裁决的依据；H4 只针对 H* 的主要缺陷。区分校准拟合、独立 search 诊断和最终任务评测，允许范围内逐步升级；对 KL/NMSE 冲突给出明确解释或补测，不由单一代理指标强行排胜负。验证第一轮 Meta/L2 被第二轮 Generation 实际使用，统计每角色输入/输出 token、工具读取、修复次数、真实测量及提前结案情况，再评估科研收益。原已保存 H2 的 search1000 与导出重载验证仍是可考虑的具体验证对象，是否采用取决于新 Brief 的目标、可比性与预算；本轮未执行。

参考的是 [Co-Scientist 的生成—讨论—演化与科研反馈](https://arxiv.org/html/2502.18864v1) 和 [Letta MemFS 的简短常驻入口、按需读取与持久化](https://docs.letta.com/concepts/memfs)。QAViT 继续使用已有 Journal、卡片和队列，没有照搬外部框架。


复现入口（从项目顶层执行；集成输出必须使用新目录）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B EXP/validation/research_refactor_20261103_r1/cpu_integration.py EXP/validation/research_refactor_20261103_r1/cpu_services_new
```

历史回放脚本 [replay_recorder.py](replay_recorder.py) 默认拒绝覆盖既有副本；再次回放需明确选择新的独立输出位置。它只复制并验证保存材料，禁止启动任何 worker。变更前源码保存在 before/，原始失败与中间验证日志保留用于解释修复过程。
