# QAViT Generation 自主循环与上下文优化（2026-11-04）

已在基线 `567f04dc2404bfa64c709cd71717a0a40df9dc92` 的最新工作区实现本轮修改，完成源码回归、历史只读回放和真实 CPU 数值集成。没有启动新的长时间 DeiT-B 贯通实验，没有修改历史结果。最终 **308 项回归全部通过**；本轮真实 LLM 调用和 GPU 实验均为 0。

## 实际实现与职责

- `core/experiment_requests.py` 删除 Generation/Evolution 常规科学审批及实验后独立 Judger 解释：程序预检、冻结、执行，Evidence/错误直接回原研究任务。删掉 `experiment_approval` 任务定义与提示词，任务目的从 14 减至 13，七个角色保持。旧审批卡 phase 仅保留归档可读性。
- 原 `request_experiment → waiting → ready` 队列就是自主循环：Generation 可以不先形成 H，自主请求零次或多次实验，每次得到结果后分析、调整或正式提交 H。问题、可选 `research_note`、完整来源和 step 持久化，重启继续同一任务。H 可显式用 `evidence_refs`、`sources.ref` 标明实际采用的依据。
- 单请求预算不足/工具失败返回可操作错误，不产生成功 Evidence；硬截止保存未完成状态，不伪造 H 或轮次完成。首次 Coding 与修订的规格澄清都能返回研究者。新代码继续走 Coding、独立 Reference、真实数值比较和 Safeguard，拒绝上限及冻结规格不变。
- 三个 H 正式形成并筛查后，仍按 H1–H2、H1–H3、H2–H3 串行结案，固定 H*，Evolution 形成 H4 后只挑战 H*。Judger 的辩论补充实验保留。无等待判断任务的 L3、plan_l3 辅助实验和已发布 interpret 仍按既定规则完成。
- `core/evidence.py` 只承担客观投影与字段查询；未引入新 Agent、评分体系、工作流协议或向量库。所统计九个支撑模块净增 188 行，主要用于查询、摘要、来源与缓存完整性；本轮简化的是调用链和重复处理，并非用代码行数替代科研价值。

| LLM 保留的科研判断 | 程序承担的确定性工作 |
|---|---|
| 发现问题、决定是否实验、科学参数和对照 | 默认执行参数、路由、冻结身份、队列与预算 |
| 机制解释、因果与替代解释、形成 H | 原始测量、客观统计、精确索引和来源绑定 |
| 科学辩论、未决选择、H4 改进 | 已结案胜负复用、父版本绑定、证据时序 |
| Recorder 综合、条件性知识与反馈 | 全部事实归档、范围/独立重复检查、原子提交及恢复 |

只适配自主实验、支持证据和 Judger 职责变化的提示词，没有大规模改写科研策略。源码与工具文档同步，差异在 `changes.patch`、`prompts.diff`，完整流程见 [PSEUDOCODE.md](PSEUDOCODE.md)。

## 同历史任务回放

读取 `deitb_static_w4a4_two_rounds_20261031_r1/run` 四个角色实际保存的最丰富请求，在原提交 sequence 和原任务快照上重建输入；两边均去掉 repair_feedback，不运行 worker。统计的是 **canonical JSON 字符数**，不是 token。构建耗时为各一次实际墙钟测量，最终测量在其他 CPU/模型测试结束后进行；不是严格隔离的微基准，不能据此承诺端到端模型加速。

| 角色 | 输入字符，前→后 | 减少 | 构建秒，前→后 | Journal 读取/校验次数，前→后 |
|---|---:|---:|---:|---:|
| Generation | 50,962 → 52,554 | -3.1% | 2.57 → 2.25 | 348 → 65 |
| Judger | 452,583 → 341,804 | 24.5% | 48.69 → 13.24 | 6,928 → 290 |
| Evolution | 379,732 → 242,246 | 36.2% | 82.22 → 20.42 | 10,633 → 76 |
| Recorder | 397,129 → 258,919 | 34.8% | 63.88 → 14.60 | 11,790 → 178 |

Generation 这份历史输入原本没有大 Evidence 正文，新增循环说明和检索接口使其略增；本轮未以缩短提示词为目标。Judger 仍保留此前主动读取的工具正文，原件按需展开后不会被悄悄删除。长度至少 80 字符的完全重复字符串所占字符：Generation 476→476；Judger 3009→3339；Evolution 4727→4807；Recorder 4571→4651；略增主要涉及来源定位等固定字符串，**没有把这一项报告成减少**。历史工具结果条数：Generation 10→10；Judger 26→26；Evolution 1→1；Recorder 2→2，回放没有新增任何工具执行或 LLM 调用。

初始 Journal 完整加载前后为 17.87/18.43 秒；仍完整核验历史事实，这部分未作跳过校验的优化。详见 [REPLAY_RESULT.json](REPLAY_RESULT.json) 和两侧原始请求。

实际热点修复：单次宿主操作复用卡片校验、变化文件签名立即重验；权限归属复用提交快照，最终授权再核验真实卡片及哈希；检索先筛元数据，空关键词不转换全部正文，选中页才构造索引；Memory 只复制最近投影；队列/工具结果只复制所需任务；投影前不复制即将替换的大型 observations。缓存结束即释放，不作为第二事实源或授权源。实验缓存也按原申请人的权限过滤，不能借缓存读取同回合对方私有证据。

## Evidence、Memory 与信息交接

大型表/数组遍历全部行，保留命名聚合指标，并提供样本数、描述性均值、正负/零值信息、极值及原始 JSON pointer。类别预览明确标记是否完整；均值不冒充合并科学指标，不自动判断指标优劣。以下是同一 Evidence 的旧/新默认观测投影，原件规模另列，避免把原件大小冒充旧 LLM 输入：

| Evidence 前缀 | 完整观测字符 | 旧投影→新投影字符 | 减少 |
|---|---:|---:|---:|
| dc1ff55c | 9,433,809 | 36,112 → 9,468 | 73.8% |
| a9731ee1 | 509,060 | 140,882 → 33,723 | 76.1% |

140 个极值入口逐一核对回原始值，原件哈希不变。真实历史中 H1_C 的 logit NMSE 比 H1_T 更大、KL 更小的冲突同时保留，不择优摘要。`read_card(select={site,token_type,metric,text,minimum,maximum})` 可按条件扫描原始 Evidence，分页返回值、标签和精确路径；不带 select 仍可完整分页读原件。选择性读取不会被冒充完整必读材料，未来证据/同回合私有材料/Reference 隔离对所有入口一致。证据统计见 [HISTORICAL_EVIDENCE.json](HISTORICAL_EVIDENCE.json)。

L0 原件全部保存；L1 状态与 incumbent 规则不变；L2 保留知识状态、适用范围、使用证据、复盘时点和后续新证据查询；L3 由 Recorder 完成科学综合。`source_refs` 是程序归档集合，`evidence_refs/sources.ref` 是研究者实际采用的支持，不能混同。默认只交付明确关联的旧知识与 L2 检索入口，不把最新版本自动替代明确引用的旧版本。`recording_status` 明示待复盘，延期批次不会假装完成，后续 Recorder 合并未综合来源。

中途回归暴露了自动 L2 列表向 Recorder 注入无关条目的问题，已改为按需入口并保留精确绑定；最终旧版本/待复盘/跨轮回归均通过。

## 验证与调用变化

修改前 297 项通过；最终 308 项通过（`full_final.log`）。新增及适配测试覆盖零/多次实验、H 前取证、重启、失败后重新设计、预算拒绝/硬截止、初次 Coding 澄清、两轮完整控制流、被采用的支持来源、极值/筛选/分页、缓存隔离及文件篡改。已有 Reference 隔离、错误舍入/非有限值拒绝、数值通过与 Safeguard 双门禁、L3 失败与资格、结案固定、Recorder 原子恢复与 L2 修订检查继续通过。

`cpu_final/INTEGRATION_RESULT.json`：耗时 34.73 秒；同一 Generation 在提交 H 前发起 6 次请求，其中 1 次精确缓存复用，5 次实际 CPU 实验形成 Evidence。覆盖 Profiling、局部探针、量化后评测、导出包重载评测，以及新增量化代码；独立 PyTorch 候选/NumPy Reference 在真实子进程完成 3 组数值比较，随后 Safeguard 和执行通过。角色决定均为脚本夹具，数值为随机小 ViT 实算，不是 ImageNet 精度或真实 LLM 科研质量证明。原始回执、读取权限、Recorder 与下一轮反馈也完成核对。

相同 6 个成功请求按旧链路需要 6 个审批及 6 个独立解释任务；新链路均为 0，Coding/Reference/Safeguard 和数值验证不减。**少 12 个常规 Judger 任务是结构对照，并非实测节省 12 次模型调用或任何 token**；本轮没有真实 LLM token 数据。详见 [CALL_COMPARISON.json](CALL_COMPARISON.json)。一次集成脚本漏传 code_artifact，被原校验拒绝；补全脚本来源参数后通过，失败材料仍保留。

结构审计：132 个 Python 文件、13 类任务、5 类服务、3 个代码接口、26 个数值工具 Schema。66 个数值/阶段/预算核心文件 SHA-256 与修改前完全一致。50,883 个历史文件的大小与 mtime_ns 无变化，关键 Evidence 另验内容哈希；没有宣称对所有历史大文件逐一做了全量哈希。

## 下一次 DeiT-B 两轮实验

`NEXT_RUN_INPUT.json` 已通过配置校验，沿用上一任务的 static W4A4 49W/97A、32 张校准、每轮相同 search/1000 子集、两轮、禁止 final、不得晋升正式完整评测 incumbent。保留六小时预算作为**未启动配置模板**，未创建新运行/预算计时，也未延长旧运行。检查了 checkpoint/清单哈希和 1,032 个校准/搜索图像路径；未重新解码全部图像或进行 DeiT 前向。资源检查时 GPU 0 的 V100 16GB 空闲、共享盘约 165GB 可用；启动时须按现有入口再核验。数据回执与边界在 `READINESS.json`。

下次明确启动时入口为：

```bash
python -B QAViT/run.py start --config EXP/validation/generation_loop_20261104_r1/NEXT_RUN_INPUT.json --name deitb_static_w4a4_generation_loop_next
```

尚存限制：历史长任务构建仍有约 13–20 秒开销，Journal 初始核验仍约十余秒；主动读取的长正文和高度异构的命名统计仍会扩大输入。未新增自动语义压缩，也未验证真实 LLM 的研究质量或实际 token 节省。下一次真实两轮运行应重点观察 Generation 是否合理选择实验/停止时机、指标冲突解释及 Recorder 对后续 H 的实际影响；本轮只建立并验证执行支撑。

## 参考取舍

采用 [Co-Scientist](https://arxiv.org/html/2502.18864v1) 的生成—讨论—演化—复盘反馈分工；借鉴 [Letta MemFS](https://docs.letta.com/concepts/memfs) 和 [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) 的精简入口与按需读取。参考 [An AI4AI Framework for Visual Token Pruning](https://arxiv.org/html/2608.07193v1) 的科学选择与确定性实例化分离，没有移植其 TPDSL 或评分设计。

[OpenAI harness engineering](https://openai.com/index/harness-engineering/) 的可导航证据与机械检查、[DeepSeek Harness 架构](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md) 的模型/运行支撑分离用于审视现有职责；没有引入插件微内核、额外 Agent 或新的审批层。这些是结合 QAViT 的实现取舍，不是对外部框架效果的复现实验。
