# 科研链路覆盖矩阵

以服务器上的 `dinov3b_static_w4a4_independent_20261121_r1` 为来源。下表中的简称对应 [collection.json](collection.json) 的键；数值作业用 task 短前缀，完整请求、父任务、版本、回执哈希、时长及尝试号见 [engineering_cases.json](engineering_cases.json)。[audit.json](audit.json) 给出来源目录、真实工具事件及成本。输入和收尾文件均属于同名 `EXP/inputs/` 子目录。以下“历史”指核对已有决策或回执，“CPU”指本版受控工程回归，“实测”指本次新模型调用或数值执行。“保留”是独立评测分区。

| 阶段 | 组件 | 预期职责 | 代表案例 | 当时来源或事后核查来源 | 验证方式 | 有效覆盖与缺口 |
|---|---|---|---|---|---|---|
| 初始化 | Brief、用户约束、模型与任务 | 按输入绑定骨干、两头、范围、运行预算、数据协议及训练约束 | initial | input、initial_state、read view 14 的 Brief 和 model_metadata | 历史请求、结构审计、data CPU | 资产路径和源版本已核；未另起一轮初始化或换模型贯通 |
| 初始资产 | 种子、完整父方案、独立 B | 区分准备期 B、官方记录与本次开发视图，按协议保留 B | initial、record | initial_state.solutions、baseline_development、Recorder 读 B 回包 c3d4e6 | 历史、knowledge/source CPU | B 正文授权缺口已修；当前断点未以 LLM 重做 B 决策 |
| 取样 | DataView、ObservedData | 解码前按 split/offset/count 和有序 ID 绑定并验实读，区分优化与评测视图 | H1 3a5b25、code_repair | 冻结请求内 diagnostic_view、原错误、独立 worker 回执 | CPU 解码计数、原 64 张 search 的实际解码、单图前向、实测 12 张 calibration | 非零 offset 和禁止跨 split 已测；原 H1 完整 64 图实验未重做 |
| 选题 | Generation focus、研究主题 | 用已有反馈与实际模型约束选择可反驳的问题，保留主题与行动记录 | initial、restore | r001/r002 focus 的输入，topic_catalog、research_topic | 历史、generation/knowledge CPU | 请求级可见；本轮未重跑选题模型或推断最优选题 |
| H1 | Generation、首个完整父方案 | 用父状态、对照和阻塞解释发展改良，可采用弱保底 | h1_block | r003/H1 a5、之前真实工程失败 | 历史、真实 Generation 单步和请求预检、data/frozen CPU | 识别旧实验未测，声明同一冻结父模型、同一非零子视图并提交新请求；四臂数值未测 |
| H2 | Generation、第二完整父方案 | 保持完整父状态，解释局部误差，提出有鉴别力的改动 | h2_parent；r001 H2 短链路 | r003/H2 a1 与 r001/H2 原请求、分支新 Evidence | 历史、实测 Generation 读反馈后提出新实验 | 新请求已观察，提出的码本比较及效用未测 |
| H3 | 灵感版本、恢复与续研 | 恢复来源、父实现和负面证据，发展、修复、组合或暂停 | h3_continue、reserved_h3 | r002/H3 a2；原始 update_inspiration 三个完成回包 | 历史、inspirations CPU；末轮保留 | 连续性和可读版本链有覆盖；未经独立盲评分 |
| 科研工具 | request_experiment、实验冻结 | 分开请求和可执行规格，将目标、控制、数据和父状态传下去 | code_repair、c8bdc0、f3d0ef | original_request、阶段规格、缺 classification 的回执 | 历史、CPU、真实短链路重绑定 | 已核对合法受理和如实拒绝；不同新请求没有旧数值 |
| 工程实施 | Coding、edit_source、提交 | 依据确切父源码及版本改动、恢复修订、交出完整可执行文件 | code_repair | read view 631、revision_feedback、旧失败工具回包 | 实测编辑＋提交、CPU 精确替换/过期/越权/重试 | 实测走宿主科研工具通道；未在同一断点做原生 CLI 编辑对照 |
| 实施审查 | Safeguard、批准绑定 | 审查当前版本实验忠实性，失效旧批准，修订后再审核 | safeguard、6bef7a→2bef01、新短链 | 原 approved/failed/revision1 与独立新审查 | 历史、实测、source CPU | 真实新批准和下游执行已测；真实新驳回后循环用历史＋CPU覆盖 |
| 执行与回传 | task_worker、Evidence、Generation | 错误归责；成功交付观测/范围/来源；反馈进入下一次真实决策 | 短链、7353e9、48d24b | 新 numerical result、EvidenceCard、Generation 实际请求和响应 | 实测、evidence_index 逐字段比较、CPU | 12 张四臂；设备错误和错字段属于原候选，冻结依赖错误回宿主 |
| 候选综合 | select、revise_candidate、B 弱保底 | 比较全部候选，允许合法修订，修订保留新旧证据口径，三个最终候选都进入计划 | synthesis、reserved_select | r002/select、候选绑定、三轮检查点；test_candidate_revision | 历史＋CPU；末轮保留 | 未发现运行中显式 revise_candidate 调用；不能声称真实新版修订提升质量 |
| L3 计划 | Generation plan_l3、三候选调度 | 同一冻结骨干双任务、独立 B、三个请求分别绑定；单个阻塞不删除兄弟候选 | plan、b5565e | r002/H2 计划、r001–003 三个检查点、r004 队列 | 历史、generation/runtime CPU | 历史 3 个完整轮次，r004 只有 H2 完成；本次未重跑 L3-dev |
| 知识发现 | Skill 目录、正文、references | 目录可发现，按需取正文、分页与准确反馈，评阅资料隔离 | h1_block 前 a2→a3 | skill_catalog；4bd66e 请求及 a3 收到的 0 字正文 | 历史逐回包、scientific_skills CPU、已有七 Skill 情景 | 原运行无 Skill 主文正文字节交付证据；检索未命中后重新打开正文用 CPU 验证 |
| 方法与方案库 | method_catalog、方案库、快照成员 | 确切目录、字段、版本及源码可恢复，同时保护未授权原件 | h2_parent、code_repair、record | trajectory_index.knowledge、source_snapshot、原始 read_asset | 历史＋knowledge/source/frozen CPU | 7 次 methods、27 次 solution_library 正文读取按历史索引口径；无显式 read_method 调用，不推断采纳或贡献率 |
| 主题、灵感与 Memory | 版本库、retrieve_memory | 在本分支 read view 中发现并读取可见版本，必要来源传到下一次调用 | h3_continue、restore、r003/H1 a3 | topic_catalog、inspiration_continuation、四次检索、三次灵感更新 | 历史真实取回、版本/分支隔离/恢复 CPU | 实际工具与上下文均有覆盖；无显式 read_topic、read_inspiration 请求，完整工具路径由受控测试覆盖 |
| 冻结生命周期 | 原生、包裹父包的自定义 loader | 构造/导出/冷载评价、跨轮验证原态、加载及局部前向 | 34987a→7353e9、package_check | r002/H2 原 Evidence、源归档、父包与 segmentation.json 的双模型哈希 | 历史构造/双任务、实测冷载＋一图前向、CPU 连续生命周期 | 一只实际嵌套包；源码不符、文件增删和身份不符受控拒绝。其他自定义格式未逐包重测 |
| 双任务证据 | 分类、分割、阶段协议、正式选择 | 骨干与任务头一致，开发视图独立；局部代理不能替代官方成绩 | plan、46fe84、历史官方收尾 | round checkpoints、final_selection、official_summary、各自 artifact SHA | 历史；dual_task_evaluation、execution_contracts、development_view CPU | 已有 50k/2k 原结果；无新增官方/开发分数；未验证跨骨干精度泛化 |
| Recorder | 总结、来源、下一轮建议与版本恢复 | 绑定审查卡原文，下一轮能读取当前问题和反证 | record→restore | 第 1 轮 MetareviewCard 与第 2 轮 focus.memory.bound_feedback | 历史四个字段逐值比较、Recorder/wakeup CPU | 下一轮确实收到 578 字建议；历史三次 Recorder，r004 未闭环；未用新版 LLM 重写复盘 |
| 并发预算 | 队列、GPU、运行段、订阅预算 | 兄弟任务独立，只有实际并行工作计重叠，未启动与失败分别计数 | 4067de a1/a2、r004 drain | interrupted/result、budget_reconciliation、operator_interventions | 历史、runtime CPU | 原 worker_lost 后同任务第二次成功；历史订阅重连数、现金账单未知 |
| 失败/停止/保存 | 监督进程、幂等入库、Recorder 排队 | 先收可用回执并登记有效证据，保留剩余任务与未完成状态 | b5565e、seq 13116→13117 | closeout_reconciliation、budget_reconciliation、停止记录 | 历史、实际状态组合的 CPU 关停补登与重复恢复 | 原运行事后人工补登，现测试补登一次、错误可见；未耗尽一次新的长科研运行 |

原版角色评分继续使用冻结的 gonly_v0_3 和 Eval-L0 的有效证据覆盖、N/A、Unknown 口径。独立评阅时可比较决策依据、实验是否可区分竞争解释、交接可执行性、实际来源和下一步处理。一条合理替代路径可以获得同样的评价。没有真实后继结果的决策，执行有效性记未测。保留分区两个案例均未用于本轮调试或新调用。原始响应、后续输出、此矩阵、评分标准和事后收尾材料均只供评阅侧使用。
