# QAViT 项目结构与科研资源盘点

盘点日期：2026-11-08。对象是当前工作区，不是只看 Git HEAD：起始提交为 `8d937f0`，包括用户此前尚未提交的 static W4A4、完整方案 Top-3 等升级，以及本次 Skill 接入。前态文件哈希与原有差异分别见 `before_manifest.json`、`preexisting_changes.patch`；本次增量见 `changes.patch`。报告依据源码、注册表与调用链，历史报告仅作验证记录的辅助来源。

QAViT 现有资源已形成“研究角色选策略与方法 → 原工具链产生证据 → Recorder 综合 → 下一轮按需利用”的体系。Skill 只增加可选的策略知识入口。静态知识、运行记忆、原始测量与完整可执行方案具有不同身份，不能互相冒充。

## 1. 完整性与省略规则

文末目录树由本次 `build_inventory.py` 扫描真实文件系统生成，包含未提交的有效文件；不是手写的理想结构。`QAViT/` 的自有源码、全部提示词、文档、配置、测试、夹具、许可证全部逐文件列出。`reference/` 列出 PDF 名称，不解析论文内容。`EXP/` 保留轻量脚本、输入、配置、报告、索引和摘要文件；历史目录不等于当前有效实现。

明确省略：`.git`；Python/工具缓存与虚拟环境；`QAViT/.runtime_pkgs` 的第三方安装树；`EXP/cache`；实验中批量的 `jobs/cards/commits/contexts/direct_calls/scripted_requests/tools/trials/evidence` 原始运行树；模型目录、checkpoint 与二进制张量/归档；实施前副本、恢复副本及历史数值库快照；运行日志；EXP 中大于 1 MiB 的结果/快照文件。被裁剪的目录在树中标注原因，同目录被略去的文件汇总数量；精确清单在 `inventory_manifest.json`。这些是本次**结构展示范围**，没有删除或迁移历史资产。

批量 cards/commits 等即使单文件较小，整体仍属于大型实验运行产物；本报告在第 5 节说明其真实组织和读取机制，而不展开成成千上万份流水文件。`configs/assets/imagenet_manifest.json` 虽被 Git 忽略，仍是有效配置，列出。清单不追入符号链接指向的外部数据/权重；外部引用见第 7 节。

仓库未配置具名资源负责人；下文“维护者”指项目维护者及其授权的代码修改。运行角色的自动写入责任另行列明，不凭目录名称猜测具体个人。

## 2. 模块职责与主调用链

| 模块 | 当前职责及边界 |
|---|---|
| `QAViT/run.py` | 唯一 CLI：start/resume/status/report/answer/extend-budget；不自行安装依赖 |
| `core/workflow.py` | 唯一 Workflow 状态；生成、筛查、辩论、两阶段选择、H4、最终实验和复盘；调用其它职责分区 |
| `core/experiment_requests.py` | 科研实验请求、Coding/Safeguard 交接、重绑定/修订、请求结果回原任务、L3 计划与有效结果 |
| `agents/interaction.py` | 按 purpose 组织上下文、可见性、分页和去重；宿主工具及科研提交绑定；本次只在这里增加目录交付 |
| `agents/roles.py`、`tasks.py`、`prompts/` | 公共/角色/任务提示词、工具和提交模型视图；12 种 purpose 映射到七角色 |
| `core/cards.py`、`journal.py` | 卡片合同、版本/引用、不可变事实、原子提交、锁、可重建投影 |
| `core/memory.py`、`evidence.py`、`evidence_scope.py` | L0–L3 记忆、词法与范围检索、客观证据投影/筛选、知识范围和比较臂 |
| `core/debate.py`、`solutions.py` | 前者处理直接交锋与排名；后者处理协议专属完整方案库，二者不是同一排名 |
| `core/execution_contract.py`、`static_w4a4.py` | 必要执行条件、量化边与码本容量；局部探索和完整候选分别核对 |
| `runtime/driver.py` | 调度、请求体、回执接收、恢复、主运行循环；Supervisor 是按需角色，不是调度器 |
| `runtime/task_queue.py`、`task_runtime.py`、`task_worker.py` | 持久任务、尝试/续调、进程启动/归属、超时与结束；host_tool 不启动模型子进程 |
| `runtime/budget.py`、`run_segments.py`、`gpu_resources.py` | 总预算/运行段、数值等待计时、设备授权与租约；Skill 不修改这些机制 |
| `runtime/model_router.py`、`subscription.py`、`codex_cli.py` | 真实订阅传输、隔离、预检、Schema、模型请求/实际回显审计；没有 Skill Router |
| `runtime/paths.py`、`identity.py`、`files.py`、`guards.py` | 自包含导入、EXP 缓存、递归源码身份、文件/运行保护 |
| `tools/experiments.py`、`experiment_worker.py` | 服务合同、实验冻结/缓存/资产检查、真实数值执行与 Evidence 构建 |
| `tools/result_contract.py`、`runtime_interface.py`、`data_receipt.py` | 实际结果/数据/后端接口与回执，不替研究者选科学问题 |
| `tools/profiling.py`、`evaluating.py`、`segmentation.py`、`classification_shards.py` | 诊断服务、分类、冻结分割头、多卡分类分片 |
| `tools/safeguards.py`、`frozen.py`、`vision_models.py`、`literature.py` | 代码接口与安全、已发布原包加载、模型适配、公开文献检索/提取 |
| `tools/ptq_agent/` | 独立数值库；此命名空间不是旧科研 Agent。backends 负责数据/模型；profiling 负责统计/会话；execution 负责插件、校准、计划、图适配、导出重载；quantization 负责基础码器 |
| `tools/model_sources/deit/` | 已保留并有许可证的模型构造源码；不复制外部 checkpoint |
| `tests/` | 确定性单元/集成、CPU 小模型、静态/接口审计；显式 validate_* 入口才可能调用真实模型或大资产 |
| `EXP/` | 全部具体输入、实验、运行记忆、生成代码、验证与交付；不能成为核心实现的导入依赖 |

核心链为 `run.py → Workflow → advance()/dispatch()/poll()`。Generation 对 H1/H2/H3 各自可多次“读取 → request_experiment → Evidence/失败 → 继续研究”，也可直接提交。三个初始 Judger 配对各自推进并并行，全部结案后持久化 H*；Evolution 只加强该父版本成为 H4；之后只比较 H4–H*，做最终父子选择、可选 L3、Recorder、下一轮。已结案裁决不改判。

实验链为 `receive_tools → experiment_request → progress_request → broker`：已有内置服务直接执行；需要代码时保存原始科学请求，Coding 给一份实现及必要选择，程序绑定，Safeguard 对照原需求审查，再执行。Evidence 由程序产生并返回原研究者。Skill 没有插入这个状态机。[源码入口](../../../QAViT/core/workflow.py)、[实验请求](../../../QAViT/core/experiment_requests.py)、[运行驱动](../../../QAViT/runtime/driver.py)。

## 3. 哪些信息进入模型上下文

| 类型 | 实际内容 | 交付/读取与时机 | 更新来源 |
|---|---|---|---|
| 每次角色调用的常驻指令 | common + 本角色 + 本任务片段、提交合同、工具 Schema | `role_spec()` 组装；角色/purpose 决定，不加载全部角色文件 | 维护者改源码；Prompt 哈希及测试基线记录增量 |
| 当前调用绑定信息 | ResearchBrief、task、剩余预算、static W4A4 描述、必需短卡/证据、反馈、当前状态/少量目录 | `context()` 按任务构造；长材料给精确深读入口；不是全部历史常驻 | 用户冻结输入、Journal、程序投影 |
| 短静态目录 | method_catalog、skill_catalog；相关角色的 complete_solutions | 自动提供定位信息；Skill 只有 INDEX，不含正文/参考 | 手工知识目录；Top-3 由程序快照生成 |
| 常用接口正文 | 实验角色的 EXPERIMENT_INPUTS.md；Coding/Safeguard 的 CODE_INTERFACES.md | 按角色直接提供，减少只为查接口而多调用一次模型 | 接口维护者 |
| 按需静态知识 | SKILL.md/references、方法卡、方案说明、执行/环境文档和来源索引 | `read_asset`；既有 UTF-8、字符分页、内容哈希与调用快照 | 维护者；研究者不会自动修改源码文档 |
| 历史记忆 | 卡片短正文、L1 状态、L2 知识版本、L3 Meta-review | `retrieve_memory` 精确范围过滤后词法排序；`read_card` 展开原件 | 程序归档事实，Recorder 写科学综合/知识修订 |
| 可执行能力 | request_experiment 及内部数值服务、Coding 接口 | 读取说明不等于执行；测量经原执行链与预算 | 工具源码、当前配置和生成代码 |
| 动态证据 | EvidenceCard、原始 profile/evaluation、冻结模型/代码身份、失败回执 | 绑定证据直接交付有界观察；其余按引用读取；统计摘要不表示已读全部行 | 数值 worker 实测/失败；Recorder 不生成测量事实 |

一次业务续调是新模型请求，已选正文会按现有逻辑保留；“按需”不意味着读过后以后永不传输。重复相同工具结果会压缩，绑定正文不会又在工具结果重复；显式 research_note 前的历史读取可转成来源入口。Skill 复用了这些规则，没有新永久缓存或隐藏记忆。

## 4. 静态策略、方法与完整方案

### 4.1 首批科研 Skill

保存于 `QAViT/docs/skills/`：一个人工短 INDEX、四份带 name/description 的 SKILL.md、九份 references。由用户提供包，后续由仓库维护者审查编辑。程序不解析 frontmatter 做自动分类；实际发现目录是 INDEX，正文按需读。目录和 frontmatter 是两处人工维护的描述，静态测试检查名称、路径、引用和可读性。

| Skill | 对应科研情境 | 本条策略参考 |
|---|---|---|
| diagnose-ptq-failure | 失效机制不清、W/A/耦合、异常与传播 | fidelity-triage；quantization-error-map；distribution-and-propagation |
| design-discriminative-experiment | 竞争解释、混杂、昂贵实验前选信息量 | experiment-patterns；metrics-and-references |
| adapt-quantization-method | 成熟方案迁移、机制改进和开放创新 | established-quantization-patterns；designing-a-remedy |
| assess-evidence-transfer | 局部结果的支持范围、推广与下一步投入 | evidence-scope；next-step-decisions |

Generation、Evolution、所有 Judger purpose、Recorder 自动看到短目录；Coding/Safeguard/Supervisor 不自动收到，但 docs 的原读取权限不缩减。读取调用为 `read_asset(asset="skills/<name>/SKILL.md")`；相对参考链接转换为 `skills/<name>/references/...`。INDEX 已说明真实工具名和参数入口，不把内部 profile 工具误当角色工具。

Skill 在决策点可选择零到数个，没有固定次序、必做实验、来源论文要求或新输出 Schema。缺目录时忽略；选错/缺文件的失败沿原任务回执返回，可继续原科研路径。H3 不受目录穷尽性限制。Skill 只提出研究策略；方法卡提供算子定义，完整方案提供起点，Memory 提供历史条件，Toolbox 提供新观测。有效认识沿 Recorder 沉淀，不自动回写静态 Skill。

原包评估集保存在 `QAViT/tests/skill_evals/ROUTING_AND_REPLAY.md`，包括正负触发、组合边界、人工回放和评分建议；它是**设计材料**，不是已通过的全部评测，也不在运行时 Skill 目录中。原交接文档与来源哈希存于本次 `package/`、`package_manifest.json`。[接入说明](../../../QAViT/docs/SCIENTIFIC_SKILLS.md)。

### 4.2 方法手册、来源索引与数值插件

`QAViT/docs/methods/index.json` 有 21 个条目，旁边 21 份方法/校准说明：rtn、uniform_full、uniform_affine、uniform_minmax、gptq、gptaq、adaround、twin_softmax、log2、log_sqrt2、twin_gelu、asymmetric_gelu、ptf、adalog、smoothquant、repq_ln、adalog_bias、integer_layernorm、integer_softmax、scale_selection、calibration_order。不是 21 个互斥的整网方案。

维护者写原理、公式、伪代码、适用问题、参数、校准顺序、组合边界和来源。实验角色通过 `context.method_catalog` 直接定位；一般 `retrieve_memory` 还可能附带少量 method_documents 入口，L1/L2 专门查询则不附带。实际读取用 `read_asset(methods/...)`，不需要模型加载或数值实验。Recorder 可从检索入口找文档，工程角色可直接读已授权 docs。

数值会话还保留 `execution/method_library.py` 的 MethodLibrary：初始化核验索引插件与真实注册表一致，冻结文本与 SHA 到该会话的 `method_library.json`；`query_method_library` 是关键词过滤，`read_method_document` 是正文读取。这条兼容通道在 profile 服务里可调用，但常规科研查资料优先直接读文档。源码中的 `require_plan_documents()` 目前没有调用点，`loop.require_method_documents` 仍是兼容配置字段，不能据其存在声称有阅读门禁。

`docs/references/ptq_methods.json` 登记八组方法来源（SmoothQuant、PTQ4ViT、FQ-ViT、RepQ-ViT、GPTQ、AdaLog、AdaRound、GPTAQ v1）、论文/官方库、固定 commit、文件 SHA、许可和历史验证索引。它不是启动时全文注入，也不是外部旧项目的运行依赖。其历史路径有溯源意义，当前实际实现以 `tools/ptq_agent/` 为准。ERQ/IGQ 等提及不代表已注册可执行插件。

当前注册表实际有权重方法 7 个、激活方法 11 个、前处理 3 个、算子 2 个；同名 RTN 等可在不同目标注册。文档新增不自动赋予执行能力。维护算法须同步 methods/插件、数值合同、序列化/重载、测试和文档。[方法说明](../../../QAViT/docs/methods/README.md)、[插件实现](../../../QAViT/tools/ptq_agent/execution/methods.py)。

### 4.3 完整方案 Top-3 与配方

静态种子是 `docs/solutions/seeds.json` 以及 gptaq.md、adalog.md、repq.md；对应 `execution/baselines.py:build_baseline()` 的完整 static W4A4 构造器。三条 ID 是 gptaq_w4a4、adalog_w4a4、repq_w4a4。初始分数为 null/待测，不能称为在当前双任务协议实测的前三名。

`SolutionLibrary.initialize()` 从种子或已有共享桶初始化；研究运行的路径是 `EXP/solutions/<protocol_hash>/top3.json`，验证运行隔离在 `<run>/solutions/<protocol_hash>/`。Journal 保存该运行当时可见的快照；`<run>/solution_library/<id_hash>.json` 物化完整配置，必要时带可复用代码。`catalog()` 只给短条目和 configuration_asset，`assets()` 纳入原来源/配置的授权哈希。**本次扫描前项目没有顶层 EXP/solutions 目录，表示共享库路径已实现，但本工作区尚未由新的正式研究运行创建该桶；不能伪写成已积累真实 Top-3。**

自动交付 complete_solutions 的 purpose 是 generate/evolve/code/safeguard/record/plan_l3；普通 screen/position/arbitrate/rank/interpret 未自动交付同一目录。研究者选择后用 read_asset 取精确配置/版本/产物。不是自动调用构造器：build_baseline 在源码中有单元数值测试调用，生产执行需研究者/Coding 明确使用。常规 Workflow 不替研究者选一个基线。

每轮最后有效完整 L3 双任务结果由 `Workflow.advance → solutions.consider_round` 考虑更新，按同协议 (mIoU, Top-1) 降序，平分保留旧项；入选/未入选都留源记录。协议键含模型、evaluation_protocol、quantization_contract、hard_constraints；跨模型/协议不合桶。此过程不改变辩论裁决，不丢弃未晋升的负结果/局部经验。

`execution/recipes.py:build_recipe()` 另保留 PTQ4ViT/RepQ/FQ-ViT 的论文配置参考，要求特定结构、位宽和来源，返回普通 v3 plan。当前生产源码没有自动调用 build_recipe；它与新严格 W4A4 的 baselines 种子不是同一层含义。Python 构造器虽可被 Coding 参考/使用，也不能作为已完成当前正式评测的证据。[方案库源码](../../../QAViT/core/solutions.py)、[静态种子说明](../../../QAViT/docs/solutions/README.md)。

## 5. Memory、Evidence 与 Recorder

### 5.1 保存、检索与可见性

Memory 实际基于**每个运行的 Journal**，没有一个另行常驻的向量数据库，也不会自动扫描整个 EXP 的历史研究。`<run>/cards/<sha256(card_id)>.json` 保存不可变卡片，`commits/` 保存原子事件；`checkpoint.json` 等是可重建派生索引。准确叶层命名由 `Journal.card_path()` 生成，读者使用卡片完整 ID，不应拼接路径猜记录。

| 层 | 实现含义 | 生产更新者与读取 |
|---|---|---|
| L0 | 所有不可变 cards/commits 原始事实 | 程序提交；按已授权 read_card/read_asset 读取 |
| L1 | 当前轮次、选择、recent hypotheses/evidence/experiments、未决问题、incumbent | 实时事实从 Journal 重建，问题随 Recorder 修订；context 短状态或 retrieve_memory(layer=L1) |
| L2 | 按 claim_id 维护的知识修订、状态、来源 Evidence 和逐臂 scope | Recorder 提案，程序校验/绑定；retrieve_memory(layer=L2) 与精确 revision_ref |
| L3 | Meta-review 的版本引用序列 | Recorder 科学综合；下一轮按绑定反馈与精确字段读取 |

这是 Memory 的 L0–L3，**不等于实验的 L1/L2/L3**，也不等于 Skill 文中的推导/检查 L0。`MemoryRetriever.retrieve()` 在角色/调用快照可见范围内，先按 kind/证据 ID/规范 scope 精确筛选，再做词法 token 排序；支持分页、after_version、来源短句柄与完整 ID。没有 embedding、向量语义检索或 LLM 检索总结。直接短正文附实际条件、状态和原件入口，未命中不证明历史不存在。

`RoleInteraction.can_read_card()`、`read_view()` 保障同回合未汇合的对方立场及私有子任务不能绕过；read_asset 也沿同一来源授权，不能遍历日志获得额外权限。新模型调用才刷新可见快照。静态 docs 可以读，不意味着全部任意 EXP 文件都能读。

### 5.2 实验事实与大型观测

实验请求/执行计划成为 ExperimentCard；worker 的 measurement/receipt 经过 `ExperimentBroker.evidence()` 绑定真实范围、资源、执行指纹和产物，生成 EvidenceCard。`measurement_source` 区分 real、synthetic_computation、test_double；数值夹具与真实数据的结论范围不同。失败/超时/工程缺口保存在任务回执，并不自动成为科学反证。

典型动态文件在 `<run>/jobs/<task>/<attempt>/`：request/receipt/llm_audit、profile_result、profile/state/method_library、profile/evidence、evaluation、冻结 model 及 manifest。生成源码在运行 code/ 或关联任务产物中。实际路径由回执给出；不能假定每种服务都有同样文件。Evidence 的 artifacts 带路径与哈希，模型/数据保留精确身份。

大型观测经 `core/evidence.py` 给有界、全量客观统计与覆盖说明、正负混合提示、极值来源及精确字段入口；这不由 LLM 重写，也不代表已读所有数值。`read_card(select=...)` 可按位点、token、metric、范围筛选原始标量，普通 field/offset/limit 可完整分页。`read_profile_evidence` 读取数值会话证据时另做本运行成功 Evidence 的绑定恢复；冻结 trial 诊断依赖显式 trial_evidence_refs，不能扫描孤立作业。

### 5.3 Recorder、反馈与沉淀

程序 `recording_batch()` 归档本批全部科学/相关工程来源；Recorder 主要提交 scientific_analysis，以及有用时的 generation_guidance、next_questions、knowledge_updates。没有逐张卡固定复述表。`prepare_metareview/apply_metareview` 绑定来源、知识 scope、比较臂、pending/版本并发布 memory_projection；有问题的可选条目与有效正文分别保存，事实归档和科学复盘完成是两种状态。

下一轮 Generation/Evolution 接收绑定反馈短正文和 required_reads；可质疑旧判断，不把它当新测量。程序 `relevant_delta` 也会发现相关新增事实，支持重调度时的陈旧输入处理。Recorder 失败可延期并保留待办，不能伪造完成的综合。

Memory 的 L1.incumbent 由 `driver.receive → memory.consider_incumbent` 在有效 L3 回执阶段更新；独立 Top-3 则在完整轮次结束后更新。两者均按双任务有效结果核验，但触发时机、协议键构成和保存目的不同。这是现状，不在本轮合并。[Memory 源码](../../../QAViT/core/memory.py)、[角色读取与交付](../../../QAViT/agents/interaction.py)。

## 6. Toolbox：角色工具、数值服务和代码接口

### 6.1 角色真正能调用什么

| 角色/purpose | 角色工具 |
|---|---|
| Generation、Evolution、Judger 所有科研 purpose | read_card、read_asset、retrieve_memory、literature_search、literature_fetch、request_experiment |
| Recorder | 前五个，不含 request_experiment |
| Coding、Safeguard、Supervisor | read_card、read_asset |

工具 Schema 在 `agents/roles.py:TOOLS`，角色映射在 `agents/tasks.py:TASKS`。profile_sensitivity 等不是上述角色的直接函数，需用 request_experiment 的 profile 服务调用；常规已有服务不用为了执行再生成代码。

### 6.2 五个数值服务与 15 个公开 profile 工具

| 服务 | 层级与作用 |
|---|---|
| profile | L1；结构/能力/位点、激活/权重分布、硬件代理、敏感性、候选/消融和证据读取 |
| tensor_probe | L1/L2；局部量化误差及单点替换后的整网回放；可用自定义 quantizer_v1 |
| quantize_evaluate | L2/L3；静态校准、量化、导出/重载、分类/分割；plan 或自定义 executor |
| evaluate_artifact | L2/L3；加载已冻结标准模型包后评测，不是通用任意包加载器 |
| custom_experiment | L1/L2/L3；新增统计/算法的 experiment_v1 实现，经原 Coding/Safeguard 链 |

公开 profile 名单由 `tools/profiling.py:PROFILE_NAMES` 唯一限定：query_structure、query_execution_capabilities、query_ptq_plugins、query_ptq_sites、profile_structure、profile_distribution、profile_hardware、profile_sensitivity、profile_plugin_sites、profile_ptq_candidates、profile_trial_diagnostics、profile_trial_ablation、read_profile_evidence、query_method_library、read_method_document。参数已随实验角色的 experiment_interface 交付；结构查询不等于精度证据，FP 路径硬件 profiling 不等于真实 INT4 kernel 加速。

数值内部 ToolRegistry/ProfileHost 还注册 compose_experiment、validate_experiment、read_trials、read_evidence_facts、read_evidence_batch、read_search_record、task_programs 等支持能力。**注册存在不等于 V2 角色直接可调用**；不在 PROFILE_NAMES 的名称不能当作公开 profile 服务请求。它们是内部数值/兼容能力，逐项实际使用需看其宿主调用点。本次不把内部全注册表伪装成新增 Agent API。

### 6.3 扩展与文献

Coding 的接口有 quantizer_v1（fit/apply）、executor_v1（计划或自定义执行器能力）、experiment_v1（execute）。可引用许可的数值模块/torch/numpy 等，在已有静态检查、保护、科学 Safeguard 和宿主结果合同下实现新算法；内置方法枚举不限制自定义 H3。`tools/frozen.py` 与 experiment_v1 的 load_frozen 支持已发布 Evidence 绑定的确切原包加载，不能把重校准的新实例称为原冻结候选。

`literature_search` 实现 Crossref 元数据搜索，最多五条，不代表读过论文；`literature_fetch` 获取公开 HTTP(S) 原件、SHA、URL 和提取文本，PDF 依赖 pypdf，首段预览不代表完整正文，后续 read_asset 分页。新文献保存在当次任务 jobs 中并授权回原任务。它不会自动更新 methods/或 Skill。此项源码已实现，本次未联网测试其服务可用性。

可执行工具由维护者修改源码并验证；一次性新实现由 Coding 写入 EXP。可复用经验先留 Memory，可复用且验证过的工程能力才由维护者提升为 QAViT 核心，当前没有自动“把研究结果安装成插件”的过程。[执行接口](../../../QAViT/docs/EXPERIMENT_INPUTS.md)、[代码接口](../../../QAViT/docs/CODE_INTERFACES.md)。

## 7. 配置、实验配方、资产与知识索引

| 位置 | 内容、维护者与实际使用 |
|---|---|
| configs/research.json | 用户研究模板：ResearchBrief、模型适配、目标/约束/空间、评测、未决输入、预算、资源和订阅；模板故意不能未经填写直接正式运行 |
| configs/models.json | 唯一七角色默认模型/effort；runtime.configuration 校验，实际配置冻结于 research_run.json。默认 Generation/Evolution gpt-6.1-sol/high，Judger/Coding/Safeguard gpt-6.1-sol/medium，Recorder gpt-6-luna/high，Supervisor gpt-6-luna/low |
| configs/deitb_backend.json | 数值服务的后端、模型/样本、格式/插件范围、缓存等；一些旧 budget/loop 字段是兼容数值配置，不是当前七角色全局调度预算 |
| configs/ade_evaluation.json | 冻结分割头、neck、输入预处理、slide 评测及 ADE 路径；定义具体评价实现 |
| configs/assets/deit_model_manifest.json、ade_head.json | 模型构造源码/权重、分割头的来源与哈希回执；由资产维护者生成/核验 |
| configs/assets/imagenet_manifest.json | 实际图像池、划分/标识；本地忽略入库，但运行仍验证清单和所用资产 |
| docs/STATIC_W4A4.md、EXECUTION.md、CODE_INTERFACES.md、EXPERIMENT_INPUTS.md | 当前合同/调用/编码说明；分别对应实际 core/tools 接口，文字不替代运行核验 |
| docs/ENVIRONMENT.md、runtime_dependencies.json | 安装来源/版本/隔离路径；依赖清单有 31 条包记录（含隔离环境中的重复包名），不是 Agent 科学知识库 |
| docs/METHODS.md、methods/index.json、references/ptq_methods.json | 能力总览、组件索引、原始来源索引，三者职责不同 |
| docs/solutions/seeds.json、execution/baselines.py、execution/recipes.py | 完整适配种子、可执行构造器、论文原配方参考，区别见 4.3 |
| EXP/inputs/deitb_training_free_static_w4a4_20261106_r1/ | 具体历史授权输入、干预、监控与研究报告；只读参考，不替换新任务 Brief |
| EXP/runs/、outputs/、validation/、implementation/ | 真实/验证运行、历史输出和实施材料；部分旧目录属于先前框架，不应按当前格式直接恢复 |
| reference/ | coscientist.pdf、coscientist_supplementary.pdf，供人参考；不是自动注入的策略或可执行流程 |
| tests/data/ | contract_golden、prompt_readiness_baseline：协议/Prompt 回归基线，不是科研记忆 |

大型资产只引用既有路径：DeiT 权重和 ADE/ImageNet 数据主要位于相邻 `ptq_training_free/` 等资产位置；模型构造已在 QAViT 内，数值实现不导入旧智能体项目。`reference/` 内 PDF 不能直接用 UTF-8 read_asset 当正文；只有 Brief 明确列入且满足授权根/文本条件的 source_documents 才可直接读取，或使用已有文献提取通道。`docs/` 则由 authorized_assets 递归发现文本。

科研知识索引并非一个总搜索引擎：Skill INDEX 管策略；methods/index 管组件；seeds/top3 管完整方案；Journal/card_versions/Memory 管运行历史；ptq_methods.json 管来源；numeric registry 管可执行能力。目录可发现不等于正文已读，来源链接不等于结果实测，配置条目不等于算法已执行。

## 8. 实现状态、验证边界与后续讨论

| 状态 | 本次可证实的内容 |
|---|---|
| 已接入且离线执行验证 | 四 Skill/九 references 经真实 host_tool 读取、源码身份、调用快照/哈希、目录去重、续调、缺目录/读失败仍可提交；详见 test_scientific_skills |
| 已有生产调用链 | Generation/Evolution 自主循环、三场并行 Judger、Coding/Safeguard、Recorder、预算、动态 Evidence/Memory、完整方案初始化/更新接口；本轮不重构 |
| 真实 Agent 验证 | 本次 live_smoke 通过生产队列、角色 Prompt、订阅和提交链；逐例读取与成功/未成功事实以 REPORT.md、live_summary.json 为准，不将评估题设当实测 Evidence |
| 可调用但非自动调用 | build_baseline/build_recipe、方法会话读取、load_frozen、自定义实现；需要研究角色/代码按科学问题选择；注册/单元测试不证明在当前正式研究中已用过 |
| 兼容残留、没有当前阅读门禁调用 | require_plan_documents()、loop.require_method_documents；旧后端 agents/tools/measurements 等字段不构成新的 V2 Skill 预算 |
| 仅资料/扩展点 | ERQ/IGQ 等未注册方法、向量检索、自动 Skill 调优/安装、自动静态知识回写、通用跨运行语义 Memory、通用真实 INT4 部署后端；不当作现有能力 |
| 尚未验证 | 全包全部行为评估、多随机重复/盲评、Skill 的统计科研增益、完整 GPU 双任务/L3、真实 INT4 性能、跨模型泛化；本次没有这些成绩 |

供下一轮讨论的具体问题如下，本轮均不改结构：

1. **目录与描述的人工同步**：Skill INDEX 与 SKILL frontmatter 重复触发语义；methods INDEX 与文档/注册表也有不同更新点。当前足够小，自动生成是否值得需要另议。
2. **方法读取双入口**：直接 read_asset 与 profile MethodLibrary 快照并存。后者仍有完整历史兼容函数及阅读门禁文字，但当前无调用；容易误以为要额外申请实验读文档。
3. **多个“最佳方案”概念**：场内裁决、L1.incumbent、独立 Top-3 各有对象/时间/协议意义，字段与说明还可更明确；不能随意合并。
4. **完整方案发现不完全统一**：普通 Judger 部分 purpose 不自动收到 complete_solutions，而其它研究/工程角色会收到。当前不为 Skill 扩充所有角色上下文；是否需要统一按需目录可另议。
5. **方案与配方层次**：recipes 的论文原配方、baselines 的严格协议适配、solutions 的已评测/待测完整条目都存在；名称相近，忠实复现与可部署改造边界需持续说明。
6. **跨运行知识可见性**：一般 Memory 只检索本运行，Top-3 却可跨研究运行共享协议桶；历史 EXP 结果很多，但不会自动成为当前检索材料。未来若扩展导入，应保持来源、范围和可见性，而不是全量灌入。
7. **历史文档累积**：README/STRUCTURE 叠加多轮交付链接，EXP 有大量早期 Prompt/协议副本。应通过当前源码入口判断现状，不能把历史 prose 或未调用函数当运行规则。
8. **上下文体积来源不同**：Skill 目录很短，但实验接口正文、方法目录、方案短目录和当前任务合同本身已占上下文。是否进一步按需化属于独立优化，不能归咎为 Skill 多余调用。
9. **术语与证据层级重名**：Memory L0–L3、实验 L1–L3、Skill 推导 L0；profile/tool/role tool；real 与 synthetic/test_double 等需按实际字段解释。

## 9. 真实文件结构

以下区域由脚本生成；`[省略...]` 是本报告的明确展示裁剪，未触碰原文件。各自有模块职责见第 2 节；EXP 各目录名保留日期/用途，脚本与报告仍属于对应历史实验。

<!-- INVENTORY_START -->
扫描时间：2026-11-08T07:05:16.774029+00:00；逐项列出 3299 个文件、810 个展开目录。省略项见规则及 manifest。

```text
Quantization_Agent_for_ViTs/
├── QAViT/
│   ├── .runtime_pkgs/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   ├── agents/
│   │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   │   ├── prompts/
│   │   │   ├── tasks/
│   │   │   │   ├── arbitrate.md
│   │   │   │   ├── code.md
│   │   │   │   ├── evolve.md
│   │   │   │   ├── generate.md
│   │   │   │   ├── interpret.md
│   │   │   │   ├── plan_l3.md
│   │   │   │   ├── position.md
│   │   │   │   ├── rank.md
│   │   │   │   ├── record.md
│   │   │   │   ├── safeguard.md
│   │   │   │   ├── screen.md
│   │   │   │   └── supervise.md
│   │   │   ├── coding.md
│   │   │   ├── common.md
│   │   │   ├── evolution.md
│   │   │   ├── generation.md
│   │   │   ├── judger.md
│   │   │   ├── recorder.md
│   │   │   ├── safeguard.md
│   │   │   └── supervisor.md
│   │   ├── interaction.py
│   │   ├── roles.py
│   │   └── tasks.py
│   ├── configs/
│   │   ├── assets/
│   │   │   ├── ade_head.json
│   │   │   ├── deit_model_manifest.json
│   │   │   └── imagenet_manifest.json
│   │   ├── ade_evaluation.json
│   │   ├── deitb_backend.json
│   │   ├── models.json
│   │   └── research.json
│   ├── core/
│   │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   │   ├── cards.py
│   │   ├── debate.py
│   │   ├── evidence.py
│   │   ├── evidence_scope.py
│   │   ├── execution_contract.py
│   │   ├── experiment_requests.py
│   │   ├── journal.py
│   │   ├── memory.py
│   │   ├── solutions.py
│   │   ├── static_w4a4.py
│   │   └── workflow.py
│   ├── docs/
│   │   ├── methods/
│   │   │   ├── adalog.md
│   │   │   ├── adalog_bias.md
│   │   │   ├── adaround.md
│   │   │   ├── asymmetric_gelu.md
│   │   │   ├── calibration_order.md
│   │   │   ├── gptaq.md
│   │   │   ├── gptq.md
│   │   │   ├── index.json
│   │   │   ├── integer_layernorm.md
│   │   │   ├── integer_softmax.md
│   │   │   ├── log2.md
│   │   │   ├── log_sqrt2.md
│   │   │   ├── ptf.md
│   │   │   ├── README.md
│   │   │   ├── repq_ln.md
│   │   │   ├── rtn.md
│   │   │   ├── scale_selection.md
│   │   │   ├── smoothquant.md
│   │   │   ├── twin_gelu.md
│   │   │   ├── twin_softmax.md
│   │   │   ├── uniform_affine.md
│   │   │   ├── uniform_full.md
│   │   │   └── uniform_minmax.md
│   │   ├── references/
│   │   │   └── ptq_methods.json
│   │   ├── skills/
│   │   │   ├── adapt-quantization-method/
│   │   │   │   ├── references/
│   │   │   │   │   ├── designing-a-remedy.md
│   │   │   │   │   └── established-quantization-patterns.md
│   │   │   │   └── SKILL.md
│   │   │   ├── assess-evidence-transfer/
│   │   │   │   ├── references/
│   │   │   │   │   ├── evidence-scope.md
│   │   │   │   │   └── next-step-decisions.md
│   │   │   │   └── SKILL.md
│   │   │   ├── design-discriminative-experiment/
│   │   │   │   ├── references/
│   │   │   │   │   ├── experiment-patterns.md
│   │   │   │   │   └── metrics-and-references.md
│   │   │   │   └── SKILL.md
│   │   │   ├── diagnose-ptq-failure/
│   │   │   │   ├── references/
│   │   │   │   │   ├── distribution-and-propagation.md
│   │   │   │   │   ├── fidelity-triage.md
│   │   │   │   │   └── quantization-error-map.md
│   │   │   │   └── SKILL.md
│   │   │   └── INDEX.md
│   │   ├── solutions/
│   │   │   ├── adalog.md
│   │   │   ├── gptaq.md
│   │   │   ├── README.md
│   │   │   ├── repq.md
│   │   │   └── seeds.json
│   │   ├── CODE_INTERFACES.md
│   │   ├── ENVIRONMENT.md
│   │   ├── EXECUTION.md
│   │   ├── EXPERIMENT_INPUTS.md
│   │   ├── METHODS.md
│   │   ├── runtime_dependencies.json
│   │   ├── SCIENTIFIC_SKILLS.md
│   │   └── STATIC_W4A4.md
│   ├── runtime/
│   │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   │   ├── __init__.py
│   │   ├── budget.py
│   │   ├── codex_cli.py
│   │   ├── configuration.py
│   │   ├── driver.py
│   │   ├── files.py
│   │   ├── gpu_resources.py
│   │   ├── guards.py
│   │   ├── identity.py
│   │   ├── model_router.py
│   │   ├── paths.py
│   │   ├── run_segments.py
│   │   ├── subscription.py
│   │   ├── task_queue.py
│   │   ├── task_runtime.py
│   │   └── task_worker.py
│   ├── tests/
│   │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   │   ├── data/
│   │   │   ├── contract_golden.json
│   │   │   └── prompt_readiness_baseline.json
│   │   ├── skill_evals/
│   │   │   └── ROUTING_AND_REPLAY.md
│   │   ├── __init__.py
│   │   ├── audit_run.py
│   │   ├── audit_structure.py
│   │   ├── fixtures.py
│   │   ├── snapshot_contracts.py
│   │   ├── support.py
│   │   ├── test_adaptive_resources.py
│   │   ├── test_branch_acl.py
│   │   ├── test_cli_versions.py
│   │   ├── test_code_asset_ownership.py
│   │   ├── test_code_repair.py
│   │   ├── test_context_cost.py
│   │   ├── test_controlled_scope.py
│   │   ├── test_evidence_handoffs.py
│   │   ├── test_execution_binding_recovery.py
│   │   ├── test_execution_contracts.py
│   │   ├── test_experiment_citation_closure.py
│   │   ├── test_experiment_inputs.py
│   │   ├── test_experiment_preflight.py
│   │   ├── test_exploration_contracts.py
│   │   ├── test_feedback_boundaries.py
│   │   ├── test_feedback_delivery.py
│   │   ├── test_feedback_overlap.py
│   │   ├── test_generation_loop.py
│   │   ├── test_infrastructure.py
│   │   ├── test_inner_json_repair.py
│   │   ├── test_interface_simplification.py
│   │   ├── test_memory_delivery.py
│   │   ├── test_memory_interface_optimization.py
│   │   ├── test_memory_state_optimization.py
│   │   ├── test_minimal_fixes.py
│   │   ├── test_numeric_imports.py
│   │   ├── test_numeric_parallel_budget.py
│   │   ├── test_parallel_requests.py
│   │   ├── test_parallel_runtime.py
│   │   ├── test_parent_evolution.py
│   │   ├── test_plan_execution_consistency.py
│   │   ├── test_plan_scientific_input.py
│   │   ├── test_read_delivery_boundaries.py
│   │   ├── test_real_research_followup.py
│   │   ├── test_recorder_poll_recovery.py
│   │   ├── test_recorder_prior_basis.py
│   │   ├── test_recorder_reference_handoff.py
│   │   ├── test_recorder_scheduling_cost.py
│   │   ├── test_request_continuation.py
│   │   ├── test_research_refactor.py
│   │   ├── test_research_simplification.py
│   │   ├── test_research_upgrade.py
│   │   ├── test_run_readiness.py
│   │   ├── test_runtime_gate_cleanup.py
│   │   ├── test_scientific_skills.py
│   │   ├── test_scope_membership.py
│   │   ├── test_serial_debate_memory.py
│   │   ├── test_service_conditions.py
│   │   ├── test_single_implementation.py
│   │   ├── test_stale_safeguard_recovery.py
│   │   ├── test_step_recovery.py
│   │   ├── test_structure.py
│   │   ├── test_structured_output_repair.py
│   │   ├── test_submission_gate_cleanup.py
│   │   ├── validate_code_service.py
│   │   ├── validate_memory_service.py
│   │   └── validate_numeric.py
│   ├── tools/
│   │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   │   ├── model_sources/
│   │   │   └── deit/
│   │   │       ├── hubconf.py
│   │   │       ├── LICENSE
│   │   │       └── models.py
│   │   ├── ptq_agent/
│   │   │   ├── backends/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── image_data.py
│   │   │   │   ├── server.py
│   │   │   │   └── task_heads.py
│   │   │   ├── execution/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adalog.py
│   │   │   │   ├── adaround.py
│   │   │   │   ├── baselines.py
│   │   │   │   ├── calibration.py
│   │   │   │   ├── capabilities.py
│   │   │   │   ├── codecs.py
│   │   │   │   ├── comparison.py
│   │   │   │   ├── contracts.py
│   │   │   │   ├── coordinate.py
│   │   │   │   ├── diagnostics.py
│   │   │   │   ├── evaluation.py
│   │   │   │   ├── gptaq.py
│   │   │   │   ├── grids.py
│   │   │   │   ├── local_comparison.py
│   │   │   │   ├── method_library.py
│   │   │   │   ├── methods.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── model.py
│   │   │   │   ├── objectives.py
│   │   │   │   ├── operators.py
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── planning.py
│   │   │   │   ├── plugin_contracts.py
│   │   │   │   ├── plugin_math.py
│   │   │   │   ├── plugin_tools.py
│   │   │   │   ├── profile_tools.py
│   │   │   │   ├── program_worker.py
│   │   │   │   ├── recipes.py
│   │   │   │   ├── runtime_state.py
│   │   │   │   ├── search_context.py
│   │   │   │   ├── sites.py
│   │   │   │   └── task_programs.py
│   │   │   ├── profiling/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapters.py
│   │   │   │   ├── contracts.py
│   │   │   │   ├── evidence.py
│   │   │   │   ├── host.py
│   │   │   │   ├── programs.py
│   │   │   │   ├── statistics.py
│   │   │   │   ├── tools.py
│   │   │   │   └── torch_profiler.py
│   │   │   ├── quantization/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── config.py
│   │   │   │   └── integer.py
│   │   │   ├── __init__.py
│   │   │   ├── configuration.py
│   │   │   ├── README.md
│   │   │   ├── registry.py
│   │   │   ├── schemas.py
│   │   │   └── storage.py
│   │   ├── classification_shards.py
│   │   ├── data_receipt.py
│   │   ├── evaluating.py
│   │   ├── experiment_worker.py
│   │   ├── experiments.py
│   │   ├── frozen.py
│   │   ├── literature.py
│   │   ├── profiling.py
│   │   ├── result_contract.py
│   │   ├── runtime_interface.py
│   │   ├── safeguards.py
│   │   ├── segmentation.py
│   │   └── vision_models.py
│   ├── README.md
│   ├── requirements.txt
│   └── run.py
├── EXP/
│   ├── cache/ [省略：版本库、缓存、第三方安装或虚拟环境]
│   ├── implementation/
│   │   ├── cli_check/
│   │   │   ├── cli_logs/
│   │   │   │   └── logs_2.sqlite
│   │   │   ├── account.json
│   │   │   ├── help.txt
│   │   │   └── models.json
│   │   ├── qavit_v2_20261026/
│   │   │   ├── before.json
│   │   │   ├── confirmed_scope.json
│   │   │   ├── implementation_manifest.json
│   │   │   ├── protection_audit.json
│   │   │   └── REPORT.md
│   │   ├── self_containment/
│   │   │   ├── read_isolation/
│   │   │   │   └── sitecustomize.py
│   │   │   ├── audit_delivery_framework.py
│   │   │   ├── capture_reference.py
│   │   │   ├── CHANGES.md
│   │   │   ├── copied_sources.json
│   │   │   ├── dependency_closure.json
│   │   │   ├── diagnose_subscription.py
│   │   │   ├── licenses_and_metadata.json
│   │   │   ├── PLAN.md
│   │   │   ├── post_search_changes.json
│   │   │   ├── repair_checks.py
│   │   │   ├── runtime_dependencies.json
│   │   │   ├── validate_migration.py
│   │   │   └── [省略文件：运行日志/进程痕迹 5 项]
│   │   ├── v2_structure_20261027/
│   │   │   ├── acceptance_contracts/
│   │   │   │   ├── profile/
│   │   │   │   │   ├── evidence/ [省略：批量实验运行产物]
│   │   │   │   │   ├── method_library.json
│   │   │   │   │   └── state.json
│   │   │   │   └── contracts.json
│   │   │   ├── after_contracts/
│   │   │   ├── after_contracts_fixed/
│   │   │   │   ├── profile/
│   │   │   │   │   ├── evidence/ [省略：批量实验运行产物]
│   │   │   │   │   ├── method_library.json
│   │   │   │   │   └── state.json
│   │   │   │   └── contracts.json
│   │   │   ├── before_contracts/ [省略：历史实现副本/快照]
│   │   │   ├── before_contracts_valid/ [省略：历史实现副本/快照]
│   │   │   ├── contracts/
│   │   │   │   ├── profile/
│   │   │   │   │   ├── evidence/ [省略：批量实验运行产物]
│   │   │   │   │   ├── method_library.json
│   │   │   │   │   └── state.json
│   │   │   │   └── contracts.json
│   │   │   ├── final_contracts/
│   │   │   │   ├── profile/
│   │   │   │   │   ├── evidence/ [省略：批量实验运行产物]
│   │   │   │   │   ├── method_library.json
│   │   │   │   │   └── state.json
│   │   │   │   └── contracts.json
│   │   │   ├── review_fixtures/
│   │   │   │   ├── budget/
│   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   │   │   ├── checkpoint.json
│   │   │   │   │   └── commit.lock
│   │   │   │   ├── ranking/
│   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   │   │   ├── checkpoint.json
│   │   │   │   │   ├── commit.lock
│   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   │   │   └── retrieval/
│   │   │   │       ├── cards/ [省略：批量实验运行产物]
│   │   │   │       ├── commits/ [省略：批量实验运行产物]
│   │   │   │       ├── checkpoint.json
│   │   │   │       └── commit.lock
│   │   │   ├── acceptance_structure.json
│   │   │   ├── before_files.json
│   │   │   ├── before_interfaces.json
│   │   │   ├── changes.json
│   │   │   ├── compare_and_inventory.py
│   │   │   ├── contract_comparison.json
│   │   │   ├── delivery.json
│   │   │   ├── delivery_structure.json
│   │   │   ├── document_links.json
│   │   │   ├── document_path_updates.json
│   │   │   ├── environment.json
│   │   │   ├── extract_tools.py
│   │   │   ├── finalize.py
│   │   │   ├── finish_dependencies.py
│   │   │   ├── history_hashes.json
│   │   │   ├── history_verification.json
│   │   │   ├── method_comparison.json
│   │   │   ├── removed_tool_framework.json
│   │   │   ├── reorganize.py
│   │   │   ├── REPORT.md
│   │   │   ├── review_cases.json
│   │   │   ├── review_cases.py
│   │   │   ├── structural_changes.json
│   │   │   ├── structure.json
│   │   │   ├── structure_audit.json
│   │   │   ├── structure_audit_fixed.json
│   │   │   ├── structure_final.json
│   │   │   └── [省略文件：二进制/模型/归档产物 1 项；运行日志/进程痕迹 24 项]
│   │   ├── build_delivery.py
│   │   ├── environment.json
│   │   ├── finalization_after.json
│   │   ├── finalization_before.json
│   │   ├── formal_start_snapshot.json
│   │   ├── monitor.py
│   │   ├── PLAN.md
│   │   ├── README.md
│   │   └── xlsx.json
│   ├── inputs/
│   │   └── deitb_training_free_static_w4a4_20261106_r1/
│   │       ├── assets_preflight.json
│   │       ├── cli167_compatibility.json
│   │       ├── cli167_live_preflight.json
│   │       ├── collect_report.py
│   │       ├── evidence_context_bounds_diagnostics.json
│   │       ├── final_framework_report_path.txt
│   │       ├── final_process_verification.json
│   │       ├── frozen_custom_package_reuse_gap.json
│   │       ├── input.json
│   │       ├── intervention_001.json
│   │       ├── intervention_002.json
│   │       ├── intervention_002.patch
│   │       ├── intervention_002_before.json
│   │       ├── intervention_003.json
│   │       ├── intervention_003.patch
│   │       ├── intervention_003_before.json
│   │       ├── intervention_004.json
│   │       ├── intervention_004.patch
│   │       ├── intervention_004_before.json
│   │       ├── intervention_005.json
│   │       ├── intervention_005.patch
│   │       ├── intervention_005_before.json
│   │       ├── intervention_005_monitor_stop.json
│   │       ├── intervention_005_stopped.json
│   │       ├── intervention_006.json
│   │       ├── intervention_006.patch
│   │       ├── intervention_006_before.json
│   │       ├── monitor.py
│   │       ├── monitor_latest.json
│   │       ├── monitor_stream.py
│   │       ├── preflight.json
│   │       ├── recorder_hotspot_profile.json
│   │       ├── recorder_scope_fix_diagnostics.json
│   │       ├── repair_handoff_gap.json
│   │       ├── research_report.txt
│   │       ├── round1_execution_facts.json
│   │       └── [省略文件：大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 7 项]
│   ├── outputs/
│   │   ├── deitb_static_w4a4_closed_loop_20261030_r1/
│   │   │   ├── ACCEPTANCE_RESULTS.json
│   │   │   ├── DELIVERY_CHECK.json
│   │   │   ├── EFFICIENCY_DETAILS.json
│   │   │   ├── FINAL_STATE.json
│   │   │   ├── KEY_CARDS.json
│   │   │   ├── MODEL_CALLS.json
│   │   │   ├── REPORT.md
│   │   │   ├── RUN_AUDIT.json
│   │   │   ├── run_config.json
│   │   │   ├── SCIENCE_AUDIT.json
│   │   │   ├── SCIENCE_AUDIT.md
│   │   │   ├── science_audit_seq000000068.json
│   │   │   ├── science_audit_seq000000227.json
│   │   │   ├── science_audit_seq000000238.json
│   │   │   ├── science_audit_seq000000323.json
│   │   │   ├── science_audit_seq000000403.json
│   │   │   ├── science_audit_snapshot.py
│   │   │   ├── SCIENCE_REVIEW_FINAL.json
│   │   │   ├── SCIENCE_REVIEW_H1.json
│   │   │   ├── SCIENCE_REVIEW_H2_H3.json
│   │   │   └── science_review_h2_h3.py
│   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│   │   │   ├── asset_preflight.json
│   │   │   ├── audit_run.py
│   │   │   ├── budget_extension_authorization.json
│   │   │   ├── changes.diff
│   │   │   ├── collect_delivery_evidence.py
│   │   │   ├── data_manifest.json
│   │   │   ├── delivery_evidence.json
│   │   │   ├── delivery_manifest.json
│   │   │   ├── diagnostic_evidence.json
│   │   │   ├── evolution_round2_handoff.json
│   │   │   ├── failed_request_recovery_preflight.json
│   │   │   ├── final_stop_audit.json
│   │   │   ├── first_blocked_checkpoint.json
│   │   │   ├── first_blocked_status.json
│   │   │   ├── first_blocked_summary.json
│   │   │   ├── invariants.json
│   │   │   ├── l3_host_failure_before.json
│   │   │   ├── l3_refreeze_audit.json
│   │   │   ├── latest_status.json
│   │   │   ├── monitor.jsonl
│   │   │   ├── monitor.py
│   │   │   ├── operator_actions.jsonl
│   │   │   ├── prepare_run.py
│   │   │   ├── prestart_identity.json
│   │   │   ├── prompts.diff
│   │   │   ├── protected_history_check.json
│   │   │   ├── PSEUDOCODE.md
│   │   │   ├── real_identity_audit.json
│   │   │   ├── recorder_failed_before.json
│   │   │   ├── REPORT.md
│   │   │   ├── REPRODUCE.md
│   │   │   ├── retry_failed_diagnostic.py
│   │   │   ├── round1_boundary.json
│   │   │   ├── round1_boundary_verified.json
│   │   │   ├── round2_l3_budget_refusal.json
│   │   │   ├── run_config.json
│   │   │   ├── run_with_boundary_stop.py
│   │   │   ├── session_budget.json
│   │   │   ├── site_catalog.json
│   │   │   ├── source_changes.json
│   │   │   ├── targeted_retry_before.json
│   │   │   ├── verify_run_invariants.py
│   │   │   ├── worker_resources.json
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 19 项]
│   │   └── deitb_static_w4a4_light_20261028_r1/
│   │       ├── interruptions/
│   │       │   ├── clarification_01_answer.json
│   │       │   ├── clarification_01_summary.json
│   │       │   ├── clarification_02_answer.json
│   │       │   └── clarification_02_summary.json
│   │       ├── audit.json
│   │       ├── delivery_summary.json
│   │       ├── final_identity.json
│   │       ├── freeze.json
│   │       ├── monitor.jsonl
│   │       ├── post_run_identity.json
│   │       ├── REPORT.md
│   │       ├── run_summary.json
│   │       ├── shutdown_check.json
│   │       └── TOOL_COVERAGE.md
│   ├── runs/
│   │   ├── deitb_static_w4a4_closed_loop_20261030_r1/
│   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   ├── code/
│   │   │   │   ├── 05b5636749f754bf60c66d426846857e1572813e7a016178e46058c79a0b41bb/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 0cbc042493b8cf75a120cbf30f5eaed3d818551a793a3b7e2ad92531efb85a1a/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   └── a24f18b84118947246ed76b0810ea50cb0a6fb6339a21201826c2480e7c2a519/
│   │   │   │       ├── implementation.py
│   │   │   │       └── manifest.json
│   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   ├── diagnostics/
│   │   │   │   ├── account_preflight_20261031T000929Z/
│   │   │   │   │   ├── diagnostic.json
│   │   │   │   │   ├── failed_A_log_classification.json
│   │   │   │   │   └── REPORT.md
│   │   │   │   ├── network_resume_check_20261031T034445Z/
│   │   │   │   │   └── network.json
│   │   │   │   ├── worker_preflight_20261031T002346Z/
│   │   │   │   │   ├── child_result.json
│   │   │   │   │   ├── diagnostic.json
│   │   │   │   │   └── REPORT.md
│   │   │   │   ├── worker_preflight_20261031T034315Z/
│   │   │   │   │   ├── child_result.json
│   │   │   │   │   └── diagnostic.json
│   │   │   │   └── worker_preflight_reproduction.py
│   │   │   ├── jobs/ [省略：批量实验运行产物]
│   │   │   ├── checkpoint.json
│   │   │   ├── commit.lock
│   │   │   ├── status.json
│   │   │   ├── summary.json
│   │   │   ├── supervisor.lock
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   ├── code/
│   │   │   │   ├── 0b61afad97ac565c9da5c828225028ecb871207c707fa34f30dc559c8cb64691/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 20d1ea9cea2b22431c61b66fb57e8803e2cb9720596f21b7e8b0bb353138060a/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 3579bd06e19a084d0c123c3b86d1c4e4326d9a1a7d2fe6af66ca73c97a83ff92/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 4a2862c2982c811cb1f27e4a94f8975fcbca997c15f57f0d1408fdc38ab0cf6f/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 526b7ba15f9528c32eb57ea413a3da018bab2f3471e046d2eb2599db310d4da2/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 8e005e2edc8136058e1c4131dd06863c8dbc285ea46684d612d62f43a9ff49e6/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── adbd45a345f4b393578985a6758b7a1022b61570ac2b917d00b2497824d2bcd8/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── c02f7190060a944ed9d997801766e9682f6fb7013c42b1b7d9e69ab1b17389b7/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── c48a4ea94e859ce928f6b6adaebf3fd806dd613f1dec48cffb9be42fa8d0213e/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── d87ecd7e34cf778b0e18657ff70650b6e14d0ff43d91c8c4257a185ba9f3c68c/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── db7f86a3fe240b64de69a06cbd4e6405a6cdae672a1ee733ae9c407cde84ba8b/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── f02a2900581b63819b38b771cde0c79a7bbbe73e636a65347c31843db37dc5e9/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── f6ab5891e0342786d1b844e387901c5fe16983d7ebcf9e1bf210533752a40c26/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   └── fa215b97a5330df9e6d1c15e2969b300d4feb413dbbdc9a728b9d3eb7e0e9c29/
│   │   │   │       ├── implementation.py
│   │   │   │       └── manifest.json
│   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   ├── jobs/ [省略：批量实验运行产物]
│   │   │   ├── checkpoint.json
│   │   │   ├── commit.lock
│   │   │   ├── status.json
│   │   │   ├── summary.json
│   │   │   ├── supervisor.lock
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   ├── deitb_static_w4a4_light_20261028_r1/
│   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   ├── jobs/ [省略：批量实验运行产物]
│   │   │   ├── operator_assets/
│   │   │   │   ├── data_view_receipt.json
│   │   │   │   └── imagenet_subset_bound_copy.json
│   │   │   ├── tool_contracts/
│   │   │   │   ├── deit_sites.json
│   │   │   │   ├── profile_distribution.json
│   │   │   │   ├── profile_hardware.json
│   │   │   │   ├── profile_plugin_sites.json
│   │   │   │   ├── profile_ptq_candidates.json
│   │   │   │   ├── profile_sensitivity.json
│   │   │   │   ├── profile_structure.json
│   │   │   │   ├── profile_trial_ablation.json
│   │   │   │   ├── profile_trial_diagnostics.json
│   │   │   │   ├── query_execution_capabilities.json
│   │   │   │   ├── query_method_library.json
│   │   │   │   ├── query_ptq_plugins.json
│   │   │   │   ├── query_ptq_sites.json
│   │   │   │   ├── query_structure.json
│   │   │   │   ├── read_method_document.json
│   │   │   │   └── read_profile_evidence.json
│   │   │   ├── checkpoint.json
│   │   │   ├── commit.lock
│   │   │   ├── status.json
│   │   │   ├── summary.json
│   │   │   ├── supervisor.lock
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   ├── deitb_static_w4a4_r1_r2gen_20261105_r1/
│   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   ├── code/
│   │   │   │   ├── 3618e35e7f9fca331c4be9f1f1fbdf6d93b4d183cd0c103d5ee170ed8aeebfad/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 6df8012422203b28797f53c592aa8a1fc89a48f2fd0c6e98a2028ef79156fefc/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 74dc5704ab47f8a00053bd0126e5a87b6df1cc40727a93e50470ee80378883bb/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 82469dca530ad51e24cf1e8e5d2d2aca6b11f5c4be402135c43217c136607e0c/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 8389f6b3c918b190de22f9f6f50274d0d7c50c63559203bee34821ec5817754c/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 95ce8339358e75f1853eb3a878ad62fc3242ac8782d015bb39e91094d75dd626/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── a00632c70cef32b7d054be7d3be115c49bbc9580999556010aa810f5886ca8c0/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── a2d1856189eb426f60e3c5318eba13ce0f648caf70dff6b420d75aaad71e117c/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── c25af238b92abef597e8f16814d0b37abbbfcd07943fc429445ebca89af97519/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── f8021ae3cb6d548e21545e477552c3afec35f343f25e344960822740c25b2146/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   └── fdde90f17aa159c2c81101ed9938d86723124fb385bec86672f12a59dfa2310e/
│   │   │   │       ├── implementation.py
│   │   │   │       └── manifest.json
│   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   ├── jobs/ [省略：批量实验运行产物]
│   │   │   ├── recording_submissions/
│   │   │   │   └── task-5a35f9544b6f5a6b44b009eb-1-0.json
│   │   │   ├── checkpoint.json
│   │   │   ├── commit.lock
│   │   │   ├── launch_provenance.json
│   │   │   ├── readiness_segment-1.json
│   │   │   ├── segment_monitor.lock
│   │   │   ├── status.json
│   │   │   ├── summary.json
│   │   │   ├── supervisor.lock
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   ├── deitb_static_w4a4_two_rounds_20261031_r1/
│   │   │   ├── fixes/
│   │   │   │   ├── 001_direct_brief/
│   │   │   │   │   ├── before_evidence.json
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── interaction.before.py
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── test_read_delivery_boundaries.before.py
│   │   │   │   │   ├── trace_replay.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 002_budget_scan_race/
│   │   │   │   │   ├── reproduction/
│   │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   │   │   │   ├── checkpoint.json
│   │   │   │   │   │   └── commit.lock
│   │   │   │   │   ├── budget.before.py
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── test_infrastructure.before.py
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│   │   │   │   ├── 003_experiment_resource_feedback/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── failure_evidence.json
│   │   │   │   │   ├── offline_replay_correction.json
│   │   │   │   │   ├── pending_request_replay.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│   │   │   │   ├── 004_bounded_evidence_delivery/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── after_delivery.json
│   │   │   │   │   ├── before_delivery.json
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   ├── test_fixture_correction.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│   │   │   │   ├── 005_vit_interface_fixture/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── cpu_candidate_diagnostic/
│   │   │   │   │   │   ├── request.json
│   │   │   │   │   │   ├── result.json
│   │   │   │   │   │   ├── scope.json
│   │   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   │   ├── before_failure.json
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 006_bulk_numeric_delivery/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── drain.json
│   │   │   │   │   ├── operator_check_corrections.json
│   │   │   │   │   ├── real_evidence_replay.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│   │   │   │   ├── 007_result_snapshot/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── after_result_traversal.json
│   │   │   │   │   ├── before_result_traversal.json
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 008_test_alignment_and_preflight_retry/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── changes.patch
│   │   │   │   │   ├── diagnosis.json
│   │   │   │   │   ├── monitor.before.py
│   │   │   │   │   ├── run_cpu_regression.before.py
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   ├── source_change_scope.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 009_native_fixture_catalog/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── before_failure.json
│   │   │   │   │   ├── change.diff
│   │   │   │   │   ├── returned_engineering_failure.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 010_irrelevant_read_filter/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── benchmark.json
│   │   │   │   │   ├── change.diff
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── measurement_identity_unchanged.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 011_reference_own_failure/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── actual_delivery.json
│   │   │   │   │   ├── change.diff
│   │   │   │   │   ├── initial_change.diff
│   │   │   │   │   ├── initial_source_after.json
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── measurement_identity_unchanged.json
│   │   │   │   │   ├── observed_failure.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│   │   │   │   ├── 012_comparison_failure_type/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── change.diff
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── measurement_identity_unchanged.json
│   │   │   │   │   ├── observed_failure.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 013_bounded_specs_sources/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── actual_first_calls.json
│   │   │   │   │   ├── actual_input_replay.json
│   │   │   │   │   ├── change.diff
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── observed_read_cost.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 014_exact_evidence_repetition/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── actual_next_call.json
│   │   │   │   │   ├── actual_replay.json
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── observed_failure.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   │   ├── 015_context_ancestor_reuse/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── actual_benchmark.json
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│   │   │   │   ├── 016_documented_delivery_bounds/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   └── source_before.json
│   │   │   │   ├── 017_resume_elapsed_deadline/
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── resume_before.py
│   │   │   │   │   └── validation.json
│   │   │   │   ├── 018_fixture_documentation/
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── EXECUTION.before.md
│   │   │   │   │   ├── fixture_fact_check.json
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   └── source_before.json
│   │   │   │   ├── 019_numeric_deadline/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── checkpoint_return.json
│   │   │   │   │   ├── failed_full_regression.json
│   │   │   │   │   ├── intervention.json
│   │   │   │   │   ├── measurement_after.json
│   │   │   │   │   ├── registration_attempt01.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 5 项]
│   │   │   │   ├── 020_read_field_feedback/
│   │   │   │   │   ├── before/ [省略：历史实现副本/快照]
│   │   │   │   │   ├── actual_read_replay_after.json
│   │   │   │   │   ├── actual_read_replay_before.json
│   │   │   │   │   ├── changes.diff
│   │   │   │   │   ├── intervention.json
│   │   │   │   │   ├── measurement_before.json
│   │   │   │   │   ├── observed_failures.json
│   │   │   │   │   ├── source_after.json
│   │   │   │   │   ├── source_before.json
│   │   │   │   │   ├── stop_intent.json
│   │   │   │   │   └── [省略文件：运行日志/进程痕迹 3 项]
│   │   │   │   └── 021_recorder_task_log_boundary/
│   │   │   │       ├── before/ [省略：历史实现副本/快照]
│   │   │   │       ├── actual_recorder_context_replay.json
│   │   │   │       ├── changes.diff
│   │   │   │       ├── intervention.json
│   │   │   │       ├── measurement_before.json
│   │   │   │       ├── root_cause.json
│   │   │   │       ├── source_after.json
│   │   │   │       ├── source_before.json
│   │   │   │       └── [省略文件：运行日志/进程痕迹 5 项]
│   │   │   ├── preflight/
│   │   │   │   ├── assets.json
│   │   │   │   ├── complete.json
│   │   │   │   ├── drain_context_fix.json
│   │   │   │   ├── drain_delivery_fix.json
│   │   │   │   ├── drain_fixture_fix.json
│   │   │   │   ├── drain_resource_fix.json
│   │   │   │   ├── judger_transport.json
│   │   │   │   ├── prepare_before_metadata_fix.py
│   │   │   │   ├── recorder_transport.json
│   │   │   │   ├── resources.json
│   │   │   │   ├── site_catalog.json
│   │   │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│   │   │   ├── run/
│   │   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   │   ├── code/
│   │   │   │   │   ├── 0ab2093b694f47225229d1f5fbb9b4f0d09e6f1b88ef5ac107d1b8914e4de52f/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 15f4e49527f63cfec72bada1b78438df1db96763bc38de1ccc74831814422a9e/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 1b199fb21fee93b49d2cc0cb56445bbdca5e462e8714c9f5e303d1d191f7d072/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 21adce0e66280bcacc47f3de52d17b5bf802426f0a54461b4b052dd3a524e823/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 3d4764554369334f57020b7a9fde3673b5d771137d25292f3609ff5e08aba467/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 41dad219a3a8b78ff9b7e6cbdb2dd7c16d501b93571aec58d5107a73f9e9ede6/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 52b36de0bbb9b25cfb5c76df84f45f806ecd9a633aee8c862909106e89665b4b/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 756c05612f18067dc9c12dc08c3d7e6ab5fce72579fc49ebb52f22a2928739bf/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 7d7319ade9bc68ea91a2cac8dcc462c779b6f7615fed9d53fbf36e6e0b4da823/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 8488dd42306ed38538dc91346403f387e369b378121aaca9f67cc1ccb9a53019/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 8747afd5c9ffaa3ec87dc6039fdd68158515aa4e8a0e4ed63a6bd9df806e1938/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── 93de5df56b73f5b544919602e7cd261e4edf7dd90e7a83eaa698bc3f0769a373/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── a84c24873b24c011510529daff7d71512d24b66845a5eee0bfae2160fe25d9ee/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── b51e3b2bfe9fe407f4ced05091f21c9cb516ebc798351626b5413867f83625b0/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── bcc306550a5ad7c77a5a7d03b1c971f4bb222fd6872a572c20eb75b51d9609bd/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── bf468c51620152a9cd609b6b23f5cb38aab3f367f457691a6e4771797082df95/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── c290a80dda9dabc76494c0e0c8c2dc48e82973084fb73a156f7052bf6b33504a/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── c42a6d7e8ab668249c2b6046c5d4f0adff6f101902b664814c5be7c483d0e55b/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── d7f0d01a24684ca23529e5c62ada406e08da5d84d24539c03e36bce79321de5b/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── dc702d2b704a9be2bcbb564f6ec0e97715eb1c9bc30ac4d1a5ffd6e9400ac204/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── dd05fdc839c8b8396af3caeca8c7296073303ae9c30de9af05b53539deb01cea/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── e1b931b8264ea10470a47f9af2383535edf929601a8662c83ddeeab73b5b5637/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── e210b109e29ea12038a3c975265edd2e8ecec1042f65ef20ac5659a423bb800c/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── f5a88b1360b8178e242209a74e890eff5aa36727a9090573820842ffa2ca9590/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   ├── f89ef275e798dbe1ccfc2adc33b3da2a812070523348a972f68124a0e94a4a70/
│   │   │   │   │   │   ├── implementation.py
│   │   │   │   │   │   └── manifest.json
│   │   │   │   │   └── f94af4bcf9b70ed0e855f75b6b1a371ee3c9a786da3005474cab93d258db5b9e/
│   │   │   │   │       ├── implementation.py
│   │   │   │   │       └── manifest.json
│   │   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│   │   │   │   ├── checkpoint.json
│   │   │   │   ├── commit.lock
│   │   │   │   ├── status.json
│   │   │   │   ├── summary.json
│   │   │   │   ├── supervisor.lock
│   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   │   ├── storage_recovery_20261101/
│   │   │   │   ├── actual_resume_confirmation.json
│   │   │   │   ├── additional_cache_candidates.json
│   │   │   │   ├── budget_after_extension.json
│   │   │   │   ├── cleanup.json
│   │   │   │   ├── combined_write_after_historical_archive.json
│   │   │   │   ├── file_creation_after_archive.json
│   │   │   │   ├── file_creation_test.json
│   │   │   │   ├── finalize_audit_archived_assets.diff
│   │   │   │   ├── finalize_audit_before_archived_assets.py
│   │   │   │   ├── historical_archive_authorization.json
│   │   │   │   ├── historical_archive_manifest.json
│   │   │   │   ├── historical_archive_preparation.json
│   │   │   │   ├── historical_archive_removal.json
│   │   │   │   ├── historical_archive_review.json
│   │   │   │   ├── launch_intent.json
│   │   │   │   ├── obsolete_codex_01501_manifest.json
│   │   │   │   ├── obsolete_codex_01501_receipt.json
│   │   │   │   ├── payload_write_test.json
│   │   │   │   ├── recurrence_trace_check.json
│   │   │   │   ├── REPORT.md
│   │   │   │   ├── restore_historical_archive.py
│   │   │   │   ├── resume_after_archive_preflight.json
│   │   │   │   ├── resume_preflight.json
│   │   │   │   ├── summarize_trace.diff
│   │   │   │   ├── summarize_trace_before.py
│   │   │   │   ├── summarize_trace_before_missing_request.py
│   │   │   │   ├── summarize_trace_missing_request.diff
│   │   │   │   ├── trace_index_check.json
│   │   │   │   └── [省略文件：二进制/模型/归档产物 2 项；运行日志/进程痕迹 10 项]
│   │   │   ├── validation/
│   │   │   │   ├── benchmark_read_ranges.py
│   │   │   │   ├── context_card_cache_baseline.txt
│   │   │   │   ├── context_card_cache_trial.json
│   │   │   │   ├── context_card_cache_trial.txt
│   │   │   │   ├── context_card_cache_trial_source.py
│   │   │   │   ├── context_profile.json
│   │   │   │   ├── context_profile.prof
│   │   │   │   ├── context_profile_top.txt
│   │   │   │   ├── cpu_all_04326b82c2048d7d.json
│   │   │   │   ├── cpu_all_13014a634c852723.json
│   │   │   │   ├── cpu_all_142b07f0559dafc7.json
│   │   │   │   ├── cpu_all_2fb6c8a744a71a72.json
│   │   │   │   ├── cpu_all_4914ed6246a5f4b1.json
│   │   │   │   ├── cpu_all_77c74f4a3c768aa6.json
│   │   │   │   ├── cpu_all_7bc1519b0445bdae.json
│   │   │   │   ├── cpu_all_aebe441a6f4b4d0b.json
│   │   │   │   ├── cpu_all_bfc4f0514141b732.json
│   │   │   │   ├── cpu_all_c6898efc5e2c2fed.json
│   │   │   │   ├── cpu_all_edccde00edcf4405.json
│   │   │   │   ├── cpu_auth_isolated_c6898.json
│   │   │   │   ├── cpu_lock_isolated_rerun.json
│   │   │   │   ├── evolution_context_profile.json
│   │   │   │   ├── evolution_context_profile.txt
│   │   │   │   ├── final_closure_check.json
│   │   │   │   ├── first_real_evidence_audit.json
│   │   │   │   ├── h3_first_output_diagnostic.json
│   │   │   │   ├── historical_check_midrun.json
│   │   │   │   ├── l2_b8c832_verified_interface.json
│   │   │   │   ├── l2_output_repetition_diagnostic.json
│   │   │   │   ├── l2_revision1_output_diff.json
│   │   │   │   ├── position_context_profile.prof
│   │   │   │   ├── read_ranges_benchmark.json
│   │   │   │   ├── recorder_blocker_diagnostic.json
│   │   │   │   ├── second_real_evidence_audit.json
│   │   │   │   ├── unit_transport.lock
│   │   │   │   └── [省略文件：运行日志/进程痕迹 29 项]
│   │   │   ├── CURRENT_PSEUDOCODE.md
│   │   │   ├── data_choice_confirmation.json
│   │   │   ├── data_manifest.json
│   │   │   ├── FINAL_AUDIT.json
│   │   │   ├── finalize_audit.py
│   │   │   ├── host_cost_samples.jsonl
│   │   │   ├── input.json
│   │   │   ├── INTERVENTIONS.md
│   │   │   ├── launch.py
│   │   │   ├── LIVE_STATUS.json
│   │   │   ├── monitor.py
│   │   │   ├── operator_actions.jsonl
│   │   │   ├── prepare.py
│   │   │   ├── PRESTART_REPORT.md
│   │   │   ├── QUOTA_STOP_REPORT.md
│   │   │   ├── REPORT.md
│   │   │   ├── resume_01.py
│   │   │   ├── resume_02.py
│   │   │   ├── resume_03.py
│   │   │   ├── resume_04.py
│   │   │   ├── resume_05.py
│   │   │   ├── resume_06.py
│   │   │   ├── resume_07.py
│   │   │   ├── resume_08.py
│   │   │   ├── resume_09.py
│   │   │   ├── resume_10.py
│   │   │   ├── resume_11.py
│   │   │   ├── resume_12.py
│   │   │   ├── resume_13.py
│   │   │   ├── resume_14.py
│   │   │   ├── resume_15.py
│   │   │   ├── resume_16.py
│   │   │   ├── resume_after_quota.py
│   │   │   ├── run_cpu_regression.py
│   │   │   ├── source_before.json
│   │   │   ├── summarize_trace.py
│   │   │   ├── TASK_TIMELINE.json
│   │   │   ├── TRACE_INDEX.json
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 3 项；运行日志/进程痕迹 42 项]
│   │   ├── deitb_static_w4a4_two_rounds_20261104_r2/
│   │   │   ├── cards/ [省略：批量实验运行产物]
│   │   │   ├── code/
│   │   │   │   ├── 04311f58a74c985bdd0770fbe53c7c4b4e68e5b80e6bddcf2096197ffe653db4/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 0e98abb03f83b1159f2477b384a06b4bd2d7a9d3a498875d573ac4fccee02929/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 116ed5fe1d382cef6dc27b8b97ab071bc898713abe43c441980cba4eea6fac89/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 14b84df7b8c6d3519158bfd9042b7fc9fa099b6d572c6a94259fa6c010ea4c17/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 33591be951e0e9e894ed4abf3c1fff84ec94fa7a6bce78eec6057ef7292ae6c1/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 41f58ec434b932700c02bb4a7b524e03d4e003b45f0e192de782da81e6e84947/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 5e669665a948ca7d330b75c55d5d8064c09ca4a3b88effef0529604ccadbe658/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 62162446e161f12e861b4f49df2bfbc6180de00bb147491f08e825cab844b064/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 6c3edfb530d09dfbc67bd4888e6ef7f71d00ddc3dfd6ddbe9e498b4a35f37efe/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 7e45c0cd0faa315eb04dcf811ee1d69bd8f98f9fb1a7252b38c056eae09561d7/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 89c9509f4410485f5fca6ff174ec89416c6a21f3e5a93fda1747bb4ae51bc7c1/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 8c2dc382855c485ab2f1b977577c4fc5ea4d24cd1acfbd0169eb8beb7e1dd0de/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 90aa7287cf63e4a031639b1a6cb361799ed26f43d779ccb9674b99a01afa74c6/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 933fa5c4fd32f9bd754ff6534f0ff5a849ca73860627f699cb3d7eb94cad4108/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 96b6844c3a4f647c0b2fd5c30e9ef1cbcbe026282c005640ee9e2b858c9f6355/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 987bf192492981cd87b821eea9e4a42bcf9a7783e11d8b5b232bc8e692868020/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── 9a1859c03841425ccc1e1174975b2cff9779df915fcf18dbed0fc234cce7ddd0/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── a2ebaedb1fa423dfa0f3e47883ec6bfeff16d9b530022dfbcb00d5a9e91a5e46/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── a883917271164856ba7b618a61ffdd8040c763d78ec1db45472297ae72237846/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── ac950ee2b9ee76db9c6aca5d9e529f05142a399646d7b24515096d88270a3569/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── b4d0cb25e44a609f02903dac067607c58470a67ba5a8d4e6b92ed41c5851f672/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── b6bc7cf6a1f4377764dab307d7db969b24ca39c4a9e2f67c2b9585f280aad489/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── bb0c396b2ac28faf786476ba9a8cde6fd471a19b19110e81d357e6efd0028a07/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── c01c23f96fa6a7eb6dea877d9dbaf2b5d114ab8106eb480304bc213d5abf6343/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── cc48c3751ab529bea94798ab3d79ba66e6e5b0678d0a123fbc3dcb75fda922cb/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── d1ab60bc4912acbb0f91f9750c0c22a3845195f5d8050a779c890c13af2c6608/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── dc4806ec58ded3a8cb5d07a14043e04fc91de723ed2adc8bcd5edb22d98de4a8/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── f4be1c6a6d272ba59be462fd95ec8922839a57b1c0404114d4d773ddcf567aeb/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── f4faff757b642bc42500fda16113632ddd38f4f5acea7e2b73667aa43a7e4886/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── fabff65795a9b53e0143c282716352dbf3124ceaf00964e117dea692ead8496b/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   ├── fc28cca864f30f2afbdabf69fac7964e77e168fe388a86a10f99ad120a4b9a64/
│   │   │   │   │   ├── implementation.py
│   │   │   │   │   └── manifest.json
│   │   │   │   └── fd409a6cfd9d7d02514dc49499a0877cede23408f8d4751b334928cb35027881/
│   │   │   │       ├── implementation.py
│   │   │   │       └── manifest.json
│   │   │   ├── commits/ [省略：批量实验运行产物]
│   │   │   ├── jobs/ [省略：批量实验运行产物]
│   │   │   ├── checkpoint.json
│   │   │   ├── commit.lock
│   │   │   ├── status.json
│   │   │   ├── summary.json
│   │   │   ├── supervisor.lock
│   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │   └── deitb_training_free_static_w4a4_20261106_r1/
│   │       ├── cards/ [省略：批量实验运行产物]
│   │       ├── code/
│   │       │   ├── 00cdf6ab873d30ab668d23259ea255f2a51e489a98a75f1ddca3e81d1c51f2f1/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 112e764cb4940c558a6d90624976b11b257e15ed34d3f456d514b067ed085741/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 157222de9c8c895fc0b3afe188649a4c856872c71a78ef0007dac702eea44c78/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 177fec9a6650f0e069b3f5cc275b145240b643418d41e72115cfa695c6111e95/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 26b900a85b783db9ce7352ecbaf03a793874431dc1c3ede5cc101f2084759fe7/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 4e647c183946909996d6045259b05a87f428a94c9d17ef0229bd921ac98df2fd/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 5b6bfa2688a2f43d3687f2b9fb5fc6c6f1b565a53fc3917ef957815ab66f3b58/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 5dff7bd62033a86577f61c9c9cc8939c8e892db97787c79f36c423c0b3e6302e/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 704f1841f0afb6c77fd52852ae5ff4a1ad1b57d6e810320f8c596afedfa677c1/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 72b15d395b46dd22bb83499fac3d2edc8c0ea13a8cb1db20db7356efb02506ac/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 72bf93bbe2c57a034deb18740cc1045b29e39702aaa96e2e4c0251a73b5441b0/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 793e5d04a342445b02c64c358d992570b82d9f15dbb6f7d26ed22884ccce0383/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 83ae74794a9a5c5c7dfd48c3e2d8d1f04018f385c6bfe71dd0e0a0ad0d4938d3/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 89aee07a5cbaa871dae0babd90baf282915488e1691ece00715d89f1b44fc099/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 8b78b59327d4ff5fe242faf7999cf66ae73e3ff8df6715b7e6375bdc5398683c/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 9697f416eae9f81cc13b8f5266fef82e2664b5ddd540c5f416a03fee13a66f22/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 96d5ec3f78160df3921b9969071e69f08a88bb7729e3ee73d20471bb7e8bde38/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── 98067013310b1bbf338714d294006e63fffb6e509d02022d4933e2f971fc8dd9/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── b3d475c4ee215e069ec35f0937d006dad39c21ce4b51e3c5c0ce7838ca0841b2/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── ba0975031462cf17bbe1c756b887574be3e656f6129de810a31197b8e68bf4b5/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── c3644d021d35a45ffa155ea0ac4c832bbda37c9049c96c4efefefdc9f909bad1/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── c7bcc7b56a99c238b3208ec2d0fa58abf99b8abd8503ac5ee48c7399a54e7255/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── c95814bc213d23769557660552c8469cbd866b4cd615fa9e9d23cf3406d4f13b/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── d4f265fb6ed058cc7992cc4cc97cb169be0b7cf5f48f6f89395aa8c0b33514e7/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   ├── d57408e3246c36fb1b55f963419d15b3f76aef685e35d234df8acfe64342baa4/
│   │       │   │   ├── implementation.py
│   │       │   │   └── manifest.json
│   │       │   └── d66229e9badcb1f9ca0baf67760acb6a7ef32112caecc93dea4f2d00b4603b86/
│   │       │       ├── implementation.py
│   │       │       └── manifest.json
│   │       ├── commits/ [省略：批量实验运行产物]
│   │       ├── jobs/ [省略：批量实验运行产物]
│   │       ├── recording_submissions/
│   │       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│   │       ├── checkpoint.json
│   │       ├── commit.lock
│   │       ├── segment_monitor.lock
│   │       ├── status.json
│   │       ├── supervisor.lock
│   │       └── [省略文件：大于1MiB的实验结果/快照 2 项]
│   └── validation/
│       ├── adaptive_finish_20261106_r1/
│       │   ├── gpu_components/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── gpu_components_r2/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── BEFORE.json
│       │   ├── before_diff.patch
│       │   ├── before_status.txt
│       │   ├── CHANGE_SCOPE.json
│       │   ├── DELIVERY.json
│       │   ├── FINAL_SOURCE_IDENTITY.json
│       │   ├── FINAL_STATE.json
│       │   ├── GPU_RESULT.json
│       │   ├── gpu_smoke.py
│       │   ├── HARDWARE_DISCOVERY.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── ROLLBACK.md
│       │   ├── STRUCTURE_AUDIT.json
│       │   ├── STRUCTURE_AUDIT_FINAL.json
│       │   └── [省略文件：二进制/模型/归档产物 1 项；大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 6 项]
│       ├── ade_buffer_guard_v1/
│       │   ├── config.json
│       │   ├── gradient_updates.json
│       │   ├── plan.json
│       │   ├── request.json
│       │   ├── started.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 1 项]
│       ├── cli166_resume_20261031_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── smoke_judger/
│       │   │   ├── cli/
│       │   │   │   ├── cli_logs/
│       │   │   │   │   └── logs_2.sqlite
│       │   │   │   ├── answer.json
│       │   │   │   ├── instructions.txt
│       │   │   │   └── schema.json
│       │   │   ├── llm_audit.json
│       │   │   ├── request.json
│       │   │   └── response.json
│       │   ├── smoke_recorder/
│       │   │   ├── cli/
│       │   │   │   ├── cli_logs/
│       │   │   │   │   └── logs_2.sqlite
│       │   │   │   ├── answer.json
│       │   │   │   ├── instructions.txt
│       │   │   │   └── schema.json
│       │   │   ├── llm_audit.json
│       │   │   ├── request.json
│       │   │   └── response.json
│       │   ├── artifact_hashes.json
│       │   ├── authorization.json
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── cli_preflight.json
│       │   ├── cli_smoke.py
│       │   ├── gpu_preflight.json
│       │   ├── launcher_result_105699.json
│       │   ├── REPORT.md
│       │   ├── research_preserved.json
│       │   ├── result.json
│       │   ├── resume_binding.json
│       │   ├── resume_runner.py
│       │   ├── smoke_summary.json
│       │   ├── source_after.json
│       │   ├── source_before.json
│       │   ├── storage_probe.json
│       │   ├── storage_probe_distributed.json
│       │   ├── storage_recheck.jsonl
│       │   ├── test_resume_runner.py
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 7 项]
│       ├── components_repairs_v1/
│       │   ├── repair_refit.json
│       │   ├── repair_sensitivity.json
│       │   ├── report.json
│       │   └── sharding.json
│       ├── components_v1/
│       │   ├── code/
│       │   │   ├── v0001/
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   └── v0002/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── direct_calls/ [省略：批量实验运行产物]
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── proposals/
│       │   │   └── p001.json
│       │   ├── web/
│       │   │   ├── test0026.json
│       │   │   └── test0027.json
│       │   ├── input_sources.json
│       │   ├── profile_schemas.json
│       │   ├── state.json
│       │   ├── summary.json
│       │   ├── task.json
│       │   └── validation.json
│       ├── components_v2/
│       │   ├── code/
│       │   │   ├── v0001/
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   └── v0002/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── direct_calls/ [省略：批量实验运行产物]
│       │   ├── interventions/
│       │   │   ├── 2026-10-24T14-57-43.993874+00-00.json
│       │   │   ├── 2026-10-24T14-58-32.005963+00-00.json
│       │   │   └── 2026-10-24T15-00-11.126584+00-00.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   ├── feedback/
│       │   │   │   └── ee0f6d94d748d1d0085d2545a5a75d6a57d7c9fc8bcc0f9b6cc8591eb9e1222d.json
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── proposals/
│       │   │   └── p001.json
│       │   ├── trials/ [省略：批量实验运行产物]
│       │   ├── web/
│       │   │   ├── test0043.json
│       │   │   └── test0044.json
│       │   ├── hardware.json
│       │   ├── input_sources.json
│       │   ├── profile_schemas.json
│       │   ├── state.json
│       │   ├── summary.json
│       │   ├── task.json
│       │   └── validation.json
│       ├── continuation_final_ade_subset/
│       │   ├── config.json
│       │   ├── gradient_updates.json
│       │   ├── plan.json
│       │   ├── README.md
│       │   ├── request.json
│       │   ├── started.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 1 项]
│       ├── continuation_final_grids/
│       │   ├── README.md
│       │   └── verification.json
│       ├── continuation_final_reload_v1/
│       │   ├── config.json
│       │   ├── gradient_updates.json
│       │   ├── plan.json
│       │   ├── README.md
│       │   ├── request.json
│       │   ├── started.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 1 项]
│       ├── continuation_memory_collision_v1/
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── input_sources.json
│       │   ├── README.md
│       │   ├── state.json
│       │   ├── task.json
│       │   └── verification.json
│       ├── continuation_v0003_grids/
│       │   ├── README.md
│       │   └── verification.json
│       ├── continuation_v0003_reload_v1/
│       │   ├── config.json
│       │   ├── gradient_updates.json
│       │   ├── plan.json
│       │   ├── README.md
│       │   ├── request.json
│       │   ├── started.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 1 项]
│       ├── contract_interfaces_20261030_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── protected_after.json
│       │   ├── PSEUDOCODE.md
│       │   ├── recent_run_audit.json
│       │   ├── REPORT.md
│       │   ├── reproduce_before.py
│       │   ├── reproduction_before.json
│       │   ├── run_tests.sh
│       │   ├── source_before.json
│       │   ├── structure_audit.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 7 项]
│       ├── controlled_run_20261105_r1/
│       │   ├── intervention01/
│       │   │   ├── after_identity.json
│       │   │   ├── before_identity.json
│       │   │   ├── before_stop.json
│       │   │   ├── changes.patch
│       │   │   ├── issues.json
│       │   │   ├── stopped_confirmation.json
│       │   │   ├── structure.json
│       │   │   ├── verification.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 5 项]
│       │   ├── intervention02/
│       │   │   ├── after_identity.json
│       │   │   ├── before_identity.json
│       │   │   ├── before_stop.json
│       │   │   ├── changes.patch
│       │   │   ├── stopped_confirmation.json
│       │   │   ├── structure.json
│       │   │   ├── verification.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 2 项]
│       │   ├── intervention03/
│       │   │   ├── after_identity.json
│       │   │   ├── before_identity.json
│       │   │   ├── before_stop.json
│       │   │   ├── changes.patch
│       │   │   ├── frozen_identity.json
│       │   │   ├── stopped_confirmation.json
│       │   │   ├── structure.json
│       │   │   ├── USER_BUDGET_AUTHORIZATION.json
│       │   │   ├── verification.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 3 项；运行日志/进程痕迹 1 项]
│       │   ├── intervention04/
│       │   │   ├── after_identity.json
│       │   │   ├── before_identity.json
│       │   │   ├── before_stop.json
│       │   │   ├── changes.patch
│       │   │   ├── frozen_identity.json
│       │   │   ├── notice_actual_request.json
│       │   │   ├── notice_delivery.json
│       │   │   ├── stopped_confirmation.json
│       │   │   ├── structure.json
│       │   │   ├── verification.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 3 项；运行日志/进程痕迹 1 项]
│       │   ├── recovery_review/
│       │   │   ├── BASELINE.json
│       │   │   ├── candidate.patch
│       │   │   └── REVIEW.md
│       │   ├── ARTIFACT_INTEGRITY.json
│       │   ├── audit_generation2.py
│       │   ├── authorize_and_resume.py
│       │   ├── before_identity.json
│       │   ├── BUDGET_AUDIT.json
│       │   ├── CALL_STATISTICS.json
│       │   ├── CHANGES_FROM_BASELINE.patch
│       │   ├── changes_launch.patch
│       │   ├── collect_delivery.py
│       │   ├── CURRENT_HANDOFF.json
│       │   ├── DELIVERY_MANIFEST.json
│       │   ├── FILESYSTEM_CLOCK_OBSERVATION.json
│       │   ├── FINAL_READ_ONLY_REVIEW.json
│       │   ├── FINAL_SOURCE_IDENTITY.json
│       │   ├── FINAL_STOP_CONFIRMATION.json
│       │   ├── GENERATION2_AUDIT.json
│       │   ├── launch.py
│       │   ├── launch_identity.json
│       │   ├── launch_manifest.json
│       │   ├── MEMORY_FINAL.json
│       │   ├── monitor.py
│       │   ├── monitor_history.jsonl
│       │   ├── monitor_latest.json
│       │   ├── observations.jsonl
│       │   ├── PRELAUNCH_VALIDATION.json
│       │   ├── PSEUDOCODE.md
│       │   ├── READ_ONLY_AUDIT.json
│       │   ├── RECORDER1_CHECK.json
│       │   ├── REPORT.md
│       │   ├── resume_intervention04.py
│       │   ├── ROOT_CAUSES.md
│       │   ├── RUN_INPUT.json
│       │   ├── scope_boundary_observations.jsonl
│       │   ├── STOP_CONFIRMATION.json
│       │   ├── structure.json
│       │   ├── structure_launch.json
│       │   ├── TERMINAL_RESUME_BEFORE.json
│       │   ├── TERMINAL_RESUME_CHECK.json
│       │   ├── TERMINAL_RESUME_CHECK_SETUP_ERROR.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 4 项；运行日志/进程痕迹 13 项]
│       ├── controls_v1/
│       │   ├── expired_run/
│       │   │   ├── contexts/ [省略：批量实验运行产物]
│       │   │   ├── memory/
│       │   │   │   ├── dynamic.json
│       │   │   │   └── static.json
│       │   │   ├── .lock
│       │   │   ├── delivery.json
│       │   │   ├── input_sources.json
│       │   │   ├── state.json
│       │   │   └── task.json
│       │   ├── gpu_group/
│       │   │   ├── code/
│       │   │   │   ├── v0001/
│       │   │   │   │   ├── experiment.py
│       │   │   │   │   ├── parameters.json
│       │   │   │   │   └── version.json
│       │   │   │   ├── v0002/
│       │   │   │   │   ├── experiment.py
│       │   │   │   │   ├── parameters.json
│       │   │   │   │   └── version.json
│       │   │   │   └── v0003/
│       │   │   │       ├── experiment.py
│       │   │   │       ├── parameters.json
│       │   │   │       └── version.json
│       │   │   ├── contexts/ [省略：批量实验运行产物]
│       │   │   ├── interventions/
│       │   │   │   └── 2026-10-24T15-09-51.873869+00-00.json
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── memory/
│       │   │   │   ├── dynamic.json
│       │   │   │   └── static.json
│       │   │   ├── proposals/
│       │   │   │   └── p001.json
│       │   │   ├── input_sources.json
│       │   │   ├── state.json
│       │   │   └── task.json
│       │   ├── sandbox/
│       │   │   └── allowed.txt
│       │   ├── tampered_code/
│       │   │   ├── experiment.py
│       │   │   ├── parameters.json
│       │   │   └── version.json
│       │   ├── protected.txt
│       │   └── report.json
│       ├── deit_preflight_v1/
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       ├── deit_preflight_v2/
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       ├── deit_preflight_v3/
│       │   ├── ade_fp_control.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       ├── deit_tools_v1/
│       │   ├── code/
│       │   │   └── v0001/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── proposals/
│       │   │   └── p001.json
│       │   ├── input_sources.json
│       │   ├── state.json
│       │   ├── task.json
│       │   └── validation.json
│       ├── draft_preflight_20261029_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── before_hashes.json
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── history_replay.json
│       │   ├── PSEUDOCODE.md
│       │   ├── replay_history.py
│       │   ├── REPORT.md
│       │   └── [省略文件：运行日志/进程痕迹 3 项]
│       ├── e2e_readiness_20261029_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── clarification_before/
│       │   │   ├── driver.py
│       │   │   ├── EXECUTION.md
│       │   │   ├── experiment_requests.py
│       │   │   └── test_code_repair.py
│       │   ├── cuda_before/
│       │   │   └── verification_worker.py
│       │   ├── during_run_before/
│       │   │   ├── experiment_requests.py
│       │   │   └── test_code_repair.py
│       │   ├── recorder_before/
│       │   │   ├── driver.py
│       │   │   ├── EXECUTION.md
│       │   │   └── test_code_repair.py
│       │   ├── reference_before/
│       │   │   ├── EXECUTION.md
│       │   │   ├── experiment_requests.py
│       │   │   └── test_code_repair.py
│       │   ├── runtime_failure_before/
│       │   │   ├── EXECUTION.md
│       │   │   ├── experiment_requests.py
│       │   │   └── test_code_repair.py
│       │   ├── before_hashes.json
│       │   ├── changes.diff
│       │   ├── clarification.diff
│       │   ├── clarification_driver_before_move.py
│       │   ├── clarification_source_review.json
│       │   ├── cuda_guard_probe.py
│       │   ├── cuda_initialization.diff
│       │   ├── cuda_source_review.json
│       │   ├── during_run.diff
│       │   ├── execution_before_layer_contract.md
│       │   ├── layer_contract.diff
│       │   ├── prestart_changes.diff
│       │   ├── READINESS.md
│       │   ├── recorder_draft.diff
│       │   ├── recorder_source_review.json
│       │   ├── reference_rebuild.diff
│       │   ├── reference_rebuild_review.json
│       │   ├── retry_h3b.py
│       │   ├── retry_rejected_cpu.py
│       │   ├── runtime_failure.diff
│       │   ├── runtime_source_review.json
│       │   ├── source_review.json
│       │   ├── storage_write_check.json
│       │   └── [省略文件：运行日志/进程痕迹 24 项]
│       ├── e2e_readiness_20261030_r1/
│       │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── cpu_tmp/
│       │   ├── preflight/
│       │   │   ├── asset_preflight.json
│       │   │   ├── cli_preflight.json
│       │   │   ├── config_check.json
│       │   │   ├── dependencies.json
│       │   │   ├── numerical_backend_comparison.json
│       │   │   ├── preflight_complete.json
│       │   │   ├── resource_before.json
│       │   │   └── site_catalog.json
│       │   ├── acceptance.json
│       │   ├── BUDGET_RECOVERY_REPORT.json
│       │   ├── BUDGET_RECOVERY_REPORT.md
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── cpu_result.json
│       │   ├── handoff_generation_card.json
│       │   ├── handoff_judger_card.json
│       │   ├── handoff_latest_meta.json
│       │   ├── handoff_old_meta.json
│       │   ├── handoff_recorder_request.json
│       │   ├── handoff_recorder_submission.json
│       │   ├── identity_before.json
│       │   ├── identity_candidate.json
│       │   ├── initial_runtime_audit.json
│       │   ├── interruption_state.json
│       │   ├── launch.json
│       │   ├── local_monitor.py
│       │   ├── local_runner.py
│       │   ├── monitor_history.jsonl
│       │   ├── monitor_latest.json
│       │   ├── monitor_process.json
│       │   ├── monitor_recovery_process_1.json
│       │   ├── operator_poll.py
│       │   ├── parser_readonly_audit.json
│       │   ├── prepare_preflight.py
│       │   ├── PRESTART_REPORT.md
│       │   ├── recovery_1.json
│       │   ├── recovery_launch_1.json
│       │   ├── reference_handoff_demo.py
│       │   ├── reference_handoff_proof.json
│       │   ├── REFERENCE_HANDOFF_REPORT.md
│       │   ├── REPORT.md
│       │   ├── rolling_budget_policy.json
│       │   ├── run_binding.json
│       │   ├── run_local.diff
│       │   ├── runner.lock
│       │   ├── runner_status.json
│       │   ├── source_before.json
│       │   ├── source_frozen.json
│       │   ├── stopped_first_monitor_latest.json
│       │   ├── stopped_first_runner_status.json
│       │   ├── structure_audit.json
│       │   ├── task_start.json
│       │   ├── test_local_budget.py
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 9 项]
│       ├── end_to_end_repair_20261105_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── evidence_handoff_fixture/
│       │   │   └── research/
│       │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │       ├── checkpoint.json
│       │   │       ├── commit.lock
│       │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── handoff_delivery/
│       │   │   ├── evidence_handoff_fixture/
│       │   │   │   └── research/
│       │   │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │   │       ├── checkpoint.json
│       │   │   │       ├── commit.lock
│       │   │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   └── evidence_handoff_metrics.json
│       │   ├── handoff_final/
│       │   │   ├── evidence_handoff_fixture/
│       │   │   │   └── research/
│       │   │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │   │       ├── checkpoint.json
│       │   │   │       ├── commit.lock
│       │   │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   └── evidence_handoff_metrics.json
│       │   ├── plan_schema_snapshot/
│       │   │   ├── method_library.json
│       │   │   ├── schema_changes.json
│       │   │   └── state.json
│       │   ├── plan_schema_snapshot_final/
│       │   │   ├── method_library.json
│       │   │   ├── schema_changes.json
│       │   │   └── state.json
│       │   ├── plan_workers_delivery/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── plan_worker_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── plan_workers_final/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── plan_worker_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── two_round_structure_delivery/
│       │   │   ├── two-rounds/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── recording_submissions/
│       │   │   │   │   ├── task-3f986a308b7be00c177262ba-1-0.json
│       │   │   │   │   └── task-80d47df90523e1aaa6ada7f1-1-0.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   └── scripted_structure_result.json
│       │   ├── asset_readability_delivery.json
│       │   ├── before_identity.json
│       │   ├── changed_files.json
│       │   ├── changed_files_delivery.json
│       │   ├── changed_files_final.json
│       │   ├── changes.patch
│       │   ├── changes_delivery.patch
│       │   ├── changes_final.patch
│       │   ├── cli_status_delivery.json
│       │   ├── cli_status_final.json
│       │   ├── DELIVERY.json
│       │   ├── delivery_identity.json
│       │   ├── evidence_handoff_metrics.json
│       │   ├── evidence_handoff_metrics.py
│       │   ├── final_identity.json
│       │   ├── final_identity_v2.json
│       │   ├── NEXT_RUN_INPUT.json
│       │   ├── PSEUDOCODE.md
│       │   ├── readiness.json
│       │   ├── readiness_check.py
│       │   ├── readiness_delivery.json
│       │   ├── readiness_final.json
│       │   ├── REPORT.md
│       │   ├── ROOT_CAUSES.md
│       │   ├── structure_delivery.json
│       │   ├── structure_final.json
│       │   ├── structure_final_v2.json
│       │   ├── validate_plan_workers.py
│       │   ├── validate_two_round_structure.py
│       │   ├── write_delivery_report.py
│       │   └── [省略文件：运行日志/进程痕迹 25 项]
│       ├── experiment_inputs_20261029_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── before_hashes.json
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── data_receipt_replay.json
│       │   ├── prompt_changes.json
│       │   ├── prompts.diff
│       │   ├── protected_hashes.json
│       │   ├── PSEUDOCODE.md
│       │   ├── replay_inputs.py
│       │   ├── REPORT.md
│       │   ├── run_tests.sh
│       │   ├── syntax_check.json
│       │   └── [省略文件：运行日志/进程痕迹 8 项]
│       ├── final_reload_v1/
│       │   ├── config.json
│       │   ├── gradient_updates.json
│       │   ├── plan.json
│       │   ├── README.md
│       │   ├── request.json
│       │   ├── started.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 1 项]
│       ├── finalization_v1/
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── rounds/
│       │   │   └── r008/
│       │   │       └── outcome.json
│       │   ├── delivery.json
│       │   ├── input_sources.json
│       │   ├── README.md
│       │   ├── regression.json
│       │   ├── state.json
│       │   └── task.json
│       ├── generation_loop_20261104_r1/
│       │   ├── after_replay/
│       │   │   ├── arbitrate_request.json
│       │   │   ├── evolve_request.json
│       │   │   ├── generate_request.json
│       │   │   ├── record_request.json
│       │   │   └── RESULT.json
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── before_replay/ [省略：历史实现副本/快照]
│       │   ├── cpu_final/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   ├── 73168374b5ab8d146d32e7d50087be61def7069e339e730884db60a7fa95865d/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   └── a6347615a0142aadb53b5f570f3bc5f4173293896549164f0fa499661daad6ed/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-eab9b23cf6df371e7264c05b-1-0.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_generation/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-e62802bb3ff3e203f4fab789-1-0.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_with_code/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   ├── 55da15795f252d4dc02d31bd44244ce8486ed4e07f255f52c8f47c2decdcff84/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   └── ba384609275e55f4b47fec0c4d3e37c50a355f33bc02cfe5669391f88abe2d80/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_with_code_r2/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   ├── 39c7662f7dcf05d2bd16aca476a05b3902c33ffe039516389f5a760b8afef0eb/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   └── 6ca5c44319b066514d614a06c69647396429415923954835b810c5833494a44c/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-1afcde2c5b576d3dc92420e8-1-0.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── delivery_replay/
│       │   │   ├── arbitrate_request.json
│       │   │   ├── evolve_request.json
│       │   │   ├── generate_request.json
│       │   │   ├── record_request.json
│       │   │   └── RESULT.json
│       │   ├── final_replay/
│       │   │   ├── arbitrate_request.json
│       │   │   ├── evolve_request.json
│       │   │   ├── generate_request.json
│       │   │   ├── record_request.json
│       │   │   └── RESULT.json
│       │   ├── optimized_replay/
│       │   │   ├── arbitrate_request.json
│       │   │   ├── evolve_request.json
│       │   │   ├── generate_request.json
│       │   │   ├── record_request.json
│       │   │   └── RESULT.json
│       │   ├── profile_record/
│       │   │   ├── record.prof
│       │   │   ├── record_request.json
│       │   │   └── RESULT.json
│       │   ├── baseline_revision.txt
│       │   ├── before_manifest.json
│       │   ├── build_report.py
│       │   ├── CALL_COMPARISON.json
│       │   ├── changes.patch
│       │   ├── complexity.json
│       │   ├── cpu_integration.py
│       │   ├── DELIVERY.json
│       │   ├── evidence_draft.py
│       │   ├── HISTORICAL_EVIDENCE.json
│       │   ├── NEXT_RUN_INPUT.json
│       │   ├── PLAN.md
│       │   ├── prompts.diff
│       │   ├── protection.json
│       │   ├── PSEUDOCODE.md
│       │   ├── READINESS.json
│       │   ├── replay_context.py
│       │   ├── REPLAY_RESULT.json
│       │   ├── REPORT.md
│       │   ├── resources_snapshot.json
│       │   ├── structure_audit.json
│       │   ├── structure_final.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 14 项]
│       ├── hessian_v1/
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── input_sources.json
│       │   ├── state.json
│       │   ├── task.json
│       │   └── tool_result.json
│       ├── interface_simplification_20261031_r1/
│       │   ├── cpu_services_01/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── harness.py
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_services_02/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── harness.py
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_services_03/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-1dc4ad9d5b415fe271bf5756-2-1.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_services_04/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-82230b29a97e1a1cc23e135d-2-1.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── input_examples/
│       │   │   ├── code_after.json
│       │   │   ├── code_before.json
│       │   │   ├── evolve_after.json
│       │   │   ├── evolve_before.json
│       │   │   ├── generate_after.json
│       │   │   ├── generate_before.json
│       │   │   ├── plan_l3_after.json
│       │   │   ├── plan_l3_before.json
│       │   │   ├── reference_after.json
│       │   │   ├── reference_before.json
│       │   │   ├── screen_after.json
│       │   │   └── screen_before.json
│       │   ├── changes.patch
│       │   ├── cpu_integration.py
│       │   ├── CURRENT_PSEUDOCODE.md
│       │   ├── delivery_check.json
│       │   ├── historical_draft_reproduction.json
│       │   ├── historical_run_after.json
│       │   ├── historical_run_before.json
│       │   ├── implementation_change.json
│       │   ├── input_comparison.json
│       │   ├── offline_audit.py
│       │   ├── offline_audit_attempt03.py
│       │   ├── REPORT.md
│       │   ├── source_after.json
│       │   ├── source_before.json
│       │   ├── static_check.json
│       │   ├── task_start.json
│       │   ├── tested_identity.json
│       │   ├── validation_summary.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 16 项]
│       ├── json_field_access_v1/
│       │   ├── fixture.json
│       │   ├── live_method.json
│       │   └── result.json
│       ├── memfs_completion_20261030_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── real_llm/
│       │   │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   │   ├── dry_chain/
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   └── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │       ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │       ├── checkpoint.json
│       │   │   │   │       ├── commit.lock
│       │   │   │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   ├── recorder_incremental/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── old_memory.json
│       │   │   │   ├── delayed_next_request.json
│       │   │   │   └── recorder_old_basis_next_request.json
│       │   │   ├── dry_chain_failed_settle/
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   └── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │       ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │       ├── checkpoint.json
│       │   │   │   │       ├── commit.lock
│       │   │   │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   ├── recorder_incremental/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── old_memory.json
│       │   │   │   ├── delayed_next_request.json
│       │   │   │   └── recorder_old_basis_diagnostic.json
│       │   │   ├── dry_materials/
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_incremental/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── initial_request.json
│       │   │   │   │   └── old_memory.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── dry_materials_failed_bootstrap/
│       │   │   │   └── judger_long/
│       │   │   │       └── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │           ├── cards/ [省略：批量实验运行产物]
│       │   │   │           ├── commits/ [省略：批量实验运行产物]
│       │   │   │           ├── checkpoint.json
│       │   │   │           ├── commit.lock
│       │   │   │           └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── dry_materials_failed_round/
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── recorder_incremental/
│       │   │   │       └── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │           ├── cards/ [省略：批量实验运行产物]
│       │   │   │           ├── commits/ [省略：批量实验运行产物]
│       │   │   │           ├── checkpoint.json
│       │   │   │           ├── commit.lock
│       │   │   │           └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── runs/
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   ├── final_task.json
│       │   │   │   │   ├── initial_request.json
│       │   │   │   │   └── read_trace.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   ├── final_task.json
│       │   │   │   │   ├── initial_request.json
│       │   │   │   │   └── read_trace.json
│       │   │   │   ├── recorder_incremental/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── recording_submissions/
│       │   │   │   │   │   │   └── task-c011537df4c396737e583ab9-3-2.json
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   ├── final_task.json
│       │   │   │   │   ├── initial_request.json
│       │   │   │   │   ├── old_memory.json
│       │   │   │   │   └── read_trace.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── acceptance_criteria.json
│       │   │   ├── artifact_integrity.json
│       │   │   ├── check_harness.py
│       │   │   ├── codex_counted.py
│       │   │   ├── harness_cpu.json
│       │   │   ├── metrics.json
│       │   │   ├── phase_ledger.json
│       │   │   ├── prepared.json
│       │   │   ├── REAL_REPORT.md
│       │   │   ├── representative_inputs.json
│       │   │   ├── semantic_results.json
│       │   │   ├── summarize_metrics.py
│       │   │   ├── transport_requests.jsonl
│       │   │   ├── validate_real.py
│       │   │   ├── verify_artifacts.py
│       │   │   └── [省略文件：运行日志/进程痕迹 5 项]
│       │   ├── build_patch.py
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── cpu_acceptance.json
│       │   ├── delivery_check.json
│       │   ├── feedback_input_after.json
│       │   ├── feedback_input_before.json
│       │   ├── feedback_input_comparison.json
│       │   ├── FEEDBACK_REPORT.md
│       │   ├── feedback_scope_comparison.py
│       │   ├── inner_json_after.json
│       │   ├── inner_json_before.json
│       │   ├── INNER_JSON_REPORT.md
│       │   ├── MERGED_REVIEW.md
│       │   ├── overlap_comparison.py
│       │   ├── overlap_input_after.json
│       │   ├── overlap_input_before.json
│       │   ├── overlap_input_comparison.json
│       │   ├── overlap_methods.txt
│       │   ├── protected_assets.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── run_cpu.sh
│       │   ├── source_before.json
│       │   ├── source_frozen.json
│       │   ├── structure.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 7 项]
│       ├── memfs_delivery_20261030_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── examples/
│       │   │   ├── scripted_run/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── code/
│       │   │   │   │   ├── 7790d225fb4f8ae245d4e106718a9b1fec799d550d8a27c0a2eeaac80d33d0eb/
│       │   │   │   │   │   ├── implementation.py
│       │   │   │   │   │   └── manifest.json
│       │   │   │   │   └── 90f0acbe24e44e5c1d378af78989f61a8b628dd6a8ff182f8febb7c6a8c1aaa6/
│       │   │   │   │       ├── implementation.py
│       │   │   │   │       └── manifest.json
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── arbitrate.before_context.json
│       │   │   ├── arbitrate.json
│       │   │   ├── evolve.before_context.json
│       │   │   ├── evolve.json
│       │   │   ├── generate.before_context.json
│       │   │   ├── generate.json
│       │   │   ├── position.before_context.json
│       │   │   ├── position.json
│       │   │   ├── screen.before_context.json
│       │   │   └── screen.json
│       │   ├── examples_final/
│       │   │   ├── scripted_run/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── code/
│       │   │   │   │   ├── 01c843d65bcd3826b73f064d803ca3e359b4deb47daa07e62d006c788b9d0252/
│       │   │   │   │   │   ├── implementation.py
│       │   │   │   │   │   └── manifest.json
│       │   │   │   │   └── bfb802bdc4a0209c0d06755d5721484035c3bf60ced05fcb2388c8c84767436d/
│       │   │   │   │       ├── implementation.py
│       │   │   │   │       └── manifest.json
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── arbitrate.before_context.json
│       │   │   ├── arbitrate.json
│       │   │   ├── code.before_context.json
│       │   │   ├── code.json
│       │   │   ├── evolve.before_context.json
│       │   │   ├── evolve.json
│       │   │   ├── experiment_approval.before_context.json
│       │   │   ├── experiment_approval.json
│       │   │   ├── generate.before_context.json
│       │   │   ├── generate.json
│       │   │   ├── interpret.before_context.json
│       │   │   ├── interpret.json
│       │   │   ├── plan_l3.before_context.json
│       │   │   ├── plan_l3.json
│       │   │   ├── position.before_context.json
│       │   │   ├── position.json
│       │   │   ├── rank.before_context.json
│       │   │   ├── rank.json
│       │   │   ├── record.before_context.json
│       │   │   ├── record.json
│       │   │   ├── reference.before_context.json
│       │   │   ├── reference.json
│       │   │   ├── safeguard.before_context.json
│       │   │   ├── safeguard.json
│       │   │   ├── screen.before_context.json
│       │   │   ├── screen.json
│       │   │   ├── supervise.before_context.json
│       │   │   └── supervise.json
│       │   ├── inputs/
│       │   │   ├── arbitrate.before_context.json
│       │   │   ├── arbitrate.json
│       │   │   ├── code.before_context.json
│       │   │   ├── code.json
│       │   │   ├── evolve.before_context.json
│       │   │   ├── evolve.json
│       │   │   ├── experiment_approval.before_context.json
│       │   │   ├── experiment_approval.json
│       │   │   ├── generate.before_context.json
│       │   │   ├── generate.json
│       │   │   ├── interpret.before_context.json
│       │   │   ├── interpret.json
│       │   │   ├── plan_l3.before_context.json
│       │   │   ├── plan_l3.json
│       │   │   ├── position.before_context.json
│       │   │   ├── position.json
│       │   │   ├── rank.before_context.json
│       │   │   ├── rank.json
│       │   │   ├── record.before_context.json
│       │   │   ├── record.json
│       │   │   ├── reference.before_context.json
│       │   │   ├── reference.json
│       │   │   ├── safeguard.before_context.json
│       │   │   ├── safeguard.json
│       │   │   ├── screen.before_context.json
│       │   │   ├── screen.json
│       │   │   ├── supervise.before_context.json
│       │   │   └── supervise.json
│       │   ├── inputs_release/
│       │   │   ├── arbitrate.before_context.json
│       │   │   ├── arbitrate.json
│       │   │   ├── code.before_context.json
│       │   │   ├── code.json
│       │   │   ├── evolve.before_context.json
│       │   │   ├── evolve.json
│       │   │   ├── experiment_approval.before_context.json
│       │   │   ├── experiment_approval.json
│       │   │   ├── generate.before_context.json
│       │   │   ├── generate.json
│       │   │   ├── interpret.before_context.json
│       │   │   ├── interpret.json
│       │   │   ├── plan_l3.before_context.json
│       │   │   ├── plan_l3.json
│       │   │   ├── position.before_context.json
│       │   │   ├── position.json
│       │   │   ├── rank.before_context.json
│       │   │   ├── rank.json
│       │   │   ├── record.before_context.json
│       │   │   ├── record.json
│       │   │   ├── reference.before_context.json
│       │   │   ├── reference.json
│       │   │   ├── safeguard.before_context.json
│       │   │   ├── safeguard.json
│       │   │   ├── screen.before_context.json
│       │   │   ├── screen.json
│       │   │   ├── supervise.before_context.json
│       │   │   └── supervise.json
│       │   ├── references/
│       │   │   ├── letta-code/
│       │   │   │   └── src/
│       │   │   │       ├── agent/
│       │   │   │       │   ├── prompts/
│       │   │   │       │   │   ├── letta_local_memfs.md
│       │   │   │       │   │   └── letta_root_memfs.md
│       │   │   │       │   ├── subagents/
│       │   │   │       │   │   └── builtin/
│       │   │   │       │   │       ├── memory.md
│       │   │   │       │   │       └── reflection-v2.md
│       │   │   │       │   ├── memory-filesystem.ts
│       │   │   │       │   └── reflection-runs.ts
│       │   │   │       └── cli/
│       │   │   │           └── helpers/
│       │   │   │               ├── reflection-completion.ts
│       │   │   │               ├── reflection-memory-v2.ts
│       │   │   │               ├── reflection-prompt.ts
│       │   │   │               └── reflection-transcript.ts
│       │   │   ├── commit.json
│       │   │   ├── memfs.html
│       │   │   ├── memory.html
│       │   │   ├── package.json
│       │   │   ├── tree.json
│       │   │   └── versions.json
│       │   ├── build_examples.py
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── delivery_check.json
│       │   ├── input_volume.json
│       │   ├── input_volume_final.json
│       │   ├── input_volume_release.json
│       │   ├── interaction_section.py
│       │   ├── new_delivery_tests.py
│       │   ├── protected_after.json
│       │   ├── protected_release.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REFERENCE_NOTES.md
│       │   ├── refresh_examples.py
│       │   ├── REPORT.md
│       │   ├── run_tests.sh
│       │   ├── source_before.json
│       │   ├── structure_audit.json
│       │   ├── structure_audit_release.json
│       │   ├── structure_audit_verified.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 22 项]
│       ├── memfs_followup_20261030_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── manifest_owner_reproduction/
│       │   │   ├── run/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── code/
│       │   │   │   │   └── c6bb49468d2618a829218fbb73bd75abd08a610481ecddc6b6cd1e6d26ac156b/
│       │   │   │   │       ├── implementation.py
│       │   │   │   │       └── manifest.json
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── request.json
│       │   │   ├── result.json
│       │   │   └── tool_result.json
│       │   ├── offline_comparison_dry/
│       │   │   ├── 000_evolve.after_context.json
│       │   │   ├── 000_evolve.before_context.json
│       │   │   ├── 001_generate.after_context.json
│       │   │   ├── 001_generate.before_context.json
│       │   │   ├── 002_interpret.after_context.json
│       │   │   ├── 002_interpret.before_context.json
│       │   │   └── comparison.json
│       │   ├── offline_comparison_dry_02/
│       │   │   ├── 000_evolve.after_context.json
│       │   │   ├── 000_evolve.before_context.json
│       │   │   ├── 001_generate.after_context.json
│       │   │   ├── 001_generate.before_context.json
│       │   │   ├── 002_interpret.after_context.json
│       │   │   ├── 002_interpret.before_context.json
│       │   │   └── comparison.json
│       │   ├── offline_comparison_release/
│       │   │   ├── 000_evolve.after_context.json
│       │   │   ├── 000_evolve.before_context.json
│       │   │   ├── 001_generate.after_context.json
│       │   │   ├── 001_generate.before_context.json
│       │   │   ├── 002_interpret.after_context.json
│       │   │   ├── 002_interpret.before_context.json
│       │   │   ├── 003_interpret.after_context.json
│       │   │   ├── 003_interpret.before_context.json
│       │   │   ├── 004_record.after_context.json
│       │   │   ├── 004_record.before_context.json
│       │   │   ├── 005_record.after_context.json
│       │   │   ├── 005_record.before_context.json
│       │   │   ├── 006_evolve.after_context.json
│       │   │   ├── 006_evolve.before_context.json
│       │   │   ├── 007_evolve.after_context.json
│       │   │   ├── 007_evolve.before_context.json
│       │   │   ├── 008_evolve.after_context.json
│       │   │   ├── 008_evolve.before_context.json
│       │   │   ├── 009_evolve.after_context.json
│       │   │   ├── 009_evolve.before_context.json
│       │   │   ├── 010_evolve.after_context.json
│       │   │   ├── 010_evolve.before_context.json
│       │   │   ├── 011_generate.after_context.json
│       │   │   ├── 011_generate.before_context.json
│       │   │   ├── 012_generate.after_context.json
│       │   │   ├── 012_generate.before_context.json
│       │   │   ├── 013_generate.after_context.json
│       │   │   ├── 013_generate.before_context.json
│       │   │   ├── 014_interpret.after_context.json
│       │   │   ├── 014_interpret.before_context.json
│       │   │   ├── 015_interpret.after_context.json
│       │   │   ├── 015_interpret.before_context.json
│       │   │   ├── 016_interpret.after_context.json
│       │   │   ├── 016_interpret.before_context.json
│       │   │   ├── 017_interpret.after_context.json
│       │   │   ├── 017_interpret.before_context.json
│       │   │   ├── 018_interpret.after_context.json
│       │   │   ├── 018_interpret.before_context.json
│       │   │   ├── 019_interpret.after_context.json
│       │   │   ├── 019_interpret.before_context.json
│       │   │   ├── 020_interpret.after_context.json
│       │   │   ├── 020_interpret.before_context.json
│       │   │   ├── 021_interpret.after_context.json
│       │   │   ├── 021_interpret.before_context.json
│       │   │   ├── 022_interpret.after_context.json
│       │   │   ├── 022_interpret.before_context.json
│       │   │   ├── 023_interpret.after_context.json
│       │   │   ├── 023_interpret.before_context.json
│       │   │   ├── 024_interpret.after_context.json
│       │   │   ├── 024_interpret.before_context.json
│       │   │   ├── 025_interpret.after_context.json
│       │   │   ├── 025_interpret.before_context.json
│       │   │   ├── 026_record.after_context.json
│       │   │   ├── 026_record.before_context.json
│       │   │   ├── 027_record.after_context.json
│       │   │   ├── 027_record.before_context.json
│       │   │   ├── 028_record.after_context.json
│       │   │   ├── 028_record.before_context.json
│       │   │   ├── 029_record.after_context.json
│       │   │   ├── 029_record.before_context.json
│       │   │   ├── 030_record.after_context.json
│       │   │   ├── 030_record.before_context.json
│       │   │   └── comparison.json
│       │   ├── real_llm/
│       │   │   ├── dry_materials/
│       │   │   │   ├── evolution/
│       │   │   │   │   └── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │       ├── checkpoint.json
│       │   │   │   │       ├── commit.lock
│       │   │   │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── judger_long/
│       │   │   │       ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │       │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │       │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │       │   ├── checkpoint.json
│       │   │   │       │   ├── commit.lock
│       │   │   │       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │       └── initial_request.json
│       │   │   ├── dry_materials_02/
│       │   │   │   ├── evolution/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   └── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │       ├── checkpoint.json
│       │   │   │   │       ├── commit.lock
│       │   │   │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   └── judger_long/
│       │   │   │       ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │       │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │       │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │       │   ├── checkpoint.json
│       │   │   │       │   ├── commit.lock
│       │   │   │       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │       └── initial_request.json
│       │   │   ├── dry_materials_03/
│       │   │   │   ├── evolution/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_conflict/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_empty/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── dry_materials_04/
│       │   │   │   ├── evolution/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_conflict/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_empty/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── dry_materials_05/
│       │   │   │   ├── evolution/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_conflict/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_empty/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── dry_materials_final/
│       │   │   │   ├── evolution/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_conflict/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_empty/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── runs/
│       │   │   │   ├── evolution/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── generation/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_delayed/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── judger_long/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_conflict/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── recording_submissions/
│       │   │   │   │   │   │   └── task-c011537df4c396737e583ab9-3-2.json
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   └── initial_request.json
│       │   │   │   ├── recorder_empty/
│       │   │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   │   ├── recording_submissions/
│       │   │   │   │   │   │   └── task-c011537df4c396737e583ab9-2-1.json
│       │   │   │   │   │   ├── checkpoint.json
│       │   │   │   │   │   ├── commit.lock
│       │   │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   │   ├── case_result.json
│       │   │   │   │   ├── final_card.json
│       │   │   │   │   ├── final_memory.json
│       │   │   │   │   └── initial_request.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── acceptance_criteria.json
│       │   │   ├── codex_counted.py
│       │   │   ├── codex_no_retry.py
│       │   │   ├── metrics.json
│       │   │   ├── phase_ledger.json
│       │   │   ├── phase_ledger_before_resume.json
│       │   │   ├── prepared.json
│       │   │   ├── REAL_REPORT.md
│       │   │   ├── retry_config_catalog.json
│       │   │   ├── retry_config_check.json
│       │   │   ├── retry_config_verified.json
│       │   │   ├── semantic_results.json
│       │   │   ├── summarize_real.py
│       │   │   ├── transport_requests.jsonl
│       │   │   ├── validate_real.py
│       │   │   └── [省略文件：运行日志/进程痕迹 8 项]
│       │   ├── asset_acl_section.py
│       │   ├── branch_acl_section.py
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── check_code_owner_proposal.py
│       │   ├── code_owner_fix.diff
│       │   ├── compare_replay_inputs.py
│       │   ├── cpu_acceptance.json
│       │   ├── delivery_check.json
│       │   ├── feedback_helper.txt
│       │   ├── interaction_code_owner_proposed.py
│       │   ├── prepare_code_owner_patch.py
│       │   ├── preservation.json
│       │   ├── protected_assets.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── reproduce_manifest_owner.py
│       │   ├── run_cpu.sh
│       │   ├── source_before.json
│       │   ├── structure_audit.json
│       │   ├── structure_final.json
│       │   ├── structure_release.json
│       │   ├── test_code_asset_ownership.py
│       │   └── [省略文件：运行日志/进程痕迹 26 项]
│       ├── memory_recorder_20261105_r1/
│       │   ├── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── before_manifest.json
│       │   ├── benchmark_projection.py
│       │   ├── changed_files.json
│       │   ├── DELIVERY.json
│       │   ├── evidence_before.py
│       │   ├── final_source_identity.json
│       │   ├── history_after.json
│       │   ├── history_after_final.json
│       │   ├── history_after_verified.json
│       │   ├── history_before.json
│       │   ├── interaction_before.py
│       │   ├── measure_history.py
│       │   ├── measure_retrieval.py
│       │   ├── projection_benchmark.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── retrieval_comparison.json
│       │   ├── structure_audit.json
│       │   ├── structure_final.json
│       │   ├── structure_verified.json
│       │   ├── task_changes.patch
│       │   └── [省略文件：运行日志/进程痕迹 22 项]
│       ├── minimal_fixes_20261028T085806Z/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── acceptance.json
│       │   ├── before_sha256.json
│       │   ├── changed_files.json
│       │   ├── changes.diff
│       │   └── [省略文件：运行日志/进程痕迹 6 项]
│       ├── parallel_debate_20261106_r1/
│       │   ├── real_cli/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   └── commit.lock
│       │   ├── changes.patch
│       │   ├── CLI_SOURCE_IDENTITY.json
│       │   ├── DELIVERY.json
│       │   ├── FINAL_SOURCE_IDENTITY.json
│       │   ├── prompts.diff
│       │   ├── PSEUDOCODE.md
│       │   ├── real_cli_probe.py
│       │   ├── REAL_CLI_RESULT.json
│       │   ├── REPORT.md
│       │   ├── STOP_CONFIRMATION.json
│       │   ├── structure_audit.json
│       │   ├── structure_audit_final.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 3 项]
│       ├── parallel_finish_20261106_r1/
│       │   ├── frozen_compare_r1/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── frozen_compare_r2/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── gpu_resume/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── gpu_smoke_r1/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── gpu_smoke_r2/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   ├── ed3d895d2b527a415c8aa2e3c7c2e803e82a53b0405099678c454d6116e500f1/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   ├── f3e2ecc7d56e8518b4a77ba3b06c8811f73afe7478905723e11eac2c75968d84/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   └── f6f0077ba5db0a1fafa3ffb7cf16383f763ce5b2d6af4f650c936edcee693421/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── gpu_smoke_r3/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   └── 09dc29862ff884728d52d63f8d594160a2cad14dee57d68ddc428f0dbce9e09e/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── DELIVERY.json
│       │   ├── FINAL_SOURCE_IDENTITY.json
│       │   ├── frozen_compare.py
│       │   ├── gpu_resume.py
│       │   ├── GPU_RESUME_COMPLETED.json
│       │   ├── gpu_resume_continue.py
│       │   ├── GPU_RESUME_RESULT.json
│       │   ├── gpu_smoke_r1_RESULT.json
│       │   ├── gpu_smoke_r2_RESULT.json
│       │   ├── gpu_smoke_r3_RESULT.json
│       │   ├── NEXT_RUN_RUNTIME.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── smoke.py
│       │   ├── structure_audit.json
│       │   ├── structure_audit_delivery.json
│       │   ├── structure_audit_final.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 15 项]
│       ├── parent_evolution_20261029_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── before_hashes.json
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── prompt_changes.json
│       │   ├── prompts.diff
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── run_tests.sh
│       │   └── [省略文件：运行日志/进程痕迹 2 项]
│       ├── qavit_v2_20261027/
│       │   ├── deit_numeric_loop/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── numeric_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── deit_numeric_loop_corrected/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── numeric_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── evaluation_adapter/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── evaluation_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── profile_adapter/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── profile_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── real_code_service/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   ├── 98887bbe16c67bc44500a790f50e8230937d091a8a143821737dde8c1018ad4c/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   ├── 988cc59a48efd75973b6c85e0a191b08370465ce04f34e186e7534f83741df42/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   └── c50634ac43c01f5cc2a6aefa68c9fccce5fbf1b90c354e3ffbb5b0098f694da6/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── memory_service_validation.json
│       │   │   ├── service_validation.json
│       │   │   ├── supervisor.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── tiny_l1_attempt01/
│       │   │   ├── capture_report.json
│       │   │   ├── gradient_audit.json
│       │   │   ├── quantizer_state.json
│       │   │   ├── request.json
│       │   │   ├── result.json
│       │   │   ├── started.json
│       │   │   └── [省略文件：运行日志/进程痕迹 1 项]
│       │   ├── tiny_numeric_loop/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── numeric_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── tiny_numeric_loop_corrected/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── numeric_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── agent_audit_final.json
│       │   ├── agent_audit_intermediate.json
│       │   ├── agent_config.json
│       │   ├── code_service_audit_final.json
│       │   ├── final_state.json
│       │   ├── recorder_model_preflight.json
│       │   ├── subscription_preflight_retry.json
│       │   ├── validation_budget.json
│       │   ├── validation_errata.json
│       │   └── [省略文件：运行日志/进程痕迹 13 项]
│       ├── qavit_v2_real_agents_20261027/
│       │   ├── cards/ [省略：批量实验运行产物]
│       │   ├── commits/ [省略：批量实验运行产物]
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── .pending-xogz068t
│       │   ├── checkpoint.json
│       │   ├── commit.lock
│       │   ├── status.json
│       │   ├── summary.json
│       │   ├── supervisor.lock
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       ├── real_research_tiny_20261031_r1/
│       │   ├── run/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-6c675e83db4a7a55de837e11-1-0.json
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── raw_inputs_manifest.json
│       │   │   ├── supervisor.lock
│       │   │   └── [省略文件：二进制/模型/归档产物 1 项；大于1MiB的实验结果/快照 1 项]
│       │   ├── CHAIN_RESULT_185.json
│       │   ├── CHAIN_RESULT_246.json
│       │   ├── changes.patch
│       │   ├── continuation_authorization.json
│       │   ├── CONTINUATION_CHECKS.json
│       │   ├── continuation_time_authorization.json
│       │   ├── CURRENT_PSEUDOCODE.md
│       │   ├── HISTORY_PROTECTION.json
│       │   ├── input.json
│       │   ├── LIVE_STATUS.json
│       │   ├── OFFLINE_TRACE_REPLAY.json
│       │   ├── preflight.json
│       │   ├── prompts.diff
│       │   ├── REPORT.md
│       │   ├── resume_validation.py
│       │   ├── RUN_ASSETS.json
│       │   ├── RUN_ASSETS_BEFORE_CONTINUATION.json
│       │   ├── run_validation.py
│       │   ├── run_validation_started.py
│       │   ├── source_after.json
│       │   ├── source_before.json
│       │   ├── structure_audit.json
│       │   ├── structure_audit_final.json
│       │   ├── structure_before.md
│       │   ├── summarize_trace.py
│       │   ├── TASK_TIMELINE.json
│       │   ├── TRACE_INDEX.json
│       │   ├── VALIDATION.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 5 项]
│       ├── recorder_closure_20261030_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── real_llm/
│       │   │   ├── dry_materials/
│       │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   ├── checkpoint.json
│       │   │   │   │   ├── commit.lock
│       │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   ├── initial_request.json
│       │   │   │   ├── material_selection.json
│       │   │   │   ├── old_memory.json
│       │   │   │   ├── PRE_CHANGE_NOTE.md
│       │   │   │   └── source_manifest.json
│       │   │   ├── final_cpu_materials/
│       │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   ├── checkpoint.json
│       │   │   │   │   ├── commit.lock
│       │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   ├── actual_request.json
│       │   │   │   ├── initial_request.json
│       │   │   │   ├── material_selection.json
│       │   │   │   ├── old_memory.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── run/
│       │   │   │   ├── deitb_static_w4a4_e2e_20261029_r1/
│       │   │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   │   ├── recording_submissions/
│       │   │   │   │   │   └── task-c011537df4c396737e583ab9-3-2.json
│       │   │   │   │   ├── checkpoint.json
│       │   │   │   │   ├── commit.lock
│       │   │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   │   ├── case_result.json
│       │   │   │   ├── final_card.json
│       │   │   │   ├── final_memory.json
│       │   │   │   ├── final_task.json
│       │   │   │   ├── initial_request.json
│       │   │   │   ├── material_selection.json
│       │   │   │   ├── old_memory.json
│       │   │   │   ├── read_trace.json
│       │   │   │   └── source_manifest.json
│       │   │   ├── acceptance_criteria.json
│       │   │   ├── actual_delivery_proof.json
│       │   │   ├── artifact_integrity.json
│       │   │   ├── basis_delivery_cpu.json
│       │   │   ├── check_basis_delivery.py
│       │   │   ├── codex_counted.py
│       │   │   ├── metrics.json
│       │   │   ├── phase_ledger.json
│       │   │   ├── prepared.json
│       │   │   ├── REAL_REPORT.md
│       │   │   ├── representative_inputs.json
│       │   │   ├── resource_final.json
│       │   │   ├── semantic_results.json
│       │   │   ├── summarize_metrics.py
│       │   │   ├── transport_requests.jsonl
│       │   │   ├── validate_real.py
│       │   │   ├── verify_artifacts.py
│       │   │   └── [省略文件：运行日志/进程痕迹 4 项]
│       │   ├── structured_output_traces/
│       │   │   ├── decision_schema/
│       │   │   │   ├── corrected/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   └── invalid/
│       │   │   │       ├── llm_audit.json
│       │   │   │       ├── raw_final.json
│       │   │   │       ├── request.json
│       │   │   │       └── result.json
│       │   │   ├── inner_json/
│       │   │   │   ├── corrected/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   └── invalid/
│       │   │   │       ├── llm_audit.json
│       │   │   │       ├── raw_final.json
│       │   │   │       ├── request.json
│       │   │   │       └── result.json
│       │   │   ├── inner_object/
│       │   │   │   ├── corrected/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   └── invalid/
│       │   │   │       ├── llm_audit.json
│       │   │   │       ├── raw_final.json
│       │   │   │       ├── request.json
│       │   │   │       └── result.json
│       │   │   ├── outer_json/
│       │   │   │   ├── corrected/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   └── invalid/
│       │   │   │       ├── llm_audit.json
│       │   │   │       ├── raw_final.json
│       │   │   │       ├── request.json
│       │   │   │       └── result.json
│       │   │   ├── outer_schema/
│       │   │   │   ├── corrected/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   └── invalid/
│       │   │   │       ├── llm_audit.json
│       │   │   │       ├── raw_final.json
│       │   │   │       ├── request.json
│       │   │   │       └── result.json
│       │   │   ├── recorder/
│       │   │   │   ├── contract_rejection/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   ├── corrected_patch/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   ├── format_rejection/
│       │   │   │   │   ├── llm_audit.json
│       │   │   │   │   ├── raw_final.json
│       │   │   │   │   ├── request.json
│       │   │   │   │   └── result.json
│       │   │   │   └── final_card.json
│       │   │   └── tool_schema/
│       │   │       ├── corrected/
│       │   │       │   ├── llm_audit.json
│       │   │       │   ├── raw_final.json
│       │   │       │   ├── request.json
│       │   │       │   └── result.json
│       │   │       └── invalid/
│       │   │           ├── llm_audit.json
│       │   │           ├── raw_final.json
│       │   │           ├── request.json
│       │   │           └── result.json
│       │   ├── build_patch.py
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── cpu_acceptance.json
│       │   ├── delivery_check.json
│       │   ├── identity_before.json
│       │   ├── prior_basis_examples.json
│       │   ├── prior_basis_examples.py
│       │   ├── prior_basis_failure_trace.json
│       │   ├── prior_basis_long_after_read.json
│       │   ├── prior_basis_long_request.json
│       │   ├── PRIOR_BASIS_REPORT.md
│       │   ├── prior_basis_short_request.json
│       │   ├── protected_assets.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── run_cpu.sh
│       │   ├── source_before.json
│       │   ├── source_frozen.json
│       │   ├── structure.json
│       │   ├── structured_output_after.json
│       │   ├── structured_output_before.json
│       │   ├── structured_output_replay.py
│       │   ├── STRUCTURED_OUTPUT_REPORT.md
│       │   ├── task_start.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 9 项]
│       ├── research_gates_20261105_r1/
│       │   ├── real_chain/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   └── 7877952b0d4f8a681032f6945572b62aa5af5bc9dd37d5645d0f0ce46ea1859b/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── service_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── real_chain_r2/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   └── ae17334c49dada1f5b9578e42607df321b4e83ec5ec596bd571fe83de9b31f6d/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── service_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── clean_source_identity.json
│       │   ├── DELIVERY.json
│       │   ├── execution_source_identity.json
│       │   ├── final_source_identity.json
│       │   ├── preflight.json
│       │   ├── PSEUDOCODE.md
│       │   ├── real_validation_checks.json
│       │   ├── REPORT.md
│       │   ├── scientific_request.txt
│       │   ├── structure_audit.json
│       │   ├── structure_final.json
│       │   ├── validated_source_identity.json
│       │   └── [省略文件：运行日志/进程痕迹 9 项]
│       ├── research_refactor_20261103_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── core_bytecode_archive/
│       │   │   ├── agents/
│       │   │   │   └── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   │   ├── core/
│       │   │   │   └── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   │   ├── runtime/
│       │   │   │   └── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   │   └── tests/
│       │   │       └── __pycache__/ [省略：版本库、缓存、第三方安装或虚拟环境]
│       │   ├── cpu_services/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-89695b8b677470704c02e10f-1-0.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_services_final/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-f7082a333ed787c132ec8bcf-1-0.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── historical_replay/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-c50a27e41c59d3b4c2e42372-1-0.json
│       │   │   ├── replay_requests/
│       │   │   │   ├── task-c50a27e41c59d3b4c2e42372.json
│       │   │   │   └── task-dbeee42a5035d399685e1c5d.json
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── supervisor.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── before_manifest.json
│       │   ├── changes.json
│       │   ├── changes.patch
│       │   ├── cleanup.json
│       │   ├── collect_changes.py
│       │   ├── complexity.json
│       │   ├── context_after_01.json
│       │   ├── context_after_final.json
│       │   ├── context_before.json
│       │   ├── cpu_integration.py
│       │   ├── DELIVERY.json
│       │   ├── measure_context.py
│       │   ├── migrate_tests.py
│       │   ├── PLAN.md
│       │   ├── prompts.diff
│       │   ├── protection.json
│       │   ├── PSEUDOCODE.md
│       │   ├── replay_migration.json
│       │   ├── replay_recorder.py
│       │   ├── REPLAY_RESULT.json
│       │   ├── replayed_metareview.json
│       │   ├── REPORT.md
│       │   ├── structure_audit.json
│       │   ├── structure_final.json
│       │   ├── update_prompts.py
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 17 项]
│       ├── research_simplification_20261031_r1/
│       │   ├── cpu_after/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-08462287216eb1d2b93c127f-2-1.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_after_final/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-ac258f0c4a0d939266863bbb-2-1.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cpu_before/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── recording_submissions/
│       │   │   │   └── task-fb78f915a4ce376b6a3873fa-2-1.json
│       │   │   ├── scripted_requests/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── INTEGRATION_RESULT.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── changes.patch
│       │   ├── collect_validation.py
│       │   ├── comparison_budget.json
│       │   ├── cpu_integration.py
│       │   ├── CURRENT_PSEUDOCODE.md
│       │   ├── delivery_check.json
│       │   ├── historical_preservation.json
│       │   ├── implementation_change.json
│       │   ├── measure_context.py
│       │   ├── position_after.json
│       │   ├── position_after_final.json
│       │   ├── position_after_final_request.json
│       │   ├── position_after_request.json
│       │   ├── position_before.json
│       │   ├── position_before_full.json
│       │   ├── position_before_full_request.json
│       │   ├── REPORT.md
│       │   ├── source_after.json
│       │   ├── source_before.json
│       │   ├── structure_audit.json
│       │   ├── structure_before.md
│       │   ├── validation_summary.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 3 项；运行日志/进程痕迹 15 项]
│       ├── research_upgrade_20261107_r1/
│       │   ├── delivery_contracts/
│       │   │   ├── profile/
│       │   │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   │   ├── method_library.json
│       │   │   │   └── state.json
│       │   │   └── contracts.json
│       │   ├── final_contracts/
│       │   │   ├── profile/
│       │   │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   │   ├── method_library.json
│       │   │   │   └── state.json
│       │   │   └── contracts.json
│       │   ├── generation_live/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── solution_library/
│       │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   ├── solutions/
│       │   │   │   └── 854b03a7696b7ea87a5f58dfa66e4f3b9df17d96e5095c43ea05b7d2bcb86632/
│       │   │   │       ├── library.lock
│       │   │   │       └── top3.json
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── status.json
│       │   │   ├── summary.json
│       │   │   ├── supervisor.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── h1_adapted_model/
│       │   │   ├── manifest.json
│       │   │   └── [省略文件：二进制/模型/归档产物 1 项]
│       │   ├── historical_numeric/ [省略：历史实现副本/快照]
│       │   ├── schema_profile/
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── DELIVERY.json
│       │   ├── generation_input.json
│       │   ├── generation_result.json
│       │   ├── generation_validation.json
│       │   ├── h1_adaptation_result.json
│       │   ├── historical_loader_result.json
│       │   ├── HYPOTHESES.json
│       │   ├── loader_identity_diff.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── source_changes.patch
│       │   ├── structure_audit.json
│       │   ├── structure_audit_delivery.json
│       │   ├── structure_audit_final.json
│       │   ├── validate_generation.py
│       │   ├── validate_h1_adaptation.py
│       │   ├── validate_historical_loader.py
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 18 项]
│       ├── rework_live_v1/
│       │   ├── checks/
│       │   │   └── v0002.json
│       │   ├── code/
│       │   │   ├── v0001/
│       │   │   │   ├── agent_submitted.py
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   └── v0002/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── interventions/
│       │   │   └── 2026-10-24T14-42-59.105481+00-00.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── llm/
│       │   │   ├── check/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0023-2686c9a66ab9.json
│       │   │   │   ├── 0024-6113aed83323.json
│       │   │   │   ├── 0025-35bcf081e4af.json
│       │   │   │   ├── 0029-c0d98e369f05.json
│       │   │   │   ├── 0030-a7508a64667b.json
│       │   │   │   ├── 0031-e24e0cbf0649.json
│       │   │   │   ├── 0032-3f86f395e2a0.json
│       │   │   │   ├── 0033-ce78bdadae89.json
│       │   │   │   ├── 0034-3232d847c053.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── code/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0017-5385240761a8.json
│       │   │   │   ├── 0018-35d9cb23c61b.json
│       │   │   │   ├── 0019-ac7aa1f93ed4.json
│       │   │   │   ├── 0020-39f5edacf121.json
│       │   │   │   ├── 0021-8cfdfa4fc54b.json
│       │   │   │   ├── 0022-2ba3af0c2bda.json
│       │   │   │   ├── 0026-29f44cd31a17.json
│       │   │   │   ├── 0027-229ba4cd4b1d.json
│       │   │   │   ├── 0028-b4acb0cb6e34.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── evaluate/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0035-84c3e7bcc933.json
│       │   │   │   ├── 0036-d1569a4ce12a.json
│       │   │   │   ├── 0037-7b7188e1b417.json
│       │   │   │   ├── 0038-4ddfdfba3116.json
│       │   │   │   ├── 0039-34ce385083a4.json
│       │   │   │   ├── 0040-20d99434e6ea.json
│       │   │   │   ├── 0041-8357de53cc64.json
│       │   │   │   ├── 0042-d5dc5ac68093.json
│       │   │   │   ├── 0043-880af134b557.json
│       │   │   │   ├── 0044-5fd3d423c8a0.json
│       │   │   │   ├── 0045-ba2c2413064f.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── lead/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0001-c451d57ef241.json
│       │   │   │   ├── 0002-969df1edfb70.json
│       │   │   │   ├── 0003-ad7a83d51de9.json
│       │   │   │   ├── 0004-36462b9ef463.json
│       │   │   │   ├── 0005-079eff4a0c2d.json
│       │   │   │   ├── 0014-1a8f01797598.json
│       │   │   │   ├── 0015-e8294d0c9edf.json
│       │   │   │   ├── 0016-03b55785b98b.json
│       │   │   │   ├── 0046-7ea57b9a67e7.json
│       │   │   │   ├── 0047-aadbc95f4a8c.json
│       │   │   │   └── subscription_state.json
│       │   │   └── method/
│       │   │       ├── .lock
│       │   │       ├── 0006-bb6030a5a6cb.json
│       │   │       ├── 0007-0095dcfe402f.json
│       │   │       ├── 0008-2af57c5bf9dc.json
│       │   │       ├── 0009-cb263277ab5f.json
│       │   │       ├── 0010-0b9e18d5270b.json
│       │   │       ├── 0011-90eeebd59480.json
│       │   │       ├── 0012-88053734f84a.json
│       │   │       ├── 0013-d78713e0386f.json
│       │   │       └── subscription_state.json
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── proposals/
│       │   │   └── p001.json
│       │   ├── rounds/
│       │   │   ├── r001/
│       │   │   │   ├── c00005.json
│       │   │   │   ├── c00013.json
│       │   │   │   ├── c00016.json
│       │   │   │   ├── c00022.json
│       │   │   │   ├── c00025.json
│       │   │   │   ├── c00028.json
│       │   │   │   ├── c00034.json
│       │   │   │   ├── c00045.json
│       │   │   │   └── outcome.json
│       │   │   └── r002/
│       │   │       └── c00047.json
│       │   ├── tools/ [省略：批量实验运行产物]
│       │   ├── .lock
│       │   ├── delivery.json
│       │   ├── input_sources.json
│       │   ├── profile_schemas.json
│       │   ├── state.json
│       │   └── task.json
│       ├── scientific_skills_20261108_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── cpu_contracts/
│       │   │   ├── profile/
│       │   │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   │   ├── method_library.json
│       │   │   │   └── state.json
│       │   │   └── contracts.json
│       │   ├── live/
│       │   │   ├── A1_skills/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── solution_library/
│       │   │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   │   ├── solutions/
│       │   │   │   │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │   │   │       ├── library.lock
│       │   │   │   │       └── top3.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── A1_without_skills/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── solution_library/
│       │   │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   │   ├── solutions/
│       │   │   │   │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │   │   │       ├── library.lock
│       │   │   │   │       └── top3.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── A4_skills/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── solution_library/
│       │   │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   │   ├── solutions/
│       │   │   │   │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │   │   │       ├── library.lock
│       │   │   │   │       └── top3.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── B1_skills/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── solution_library/
│       │   │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   │   ├── solutions/
│       │   │   │   │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │   │   │       ├── library.lock
│       │   │   │   │       └── top3.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── C1_skills/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── solution_library/
│       │   │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   │   ├── solutions/
│       │   │   │   │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │   │   │       ├── library.lock
│       │   │   │   │       └── top3.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   ├── D1_skills/
│       │   │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   │   ├── solution_library/
│       │   │   │   │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │   │   │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │   │   │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │   │   ├── solutions/
│       │   │   │   │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │   │   │       ├── library.lock
│       │   │   │   │       └── top3.json
│       │   │   │   ├── checkpoint.json
│       │   │   │   ├── commit.lock
│       │   │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   │   └── D_policy_skills/
│       │   │       ├── cards/ [省略：批量实验运行产物]
│       │   │       ├── commits/ [省略：批量实验运行产物]
│       │   │       ├── jobs/ [省略：批量实验运行产物]
│       │   │       ├── solution_library/
│       │   │       │   ├── 29bcde579ddf720c898e32860da28904c0c04ace4ebd4f0afc128af6ed6e5d63.json
│       │   │       │   ├── d9714a16e452e2bd0741da46a9110e26e69a59b7411acb4138fe1de8794502f5.json
│       │   │       │   └── e430deab30c52bb0eaca40cc5d682eb6d3f9c4a83779b763a79634bb973d97d9.json
│       │   │       ├── solutions/
│       │   │       │   └── 34fbbb9c215f92300233f299ca8b7490c91783819cbe1b31bf75f605be90ce9c/
│       │   │       │       ├── library.lock
│       │   │       │       └── top3.json
│       │   │       ├── checkpoint.json
│       │   │       ├── commit.lock
│       │   │       └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── package/
│       │   │   ├── INDEX.original.md
│       │   │   ├── INTEGRATION_HANDOFF.md
│       │   │   └── ROUTING_AND_REPLAY.original.md
│       │   ├── A1_skills.json
│       │   ├── A1_without_skills.json
│       │   ├── A4_skills.json
│       │   ├── B1_skills.json
│       │   ├── before_manifest.json
│       │   ├── build_inventory.py
│       │   ├── C1_skills.json
│       │   ├── changes.json
│       │   ├── changes.patch
│       │   ├── context_cost.json
│       │   ├── D1_skills.json
│       │   ├── D_policy_skills.json
│       │   ├── DELIVERY.json
│       │   ├── document_checks.json
│       │   ├── final_source_identity.json
│       │   ├── initial_commit.txt
│       │   ├── initial_status.txt
│       │   ├── inventory_manifest.json
│       │   ├── live_D_policy_summary.json
│       │   ├── live_smoke.py
│       │   ├── live_summary.json
│       │   ├── live_validation.json
│       │   ├── package_manifest.json
│       │   ├── preexisting_changes.patch
│       │   ├── PROJECT_RESOURCE_INVENTORY.md
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── structure_audit.json
│       │   ├── structure_audit_final.json
│       │   └── [省略文件：运行日志/进程痕迹 7 项]
│       ├── self_containment_budget_v1/
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── continuation.json
│       │   ├── input_sources.json
│       │   ├── README.md
│       │   ├── state.json
│       │   ├── task.json
│       │   └── verification.json
│       ├── self_containment_components_v1/
│       │   ├── code/
│       │   │   ├── v0001/
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   └── v0002/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── direct_calls/ [省略：批量实验运行产物]
│       │   ├── interventions/
│       │   │   └── 2026-10-25T02-48-57.307373+00-00.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   ├── feedback/
│       │   │   │   └── b8b4b75601bb7ca3f9e3c8a8fafef6500eb4e7926a56a108afaf889cb10c37a8.json
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── proposals/
│       │   │   └── p001.json
│       │   ├── trials/ [省略：批量实验运行产物]
│       │   ├── web/
│       │   │   ├── test0043.json
│       │   │   └── test0044.json
│       │   ├── hardware.json
│       │   ├── input_sources.json
│       │   ├── profile_schemas.json
│       │   ├── README.md
│       │   ├── state.json
│       │   ├── task.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 2 项]
│       ├── self_containment_controls_v1/
│       │   ├── expired_run/
│       │   │   ├── contexts/ [省略：批量实验运行产物]
│       │   │   ├── memory/
│       │   │   │   ├── dynamic.json
│       │   │   │   └── static.json
│       │   │   ├── .lock
│       │   │   ├── delivery.json
│       │   │   ├── input_sources.json
│       │   │   ├── state.json
│       │   │   └── task.json
│       │   ├── gpu_group/
│       │   │   ├── code/
│       │   │   │   └── v0001/
│       │   │   │       ├── experiment.py
│       │   │   │       ├── parameters.json
│       │   │   │       └── version.json
│       │   │   ├── contexts/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── memory/
│       │   │   │   ├── dynamic.json
│       │   │   │   └── static.json
│       │   │   ├── proposals/
│       │   │   │   └── p001.json
│       │   │   ├── input_sources.json
│       │   │   ├── state.json
│       │   │   └── task.json
│       │   ├── sandbox/
│       │   │   └── allowed.txt
│       │   ├── tampered_code/
│       │   │   ├── experiment.py
│       │   │   ├── parameters.json
│       │   │   └── version.json
│       │   ├── protected.txt
│       │   ├── README.md
│       │   └── report.json
│       ├── self_containment_hessian_v1/
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── input_sources.json
│       │   ├── README.md
│       │   ├── state.json
│       │   ├── task.json
│       │   └── tool_result.json
│       ├── self_containment_live_v1/
│       │   ├── checks/
│       │   │   ├── v0002.json
│       │   │   └── v0003.json
│       │   ├── code/
│       │   │   ├── v0001/
│       │   │   │   ├── agent_submitted.py
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   ├── v0002/
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   └── v0003/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── interventions/
│       │   │   ├── 2026-10-25T02-45-42.247917+00-00.json
│       │   │   └── 2026-10-25T02-47-09.858041+00-00.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── llm/
│       │   │   ├── check/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0018-f35e5f131d18.json
│       │   │   │   ├── 0022-e3f9bf221e64.json
│       │   │   │   ├── 0023-c452ade2f19d.json
│       │   │   │   ├── 0024-87f045cd3de1.json
│       │   │   │   ├── 0057-7bdbaa64e9a2.json
│       │   │   │   ├── 0058-26df0f80cf9b.json
│       │   │   │   ├── 0059-cbf00b39222c.json
│       │   │   │   ├── 0060-d27a9977a6f8.json
│       │   │   │   ├── 0061-a99dc2094d24.json
│       │   │   │   ├── 0062-41c9caf40899.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── code/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0012-cab3cffc5a1a.json
│       │   │   │   ├── 0013-85bf6a43f503.json
│       │   │   │   ├── 0014-9c8738426e58.json
│       │   │   │   ├── 0015-7442953cf398.json
│       │   │   │   ├── 0016-ebf2d288b478.json
│       │   │   │   ├── 0017-8434e1a1cfca.json
│       │   │   │   ├── 0019-4d8d704adbca.json
│       │   │   │   ├── 0020-5381e22914b6.json
│       │   │   │   ├── 0021-1d208caa5d8d.json
│       │   │   │   ├── 0053-7aa38600ede5.json
│       │   │   │   ├── 0054-1a5be3b989fd.json
│       │   │   │   ├── 0055-a977d93a4450.json
│       │   │   │   ├── 0056-08947b8f8bd3.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── evaluate/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0025-cbb74b3c70da.json
│       │   │   │   ├── 0026-36ac7486ca1c.json
│       │   │   │   ├── 0027-669a5d3dc343.json
│       │   │   │   ├── 0028-6a7d6fba211c.json
│       │   │   │   ├── 0029-a5273d737feb.json
│       │   │   │   ├── 0030-c5d3dcea066f.json
│       │   │   │   ├── 0031-3a43838d45d3.json
│       │   │   │   ├── 0032-ba144d15bb4f.json
│       │   │   │   ├── 0033-bce2f8856389.json
│       │   │   │   ├── 0063-10ee741e8209.json
│       │   │   │   ├── 0064-c0e40ad9e9b6.json
│       │   │   │   ├── 0065-fcb4796d9a0f.json
│       │   │   │   ├── 0066-b2d96ab2ea18.json
│       │   │   │   ├── 0067-1e478cd398b8.json
│       │   │   │   ├── 0068-1c8c14ad72a0.json
│       │   │   │   ├── 0069-6d7cba0c0f29.json
│       │   │   │   ├── 0070-eba5d2270738.json
│       │   │   │   ├── 0071-64f426147ca5.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── lead/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0001-433c5ba6f004.json
│       │   │   │   ├── 0002-6155790af28a.json
│       │   │   │   ├── 0011-429c5f98192b.json
│       │   │   │   ├── 0034-57539f9bbaad.json
│       │   │   │   ├── 0044-8731d44e1fc2.json
│       │   │   │   ├── 0045-57ef1bd6e8b7.json
│       │   │   │   ├── 0046-0f3ca2dc2272.json
│       │   │   │   ├── 0047-30f371eae384.json
│       │   │   │   ├── 0048-99046df15685.json
│       │   │   │   ├── 0049-90d255b97dae.json
│       │   │   │   ├── 0050-aa0678b5c48d.json
│       │   │   │   ├── 0051-223c1cbae787.json
│       │   │   │   ├── 0052-a1a472f763ff.json
│       │   │   │   ├── 0072-dd04cc11b771.json
│       │   │   │   └── subscription_state.json
│       │   │   └── method/
│       │   │       ├── .lock
│       │   │       ├── 0003-d85634226a5f.json
│       │   │       ├── 0004-581505f36809.json
│       │   │       ├── 0005-474330b04bfc.json
│       │   │       ├── 0010-448b13b6595d.json
│       │   │       ├── 0035-c63f9743dc4e.json
│       │   │       ├── 0036-873913880d08.json
│       │   │       ├── 0037-0671adbae9d2.json
│       │   │       ├── 0038-c4a1aa079b17.json
│       │   │       ├── 0039-de18ccc6ea41.json
│       │   │       ├── 0040-93278e543658.json
│       │   │       ├── 0041-dd5ac3521a1b.json
│       │   │       ├── 0042-c88f99686f8a.json
│       │   │       ├── 0043-6b257e98e8b8.json
│       │   │       ├── subscription_state.json
│       │   │       └── [省略文件：大于1MiB的实验结果/快照 4 项]
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   ├── feedback/
│       │   │   │   └── 4dc9ac665eeeced7dba62c3795b51149fdbd5cfe41584a32549922f35c1b51e4.json
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── proposals/
│       │   │   ├── p001.json
│       │   │   └── p002.json
│       │   ├── rounds/
│       │   │   ├── r001/
│       │   │   │   ├── c00002.json
│       │   │   │   ├── c00010.json
│       │   │   │   ├── c00011.json
│       │   │   │   ├── c00017.json
│       │   │   │   ├── c00018.json
│       │   │   │   ├── c00021.json
│       │   │   │   ├── c00024.json
│       │   │   │   ├── c00033.json
│       │   │   │   └── outcome.json
│       │   │   ├── r002/
│       │   │   │   ├── c00034.json
│       │   │   │   ├── c00043.json
│       │   │   │   ├── c00052.json
│       │   │   │   ├── c00056.json
│       │   │   │   ├── c00062.json
│       │   │   │   ├── c00071.json
│       │   │   │   └── outcome.json
│       │   │   └── r003/
│       │   │       └── c00072.json
│       │   ├── tools/ [省略：批量实验运行产物]
│       │   ├── trials/ [省略：批量实验运行产物]
│       │   ├── .lock
│       │   ├── delivery.json
│       │   ├── input_sources.json
│       │   ├── profile_schemas.json
│       │   ├── README.md
│       │   ├── state.json
│       │   ├── summary.json
│       │   └── task.json
│       ├── self_containment_migration_v1/
│       │   ├── checks/
│       │   │   └── v0008.json
│       │   ├── code/
│       │   │   └── v0008/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── classification_comparison.json
│       │   ├── README.md
│       │   └── task.json
│       ├── self_containment_migration_v2/
│       │   ├── checks/
│       │   │   └── v0008.json
│       │   ├── code/
│       │   │   └── v0008/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── classification_comparison.json
│       │   ├── README.md
│       │   └── task.json
│       ├── self_containment_migration_v3/
│       │   ├── checks/
│       │   │   └── v0008.json
│       │   ├── code/
│       │   │   └── v0008/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── ade_result.json
│       │   ├── classification_comparison.json
│       │   ├── import_origins.json
│       │   ├── migration.json
│       │   ├── README.md
│       │   └── task.json
│       ├── self_containment_reference_v1/
│       │   ├── README.md
│       │   └── [省略文件：二进制/模型/归档产物 1 项]
│       ├── self_containment_reference_v2/
│       │   ├── profile/
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── ade_reference.json
│       │   ├── README.md
│       │   ├── resolved_ade_model.json
│       │   ├── resolved_ade_pipeline.json
│       │   └── [省略文件：二进制/模型/归档产物 1 项；大于1MiB的实验结果/快照 1 项]
│       ├── self_containment_reload_v1/
│       │   ├── config.json
│       │   ├── gradient_updates.json
│       │   ├── plan.json
│       │   ├── README.md
│       │   ├── request.json
│       │   ├── started.json
│       │   └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 1 项]
│       ├── self_containment_repairs_v1/
│       │   ├── README.md
│       │   ├── repair_refit.json
│       │   ├── repair_w4.json
│       │   ├── report.json
│       │   └── sensitivity_rows.json
│       ├── serial_debate_memory_20261029_r1/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── before_hashes.json
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── prompt_changes.json
│       │   ├── prompts.diff
│       │   ├── protected_hashes.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── run_tests.sh
│       │   ├── syntax_check.json
│       │   └── [省略文件：运行日志/进程痕迹 4 项]
│       ├── single_implementation_20261105_r1/
│       │   ├── real_chain/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   └── ecf2b566c9cfb246b6ddf984ab26cdd135842742eba0fed73a21746c1b21dbb4/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── service_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── real_chain_final/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   └── b38234d89a41597eb64baa4a2f0391c5dffe07283061d73fc2287423ee40e292/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── service_validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── cli_help.txt
│       │   ├── DELIVERY.json
│       │   ├── llm_preflight.json
│       │   ├── PSEUDOCODE.md
│       │   ├── REPORT.md
│       │   ├── structure_audit.json
│       │   ├── structure_audit_final.json
│       │   ├── structure_delivery.json
│       │   ├── structure_verified.json
│       │   └── [省略文件：运行日志/进程痕迹 17 项]
│       ├── smoke_live_v1/
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── interventions/
│       │   │   └── schema_adapter.json
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── llm/
│       │   │   ├── code/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0012-51dc9c064a89.json
│       │   │   │   ├── 0013-31a1fc135e1f.json
│       │   │   │   ├── 0014-a3bd5b746f7e.json
│       │   │   │   ├── 0015-aec388bd1047.json
│       │   │   │   ├── 0016-50434d16f13e.json
│       │   │   │   ├── 0017-cc9322342a99.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── lead/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0001-91c395ba585c.json
│       │   │   │   ├── 0002-c204113f03fe.json
│       │   │   │   ├── 0003-14f03d67243c.json
│       │   │   │   ├── 0004-dbabdf06bf59.json
│       │   │   │   ├── 0005-92057b61b759.json
│       │   │   │   ├── 0006-bc09191d5f13.json
│       │   │   │   ├── 0007-ad674c3af6fa.json
│       │   │   │   ├── 0010-3ed4b59d6c53.json
│       │   │   │   ├── 0011-f8d1392c336f.json
│       │   │   │   ├── 0018-2e22f7e416ce.json
│       │   │   │   └── subscription_state.json
│       │   │   └── method/
│       │   │       ├── .lock
│       │   │       ├── 0008-76f2aecd0798.json
│       │   │       ├── 0009-a2f6eec0bba7.json
│       │   │       └── subscription_state.json
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── proposals/
│       │   │   └── p001.json
│       │   ├── rounds/
│       │   │   ├── r001/
│       │   │   │   ├── c00007.json
│       │   │   │   ├── c00009.json
│       │   │   │   ├── c00011.json
│       │   │   │   ├── c00017.json
│       │   │   │   └── outcome.json
│       │   │   └── r002/
│       │   │       └── c00018.json
│       │   ├── tools/ [省略：批量实验运行产物]
│       │   ├── .lock
│       │   ├── delivery.json
│       │   ├── input_sources.json
│       │   ├── state.json
│       │   └── task.json
│       ├── smoke_live_v2/
│       │   ├── checks/
│       │   │   ├── v0001.json
│       │   │   └── v0002.json
│       │   ├── code/
│       │   │   ├── v0001/
│       │   │   │   ├── experiment.py
│       │   │   │   ├── parameters.json
│       │   │   │   └── version.json
│       │   │   └── v0002/
│       │   │       ├── experiment.py
│       │   │       ├── parameters.json
│       │   │       └── version.json
│       │   ├── contexts/ [省略：批量实验运行产物]
│       │   ├── jobs/ [省略：批量实验运行产物]
│       │   ├── llm/
│       │   │   ├── check/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0018-144f13592a50.json
│       │   │   │   ├── 0019-5c605e2a9f6c.json
│       │   │   │   ├── 0020-9589ec8d0ab6.json
│       │   │   │   ├── 0021-b3df1946c6bd.json
│       │   │   │   ├── 0022-e6092bdfd30d.json
│       │   │   │   ├── 0043-9fda530ce283.json
│       │   │   │   ├── 0044-80d2414ceb67.json
│       │   │   │   ├── 0045-b7f9fa256849.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── code/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0011-c320dceb47f0.json
│       │   │   │   ├── 0012-516d0500f7c3.json
│       │   │   │   ├── 0013-327638ba3904.json
│       │   │   │   ├── 0014-258b2847a0f2.json
│       │   │   │   ├── 0015-373977904bd9.json
│       │   │   │   ├── 0016-b20bf3000636.json
│       │   │   │   ├── 0017-209f9a12c919.json
│       │   │   │   ├── 0041-92fb163233dd.json
│       │   │   │   ├── 0042-8f658486eda8.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── evaluate/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0023-fcaccb7623f6.json
│       │   │   │   ├── 0024-53c41b883a7b.json
│       │   │   │   ├── 0025-9e87340d561c.json
│       │   │   │   ├── 0026-03aec1f6cc82.json
│       │   │   │   ├── 0027-410c75c9708f.json
│       │   │   │   ├── 0028-85d1af9f9df6.json
│       │   │   │   ├── 0029-50dd07f4e023.json
│       │   │   │   ├── 0030-15173d690165.json
│       │   │   │   ├── 0031-9f321e967e79.json
│       │   │   │   ├── 0046-527c1a6cd360.json
│       │   │   │   ├── 0047-26cdc76cfcdb.json
│       │   │   │   ├── 0048-72e88cd53ee7.json
│       │   │   │   ├── 0049-6de81374b50e.json
│       │   │   │   ├── 0050-cebc794b5200.json
│       │   │   │   ├── 0051-d19483e26342.json
│       │   │   │   ├── 0052-66a1adbe95a1.json
│       │   │   │   ├── 0053-138aa54fe6a1.json
│       │   │   │   ├── 0054-71451c2dcb79.json
│       │   │   │   └── subscription_state.json
│       │   │   ├── lead/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0001-df2fde0e306c.json
│       │   │   │   ├── 0002-764a5f2d7bfd.json
│       │   │   │   ├── 0008-d8b34a7ad19d.json
│       │   │   │   ├── 0009-3ec856206552.json
│       │   │   │   ├── 0010-addf69525c7d.json
│       │   │   │   ├── 0032-c09f77b8afbb.json
│       │   │   │   ├── 0036-73a947a288ae.json
│       │   │   │   ├── 0037-70e117b78bfc.json
│       │   │   │   ├── 0038-4fe8e0a9d2cb.json
│       │   │   │   ├── 0039-0424514b73ae.json
│       │   │   │   ├── 0040-ee7d91c1908d.json
│       │   │   │   ├── 0055-3a03fb83ba3f.json
│       │   │   │   └── subscription_state.json
│       │   │   └── method/
│       │   │       ├── .lock
│       │   │       ├── 0003-b4afd1ac8daa.json
│       │   │       ├── 0004-329a77c01c8b.json
│       │   │       ├── 0005-9934c32f975b.json
│       │   │       ├── 0006-2526d2114028.json
│       │   │       ├── 0007-c6e223bc2772.json
│       │   │       ├── 0033-db63e3a83d7f.json
│       │   │       ├── 0034-1ff4fb31d0e7.json
│       │   │       ├── 0035-66760b5c4af6.json
│       │   │       └── subscription_state.json
│       │   ├── memory/
│       │   │   ├── dynamic.json
│       │   │   └── static.json
│       │   ├── profile/
│       │   │   ├── feedback/
│       │   │   │   └── 2e36a498b2ced980676989a00ab9f54aafb889ed78db037d755fcea68f2b9cae.json
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── proposals/
│       │   │   ├── p001.json
│       │   │   └── p002.json
│       │   ├── rounds/
│       │   │   ├── r001/
│       │   │   │   ├── c00002.json
│       │   │   │   ├── c00007.json
│       │   │   │   ├── c00010.json
│       │   │   │   ├── c00017.json
│       │   │   │   ├── c00022.json
│       │   │   │   ├── c00031.json
│       │   │   │   └── outcome.json
│       │   │   ├── r002/
│       │   │   │   ├── c00032.json
│       │   │   │   ├── c00035.json
│       │   │   │   ├── c00040.json
│       │   │   │   ├── c00042.json
│       │   │   │   ├── c00045.json
│       │   │   │   ├── c00054.json
│       │   │   │   └── outcome.json
│       │   │   └── r003/
│       │   │       └── c00055.json
│       │   ├── tools/ [省略：批量实验运行产物]
│       │   ├── trials/ [省略：批量实验运行产物]
│       │   ├── .lock
│       │   ├── delivery.json
│       │   ├── input_sources.json
│       │   ├── profile_schemas.json
│       │   ├── state.json
│       │   ├── summary.json
│       │   └── task.json
│       ├── storage_recovery_20261031_r1/
│       │   ├── before.json
│       │   ├── bytecode_inventory.json
│       │   ├── cleanup_progress.json
│       │   ├── cpu_result.json
│       │   ├── delivery_check.json
│       │   ├── finalize_resume_report.py
│       │   ├── first_experiment_frozen_audit.json
│       │   ├── first_experiment_rejection_audit.json
│       │   ├── freeze_resume.py
│       │   ├── h4_screen_actual_delivery.json
│       │   ├── INTERFACE_BLOCKER.md
│       │   ├── launch.json
│       │   ├── monitor_checkpoint_1.json
│       │   ├── monitor_checkpoint_2.json
│       │   ├── monitor_process.json
│       │   ├── pre_resume_monitor_latest.json
│       │   ├── pre_resume_runner_status.json
│       │   ├── private_backup_receipt.json
│       │   ├── probe_before.json
│       │   ├── protected_after.json
│       │   ├── protected_summary.json
│       │   ├── recover_storage.py
│       │   ├── REPORT.md
│       │   ├── resume_audit.json
│       │   ├── resume_binding_receipt.json
│       │   ├── resume_preflight.json
│       │   ├── resumed_acceptance.json
│       │   ├── resumed_calls.json
│       │   ├── resumed_key_cards.json
│       │   ├── resumed_state.json
│       │   ├── RUN_RESUME_REPORT.md
│       │   ├── science_review_h1_h3.json
│       │   ├── science_review_h2_h3_positions.json
│       │   ├── science_review_h4.json
│       │   ├── source_identity.json
│       │   ├── Stable-08d4889f9ec4a1685d257b9b95de036c8e1ce1e5_receipt.json
│       │   ├── Stable-110a328ea54b42367b803ec53ee0bf52ef26b419_receipt.json
│       │   ├── storage_result.json
│       │   ├── verify_storage.py
│       │   ├── write_verification.json
│       │   └── [省略文件：二进制/模型/归档产物 3 项；大于1MiB的实验结果/快照 2 项；运行日志/进程痕迹 5 项]
│       ├── subscription_live_v1/
│       │   ├── llm/
│       │   │   ├── lead/
│       │   │   │   ├── .lock
│       │   │   │   ├── 0001-f215512f9ccc.json
│       │   │   │   └── subscription_state.json
│       │   │   └── method/
│       │   │       ├── .lock
│       │   │       ├── 0001-dbf057944394.json
│       │   │       └── subscription_state.json
│       │   ├── lead.json
│       │   └── method.json
│       ├── throughput_20261028T133200Z/
│       │   ├── ade_check/
│       │   │   ├── raw.json
│       │   │   └── validation.json
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── builtin_service_checks/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── manual_model/
│       │   │   │   ├── manifest.json
│       │   │   │   └── [省略文件：二进制/模型/归档产物 1 项]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── catalog_session/
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── host_checks/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── output/
│       │   │   └── PSEUDOCODE.md
│       │   ├── permission_checks/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── real_data_checks/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── real_profile_catalog/
│       │   │   ├── method_library.json
│       │   │   └── state.json
│       │   ├── registry_checks/
│       │   │   ├── profile/
│       │   │   │   ├── calls/
│       │   │   │   │   └── c9a8a6865aaaaf03b25bcfa456bd2be68fd8503430892ac38b0bf9ca55f4d548.json
│       │   │   │   ├── evidence/ [省略：批量实验运行产物]
│       │   │   │   ├── programs/
│       │   │   │   │   ├── manual_structure-9fa5dda4489f193fbc4aff3da2b58526b6d802c217bcaef23dcc3d12e6bb981b.json
│       │   │   │   │   ├── manual_structure-9fa5dda4489f193fbc4aff3da2b58526b6d802c217bcaef23dcc3d12e6bb981b.py
│       │   │   │   │   └── run-6e398c63972e491296824655069d5a2d.json
│       │   │   │   ├── method_library.json
│       │   │   │   └── state.json
│       │   │   ├── results.json
│       │   │   └── summary.json
│       │   ├── run_tool_contracts/
│       │   │   ├── deit_sites.json
│       │   │   ├── profile_distribution.json
│       │   │   ├── profile_hardware.json
│       │   │   ├── profile_plugin_sites.json
│       │   │   ├── profile_ptq_candidates.json
│       │   │   ├── profile_sensitivity.json
│       │   │   ├── profile_structure.json
│       │   │   ├── profile_trial_ablation.json
│       │   │   ├── profile_trial_diagnostics.json
│       │   │   ├── query_execution_capabilities.json
│       │   │   ├── query_method_library.json
│       │   │   ├── query_ptq_plugins.json
│       │   │   ├── query_ptq_sites.json
│       │   │   ├── query_structure.json
│       │   │   ├── read_method_document.json
│       │   │   └── read_profile_evidence.json
│       │   ├── service_checks/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── service_checks_v2/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   ├── 24c714a24426f7c48b083e03854133ede55c83254d3aa6b1842b2100670e25eb/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   ├── 2a3729ed07c578598d3f76b1314aa262eca95c4b06e46ab9c44551416181607e/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   ├── 52b130b6fcec7a10bcec4f9a2231190fea21dd83d48d98c92319ed0f3435b111/
│       │   │   │   │   ├── implementation.py
│       │   │   │   │   └── manifest.json
│       │   │   │   └── 630d6ce4adb6f5b62be065e838a4178d0bb263d519b424dd54aac97bb079f2e9/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── service_checks_v3/
│       │   │   ├── cards/ [省略：批量实验运行产物]
│       │   │   ├── code/
│       │   │   │   └── 1434f8df2a33671f238be07cde239ad7568147d79244ba6ee047c9df0bee4e84/
│       │   │   │       ├── implementation.py
│       │   │   │       └── manifest.json
│       │   │   ├── commits/ [省略：批量实验运行产物]
│       │   │   ├── jobs/ [省略：批量实验运行产物]
│       │   │   ├── checkpoint.json
│       │   │   ├── commit.lock
│       │   │   ├── validation.json
│       │   │   └── [省略文件：大于1MiB的实验结果/快照 1 项]
│       │   ├── audit_run.py
│       │   ├── before_identity.json
│       │   ├── changes.diff
│       │   ├── changes.json
│       │   ├── check_ade.py
│       │   ├── check_builtin_services.py
│       │   ├── check_real_data.py
│       │   ├── check_registry.py
│       │   ├── check_services.py
│       │   ├── check_services_v3.py
│       │   ├── deit_sites.json
│       │   ├── finalize_report.py
│       │   ├── host_tool_checks.json
│       │   ├── imagenet_subset.json
│       │   ├── input.json
│       │   ├── late_contract_check.json
│       │   ├── launch_frozen.py
│       │   ├── permission_checks.json
│       │   ├── preparation_report.json
│       │   ├── prepare_input.py
│       │   ├── real_profile_catalog.json
│       │   ├── tool_catalog.json
│       │   ├── watch_run.py
│       │   └── [省略文件：运行日志/进程痕迹 15 项]
│       ├── two_rounds_20261104_r2/
│       │   ├── before/ [省略：历史实现副本/快照]
│       │   ├── fixture_config_proposal/
│       │   │   ├── proposed/
│       │   │   │   └── QAViT/
│       │   │   │       └── tools/ [省略：批量实验运行产物]
│       │   │   ├── check_proposal.py
│       │   │   ├── checkpoint_run_audit.json
│       │   │   ├── identities.json
│       │   │   ├── pause.json
│       │   │   ├── production_identity_unchanged.json
│       │   │   ├── PROPOSAL.md
│       │   │   ├── proposal.patch
│       │   │   ├── stopped_tasks.json
│       │   │   └── [省略文件：运行日志/进程痕迹 3 项]
│       │   ├── intervention01/
│       │   │   ├── before/ [省略：历史实现副本/快照]
│       │   │   ├── changes.patch
│       │   │   ├── record.json
│       │   │   ├── stopped_status.json
│       │   │   ├── stopped_summary.json
│       │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│       │   ├── intervention02/
│       │   │   ├── before/ [省略：历史实现副本/快照]
│       │   │   ├── after_identity.json
│       │   │   ├── changes.patch
│       │   │   ├── record.json
│       │   │   ├── structure_audit.json
│       │   │   └── [省略文件：运行日志/进程痕迹 3 项]
│       │   ├── intervention03/
│       │   │   ├── before/ [省略：历史实现副本/快照]
│       │   │   ├── after_identity.json
│       │   │   ├── before_identity.json
│       │   │   ├── changes.patch
│       │   │   ├── pause.json
│       │   │   ├── record.json
│       │   │   ├── structure_audit.json
│       │   │   └── [省略文件：运行日志/进程痕迹 2 项]
│       │   ├── intervention04/
│       │   │   ├── before/ [省略：历史实现副本/快照]
│       │   │   ├── after_identity.json
│       │   │   ├── before_identity.json
│       │   │   ├── changes.patch
│       │   │   ├── pause.json
│       │   │   ├── record.json
│       │   │   ├── stopped_tasks.json
│       │   │   ├── structure_audit.json
│       │   │   └── [省略文件：运行日志/进程痕迹 7 项]
│       │   ├── intervention05/
│       │   │   ├── before/ [省略：历史实现副本/快照]
│       │   │   ├── after_identity.json
│       │   │   ├── authorization.json
│       │   │   ├── before_identity.json
│       │   │   ├── changes.patch
│       │   │   ├── record.json
│       │   │   ├── structure_audit.json
│       │   │   └── [省略文件：运行日志/进程痕迹 3 项]
│       │   ├── llm_smoke/
│       │   │   ├── cli/
│       │   │   │   ├── instructions.txt
│       │   │   │   └── schema.json
│       │   │   └── llm_audit.json
│       │   ├── llm_smoke_absolute/
│       │   │   ├── cli/
│       │   │   │   ├── cli_logs/
│       │   │   │   │   └── logs_2.sqlite
│       │   │   │   ├── answer.json
│       │   │   │   ├── instructions.txt
│       │   │   │   └── schema.json
│       │   │   └── llm_audit.json
│       │   ├── numeric_smoke/
│       │   │   ├── capture_report.json
│       │   │   └── quantizer_state.json
│       │   ├── after_manifest.json
│       │   ├── base_commit.txt
│       │   ├── before_manifest.json
│       │   ├── changes.json
│       │   ├── changes.patch
│       │   ├── changes_all.json
│       │   ├── changes_all.patch
│       │   ├── current_manifest.json
│       │   ├── DELIVERY.json
│       │   ├── execution_index.json
│       │   ├── INDEX.md
│       │   ├── input.json
│       │   ├── input_source.json
│       │   ├── inspect_run.py
│       │   ├── llm_preflight.json
│       │   ├── llm_smoke_absolute_result.json
│       │   ├── llm_smoke_result.json
│       │   ├── numeric_preflight.json
│       │   ├── preexisting.patch
│       │   ├── preexisting_status.txt
│       │   ├── preflight_numeric.py
│       │   ├── PROCESS_NOTES.md
│       │   ├── prompt_manifest.json
│       │   ├── PSEUDOCODE.md
│       │   ├── recorder_preflight.json
│       │   ├── REPORT.md
│       │   ├── resources_preflight.txt
│       │   ├── run_audit.json
│       │   ├── statistics.json
│       │   ├── structure_audit.json
│       │   ├── summarize_delivery.py
│       │   └── [省略文件：运行日志/进程痕迹 9 项]
│       ├── assets_final.json
│       ├── audit_formal.py
│       ├── audit_workflow.py
│       ├── confusion_matrix_payload_check.json
│       ├── confusion_matrix_payload_live.json
│       ├── continuation_assets_final.json
│       ├── continuation_cache_paths.json
│       ├── continuation_delivery_integrity.json
│       ├── continuation_formal_binding_final.json
│       ├── continuation_native_dependency_audit.json
│       ├── continuation_read_isolation_evidence.json
│       ├── deit_preflight.py
│       ├── deit_preflight_v2.py
│       ├── deit_preflight_v3.py
│       ├── formal_binding_audit.json
│       ├── formal_binding_final.json
│       ├── latest_interface_checks.json
│       ├── log_code_cardinality.json
│       ├── pre_formal_gate.json
│       ├── README.md
│       ├── resume_status_check.json
│       ├── rework_task.json
│       ├── secondary_numeric_checks.json
│       ├── self_containment_formal_round9_audit.json
│       ├── self_containment_formal_round9_audit_v2.json
│       ├── self_containment_gate.json
│       ├── self_containment_live_task.json
│       ├── SELF_CONTAINMENT_REPORT.md
│       ├── self_containment_source_path_audit.json
│       ├── self_containment_subscription_diagnostic.json
│       ├── sensitivity_fixed.json
│       ├── sensitivity_numerical_check.json
│       ├── sensitivity_rows_w4.json
│       ├── sensitivity_w4.json
│       ├── tiny_fixture.py
│       ├── twin_grid_independent_audit.json
│       ├── validate_components.py
│       ├── validate_controls.py
│       ├── validate_deit_tools.py
│       ├── validate_delivery_grids.py
│       ├── validate_hessian.py
│       ├── validate_remaining.py
│       ├── validate_self_containment_controls.py
│       ├── validate_self_containment_hessian.py
│       ├── validate_sensitivity.py
│       ├── validate_sensitivity_w4.py
│       ├── workflow_audit.json
│       └── [省略文件：大于1MiB的实验结果/快照 1 项；运行日志/进程痕迹 34 项]
├── reference/
│   ├── coscientist.pdf
│   └── coscientist_supplementary.pdf
├── .git/ [省略：版本库、缓存、第三方安装或虚拟环境]
├── .gitignore
├── AGENTS.md
├── README.md
└── STRUCTURE.md
```
<!-- INVENTORY_END -->
