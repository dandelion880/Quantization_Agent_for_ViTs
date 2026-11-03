# QAViT V2 结构整理与离线验证

2026-10-27。本次已实际完成 V2 单实现整理、依赖提取与离线验证。科研策略、角色职责、排名、Memory、预算策略及量化/评测算法保持原有行为；下文列出的科研问题未修复。没有真实 LLM 调用、GPU 实验、完整数据评测、新科研运行或历史续跑。

日常阅读从 [QAViT README](../../../QAViT/README.md) 和 [STRUCTURE](../../../STRUCTURE.md) 开始。本报告保存本次变更与验收证据，不替代长期入口说明。

## 1. 依赖判断与处理

从旧 `run.py → core/research_workflow.py` 追踪到队列 worker、实验 worker、backend factory、字符串资源目录、模型构造源码和 Profiling 初始化链后分类处理。普通导入清单见 [before_interfaces.json](before_interfaces.json)，修改前文件哈希见 [before_files.json](before_files.json)，原件保存在 [before_source.tar.gz](before_source.tar.gz)。

| 分类 | 处理与理由 |
|---|---|
| 纯 V2 | 保留 cards、journal、debate、memory、实验合同、Safeguard、评测和模型适配；拆分科研编排，不重新实现科研判断 |
| V2 借用的旧公共能力 | 提取订阅传输、路径/文件读写、保护函数、完整 Profiling 注册、计划展开与数值执行；清除旧对象构造和 monkeypatch |
| 纯 MVP | 删除五角色 `agents/prompts.json`、旧 roles/workflow 内容、旧 CLI 分支、toolbox/research/worker 中不再需要的逻辑；删除 deitb_free/deitb_continue/smoke/llm 专用配置。旧通用文件名由当前 V2 使用，不保留兼容入口或 legacy 目录 |
| 旧数值库内的智能体框架 | 删除 `agents.py`、`orchestrator.py`、`memory.py`、旧选优 evaluation/reporting、Profiling session/recommendations 和旧 llm 包；删除无 V2 调用路径的 toy backend/plugin。实际量化与工具代码保留 |
| 实验与一次性资产 | 历史 EXP 内容原位保留；本次快照、对照脚本、失败记录、CPU 输出和回执全在本目录；可复用测试与小型黄金夹具在 QAViT/tests |
| 第三方与只读资产 | `.runtime_pkgs/`、DeiT 源码与 LICENSE 保留，未安装或升级依赖；数据集、母模型与冻结 head 不迁移、不覆盖，配置仍只读引用原位置 |

逐文件删除/新增/修改及最终 SHA256 见 [changes.json](changes.json)。文件改名计入删除与新增；`core/workflow.py`、`agents/roles.py` 的同名文件是以当前 V2 内容替换旧实现，并非保留 MVP。

## 2. 主要路径对应与解耦结果

下表路径均相对 QAViT。

| 旧位置 | 当前位置/结果 |
|---|---|
| `core/research_workflow.py` | `core/workflow.py` 主阶段；`core/experiment_requests.py` 实验子流程；`agents/interaction.py` 上下文/工具/提交；`runtime/driver.py` 调度/轮询/恢复/报告 |
| `agents/research_roles.py`、`research_prompts.json` | `agents/roles.py`、`tasks.py`、`prompts/common.md`、七角色与十四任务 Markdown |
| `core/model_router.py`、`core/llm.py`、数值库订阅客户端 | `runtime/model_router.py`、`subscription.py`、`codex_cli.py`；正常构造独立 SubscriptionTransport，保留命令、解码、预检和审计 |
| `core/common.py` | `runtime/paths.py`、`files.py`、`identity.py`；MVP 汇总逻辑删除 |
| `core/{budget,task_queue,task_runtime,task_worker}.py` | `runtime/` 同名文件；启动路径、导入与恢复关联同步更新 |
| `tools/research_worker.py`、旧 `tools/worker.py` 保护函数 | `tools/experiment_worker.py`、`runtime/guards.py` |
| `tools/research.py`、`unlimited_profile_host()` | `tools/profiling.py` + `ptq_agent/profiling/host.py` 的 ProfileHost |
| `ExecutableProfilingSession._initialize()` 及父类注册链 | `ptq_agent/registry.py`、`profiling/tools.py`、`execution/profile_tools.py`；计划视图移至 `execution/planning.py` |
| `PluginExecutionAgent`、`ExecutionAgent` | 数值职责命名为 `QuantizationExecutor`、`RTNExecutor`；构造参数和计算方法保持 |
| `configs/research_v2.json`、各处角色模型默认值 | `configs/research.json` 模板 + 唯一默认来源 `configs/models.json`；完整显式运行覆盖仍冻结到运行清单 |
| `tests/*v2*` | `test_infrastructure.py`、`support.py`、`audit_run.py`、`validate_{numeric,code_service,memory_service}.py`；没有保留旧文件转发 |

ProfileHost 不构造 Agent、LLM client、Memory、Orchestrator 或选优对象。完整初始化仍注册特殊位点、插件、执行许可、RTN 敏感度、计划展开、方法资料、任务程序、候选诊断和证据索引。原注册表的 31 项中保留 26 项 Schema；删除未向 V2 公开且不被其工具实现调用的 `read_memory/profile/validate_candidate/run_trial/compare_trials`。V2 公开的 15 项 Profiling 工具目录没有扩大或缩小。测量计数、缓存键、证据格式和哈希核验保留；活动时间检查和数值工具调用预留仍有实际作用，未顺带删掉预算行为。

数值后端删除无执行用途的 `llm`、`backend.research_root`，不再提供依赖相邻旧项目的 `verified_audit` 构造分支。当前 manifest v1/v2 后端和 V2 覆盖后的配置效果一致；原 `validate_loop()` 及资源限制断言保留。固定层等已被 V2 worker 替换的旧模板值改为当前有效默认值；原模板中不符合自身 Schema 的可选 `min_top1: null` 改为省略，进入 V2 后的有效值未变。

没有剩余的结构依赖阻点需要保留旧框架。保留的 `ptq_agent` 是数值工具包名称，计划 v2/v3、卡片版本和历史证据字段仍有用途；其旧格式字段不代表还有旧 Agent 循环。

## 3. 保持项与必要变化

- 14 个任务的最终角色 Prompt 字节一致。Markdown 以软换行便于编辑，组装时仅移除片段内新增换行。任务目的、角色、输出类型、允许工具、片段和审查阶段集中到 `agents/tasks.py`。
- 方法资料内的 `src/ptq_agent` 指引改为 `tools/ptq_agent`，来源链接改为项目内 `docs/references/ptq_methods.json`。方法正文的数学含义未改；方法库返回文本的路径与文档哈希会变化，详见 [document_path_updates.json](document_path_updates.json)。来源索引的外部地址是历史证据出处，不是运行 import。
- `code_identity()` 递归覆盖自有 Python、Prompt、配置、接口文档、模型源码和测试；`measurement_implementation()` 递归覆盖工具、运行设施、执行配置及数值库。新增子目录纳入身份，快照逐文件重新核验 SHA256。
- EXP、安装目录、缓存不进入源码身份；大型 ImageNet 清单由执行输入的资产哈希绑定。路径/源码/配置字段变化使新身份及关联证据 ID 自然变化，没有跳过哈希检查或修改历史快照来适配。
- 输出仍归 EXP；测试临时目录自动清理。旧源码目录中确认无用途的字节码缓存已清理，第三方依赖目录未作缓存删除。

## 4. 当前流程伪代码

这是当前源码的导航，不代表新增科研门槛；实现分别位于 workflow、experiment_requests、interaction、driver。

```text
输入：ResearchBrief、运行配置、硬预算；新运行保存有效配置及递归源码快照
恢复：读取 Journal 的 cards/commits；校验配置/代码身份；重建队列、Memory、预算
      接收已持久化的 worker 回执；显式 intervention/retry 留事件，不重置预算

while 预算允许：                         # runtime/driver.py
    poll 并接收完成回执；推进等待/失败/重试状态
    advance 当前轮：                      # core/workflow.py
        Generation 分别产生 H1、H2、H3；Judger 筛查资格
        推进前三场配对：双方独立 position → Judger 裁决，最多五次交流
        Evolution 依据首阶段记录产生 H4；筛查后推进另三场配对
        六场均有终态后提交排名；保留原未决/循环结果处理
        若启用 L3 且有选择：申请 L3；等待申请终态（原实现不强制 L2 前置）
        Recorder 提交 Meta-review → 更新 Memory → round_finished → 下一轮
    dispatch 就绪任务：工具/申请由主进程推进，其余交持久 worker
    无可推进任务时报告 blocked/needs_clarification；超预算/中断保存状态

实验申请：                               # core/experiment_requests.py
    非 Judger 申请先经科学审批 → 冻结 ExperimentCard 和执行资产
    需要代码时：Coding 候选 与 Safeguard 独立参考 → 数值验证 → Safeguard 审查
    复用匹配且完整的证据，或执行 worker → EvidenceCard 原始事实
    Judger 解释原始证据 → 申请完成；解释与测量分开保存

提交：校验任务合同/引用 → Journal 原子提交 → 更新队列与 Memory 投影
Memory：L0 事实记录；L1 运行/候选状态；L2 带状态知识；L3 Meta-review
        incumbent 的完整协议校验按既有实现执行，不因本次拆分放宽
```

## 5. 验证命令、结果与边界

从项目顶层执行，所有结果在本目录。下列命令是本次实际离线检查；重新执行快照/审计时需另取新的输出路径。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
python -B QAViT/tests/audit_structure.py --output EXP/implementation/v2_structure_20261027/structure_final.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B QAViT/tests/snapshot_contracts.py --output EXP/implementation/v2_structure_20261027/contracts
python -B EXP/implementation/v2_structure_20261027/compare_and_inventory.py
python -B EXP/implementation/v2_structure_20261027/review_cases.py
```

| 检查 | 实际结果与证据 |
|---|---|
| 修改前基线 | 原 23 项基础设施测试通过：[before_tests.log](before_tests.log) |
| 修改后确定性测试 | 29 项通过：[checks.log](checks.log)。包含模拟两轮、卡片/队列/原子持久化/恢复、预算延续、独立数值参考与错误舍入拒绝 |
| 结构与资源 | 100 个自有/附带模型 Python 文件语法检查；98 个自有模块导入；worker、动态源码清单、fixture factory、模型构造源哈希、外部资产路径检查通过。源码快照 161 文件：[structure_final.json](structure_final.json) |
| 有效输入与测量对照 | 24 项比较通过：[contract_comparison.json](contract_comparison.json)。14 个最终 Prompt、当前有效配置、26 个保留 Schema 一致；固定 CPU distribution、RTN sensitivity、post-softmax 原始数值结果一致，重复命中缓存不增加测量计数 |
| 科研/传输代码核对 | 31 个原科研方法和 3 个传输方法的 AST 对照通过：[method_comparison.json](method_comparison.json)。只归一化任务映射集中化和模块路径，不忽略科学分支/数值/断言 |
| 专项风险 | 新测试覆盖完整工具初始化及真实隔离 program worker、微型 CPU 模型量化导出/重载逐值一致、FP 母模型不变、递归身份与篡改拒绝、伪订阅审计/模型不符拒绝；未调用真实服务 |
| 任意工作目录入口 | 从 `/tmp` 用绝对路径运行 `python -B …/QAViT/run.py --help` 成功；没有创建科研运行 |
| 历史与资产保全 | 原清单内 6,503 个历史文件哈希不变；资产描述、模型构造源码及 LICENSE 不变；顶层仍为五项：[history_verification.json](history_verification.json) |

对照明确排除的变化只有：旧未使用配置字段、路径/源码身份及其派生 ID，以及实时空闲内存元数据；没有将数值容差放宽来通过测试。方法资料路径修正后另做完整工具初始化与导入/链接检查，见 [documentation_checks.log](documentation_checks.log) 和 [document_links.json](document_links.json)。

实施中出现并已修复的结构问题包括：提取时漏掉 ToolDefinition 的 dataclass 装饰器、旧测试模块引用、原数值模板可选 null 字段、测试夹具统计名和任务程序返回类型错误。相关失败日志留在本目录；未删除有效断言。`validate_loop()` 的原资源校验完整保留，并新增非法资源值拒绝测试。

未验证：真实模型服务、研究结论质量、GPU/完整 ImageNet/ADE 评测、全部数值算法组合、真实中断时的所有操作系统故障。大型权重/head 只核验路径，本轮未加载或重新计算其全量哈希；真正执行时仍走已有冻结/加载校验。模型构造源码已查哈希，没有构造 DeiT 大模型。当前可选 `pypdf` 未安装，PDF 抽取未验证；文献网络访问未执行。环境版本见 [environment.json](environment.json) 和 [长期环境说明](../../../QAViT/docs/ENVIRONMENT.md)。

## 6. 留待人工审核的原有问题

以下均未修复。可复现控制流证据来自 [review_cases.py](review_cases.py) / [review_cases.json](review_cases.json)，其中模拟输入明确标注；静态推论另行标明，不冒充真实测量。

| 问题与代码位置 | 已有证据及影响 | 建议下一步验证 |
|---|---|---|
| 排名：`core/debate.py:29 ranking_table`；`agents/interaction.py:209 submit` | 脚本中 H1 三胜、H4 零胜三负，仅 H2/H3 未决；`requires_provisional_basis=true` 时选择 H4 且 provisional=true 仍被接受。未决标记可能绕过直接优胜约束 | 人工确定未决与可选集合关系，再补 tie/cycle/ineligible/修订裁决矩阵测试 |
| 证据来源绑定：`tools/experiment_worker.py:343 run_experiment`（provenance 合并）；`tools/experiments.py:211 evidence` | 静态源码中 `**payload['scope']` 最后合并，可覆盖 protocol_hash 等同名来源字段；Python 字典小例复现覆盖，未执行真实 worker。回执/资产校验不等于逐字段来源校验 | 用固定 CPU worker 验证保留字段冲突、factory 来源类别、协议与模型身份绑定；先界定哪些字段可信 |
| 缓存与解释：`tools/experiments.py` 的 `execution_fingerprint/cached`；`core/experiment_requests.py:progress_request` | 缓存主要按执行身份复用，问题/解释范围不在同一键内；复用 Evidence 保留原实验引用，新解释绑定当前申请。当前机制保留，是否足以防止跨语境推断需审核 | 同执行、不同问题/范围的双申请，检查原始事实与新解释各自引用及适用范围 |
| Recorder：`core/memory.py:97 validate_metareview`；`core/cards.py:136 validate_payload` | 脚本证实空 judgment_audit 与空 knowledge_updates 可通过；逐项校验未要求覆盖本轮全部科研判断，可能缺少应有审计 | 明确必须审查的判断集合，再检验遗漏、重复、无证据与跨轮引用；空更新是否允许由科研要求决定 |
| Memory 检索：`core/memory.py:192 retrieve`、`:230 relevant_delta`；`runtime/driver.py:43 dispatch` | `after_version=当前版本` 时，非显式旧 HypothesisCard 不再出现在 cards 中；L1/L2/必读 Meta-review 仍存在，不能据此称全部记忆丢失。刷新事件与实际重新推理效果也需区分 | 同任务多轮检索：新事实、旧相关事实、显式引用、scope 匹配、freshness 事件和实际上下文逐项比对 |
| 预算：`runtime/budget.py:53 check/:78 reserve`；`runtime/driver.py:214 run` | 剩余 100 秒、申请声明 101 秒即抛 BudgetExhausted（脚本证实）；从 launch 冒泡至 run 的路径会结束运行，是静态控制流推论，可能尚有可执行短任务 | 假时钟下并列长/短任务，区分单请求不可容纳与全局耗尽，再决定策略 |
| 内外预算边界：`ptq_agent/profiling/host.py:50 check_time`；`execution/search_context.py:268 submission_budget` | 原数值工具仍有活动时间及调用预留判断。本次保留这些分支，不能声称工具只受一个全局截止时间约束 | 假时钟验证延长全局预算、工具局部上限、候选诊断预留和恢复耗时关系；需人工决定统一方式 |

## 7. 历史索引与文件管理

历史只读入口：[V2 初次实施](../qavit_v2_20261026/REPORT.md)、[V2 脚本验证](../../validation/qavit_v2_20261027/)、[历史真实 Agent 验证](../../validation/qavit_v2_real_agents_20261027/)、[旧运行一](../../runs/deitb_static_w4a4_free_20261024_r1/)、[旧运行二](../../runs/deitb_static_w4a4_free_continue_20261025_r1/)。历史结果的真实性与验证边界以各自原记录为准，本次没有重新背书。

新任务使用当前版本创建新运行；不承诺直接续跑重构前 V2，也不为历史兼容保留旧调度器。当前版本自身的队列恢复、预算延续、代码 intervention 和证据追踪保留。

本目录的迁移/对照脚本仅为一次性实施证据，框架不导入它们；中间失败日志和原件快照保留追溯价值。测试可复用程序在 QAViT/tests，执行产物没有混入 QAViT。结构整理完成不表示 V2 科研问题已经解决。
