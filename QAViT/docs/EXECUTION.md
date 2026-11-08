# 运行代码与工具接口

## QAViT v2 科研协议

适用于 `schema_version=qavit.research/2`；这是当前唯一科研入口。数值计划的 schema_version=2/3 是模型包协议，与科研框架版本独立。

卡片合同位于 `core/cards.py`，版本为 `qavit.cards/2`。Agent 提交科研 payload；宿主补齐任务固定的 round/slot/type/phase/side、预测 ID、唯一绑定的前轮反馈引用，以及卡片 ID、版本、作者、来源和哈希。显式身份冲突仍拒绝。memory.source_handles 的短引用（如 C42）可用于结构化引用与 read_card；宿主按固定快照解析为完整 ID，不猜错别字或解析自由文本。允许完整数学推导、多条预测、来源和扩展字段，不限制科研正文长度。单次 CLI 的完整响应仍受传输保护限制，超限时拒绝不完整结果，不静默截断科研卡片。

ResearchBrief 分开保存用户硬约束、搜索空间、基准模型和评测协议。hard_constraints 的数值/字符串/布尔要求相等，列表表示允许集合，`{min,max}` 为数值范围，notes 保存自然语言。未指定参数不等于4位、某个固定粒度、32个校准样本或禁止训练。model.adapter 明确本次基准；允许研究不同模型时，用户可声明 model.allowed_adapters。不能在实验时偷偷替换基准。

必需执行约束统一为 `ResearchBrief.quantization_contract` 与冻结后的 `ExperimentCard.execution.quantization_contract`，由 `core/execution_contract.py` 独立核对。现有 `weight_bits/activation_bits`、`weight_spec/activation_spec`、`weight_granularity/activation_granularity`、`weight_axis/activation_axis`、`granularity.{weight,activation}`、`quantization_scope.{sites,kinds}` 和平铺 execution 字段可无歧义归并；同义字段冲突会被拒绝。`fixed_quantization` 中已支持的字段也进入合同；未知必需字段、自然语言 exceptions/parameter_contract 不再被静默忽略，须改为结构化字段或放入仅供科学审查的 notes。旧历史卡片不自动迁移。

| quantization_contract 字段 | 宿主核验内容 |
|---|---|
| required_sites、required_site_targets、required_kinds | 精确 site_id 数组、site_id→weight/activation 目标数组、明确 kind 数组；缺位点、未知/空展开、重复位点及必需目标留 FP 均拒绝。未单列 targets 时按已声明规格或实际计划中的目标核对，不因 Linear/Conv2d 自动要求同时干预 W/A。 |
| specs、methods、calibrations | 按 weight/activation 映射到规格字段、方法名、observer；规格包括 bits/format/granularity/axis/group_size/symmetric/scale。 |
| site_overrides | site_id→layer 的 W/A 规格、方法、observer 或 options；逐字段核对，observer 覆盖通用 observer。 |
| execution、preprocess、operators、unlisted_modules | 实际执行方式、计算精度、校准方式、阶段顺序和预处理/算子声明；必须与冻结选择一致。 |
| parameters、parameter_bindings | parameters 由程序镜像到冻结记录并实际传入实现。parameter_bindings 是内置计划可选的字段对应关系；提供时逐项检查 parameter、site、target、field、as_list，省略不阻塞自定义算法。内置能力校验发生在实际使用内置执行器时。 |
| fp_exceptions | 对象：sites 为 site_id→目标数组，modules 为完整模块路径数组，parameters 为参数/buffer 完整名称数组；宿主核对计划及实际模型状态，不从文字猜例外。 |

Brief 的全网位宽、覆盖及最终评测要求属于最终候选，在既有 L3 阶段核验。L1/L2 探索按本次显式执行约束核验，允许局部干预、消融、浮点对照后观察整网输出；观察范围不产生干预范围声明。最终研究目标仍保存在 research_quantization_contract。位点种类展开等依赖真实模型目录的检查在 worker 加载模型后执行。自定义方法名、observer 和参数由批准源码解释；内置计划的枚举只限制实际使用内置执行器的请求。

HypothesisCard 科研核心是 title、research_question、mechanism、intervention。预测、来源、机制关系、实现说明、验证设计、scope 和 feedback_used 均可省略或按需表达，不要求为满足格式编造内容。H1/H2/H3 分别是 existing_method/engineering/new_algorithm；H4 为 evolved，保留 improvement_objective 与固定 H* 的 parent_refs 关系；身份、轮次和唯一父引用由程序绑定。前轮反馈由程序交付，是否采纳及如何改进由研究者判断，不强制填写 feedback_used。

ReviewDecisionCard 保留 rationale 及本任务的实质决定。screening 的 eligible 必需；不合格时必须说明 hard_rejection_reasons，合格时可省略并补为空列表。similarity_analysis、evidence_gaps、uncertainties、论据/反驳/回应/预测数组等按需表达。裁决 outcome 为 A/B/tie/undetermined/continue，第5回合不能 continue；胜负依据可统一写在 rationale，decisive_reason 和 next_questions 可省略；continuation_assessment 可省略；若显式提供，仍核对其与未决结案的一致性。选择和 H4 关系、已结案不可改判及证据权限保持不变。

可选科研条目逐项处理：格式或引用有问题的条目连同具体问题保存在 submission_issues，不成为有效科学支持，其余有效科研产出可提交。顶层明确证据引用、固定身份或实质决定冲突仍返回准确错误。Recorder 核心只需 scientific_analysis；知识更新有问题时保存在 pending_knowledge_updates，不发布为已验证知识。

两阶段均复用 rank/Judger 与 ReviewDecisionCard：`rNNN/select_parent` 的 selection_stage=parent，仅用 H1-H2、H1-H3、H2-H3 选定并持久化 H*；`rNNN/rank` 的 selection_stage=final，仅从 H4 与 H* 中确定最终候选。卡片绑定 candidate_refs、selected_hypothesis_ref 的完整 ID，最终卡另绑定 parent_selection_ref。禁止将四场合并为四候选全局排名。只有 final 选择写入 Memory 的 round_selection，并交给 L3、Recorder 与下一轮反馈；两个选择均进入 Recorder 科研上下文，是否逐项分析由科研需要决定。

已有全胜者、仅余一个合格方案或没有合格方案时，程序直接引用已提交裁决/筛查完成现有 rank 任务，选择卡 author=program、selection_source=existing_verdicts_and_screening，不产生新的 LLM 判断或伪造测量。未决时仍由 Judger 给出 selected_slot 和选择理由，在当前阶段的直接结果第一组中选择，不能忽略明确胜负。provisional 缺省时从已有直接结果确定，显式冲突仍拒绝；selection_basis 可省略，程序原样使用 rationale 交接选择依据。H4 不合格保留合格父方案；初始全部不合格则两个选择均为空，跳过 Evolution 和 L3，正常 Recorder 与轮次收尾。

初始 H1–H2、H1–H3、H2–H3 三场独立推进，各自按需取证、讨论和结案；全部提交 match_closed 后才统一选择 H*。恢复时沿各场已有任务和结案事件继续，不重复结案。之后仍只比较 H4–H*。每场最终裁决包括 tie/undetermined 一经结案即固定，拒绝非空改判请求，不再生成改判字段或修订卡片。旧记录保持可读，不改写历史。每对最多五次交锋；本场取证、真实结果核验及 Recorder 科研知识更新保留，后到证据不能改写已结案结果。

默认 runtime.llm_concurrency=4（可设1–4）；Judger最多3个，每场同时1个；其它角色合计单通道。项目订阅锁保持3+1。数值实验使用独立 experiment_concurrency 上限，新模板为 auto，解析后等于允许 GPU 池大小（CPU 路径为1）；显式整数上限保持有效。原非数值worker合计上限llm_concurrency+1保留，数值不占这些槽。不以相同模型checkpoint或全局numeric锁串行化独立实验。GPU按空闲情况分配，以 UUID 文件租约跨进程互斥；worker及分片继承租约，忙卡留在ready队列，不占尝试/重试次数。Coding→Safeguard→执行→原请求Evidence链不变。旧运行显式并发1仍遵守原配置。

新运行 runtime.allowed_gpus="auto"（省略时同义）从 CUDA 实际可见 UUID、容器设备访问和分配环境取交集，不直接使用主机全部 NVML 设备。支持 CUDA_VISIBLE_DEVICES 的实际逻辑映射及 SLURM_JOB_GPUS/SLURM_STEP_GPUS 分配范围；NVIDIA_VISIBLE_DEVICES 的 UUID 范围也会收窄。允许设备可在同一研究 JSON 中显式设为 NVML 编号或完整 GPU UUID 数组，[] 禁用 GPU；显式范围仍与实际可见性取交集。自动探测不可靠时可集中提供 runtime.gpu_inventory=[{"index":编号,"uuid":"完整GPU-UUID","device_node":"/dev/nvidia设备minor"}]，以 CUDA_VISIBLE_DEVICES=GPU-UUID,... 明确可见性；不会猜测不可靠的编号对应关系。新运行冻结解析后的 allowed_gpus、gpu_bindings 和 gpu_selection；恢复不扩大冻结池、不改配置身份，当前设备仅按冻结 UUID 匹配，即使 NVML 编号或设备 minor 改变也不换卡。暂时不可见的冻结设备保留排队；资源范围确需改变时使用新的研究输入。

resources.gpu 的数字是本运行冻结的 NVML 编号标签；CUDA worker 通过 UUID 只暴露所租主卡，其模型设备统一为逻辑 cuda:0。容器 /dev/nvidia minor 另行按 UUID 对应，不能由编号拼猜。每个运行进程仅解析一次设备身份（恢复进程重新解析），避免反复 CUDA 初始化造成瞬时占用；排队只刷新占用和租约，不支持运行中热插拔扩池。ADE 与其它后端共用此路径，不再对旧物理编号另设限制。完整 result.json 写出后继续保留任务和资源占用，等待登记的 worker/子进程退出；默认30秒退出宽限后仅清理这些已完成进程，再接收原回执，恢复也沿用此状态。科学结果不因正常退出收尾变成失败，设备不会在进程仍存活时释放。

resources.gpu 可选，指定时固定主卡；不指定时从 allowed_gpus 自动选择。默认每实验一卡；可选 gpu_count 是尽力使用的卡数，不能突破授权集合。当前仅 server 后端、原生非训练 quantize_evaluate/evaluate_artifact 的纯 classification 评测支持实验内多卡：单卡prepare/校准/导出一次，分片重载同一冻结包，复用逐图evaluate和aggregate，按整数correct counts与FP64误差累加器汇总，核对模型、请求身份及完整不重复样本。仅在必需指标可汇总时启用；latency_ms、measured_speedup、CUDA显存等必需指标及无法解释的组合 required_results schema 自动选择原单卡评测，保留全部要求。局部探针、profiling、factory/fixture、自定义executor、ADE或混合评测仍沿单卡；少卡时减卡或单卡，忙卡等待。配置 experiment_concurrency=1 可限制跨实验数值并发，gpu_count=1 可限制实验内部多卡，均不改变 LLM 并发。实际卡和回退原因写入回执；分片原始结果归档，不填造未采集的性能统计，不从QDQ耗时声称部署加速。

新模板明确启用 budget.clock_mode="active_segments"、budget.exclude_numeric_wait=true。累计智能体预算扣除纯数值执行/资源排队/结果等待的区间，并行推理、Coding、审核等期间仍扣时，重叠只计一次。数值 estimated_seconds 沿用单任务超时，不再与智能体剩余时长比较；已启动实验不会被总时长或收尾预留截断，余额不足时先接收它们，不再启动新工作。外层监控跟随同一余额，暂停只在所有自有进程退出后结算，恢复不重置消费；等待检查点之后的未知崩溃间隔保守计时。人工停止、磁盘限制及单任务超时保留。旧未启用的连续/活动段预算保持原语义。

每次调用保留独立提交快照及同回合A/B隔离。自动最新发现与数据校验引用仅来自当前任务已关联的证据，自身结果及明确引用支持仍交付；公开Memory按需读取权限不变，不自动灌入无关外场测量。

### 任务输入来源

| 任务 | 必需交接材料 | 历史读取与限制 |
|---|---|---|
| generate / evolve | ResearchBrief、分配槽位、与当前对象明确关联的问题；generate 另获任务已绑定前轮 Meta 的 generation_guidance/next_questions/recurring_failures/blind_spots；evolve 获固定 H*、选择理由、批评及证据 | 当前正文和支持证据有界直供，长正文入口在 required_reads；完整 Meta-review、L2 和非绑定正文按需读取 |
| screen / position / arbitrate / rank | 待审候选与阶段、当前/最近争点、新返回结果；position 指定一方，arbitrate 接收已汇合双方；rank 仅接收当前阶段结果 | 全部授权记录可分页发现和读取；对方本 exchange 未汇合立场不可读，结案不可改判 |
| plan_l3 | final 选择卡、最终候选完整 ID、选择依据及明确引用的支持 Evidence 观测与条件、Brief 协议、数据回执与预算回执 | 原件按需精读；不拿初始 H* 代替最终候选，有效 L3 不因解释失败重复执行 |
| code / safeguard | 不可改写的原始需求、执行条件；code 获当前源码及失败反馈，safeguard 获当前唯一源码 | 按任务授权读取公共接口及相关来源 |
| interpret / record | 实测 Evidence 与对应定义；record 交付未综合的假设、实测证据、裁决/选择/解释与工程失败摘要 | 旧知识按需读取；长正文可分页定位；结果解释不覆盖测量，知识更新不改比赛裁决 |

具备 request_experiment 权限的任务均获得 data_receipt（包括生成/演化/规划），其它任务按 purpose 交付绑定材料。回执派生自有效后端配置、明确 overrides、数据清单及本次运行匹配的 Evidence 数据身份，无独立可编辑配置。available_data 标示池容量与半开区间 `[0,N)`，selected_data 仅转述草案/冻结执行视图；整个视图为 null 表示未声明，可用容量为 null 表示未知，并列 unknowns。视图内显式 count=null 保留原接口的完整/剩余范围语义。清单 SHA、配置数量、已有校验的范围分别说明；不加载模型、不重读全部图像，也不把目录/清单检查当作运行时验证。配置/清单路径用于溯源，不因此扩大 read_asset 的读取权限。

budget_receipt 来自一次 BudgetLedger 快照：remaining_seconds 为当前计时模式的余额，research_remaining_seconds 已扣 closeout_reserve_seconds 与 shutdown_reserve_seconds；available_seconds 保留为前者的兼容字段。启用 exclude_numeric_wait 时，数值耗时无需小于该余额，deadline_unix 随纯数值等待后移；历史显式预算模式按原口径执行。未结算 estimated_seconds 按类型汇总，现有 check 不再次扣未结算估计，LLM 的估计为超时上限；输入展示不新建额度或预留规则。实际启动仍执行原预算、磁盘与资源检查。read_asset 限本次授权来源、reference、QAViT/docs，以及 code/safeguard 的明确公共数值接口白名单；不自动开放其它 EXP、配置或源码目录。

### ExperimentCard 与真实服务

唯一公开入口仍是 request_experiment；plan_l3.submit 使用同一草案。模型最小输入为 question 与 measurements，科学约束按需给 controls；prediction、decision_impact 可省略。原始需求先独立保存，Coding 交付后再生成完整执行卡。宿主维护任务、身份、哈希、版本、资源、默认值与参数镜像；明确冲突返回具体字段。资源默认来自当前运行，GPU 仅在授权设备唯一时补齐，不猜未知科学条件。新代码不要求独立参考、测试夹具或双实现容差。

execution 的 service 和必要科学参数可由已有服务请求直接指定，或由 Coding 在原始约束内完成 execution_choices。程序补齐确定的 backend、seed、training 约束和 resolved_choices 镜像；明确冲突拒绝。位宽、signed、覆盖、粒度、轴、校准规模和方法遵循原始科学约束；宿主不猜未知定义。配置、模型、数据及实现仍以哈希绑定，数据视图不改旧模型包身份。

校准以 `execution.calibration` 为执行视图的权威来源，`resolved_choices.calibration_samples` 保留为内部一致性镜像。草案可只声明 `calibration.count`；宿主生成镜像。兼容只声明正整数 `calibration_samples` 的草案，机械补齐缺失的 count；两者显式冲突时拒绝并指出字段和值。count 必须是正整数或显式 null（沿用数据视图从 offset 到 split 末尾的语义），缺失不再隐式代表完整视图。`tensor_probe` 和 `quantize_evaluate` 的 worker 固定读取 calibration split，故可补齐缺失 split；显式其它 split 会被拒绝。自定义实验未固定 split，须明确声明。实际需要校准且本次没有给出用量时，可从已明确的 hard_constraints/evaluation_protocol.calibration_samples 补齐；探索中明确给出的不同用量不被最终候选默认覆盖。无校准操作不生成校准视图；不会把后端样本池容量当作本次实测用量。已有扩展字段保留。

`plan_l3` 完成和 `request_experiment` 入队前，与冻结入口共用纯参数预检（卡片合同、执行选择、校准视图及服务层级）。已提供字段的结构、镜像冲突、服务层级及已知资源/授权边界在固定原请求前检查，输入错误直接回到原提交任务局部修复；尚未决定的实现选择仍由 Coding 完成，依赖实际模型目录的检查延至 worker。预检不读取模型/数据资产、不冻结执行卡片、不启动实验或占用实验预算。可修复错误进入原任务现有 `repair_feedback`，附错误、原始响应（含草案或工具批次）及回执路径，当前失败步骤最多连续修复两次；合法工具动作结束、进入新研究步骤后清除已解决的 repair_feedback 并重置该连续计数。累计 contract_error_count、历史错误回执、总预算和代码实现修订上限保留；不确定的规格使用 clarify。工具批次先全部预检，避免修复时重复入队前半批实验。冻结时仍核验资产、证据绑定和当前预算；预检通过不代表实际数据、数值实现或实验验证通过。

| service | 配置与含义 |
|---|---|
| profile | L1；提供 tool、arguments，复用 `tools/profiling.py` 的 PROFILE_NAMES；保留已验证工具，移除旧调用/测量计数门禁。 |
| tensor_probe | L1/L2；提供 site、method、parameters、calibration、evaluation。真实捕获位点并测输入/局部输出误差；L2 额外执行单边替换的整网小样本回放，不冒充全模型量化精度。 |
| evaluate_artifact | L2/L3；明确 model_dir，加载冻结量化包后运行 evaluations，不导入旧指标。 |
| quantize_evaluate | L2/L3；现有算法交 execution.plan；新增实现走 Coding/Safeguard。共用独立覆盖审计、导出重载和评测。 |
| custom_experiment | L1/L2/L3；开发新的统计或算法，必须经过单份实现、执行条件绑定及 Safeguard 科学审查。 |

`read_profile_evidence` 的 `arguments.evidence` 使用先前成功测量返回的 `evidence/...json` 引用。宿主从本次运行已提交的 EvidenceCard 冻结原始测量和工具回执的哈希绑定，在新作业恢复成功登记后分页读取，不重新测量；跨运行、未提交、越界、损坏或模型身份不匹配的证据不获授权。

服务参数、Profiling 常用接口和现有方法目录见 [EXPERIMENT_INPUTS.md](EXPERIMENT_INPUTS.md)，作为 context.experiment_interface 直接提供给有实验权限的角色，无需反复读取整本文档。

单实现交付、原始需求与执行记录绑定、真实数据接口及结果检查见 [CODE_INTERFACES.md](CODE_INTERFACES.md)。工程角色直接接收该接口正文。

### Evidence 与四层 Memory

EvidenceCard 只由程序从真实工作进程回执构建，保存数值、配置、模型状态、代码、数据划分、种子、资源和原始结果引用。measurement_source 区分 real、synthetic_computation、test_double；随机模型真实运算不代表真实任务精度。科研解释保存在原研究者的 HypothesisCard、ReviewDecisionCard 或 Recorder 的 MetareviewCard 中，不能覆盖测量。

Generation/Evolution 可在正式 H 之前执行零次或多次“问题/缺口→实验→解释→调整”循环。无需预先提交 H；程序完成参数/范围、预算、来源与可执行性检查后冻结并执行，不再创建常规 experiment_approval，也不另调 Judger 解释。Evidence 与失败回执直接返回等待请求的原 Generation/Evolution/Judger 任务，step 递增，保留问题、可选 research_note 及来源，下一次调用刷新可见快照。LLM 决定科学参数、解释结果与是否继续；程序补默认值并处理路由、状态、预算、缓存和原件。HypothesisCard 可用 evidence_refs 与 sources.ref 精确绑定实际采用的依据，不把全部已归档实验视为支持。

单请求预算不足、执行失败、Coding/Safeguard 科学澄清和实现修订耗尽均返回原科研请求者，不等同全局耗尽或假说反证。等待中的 Generation/Evolution/Judger 按现有任务续调；程序根据已提交 plan_l3 创建的最终 L3 则由新的同角色 plan_l3/revisionN 接收原需求、失败或澄清及来源，再产生新的 L3/revisionN。已完成计划和冻结请求保持不变；恢复运行沿已有版本继续。可恢复工程故障仍按现有分类有限重试，新实现重新绑定条件并经 Safeguard，不重复未修改的失败源码。无法在已有范围内继续时，Judger 可用可选 defer_reason 如实说明未验证，正常进入 Recorder；只有用户才能决定的信息才进入人工澄清。

所有服务与请求来源的有效 Evidence 保存、规定数值协议核验完成后立即返回，不再为实验回执自动创建 interpret。原研究步骤或正常轮末 Recorder 承担解释；恢复时取消尚未启动且已无依赖必要的旧 interpret；已运行的 interpret 单独结算，均不撤销有效测量，也不触发重复执行。缓存保留原 execution_id 并标记 cache_reused，不冒充独立重复。报告与 CLI 分列 workflow_completed、formal_validation_success、numerical_validation_status、scientific_review_complete；每轮 l3_outcome 同时给出执行、解释和计划状态。completed 仅代表流程结束，不代表缺失或延期的 L3 已通过。三个初始 H 均形成并筛查后，才进入既定三场配对与 H4 父子挑战。

新 Evidence.scope 使用 `qavit.evidence_scope/1`：由冻结模型身份及宿主 execution_facts 生成 model/model_identity_sha256、measurement_kind、measurement_source、实际 site/quantization/位宽、parameters 和 data。完整模型数据范围包含导出包实际校准 ID 和本次评测 ID；局部激活探针不因研究目标 W4A4 而获得 weight_bits=4。自定义统计仅记录实际读取和前向访问、冻结参数，不把无法独立核验的量化目标提升为已执行事实。申请 scope 与 resolved_choices 仅存 provenance.research_target。

Recorder 实证知识的完整 scope 由真实引用证据绑定模型、位点、位宽、参数及数据边界。模型可完全省略 scope 或只给科学条件；缺失的确定身份由程序补齐，明确冲突逐条记录。不同条件的比较按 scope.comparison 逐份保存 evidence_ref 和真实 scope，不取第一份或合并范围；比较差异、解释和限制写在既有 text/rationale。无法绑定的条目保留为待核实，不影响其他知识和科学分析。精确检索仍使用同一规范 scope，词法排序及四层结构不变。

内置实验指纹覆盖实际 execution、L1/L2/L3 层级及计算资源，不以科学问题或控制说明的不同措辞区分相同计算。插件可读取完整正文，因而将其全部实际输入也纳入指纹。相同有效测量可复用；明确重复验证产生新的 execution_id，缓存命中不能计为重复验证。方法与假设的科学重复性由结构信息、来源、Memory 和 Judger 审查共同判断。

L0 为不可变卡片/提交；L1 根据 Journal 维护本轮真实假说、实验、证据、选择与 incumbent，不能由 Recorder 引用子集覆盖；L2 为有状态、范围和出处的知识；L3 为每轮 Meta-review。Recorder 维护科研判断，程序检查来源并保护原始事实。knowledge_updates 可为空；每项科研输入是 text、status、evidence_refs、rationale。新 claim_id 可由程序生成；scope 可完全省略或部分提交，程序从引用证据绑定可确定字段；比较条件逐份绑定，冲突条目保留待核实。持久化项仍包含完整 claim_id、text、status、scope、evidence_refs、rationale；status 为 hypothesis/observed_once/replicated/refuted/conditional/abandoned。异质条件的对照使用 conditional 等适当状态，不因两个执行就视为重复验证。replicated 需要相同条件下不同执行与 replication_analysis；更新已有 claim_id 时，程序绑定 supersedes 为当前 revision_ref；显式冲突局部处理。judgment_audit 为可选的重点判断分析，条目含 judgment_ref、assessment、evidence_refs、reasoning，不要求穷举。Recorder 应解释预判与实测差异、指标价值、失败机制和下一轮问题。next_questions 只新增未决问题，省略或空数组均不清空旧问题；可选 resolved_questions/abandoned_questions 明确移除已解决/放弃的问题，无关闭审批。pending 沿 claim_id 与旧 revision_ref 修正、发布或 abandoned，原稿保留，当前待办只呈现仍待处理项，不启动额外修正文书调用。

L3 失败、取消或拒绝也进入 Recorder；`task.l3_outcome` 默认交付请求/执行状态、子任务状态与失败原因、原文入口、有效证据及缺口；完整工程结果和长日志按需读取。报告的 `status=completed`、`workflow_completed=true` 仅表示流程完成。`formal_validation_success` 另行核验本轮冻结规格、执行指纹、协议及原始资产；已完成请求没有有效 L3 证据仍为 false。报告总值要求流程完成、至少一项 L3 申请且所有申请验证成功；各轮保留独立结果。该标志不表示假设正确、达到性能目标或完成了规格以外的评测。

Recorder 只需提交 scientific_analysis；generation_guidance、next_questions 按需表达；可补充有证据支持的知识、关键判断分析、指标适用性、失败模式和盲点。程序绑定全部 source_refs、recording_batch_id 与 analysis_scope；来源覆盖表示纳入批次综合，不宣称每项经过科研认可。新提交不使用 reviewed_refs；兼容收到的旧字段时忽略，既不认定已审，也不拒绝有效复盘。

incumbent 独立保留现有同模型、同协议、完整分类及分割证据和指标比较门槛；这些资格条件不倒推为所有 L3 的必做任务。Brief 的 formal_incumbent_allowed=false 明确禁止晋升；仅分类或子集协议可以完成本次 L3，不能因此宣称通过 incumbent 检查。

Memory 的唯一事实来源仍为 Journal 和不可变卡片；目录、输入和四层状态都是可重建投影。参考 Letta MemFS 的简短常驻目录和按需读取，不引入文件系统记忆库、向量数据库、新智能体或额外 LLM 筛选。MemFS 提供设计参考，QAViT 的提交事实与科学综合状态分别维护；参考记录留在 EXP。

按 purpose 交付明确处理对象，正文不会同时出现在 task、memory、tool_results 和 messages。自动历史目录仅给4条与分页入口；主动 retrieve_memory 沿用默认12条可用正文。已限定类型或证据ID的查询不重复附方法文档：

| purpose | 默认必需材料 |
|---|---|
| generate | ResearchBrief、本轮类型/要求、简短目录、明确关联或分配的反馈；全局复盘及 L2 按需读 |
| screen、position、arbitrate、rank、interpret、plan_l3 | 本次待审方案/比较对象、当前或最近争点、明确引用的判断/测量、本任务新返回的实验与对应冻结定义；其他场次和旧正文按需读 |
| evolve | 唯一 H* 完整方案放在 task.parent_hypothesis，一处提供；选择依据、明确批评与绑定证据；不再次默认注入完整最新 Meta-review |
| code、safeguard | 原始需求、已知执行条件及当前实现/修订反馈；Safeguard 另获绑定后的完整执行卡 |
| record | 本批假设、Evidence/冻结定义、裁决/选择/解释；工程计数与失败原因；本批关键立场/争点直接交付，完整来源按需读 |
| supervise | 当前任务、流程状态、预算和异常反馈，历史科研正文按需读 |

短必需正文直接提供；超过展示阈值的必需卡片/字段/源码在 `memory.required_reads` 或 `required_assets` 列出精确引用、字段、哈希、长度和 next_offset。它们标记优先交付的材料和读取入口；分页完成度不作为提交条件，研究者按科学判断读取所需内容。页面在无状态续调中保留一次。读取记录只证明程序交付过文本，不能证明 LLM 已理解。

`retrieve_memory` 默认返回 `entries/offset/total/next_offset` 和程序组织的可用短正文：关键结果、真实条件、知识状态、重要限制、精确来源与展开入口。省略 query、`{}` 或 `browse=true` 均可浏览；`kind` 过滤类型，`layer=L1` 获取当前研究状态，`layer=L2` 获取当前知识修订。`text` 保留词法排序，其余字段精确匹配 scope；比较知识明确标注匹配的证据组与其他组的不同条件，不把相关经验混作相同条件。`after_version` 为增量界限，limit 为 1–50。短正文不是完整原件，也不自动证明结论；无新增 LLM 调用。方法目录复用 `docs/methods/index.json`，提供标题、适用目标和直接 `read_asset` 文档路径，查阅文档无需实验。

大型 Evidence 的客观统计视图也受展示预算控制：保留可交付的结果和条件，对未展示部分列出大小、精确字段/选择读取入口及不完整标志。全体统计、极值、符号冲突只代表描述性统计，不推导科学结论；必要反证和未展开材料不静默丢弃。`read_card(select={site,token_type,metric,...})` 可精确展开原始标量。旧工具正文在研究进展后转为短回执，沿用 research_note 保存当前解释与下一步理由；原工具页和来源仍可读。预算仅用于组装和展示，不限制科学提交。

`read_card(ref, field?, offset?, limit?)`：field 是 JSON pointer（如 `/payload/observations`）；短卡默认返回完整卡片，长卡或指定 field/分页参数返回 canonical JSON 文本页面和 sha256/next_offset。`read_asset` 按字符分页读取 UTF-8 文本。两者单页最多 12000 字符，可用现有 tool_calls 批次同时读多卡、多字段、多页。query 可省略或为空对象；显式 text/ID/过滤值类型校验保持；已知输入错误继续返回原任务并沿用有限修复，未知程序异常不吞掉。

订阅传输保留原 `name/arguments_json` 封套；多个独立工具用 `name="__tool_batch__"`，arguments_json 解码为 `{"calls":[{"name":"read_card","arguments":{"ref":"完整ID","offset":0}},...]}`。宿主解码为原有 tool_calls 列表，所有参数/实验草案先预检，再各自入队；运行期某项无权限/失败不吞掉同批其他结果。不得夹带决策、嵌套批次或同批依赖；工具预算为零时批次也不开放。格式和合同错误共用当前失败步骤最多两次连续修复；CLI 完成/身份/隔离检查仍在之前，整批调用的已知失败用量正常结算。

每次模型调用固定 `read_view.sequence`；目录、提醒、搜索、读卡和资产权限都先按这个提交快照过滤。工具批次沿用发起调用的快照；下一次调用（含实验回传）再刷新。Generation/Judger/Evolution 可发现此前提交的 ExperimentCard、EvidenceCard、公开论证/interpretation 及文献来源，不等待 Recorder。资产访问仅允许同一快照中明确关联的文件及固定项目资料，并绑定哈希；不能读取运行目录下任意日志/新文件绕过隔离。同回合对方立场、其私有子任务与间接引用均隔离。

Recorder 复用轮末及预算收尾的轻量同步任务。批次输入集中于新增假说及变化、实验目的/核心结果、关键裁决争点、相关旧依据与影响判断的失败。仅工程状态补记而无新增待综合科研材料时不单独启动 Recorder，工程记录照常归档；失败延期的科学材料继续 pending，不能标成已复盘。程序从已提交的 recorder_batch_completed.source_refs 重建已综合集合，批次按完整对象版本冻结，迟到产出不会被跳过。来源绑定、索引、游标、去重和完整原件均由程序维护；模型不填写覆盖清单。Meta-review、Memory 投影、批次完成事件与 TaskCard 原子发布，存储失败不推进游标；recording_submissions 只为已验证待发布草稿提供恢复入口。

事实归档与科研复盘分别报告：recording_status.facts=archived_in_journal；scientific_review=pending/completed。复盘的 completed 表示没有待复盘的科学材料；纯工程补记仅归档，无需发布 Meta-review，也不是逐条科研认可。Recorder 的有限修复耗尽、取消或请求澄清会记录 recorder_deferred，保留未综合来源，不伪造 Meta/L2。正常轮次可继续；下一轮 Generation 明确收到复盘缺口与前轮假设/证据/选择，下一批 Recorder 自动合并旧来源。同一失败批次不在收尾无限重试，有新增科学材料或相关失败时才重新复盘；新增成功工程快照不会单独重试延期批次。可以通过现有显式重试入口处理留下的任务。

报告同时提供 workflow_completed、scientific_review_complete、formal_validation_success，三者语义独立。未完成复盘不会单独被误报成存储损坏；真实持久化错误仍标 recovery_required。硬预算耗尽或中断时保存 recorder_pending/pending_recording，不额外启动 LLM，不扩大原截止时间/调用额度。源码身份和历史恢复约束不变。

## 数值计划与修改入口

内置执行器使用 `executor_v1.build_plan(backend, cfg, experiment_payload, parameters)` 返回的数值计划：
`schema_version=3, unlisted_modules="keep_fp", execution, preprocess, layers, unresolved`。
`execution` 明确 mode、compute_dtype、activation_calibration、order，以及需要时的 within_stage；每个 layer 分别声明 W/A 规格和方法。
科学选择来自冻结 ExperimentCard/ResearchBrief，不从旧实验继承固定位宽、覆盖或质量目标。

实际使用内置执行器时，宿主先通过统一 normalizer 补齐唯一确定的 bits、空 axis/group_size、禁用目标的空 method 等，再按内置计划 Schema、未解决项、冻结字段和设备能力检查，再执行捕获、prepare、audit。提供 make_executor 的自定义实现使用其自己的计划表示和已批准接口；宿主仍检查真实输入、基准不变性、导出重载及结果。内置枚举不充当新算法注册表。

以下字段直接对应 `contracts.SPEC`、`plugin_contracts.plan_schema/site_id`，是接口编码说明，不增加科研默认值。`layers` 是条目列表，不是以位点为键的字典；v3 不使用单一 `method` 字段。`layers=[]` 可表示纯浮点对照，不伪称发生量化。顶层 `preprocess`、`unresolved` 必须是列表；无已声明变换或未决项时分别为 `[]`。`evaluation`、`operators` 为可选字段，没有声明需求可省略。

| layer 字段 | 合同 |
|---|---|
| module、kind | 必填字符串；module 为真实模块名，kind 与 site_catalog 条目一致。Linear/Conv2d 使用 `linear`/`conv2d`；注意力纯激活使用 `attention_query`、`attention_key`、`attention_value`、`post_softmax` 等实际目录 kind。 |
| weight、activation | 两者均必填；值为 SPEC 对象或 null。纯激活位点 weight 必须为 null。 |
| weight_method、activation_method | 两者均必填；目标非空时为已注册方法名，目标为空时为 null。分别与各目标规格对应。 |
| reason、evidence | 可选；reason 可为空字符串，evidence 可为空列表。有实际依据时填写，不为逐位点说明编造来源。 |
| weight_calibration、activation_calibration | 可选的独立 observer 对象；目标为 null 时禁止提供对应字段。`strategy` 必填；minmax 可写 `{"strategy":"minmax"}`，percentile 写 `{"strategy":"percentile","percentiles":[p]}`，p 取冻结规格，不使用库内缺省搜索列表。 |
| options | 可选对象；没有方法特定选项时可省略或 `{}`。 |

每个非空 SPEC 的七个字段全部必填：`format`（格式字符串）、`bits`（整数且与格式一致）、`granularity`（如 per_tensor/per_channel）、`axis`（非负整数或 null）、`group_size`（正整数或 null）、`symmetric`（布尔）、`scale`（minmax/none）。per_tensor 的 axis/group_size 均为 null；per_channel 权重的 axis=0、group_size=null。uniform_full 使用 INT 格式，uniform_affine 使用 UINT 格式；静态 uniform 网格的 SPEC.scale 仍为 `"minmax"`，percentile 策略放在对应 `*_calibration`，不能写进 SPEC.scale。具体位宽、符号、粒度和策略仍来自冻结定义与方法文档。schema 拒绝未声明额外键；可直接导入 `ptq_agent.execution.contracts.PLUGIN_PLAN_SCHEMA` 和 `ptq_agent.schemas.validate` 检查返回结构。

`site_catalog(backend,cfg)` 返回 `{site_id: info}` 字典。info 包含 `name`（模块名）、`kind`、`protected`、`semantic`、`edge`、`weight_shape`、`activation_shape`（可为 null）、`activation_channel_axis` 等元数据。`site_id(layer)` 对 linear/conv2d 返回 module，其余返回 `module + "::" + kind`；因此注意力 Q/K/V/概率条目的 module 是 attention 容器名，不能把完整带 `::kind` 的 ID 填入 module。目录说明哪些位点可用，不授权自动扩大冻结覆盖；保护、重复边、完整覆盖和实际执行仍由原校验器检查。

- `tools/ptq_agent/execution/contracts.py`、`plugin_contracts.py`：格式、计划、算子及校准合同。
- `ptq_agent.execution.sites.site_catalog(backend, cfg)`：真实位点及保护范围。
- `ptq_agent.execution.model.RTNExecutor`：统一 v2/v3 适配入口；v3 委托 `pipeline.QuantizationExecutor`，v2 继续使用既有实现。prepare/audit/export/load 是数值操作，没有 LLM 或选优循环。
- `ptq_agent.execution.model.RTNExecutor`：基础 RTN 计划及模型加载，保留数值计划 v2/v3 语义。
- `ptq_agent.execution.planning.expand_groups(...)`：只展开明确分组，不选择算法。
- `plugin_math.py`、`grids.py`、`calibration.py`、`adaround.py`、`gptaq.py`、`adalog.py`：具体数值方法；方法说明在 `docs/methods/`。

可选 `make_executor(backend, cfg, data)` 实现同一 prepare/audit/export/load 接口。源码随运行保存，重载仍校验模型、配置和权重身份。
`tools/profiling.py` 提供 15 个 V2 可请求的 Profiling 工具；纯工具会话在 `profiling/host.py`，完整注册在 `execution/profile_tools.py`。
其内部注册还保留计划展开、任务程序和证据索引等支持能力。V2 调用白名单没有扩大；旧 read_memory/compare_trials 等 MVP 工具已移除。

`tools/evaluating.py` 使用逐图 batch=1 的实际数据视图；subset 与完整数据评测明确分开。QDQ 延迟是参考实现开销，不等于低比特硬件加速。
`tools/segmentation.py` 保留既有 DeiT 冻结 Neck/UPerHead、150 类和 ADE 预处理/滑窗协议；设备沿用当前运行冻结的授权 GPU 池与 UUID 映射，不再固定物理 GPU 1/2/3。ADE 及混合评测继续按上述兼容路径使用单卡。
原 log2/log_sqrt2/AdaLog 参考网格可能包含额外下溢零；本次未改公式，不能仅凭方法名声称满足特定码本约束。

worker 从 `runtime/guards.py` 安装 Landlock 和梯度审计，只向当前 job 与 EXP 临时目录写入。training-free 允许正常只读 autograd.grad/backward 诊断，调用如实记录，不按调用文件放行；optimizer.step 仍禁止，浮点基准的权重、buffer、模式和梯度状态仍受保护。
生成代码的结构检查、真实执行检查和 Safeguard 审查都有明确覆盖边界，不能当作算法对任意输入正确的证明。

## 读取、修复与反馈边界

`present_card` 是投影，不等于整卡：`read_manifest.refs` 仅表示完整正文，`projected_refs` 表示省略工程封套/身份的短视图。展开省略字段后，续调保留实际工具结果；只对内容等价的整卡、字段或已交付页面去重。工程短视图同时保留队列 state 和有界的 result.status/reason/message，处理完成不表示实验通过。

Evidence 的完整客观投影直接交付，即使统计项多而超过原展示阈值，也不退回整份原始 payload 的强制分页。超过单页长度的原文字符串保留精确字段的分页入口，选择性查询不冒充完整交付；相邻大数组、状态和长集合的原件仍按需读取。全部命名指标、统计与来源入口保留，不按科学价值筛选字段。

Recorder 完全缺失或部分填写 scope 均由程序按引用证据补齐确定字段；冲突及不同证据条件不能自动改写。每条知识独立归一和校验，失败条目进入 pending_knowledge_updates 并带具体问题，不进入已验证知识。科学分析、有效知识和原始事实保留。repair_patch 仅替换必要顶层输出；tensor_probe 等参数镜像由宿主统一核对。

冻结 ExperimentCard 的 measurement_implementation 文件哈希清单以原字段引用和摘要哈希交付；Evidence 中与该冻结卡身份及 execution 完全相同的 resolved_execution 以精确引用交付，冻结实验仍为必需对象。观测数值、scope、执行身份、来源、数据选择和原始资产入口保留；不匹配的旧执行正文不折叠。长科研正文和工程身份原件按需读取；执行检查使用原始记录，不以模型阅读进度替代事实核验。

读取记录按原文哈希、分页内容、连续范围标记交付覆盖；读完根对象或祖先字段可满足子字段。缺页不冒充完整交付，也不阻断有效科研提交；来源身份与权限独立核验。TaskCard 续调新增版本时，仅同一逻辑任务且内容未变的固定 inputs/context 字段可沿用旧读取，不能将旧版整卡当作新版整卡。

同回合立场边界沿宿主持久化 parent_task 传给实验申请、Coding、执行及解释等子任务。资产权限取自宿主产出、程序 Evidence 与已绑定代码清单的可信归属；可见正文中的路径提及不授予文件权限，已发布数值任务的原始结果随其 Evidence 开放，须匹配生产任务 outputs、experiment_ref 及资产哈希；当前源码及其他分支不因路径提及而开放。

权限判断先排除无读取资格的 TaskCard，再检查同回合引用隔离；一次引用遍历只加载每个祖先一次。没有跨调用缓存或新的授权来源，实际读取仍校验卡片与资产哈希。

全局最新复盘仅提供可选入口。反馈绑定沿用 purpose 选定的处理对象；共享 Evidence 不会把另一方案的审查变成当前任务的必读全文。已经转为按需历史的较早交锋保留入口，明确的问题、条件和 rationale 原文继续交付；无法可靠区分是否解决时保守保留。归属不明的 Evidence-only 审查仍完整保留并标明原因。对应 judgment_audit、引用证据的知识及宿主分配的 feedback_refs/supervisor_feedback_refs，短字段直接交付，长字段提供优先读取入口；关联关系由明确引用决定，与长度及全库写入顺序无关。conditional 保留适用条件，不统一标成反证。L2 文本查询按原词法相关性排序，保留范围/版本/权限过滤与固定快照下稳定分页。

默认反馈字段与后续同版本整卡、祖先字段或完整分页覆盖等价时，正文只留在完整交付位置；默认入口保留来源、版本和位置，工具读取记录不删除。缺页、错误哈希或仅同 ID 的非等价结果不构成覆盖；展开的新字段必须出现在下一次实际模型请求中。

订阅返回先通过 CLI 完成状态、身份、工具隔离及 stdout/final 一致性检查，再校验模型最终输出。外层 JSON/Schema、内层 JSON/非对象、工具参数及角色动作 Schema 的已知错误，携带原始输出、错误阶段、字段位置和任务/执行身份，进入原任务现有共享的最多两次 repair_feedback；不另走传输重试。内层 JSON 错误保留 llm_inner_json_error，其他最终输出结构错误为 llm_structured_output_error。重复键、非有限数和非法转义仍严格拒绝，不补括号、不猜字段或更改科学内容。失败调用的可获得用量保存在 audit，并随这些已分类错误进入 worker 回执和预算结算。重新输出仍须通过全部科研合同；CLI 事件损坏、缺完成回执、鉴权、额度、身份、隔离和未知程序异常保持独立处理，不全局捕获解析或 Schema 异常。

Recorder 完整提交被合同拒绝后，同任务后续输出格式失败单独记录原文与定位，保留已有完整被拒草稿；重发合法 repair_patch 仍可合并该草稿，并再次通过全部合同。该连续性只属于原任务，不能从其他任务借用草稿；格式和科学合同修复在当前失败步骤共用两次连续额度；未进入新步骤时保留被拒草稿。

Recorder 的 recording_batch.source_refs 是新增整理覆盖集合，不是可读取依据的边界。当前批次正文已精确点名 claim_id/revision_ref 且直接引用对应旧 Metareview 时，只交付该 knowledge_updates 条目：短正文进入 memory.bound_feedback，长正文进入 required_reads；既有字段/祖先/完整分页覆盖可满足读取，不要求重复读取完整直接交付的正文。旧参考不加入本批 source_refs，其他历史仍按需。精确关联优先使用成对结构字段，无结构字段时兼容 claim_id=…、revision_ref=… 文本标注及外围句号、括号、引号/Markdown 反引号；ID 内部字符与显式版本不修补。Generation/Judger 等正常提交会核验明确点名的旧版本、claim 唯一性与本次调用权限，并将该版本保存为来源卡的直接 refs，Recorder 再次核对直接引用。不沿全部引用扩张，不凭相似文本或缺失版本推断，也不附送旧复盘其他无关正文。可见范围、权限、固定版本及科研 scope 校验保持不变。

代码 manifest 证明文件身份，不独立授予权限：code 的已提交产物绑定生产任务，不能由公开实验或其他正文路径覆盖私有生产者。无任务归属的兼容交接仅接受 Safeguard 的确切 candidate 与 Coding 的 previous_candidate；默认源码交付也经过同一资产权限校验。

必需 Brief 的完整正文在 14000 字符内直接交付。Evidence 大型列表、密集数组和模块表由 core/evidence.py 扫描全量叶子：保留单独命名的聚合指标，给出每种统计的样本数、描述性均值、负值/零值数、极值及精确 JSON pointer，附原字段哈希和读取入口；不取前 N 行代替整体，均值不冒充合并实验指标。模块索引用于同类指标分组，原位点与极值路径不丢失。类别预览明确说明是否完整，全部值仍可检索。小视图直接交付；冻结规格、科研正文与必需源码仍按原有完整交付/分页规则处理。

read_card(ref, select={site,token_type,metric,text,minimum,maximum}, field?, offset?, limit?) 可扫描原始 Evidence 全字段，按位点/token/指标通配或数值范围定位，返回带标签、精确路径与原字段哈希的叶子分页；默认 observations，limit 最大 50。过滤条件同时成立；取消 select 即按原接口完整分页读原件。检索不改原始结果，不扩大提交快照或 同回合立场权限。Hypothesis/Review 使用者自行判断指标意义与对照优劣。

请求构造/工具读取/提交使用一次宿主操作范围的 Journal 校验缓存；每次读取仍检查文件 inode/大小/mtime/ctime，变化立即重新校验，调用结束释放缓存。返回独立副本避免别名修改；恢复和 audit 完整核验。资产归属从已提交任务快照复用，真正授予权限前仍核验卡片及文件哈希；私有来源不能被可见路径提及洗白。目录先用已有元数据筛选、仅为结果页构造条目；无关键词不扫描巨型正文。队列 get 只复制指定任务。实验缓存也按原研究者在交接时的权限过滤，不能复用同回合对方私有测量。

knowledge_directory 提供有限的 L2 目录、知识状态和复盘时点，保留原条目与其支持证据，check_new_evidence 定位该判断之后的证据；出现新事实不自动推断旧知识失效。recording_status 显示待复盘缺口。source_refs 表示程序归档，evidence_refs/sources.ref 表示研究者实际采用的支持，二者不混同。


当前对象的证据交付使用同一规则：从已绑定 Hypothesis、Review、Experiment 的明确支持引用追加 Evidence 观测和各自冻结条件，覆盖筛查、交锋、选择、演化、plan_l3 及 Coding/Safeguard。不递归展开普通历史 refs；Evidence 的条件卡不会再次触发旧 trial 链。按提交快照、角色可见性及同回合隔离过滤，同一来源只交付一次；大型结果保持既有统计投影、覆盖说明和精确 read_card 展开入口。Recorder 继续使用增量批次，不能借交接规则重新注入全历史。

当前 H1 为成熟完整方案驱动，H2 为完整方案上的真实瓶颈改进，H3 保持开放创新；旧 existing_method/engineering/new_algorithm 卡片枚举保留。H1/H2 自然说明起点和改动，不增固定措辞或审核门槛。Top-3 独立于 Memory，按协议分桶持久化在 EXP/solutions，每轮完成后有效双任务按 (ADE20K mIoU, ImageNet Top-1) 降序更新；待测种子保留空成绩。完整源码、配置、协议和版本可按 configuration_asset 获取，失败与未入选发现仍由 Recorder/Memory 保存。统一 [W4A4协议](STATIC_W4A4.md) 只在最终完整方案检查全覆盖。
