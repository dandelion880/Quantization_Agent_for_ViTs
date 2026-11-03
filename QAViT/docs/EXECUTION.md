# 运行代码与工具接口

## QAViT v2 科研协议

适用于 `schema_version=qavit.research/2`；这是当前唯一科研入口。数值计划的 schema_version=2/3 是模型包协议，与科研框架版本独立。

卡片合同位于 `core/cards.py`，版本为 `qavit.cards/2`。Agent 提交科研 payload；宿主补齐任务固定的 round/slot/type/phase/side、预测 ID、唯一绑定的前轮反馈引用，以及卡片 ID、版本、作者、来源和哈希。显式身份冲突仍拒绝。memory.source_handles 的短引用（如 C42）可用于结构化引用与 read_card；宿主按固定快照解析为完整 ID，不猜错别字或解析自由文本。允许完整数学推导、多条预测、来源和扩展字段，不限制科研正文长度。单次 CLI 的完整响应仍受传输保护限制，超限时拒绝不完整结果，不静默截断科研卡片。

ResearchBrief 分开保存用户硬约束、搜索空间、基准模型和评测协议。hard_constraints 的数值/字符串/布尔要求相等，列表表示允许集合，`{min,max}` 为数值范围，notes 保存自然语言。未指定参数不等于4位、某个固定粒度、32个校准样本或禁止训练。model.adapter 明确本次基准；允许研究不同模型时，用户可声明 model.allowed_adapters。不能在实验时偷偷替换基准。

必需执行约束统一为 `ResearchBrief.quantization_contract` 与冻结后的 `ExperimentCard.execution.quantization_contract`，由 `core/execution_contract.py` 独立核对。现有 `weight_bits/activation_bits`、`weight_spec/activation_spec`、`weight_granularity/activation_granularity`、`weight_axis/activation_axis`、`granularity.{weight,activation}`、`quantization_scope.{sites,kinds}` 和平铺 execution 字段可无歧义归并；同义字段冲突会被拒绝。`fixed_quantization` 中已支持的字段也进入合同；未知必需字段、自然语言 exceptions/parameter_contract 不再被静默忽略，须改为结构化字段或放入仅供科学审查的 notes。旧历史卡片不自动迁移。

| quantization_contract 字段 | 宿主核验内容 |
|---|---|
| required_sites、required_site_targets、required_kinds | 精确 site_id 数组、site_id→weight/activation 目标数组、明确 kind 数组；缺位点、未知/空展开、重复位点及必需目标留 FP 均拒绝。required_sites 未单列 targets 时，Linear/Conv2d 要求 W/A，其余位点要求 A。 |
| specs、methods、calibrations | 按 weight/activation 映射到规格字段、方法名、observer；规格包括 bits/format/granularity/axis/group_size/symmetric/scale。 |
| site_overrides | site_id→layer 的 W/A 规格、方法、observer 或 options；逐字段核对，observer 覆盖通用 observer。 |
| execution、preprocess、operators、unlisted_modules | 实际执行方式、计算精度、校准方式、阶段顺序和预处理/算子声明；必须与冻结选择一致。 |
| parameters、parameter_bindings | 冻结 parameters 与 execution.parameters 一致；完整模型服务的每个参数须绑定到实际计划。绑定项含 parameter、site、target、field、as_list；当前支持 observer 的 percentiles/scale_values/zero_values/factors。新增执行语义须扩展核验器，不能只自报相同参数。 |
| fp_exceptions | 对象：sites 为 site_id→目标数组，modules 为完整模块路径数组，parameters 为参数/buffer 完整名称数组；宿主核对计划及实际模型状态，不从文字猜例外。 |

这些字段不提供模型、覆盖数量、位宽或参数默认值。完整模型服务继承本次 Brief 的覆盖要求；局部 tensor_probe 只核验实际激活干预，其余研究目标单独保存在 research_quantization_contract。自定义统计不能凭研究目标声明已完成整网量化。宿主无法核验的必需语义须明确修订合同，不以放宽门禁继续。

HypothesisCard 必含研究问题、机制、干预、预测、实现、验证、sources、mechanism_relations、scope。预测包含 statement、measurement、falsification。H1/H2/H3 分别是 existing_method/engineering/new_algorithm；H4 为 evolved，另含 parent_refs 与 improvement_objective。H4 的 parent_refs 可省略，由程序固定为初始选择已绑定的 H* 完整卡片 ID（含版本）的单元素数组；恢复或同槽位后续版本不能替换它。Evolution 获得完整父卡片、选择理由、相关批评与证据，只加强 H*。后续 Generation 的 feedback_used 必须说明如何使用任务已绑定前轮反馈；只有一个绑定版本时 metareview_ref 可省略，change 仍须由模型说明。

ReviewDecisionCard 共同字段是 phase、rationale、evidence_refs、uncertainties。screening 还需布尔 eligible、非空字符串项数组 hard_rejection_reasons、非空字符串 similarity_analysis、字符串数组 evidence_gaps；拒绝时理由数组非空，合格时理由数组为空，空白字符不算理由。position 还需 side、arguments、objections、responses_to_prior、key_predictions；verdict 还需 outcome、decisive_reason、changed_judgments、next_questions。outcome 为 A/B/tie/undetermined/continue，第5回合不能 continue。新提交的 undetermined 须给 continuation_assessment：experiment_available、next_dispute_available 布尔值及非空 reason，判断当前是否还有可行且有区分力的实验/下一争议；声明可实验时先取证，不足5回合且可继续争议时不能提前未决。宿主只校验自述的一致性，不替裁决者判断科学区分力；原历史结案不重新验证或改判。程序把不合格配对记为 ineligible，不虚构辩论或胜负。

两阶段均复用 rank/Judger 与 ReviewDecisionCard：`rNNN/select_parent` 的 selection_stage=parent，仅用 H1-H2、H1-H3、H2-H3 选定并持久化 H*；`rNNN/rank` 的 selection_stage=final，仅从 H4 与 H* 中确定最终候选。卡片绑定 candidate_refs、selected_hypothesis_ref 的完整 ID，最终卡另绑定 parent_selection_ref。禁止将四场合并为四候选全局排名。只有 final 选择写入 Memory 的 round_selection，并交给 L3、Recorder 与下一轮反馈；两个选择均进入 Recorder 科研上下文，是否逐项分析由科研需要决定。

已有全胜者、仅余一个合格方案或没有合格方案时，程序直接引用已提交裁决/筛查完成现有 rank 任务，选择卡 author=program、selection_source=existing_verdicts_and_screening，不产生新的 LLM 判断或伪造测量。未决时仍由 Judger 给出 selected_slot、provisional=true 和非空 selection_basis；在当前阶段的直接结果第一组中选择，不能忽略明确胜负。H4 不合格保留合格父方案；初始全部不合格则两个选择均为空，跳过 Evolution 和 L3，正常 Recorder 与轮次收尾。

初始三场严格按 H1–H2 → H1–H3 → H2–H3 串行，前一场提交 match_closed 后才发布下一场任务；恢复时沿已有结案事件继续，不重复结案。之后仍只比较 H4–H*。每场最终裁决包括 tie/undetermined 一经结案即固定，拒绝非空改判请求，不再生成改判字段或修订卡片。旧记录保持可读，不改写历史。每对最多五次交锋；本场取证、数值验证及 Recorder 科研知识更新保留，后到证据不能改写已结案结果。

### 任务输入来源

| 任务 | 必需交接材料 | 历史读取与限制 |
|---|---|---|
| generate / evolve | ResearchBrief、分配槽位、与当前对象明确关联的问题；generate 另获任务已绑定前轮 Meta 的 generation_guidance/next_questions/recurring_failures/blind_spots；evolve 获固定 H*、选择理由、批评及证据 | 短必需正文直供，长必需正文在 required_reads；完整 Meta-review、L2 和非绑定正文按需读取 |
| screen / position / arbitrate / rank | 待审候选与阶段、当前/最近争点、新返回结果；position 指定一方，arbitrate 接收已汇合双方；rank 仅接收当前阶段结果 | 全部授权记录可分页发现和读取；对方本 exchange 未汇合立场不可读，结案不可改判 |
| plan_l3 | final 选择卡、最终候选完整 ID、选择依据、Brief 协议、数据回执与预算回执 | 原检索及按需读取；不拿初始 H* 代替最终候选，不重复申请同一 L3 |
| experiment_approval | 原草案、请求方、Brief、数据/预算回执；selected_data 来自此草案 | 原检索及按需读取；批准不改写草案，不虚构冻结卡片 |
| code / reference / safeguard | 冻结 ExperimentCard；code 为候选实现，reference 为独立规格，safeguard 获得两份代码及数值验证 | Reference 仅能读授权规格/Brief 原件及明确项目资料；不能读候选、共享科研目录或数据回执 |
| interpret / record | 实测 Evidence 与对应定义；record 交付未综合的假设、实测证据、裁决/选择/解释与工程失败摘要 | 旧知识按需读取；长必需正文须分页读完；结果解释不覆盖测量，知识更新不改比赛裁决 |

具备 request_experiment 权限的任务均获得 data_receipt（包括规划/审批），其它任务按 purpose 交付绑定材料。回执派生自有效后端配置、明确 overrides、数据清单及本次运行匹配的 Evidence 数据身份，无独立可编辑配置。available_data 标示池容量与半开区间 `[0,N)`，selected_data 仅转述草案/冻结执行视图；整个视图为 null 表示未声明，可用容量为 null 表示未知，并列 unknowns。视图内显式 count=null 保留原接口的完整/剩余范围语义。清单 SHA、配置数量、已有校验的范围分别说明；不加载模型、不重读全部图像，也不把目录/清单检查当作运行时验证。配置/清单路径用于溯源，不因此扩大 read_asset 的读取权限。

budget_receipt 来自一次 BudgetLedger 快照：remaining_seconds 为硬截止余额，research_remaining_seconds 已扣 closeout_reserve_seconds 与 shutdown_reserve_seconds；available_seconds 保留为前者的兼容字段。未结算 estimated_seconds 按类型汇总，已占用调用额度及已结算数量可见。现有 check 不再次扣未结算估计，LLM 的估计为超时上限；输入展示不新建额度或预留规则。实际启动仍执行原预算、磁盘与资源检查。read_asset 限本次授权来源、reference、QAViT/docs，以及 code/reference/safeguard 的明确公共数值接口白名单；不自动开放其它 EXP、配置或源码目录。

### ExperimentCard 与真实服务

唯一公开入口仍是 request_experiment；plan_l3.submit 使用同一草案 Schema 和预检，不新增 profile/evaluate 等平行工具。内部冻结 payload 仍完整包含 level、question、prediction、measurements、controls、decision_impact、execution、resources、replication、scope、validation。Agent 负责问题、预测/判别标准、科学选择及 estimated_seconds/GPU 需求；宿主补卡片身份、资产哈希、确定的服务接口和运行资源字段。resources.cpu_threads 省略使用当前运行配置；resources.gpu 省略且实际 server 后端为 cuda/cuda:0 时，冻结程序仅在当前运行 allowed_gpus 恰有一个设备时补全，保留原授权检查与 GPU 资源锁；不唯一时返回明确错误，不猜科学参数或覆盖显式 GPU；replication 省略表示未申请独立重复（requested=false），申请重复仍须完整目的。无新增代码时 validation 可省略为空对象；新增代码不得省略数学与接口验证规格。显式冲突仍拒绝，不覆盖科学选择。

execution 明确 service、整数 seed 与科学执行参数；resolved_choices 可省略，由程序从这些参数生成一致性镜像。backend 省略时使用 Brief 明确的适配器；training 仅在 Brief.hard_constraints.training 为明确布尔值时可省略，否则须选择。tensor_probe 的 activation_bits 可由明确 parameters.bits 生成镜像；两处显式冲突仍拒绝。quantize_evaluate 未提供现有 plan 时，以及 custom_experiment 的 requires_code=true 与接口可推导；tensor_probe 新量化器仍须显式 requires_code=true。位宽、signed、量化范围、粒度/轴、校准规模和方法不得猜测。training=true 允许冻结规格定义的训练，但不得改变 FP 基准；已有 training-free 验证不代表 QAT 已验证。配置、模型、数据及实现文件仍以哈希绑定，数据视图不改旧模型包身份。

校准以 `execution.calibration` 为执行视图的权威来源，`resolved_choices.calibration_samples` 保留为内部一致性镜像。草案可只声明 `calibration.count`；宿主生成镜像。兼容只声明正整数 `calibration_samples` 的草案，机械补齐缺失的 count；两者显式冲突时拒绝并指出字段和值。count 必须是正整数或显式 null（沿用数据视图从 offset 到 split 末尾的语义），缺失不再隐式代表完整视图。`tensor_probe` 和 `quantize_evaluate` 的 worker 固定读取 calibration split，故可补齐缺失 split；显式其它 split 会被拒绝。自定义实验未固定 split，须明确声明。不会从后端样本池、评测协议样本上限、自然语言或全局默认配置猜测本次科学参数；已有扩展字段保留。

`plan_l3` 完成和 `request_experiment` 入队前，与冻结入口共用纯参数预检（卡片合同、执行选择、校准视图及服务层级）。预检不读取模型/数据资产、不冻结卡片、不启动实验或占用实验预算。可修复错误进入原任务现有 `repair_feedback`，附错误、原始响应（含草案或工具批次）及回执路径，最多修复两次；不确定的规格使用 clarify。工具批次先全部预检，避免修复时重复入队前半批实验。冻结时仍核验资产、证据绑定和当前预算；预检通过不代表实际数据、数值实现或实验验证通过。

| service | 配置与含义 |
|---|---|
| profile | L1；提供 tool、arguments，复用 `tools/profiling.py` 的 PROFILE_NAMES；保留已验证工具，移除旧调用/测量计数门禁。 |
| tensor_probe | L1/L2；提供 site、method、parameters、calibration、evaluation。真实捕获位点并测输入/局部输出误差；L2 额外执行单边替换的整网小样本回放，不冒充全模型量化精度。 |
| evaluate_artifact | L2/L3；明确 model_dir，加载冻结量化包后运行 evaluations，不导入旧指标。 |
| quantize_evaluate | L2/L3；现有算法交 execution.plan；新增实现走 Coding/Reference/Safeguard。共用独立覆盖审计、导出重载和评测。 |
| custom_experiment | L1/L2/L3；开发新的统计或算法，必须经过冻结规格、独立数值对照及三类 Safeguard 审查。 |

`read_profile_evidence` 的 `arguments.evidence` 使用先前成功测量返回的 `evidence/...json` 引用。宿主从本次运行已提交的 EvidenceCard 冻结原始测量和工具回执的哈希绑定，在新作业恢复成功登记后分页读取，不重新测量；跨运行、未提交、越界、损坏或模型身份不匹配的证据不获授权。

服务参数、Profiling 常用接口和现有方法目录见 [EXPERIMENT_INPUTS.md](EXPERIMENT_INPUTS.md)，作为 context.experiment_interface 直接提供给有实验权限的角色，无需反复读取整本文档。

新增代码的数学案例、独立实现、真实数据接口、接口夹具和结果验证见 [CODE_INTERFACES.md](CODE_INTERFACES.md)。工程角色直接接收该接口正文。

### Evidence 与四层 Memory

EvidenceCard 只由程序从真实工作进程回执构建，保存数值、配置、模型状态、代码、数据划分、种子、资源和原始结果引用。measurement_source 区分 real、synthetic_computation、test_double；随机模型真实运算不代表真实任务精度。科研解释保存在 Judger 判断卡片中，不能覆盖测量。

提交判断卡的原 Judger 通过工具主动申请的实验，在执行或有效缓存返回后直接完成请求，由等待该请求的原任务继续比较预测与实测、解释判断变化，并按原合同提交。此路径不再新建独立 interpret 任务；事实仍在 EvidenceCard，解释仍在原任务的 ReviewDecisionCard。 执行失败时请求回执同时保留失败子任务的真实错误，原请求者无需再追读不可见的底层任务编号。所有原判断及请求链由程序归档，Recorder 可按需分析具体 position，无需逐项重复审计。Generation/Evolution 的科学审批与独立 Judger 解释、plan_l3 的辅助实验、无等待请求者的 L3 解释保持；plan_l3 本身只提交草案，不能代替判断卡。已发布的 interpret 任务继续完成，不在恢复时取消或重复创建。新增代码的 Coding、独立 Reference、数值验证与 Safeguard、失败处理保持。

新 Evidence.scope 使用 `qavit.evidence_scope/1`：由冻结模型身份及宿主 execution_facts 生成 model/model_identity_sha256、measurement_kind、measurement_source、实际 site/quantization/位宽、parameters 和 data。完整模型数据范围包含导出包实际校准 ID 和本次评测 ID；局部激活探针不因研究目标 W4A4 而获得 weight_bits=4。自定义统计仅记录实际读取和前向访问、冻结参数，不把无法独立核验的量化目标提升为已执行事实。申请 scope 与 resolved_choices 仅存 provenance.research_target。

Recorder 实证知识必须保留测量的模型、位点、位宽、量化规格、参数和数据边界；多元素列表不得包含未测对象，省略限制同样拒绝。历史缺少规范范围的证据继续可读，不能据其自填标签新增实证知识。精确检索共用同一 scope，使用完整字段路径，不再用后缀匹配到 research_target；词法排序和四层结构不变。

内置实验指纹覆盖实际 execution、L1/L2/L3 层级及计算资源，不以科学问题或控制说明的不同措辞区分相同计算。插件可读取完整正文，因而将其全部实际输入也纳入指纹。相同有效测量可复用；明确重复验证产生新的 execution_id，缓存命中不能计为重复验证。方法与假设的科学重复性由结构信息、来源、Memory 和 Judger 审查共同判断。

L0 为不可变卡片/提交；L1 实时状态、近期证据、选择与 incumbent；L2 为有状态、范围和出处的知识；L3 为每轮 Meta-review。Recorder 维护科研判断，程序检查来源并保护原始事实。knowledge_updates 可为空；每项科研输入是 text、status、evidence_refs、rationale。新 claim_id 可由程序生成；所引 Evidence 的范围完全相同时 scope 可精确复制，否则须显式声明并核验，不能扩大测量范围。持久化项仍包含完整 claim_id、text、status、scope、evidence_refs、rationale；status 为 hypothesis/observed_once/replicated/refuted/conditional。replicated 需要两个不同执行与 replication_analysis；知识更新还需 supersedes 指向上一 revision_ref。judgment_audit 为可选的重点判断分析，条目含 judgment_ref、assessment、evidence_refs、reasoning，不要求穷举。Recorder 应解释预判与实测差异、指标价值、失败机制和下一轮问题。

L3 失败、取消或拒绝也进入 Recorder；`task.l3_outcome` 默认交付请求/执行状态、子任务状态与失败原因、原文入口、有效证据及缺口；完整工程结果和长日志按需读取。报告的 `status=completed`、`workflow_completed=true` 仅表示流程完成。`formal_validation_success` 另行核验本轮冻结规格、执行指纹、协议及原始资产；已完成请求没有有效 L3 证据仍为 false。报告总值要求流程完成、至少一项 L3 申请且所有申请验证成功；各轮保留独立结果。该标志不表示假设正确、达到性能目标或完成了规格以外的评测。

Recorder 只需提交 scientific_analysis、generation_guidance、next_questions；可补充有证据支持的知识、关键判断分析、指标适用性、失败模式和盲点。程序绑定全部 source_refs、recording_batch_id 与 analysis_scope；来源覆盖表示纳入批次综合，不宣称每项经过科研认可。新提交不使用 reviewed_refs。

incumbent 独立保留现有同模型、同协议、完整分类及分割证据和指标比较门槛；这些资格条件不倒推为所有 L3 的必做任务。Brief 的 formal_incumbent_allowed=false 明确禁止晋升；仅分类或子集协议可以完成本次 L3，不能因此宣称通过 incumbent 检查。

Memory 的唯一事实来源仍为 Journal 和不可变卡片；目录、输入和四层状态都是可重建投影。参考 Letta MemFS 的简短常驻目录和按需读取，不引入文件系统记忆库、向量数据库、新智能体或额外 LLM 筛选。MemFS 提供设计参考，QAViT 的提交事实与科学综合状态分别维护；参考记录留在 EXP。

按 purpose 交付明确处理对象，正文不会同时出现在 task、memory、tool_results 和 messages：

| purpose | 默认必需材料 |
|---|---|
| generate | ResearchBrief、本轮类型/要求、简短目录、明确关联或分配的反馈；全局复盘及 L2 按需读 |
| screen、position、arbitrate、rank、interpret、plan_l3 | 本次待审方案/比较对象、当前或最近争点、明确引用的判断/测量、本任务新返回的实验与对应冻结定义；其他场次和旧正文按需读 |
| evolve | 唯一 H* 完整方案放在 task.parent_hypothesis，一处提供；选择依据、明确批评与绑定证据；不再次默认注入完整最新 Meta-review |
| code、safeguard | 冻结 ExperimentCard、当前实现/修订材料、验证回执；代码文件沿用 source_sha256 身份 |
| reference | 仅本任务冻结规格、ResearchBrief 和项目接口资料；与 safeguard 共用角色也不共享候选源码、反馈、数值结果或科研目录 |
| record | 本批假设、Evidence/冻结定义、裁决/选择/解释；工程计数与失败原因；完整来源索引、立场等材料按需读 |
| supervise | 当前任务、流程状态、预算和异常反馈，历史科研正文按需读 |

短必需正文直接提供；超过展示阈值的必需卡片/字段/源码在 `memory.required_reads` 或 `required_assets` 列出精确引用、字段、哈希、长度和 next_offset。它们是必读材料，不能作为可选链接跳过；提交前宿主检查完整读取进度。页面在无状态续调中保留一次。读取记录只证明程序交付过文本，不能证明 LLM 已理解。

`retrieve_memory` 返回 `entries/offset/total/next_offset` 目录，**不再返回全量 L2 或选中卡片正文**。条目含标题、类型、版本、scope、scope_basis 与 scope_verified（区分已核验测量和声明/历史范围；长范围为精确字段入口）、来源引用与读卡入口。`query.browse=true` 浏览所有授权科研记录，`kind` 过滤类型，`layer=L2` 定位知识当前修订及 scope，`text` 保持原词法排序，其余字段共用规范 scope 精确匹配。`after_version` 为可选增量界限，limit 为 1–50。无需每次调用先检索；索引不是已读正文，相关性也不是证据支持。

`read_card(ref, field?, offset?, limit?)`：field 是 JSON pointer（如 `/payload/observations`）；短卡默认返回完整卡片，长卡或指定 field/分页参数返回 canonical JSON 文本页面和 sha256/next_offset。`read_asset` 按字符分页读取 UTF-8 文本。两者单页最多 12000 字符，可用现有 tool_calls 批次同时读多卡、多字段、多页。query 必须是非空对象，text/ID/过滤值类型与非空校验保持；已知输入错误继续返回原任务并沿用有限修复，未知程序异常不吞掉。

订阅传输保留原 `name/arguments_json` 封套；多个独立工具用 `name="__tool_batch__"`，arguments_json 解码为 `{"calls":[{"name":"read_card","arguments":{"ref":"完整ID","offset":0}},...]}`。宿主解码为原有 tool_calls 列表，所有参数/实验草案先预检，再各自入队；运行期某项无权限/失败不吞掉同批其他结果。不得夹带决策、嵌套批次或同批依赖；工具预算为零时批次也不开放。格式错误共用原任务最多两次修复；CLI 完成/身份/隔离检查仍在之前，整批调用的已知失败用量正常结算。

每次模型调用固定 `read_view.sequence`；目录、提醒、搜索、读卡和资产权限都先按这个提交快照过滤。工具批次沿用发起调用的快照；下一次调用（含实验回传）再刷新。Generation/Judger/Evolution 可发现此前提交的 ExperimentCard、EvidenceCard、公开论证/interpretation 及文献来源，不等待 Recorder。资产访问仅允许同一快照中明确关联的文件及固定项目资料，并绑定哈希；不能读取运行目录下任意日志/新文件绕过隔离。同回合对方立场、其私有子任务与间接引用均隔离。

Recorder 复用轮末及预算收尾任务。程序从已提交的 recorder_batch_completed.source_refs 重建已综合集合，批次按完整对象版本冻结，迟到产出不会被跳过。来源绑定、索引、游标、去重和完整原件均由程序维护；模型不填写覆盖清单。Meta-review、Memory 投影、批次完成事件与 TaskCard 原子发布，存储失败不推进游标；recording_submissions 只为已验证待发布草稿提供恢复入口。

事实归档与科研复盘分别报告：recording_status.facts=archived_in_journal；scientific_review=pending/completed。复盘的 completed 表示批次科研综合已发布，不是逐条认可。Recorder 的有限修复耗尽、取消或请求澄清会记录 recorder_deferred，保留未综合来源，不伪造 Meta/L2。正常轮次可继续；下一轮 Generation 明确收到复盘缺口与前轮假设/证据/选择，下一批 Recorder 自动合并旧来源。同一失败批次不在收尾无限重试，有新来源时才形成新批次。可以通过现有显式重试入口处理留下的任务。

报告同时提供 workflow_completed、scientific_review_complete、formal_validation_success，三者语义独立。未完成复盘不会单独被误报成存储损坏；真实持久化错误仍标 recovery_required。硬预算耗尽或中断时保存 recorder_pending/pending_recording，不额外启动 LLM，不扩大原截止时间/调用额度。源码身份和历史恢复约束不变。

## 数值计划与修改入口

`executor_v1.build_plan(backend, cfg, experiment_payload, parameters)` 返回普通数值计划：
`schema_version=3, unlisted_modules="keep_fp", execution, preprocess, layers, unresolved`。
`execution` 明确 mode、compute_dtype、activation_calibration、order，以及需要时的 within_stage；每个 layer 分别声明 W/A 规格和方法。
科学选择来自冻结 ExperimentCard/ResearchBrief，不从旧实验继承固定位宽、覆盖或质量目标。

候选和独立参考返回 plan 后，宿主先做完整 JSON Schema、未解决项和冻结字段核对，再做设备能力、实际捕获、prepare、audit；执行阶段重复同一核验。数学输出偶然相同不能补救缺字段、删位点或错误计划。

以下字段直接对应 `contracts.SPEC`、`plugin_contracts.plan_schema/site_id`，是接口编码说明，不增加科研默认值。`layers` 是条目列表，不是以位点为键的字典；v3 不使用单一 `method` 字段。顶层 `preprocess`、`unresolved` 必须是列表；无已声明变换或未决项时分别为 `[]`。`evaluation`、`operators` 为可选字段，没有声明需求可省略。

| layer 字段 | 合同 |
|---|---|
| module、kind | 必填字符串；module 为真实模块名，kind 与 site_catalog 条目一致。Linear/Conv2d 使用 `linear`/`conv2d`；注意力纯激活使用 `attention_query`、`attention_key`、`attention_value`、`post_softmax` 等实际目录 kind。 |
| weight、activation | 两者均必填；值为 SPEC 对象或 null。纯激活位点 weight 必须为 null。 |
| weight_method、activation_method | 两者均必填；目标非空时为已注册方法名，目标为空时为 null。分别与各目标规格对应。 |
| reason、evidence | 必填；reason 为字符串，evidence 为至少一个字符串的列表，引用实际冻结依据/文档，不把方法依据伪装为测量。 |
| weight_calibration、activation_calibration | 可选的独立 observer 对象；目标为 null 时禁止提供对应字段。`strategy` 必填；minmax 可写 `{"strategy":"minmax"}`，percentile 写 `{"strategy":"percentile","percentiles":[p]}`，p 取冻结规格，不使用库内缺省搜索列表。 |
| options | 可选对象；没有方法特定选项时可省略或 `{}`。 |

每个非空 SPEC 的七个字段全部必填：`format`（格式字符串）、`bits`（整数且与格式一致）、`granularity`（如 per_tensor/per_channel）、`axis`（非负整数或 null）、`group_size`（正整数或 null）、`symmetric`（布尔）、`scale`（minmax/none）。per_tensor 的 axis/group_size 均为 null；per_channel 权重的 axis=0、group_size=null。uniform_full 使用 INT 格式，uniform_affine 使用 UINT 格式；静态 uniform 网格的 SPEC.scale 仍为 `"minmax"`，percentile 策略放在对应 `*_calibration`，不能写进 SPEC.scale。具体位宽、符号、粒度和策略仍来自冻结定义与方法文档。schema 拒绝未声明额外键；可直接导入 `ptq_agent.execution.contracts.PLUGIN_PLAN_SCHEMA` 和 `ptq_agent.schemas.validate` 检查返回结构。

`site_catalog(backend,cfg)` 返回 `{site_id: info}` 字典。info 包含 `name`（模块名）、`kind`、`protected`、`semantic`、`edge`、`weight_shape`、`activation_shape`（可为 null）、`activation_channel_axis` 等元数据。`site_id(layer)` 对 linear/conv2d 返回 module，其余返回 `module + "::" + kind`；因此注意力 Q/K/V/概率条目的 module 是 attention 容器名，不能把完整带 `::kind` 的 ID 填入 module。目录说明哪些位点可用，不授权自动扩大冻结覆盖；保护、重复边、完整覆盖和实际执行仍由原校验器检查。

- `tools/ptq_agent/execution/contracts.py`、`plugin_contracts.py`：格式、计划、算子及校准合同。
- `ptq_agent.execution.sites.site_catalog(backend, cfg)`：真实位点及保护范围。
- `ptq_agent.execution.pipeline.QuantizationExecutor`：prepare/audit/export/load；是数值执行器，没有 LLM 或选优循环。
- `ptq_agent.execution.model.RTNExecutor`：基础 RTN 计划及模型加载，保留数值计划 v2/v3 语义。
- `ptq_agent.execution.planning.expand_groups(...)`：只展开明确分组，不选择算法。
- `plugin_math.py`、`grids.py`、`calibration.py`、`adaround.py`、`gptaq.py`、`adalog.py`：具体数值方法；方法说明在 `docs/methods/`。

可选 `make_executor(backend, cfg, data)` 实现同一 prepare/audit/export/load 接口。源码随运行保存，重载仍校验模型、配置和权重身份。
`tools/profiling.py` 提供 15 个 V2 可请求的 Profiling 工具；纯工具会话在 `profiling/host.py`，完整注册在 `execution/profile_tools.py`。
其内部注册还保留计划展开、任务程序和证据索引等支持能力。V2 调用白名单没有扩大；旧 read_memory/compare_trials 等 MVP 工具已移除。

`tools/evaluating.py` 使用逐图 batch=1 的实际数据视图；subset 与完整数据评测明确分开。QDQ 延迟是参考实现开销，不等于低比特硬件加速。
`tools/segmentation.py` 保留既有 DeiT 冻结 Neck/UPerHead、150 类和 ADE 预处理/滑窗协议；现有物理 GPU 1/2/3 限制未更改。
原 log2/log_sqrt2/AdaLog 参考网格可能包含额外下溢零；本次未改公式，不能仅凭方法名声称满足特定码本约束。

worker 从 `runtime/guards.py` 安装 Landlock 和梯度审计，只向当前 job 与 EXP 临时目录写入。
生成代码的结构检查、有限数值案例和 Safeguard 审查都有明确覆盖边界，不能当作算法对任意输入正确的证明。

## 读取、修复与反馈边界

`present_card` 是投影，不等于整卡：`read_manifest.refs` 仅表示完整正文，`projected_refs` 表示省略工程封套/身份的短视图。展开省略字段后，续调保留实际工具结果；只对内容等价的整卡、字段或已交付页面去重。工程短视图同时保留队列 state 和有界的 result.status/reason/message，处理完成不表示实验通过。

Evidence 的短正文阈值按呈现后的 payload 计算；卡片 ID、时间戳等封套不会使本可直接交付的数值与 scope 退回整份原始 payload 分页。全部呈现字段仍交付，科学长正文门禁不变。

Recorder 草案的知识 scope 可在所引 Evidence 范围完全相同时省略，由宿主精确复制；冻结 MetareviewCard 仍含完整 scope 并接受原严格校验。显式错误范围不覆盖修正，不同配置的对照不自动合并范围；可分列知识或保留在分析/反馈。repair_patch 只替换顶层字段，数组须完整替换，错误路径不进入可恢复草稿。tensor_probe 中与 parameters 一致的 granularity 字符串镜像按唯一激活目标规范化；其它服务的目标歧义仍拒绝。

冻结 ExperimentCard 的 measurement_implementation 文件哈希清单以原字段引用和摘要哈希交付；Evidence 中与该冻结卡身份及 execution 完全相同的 resolved_execution 以精确引用交付，冻结实验仍为必需对象。观测数值、scope、执行身份、来源、数据选择和原始资产入口保留；不匹配的旧执行正文不折叠。长科研正文仍须完整读取，工程身份原件按需读取，提交与执行门禁不使用投影替代原件。

必读覆盖核验原文哈希、分页内容、连续范围；读完根对象或祖先字段可满足子字段。缺页或版本/哈希不匹配不能放行。TaskCard 续调新增版本时，仅同一逻辑任务且内容未变的固定 inputs/context 字段可沿用旧读取，不能将旧版整卡当作新版整卡。

同回合立场边界沿宿主持久化 parent_task 传给审批、Coding、执行及解释等子任务；Reference 另受更严格限制。资产权限取自宿主产出、程序 Evidence 与已绑定代码清单的可信归属；可见正文中的路径提及不授予文件权限，已发布数值任务的原始结果随其 Evidence 开放，须匹配生产任务 outputs、experiment_ref 及资产哈希；候选源码、独立参考及其他分支不因路径提及而开放。

权限判断先排除无读取资格的 TaskCard，再检查同回合引用隔离；一次引用遍历只加载每个祖先一次。没有跨调用缓存或新的授权来源，实际读取仍校验卡片与资产哈希。

全局最新复盘仅提供可选入口。反馈绑定沿用 purpose 选定的处理对象；共享 Evidence 不会把另一方案的审查变成当前任务的必读全文。已经转为按需历史的较早交锋保留入口，明确的问题、条件和 rationale 原文继续交付；无法可靠区分是否解决时保守保留。归属不明的 Evidence-only 审查仍完整保留并标明原因。对应 judgment_audit、引用证据的知识及宿主分配的 feedback_refs/supervisor_feedback_refs，短字段直接交付，长字段必读；是否必读由明确关系决定，与长度及全库写入顺序无关。conditional 保留适用条件，不统一标成反证。L2 文本查询按原词法相关性排序，保留范围/版本/权限过滤与固定快照下稳定分页。

默认反馈字段与后续同版本整卡、祖先字段或完整分页覆盖等价时，正文只留在完整交付位置；默认入口保留来源、版本和位置，工具读取记录不删除。缺页、错误哈希或仅同 ID 的非等价结果不构成覆盖；展开的新字段必须出现在下一次实际模型请求中。

订阅返回先通过 CLI 完成状态、身份、工具隔离及 stdout/final 一致性检查，再校验模型最终输出。外层 JSON/Schema、内层 JSON/非对象、工具参数及角色动作 Schema 的已知错误，携带原始输出、错误阶段、字段位置和任务/执行身份，进入原任务现有共享的最多两次 repair_feedback；不另走传输重试。内层 JSON 错误保留 llm_inner_json_error，其他最终输出结构错误为 llm_structured_output_error。重复键、非有限数和非法转义仍严格拒绝，不补括号、不猜字段或更改科学内容。失败调用的可获得用量保存在 audit，并随这些已分类错误进入 worker 回执和预算结算。重新输出仍须通过全部科研合同；CLI 事件损坏、缺完成回执、鉴权、额度、身份、隔离和未知程序异常保持独立处理，不全局捕获解析或 Schema 异常。

Recorder 完整提交被合同拒绝后，同任务后续输出格式失败单独记录原文与定位，保留已有完整被拒草稿；重发合法 repair_patch 仍可合并该草稿，并再次通过全部合同。该连续性只属于原任务，不能从其他任务借用草稿；格式和科学合同修复共用两次额度。

Recorder 的 recording_batch.source_refs 是新增整理覆盖集合，不是可读取依据的边界。当前批次正文已精确点名 claim_id/revision_ref 且直接引用对应旧 Metareview 时，只交付该 knowledge_updates 条目：短正文进入 memory.bound_feedback，长正文进入 required_reads；既有字段/祖先/完整分页覆盖可满足读取，不要求重复读取完整直接交付的正文。旧参考不加入本批 source_refs，其他历史仍按需。精确关联优先使用成对结构字段，无结构字段时兼容 claim_id=…、revision_ref=… 文本标注及外围句号、括号、引号/Markdown 反引号；ID 内部字符与显式版本不修补。Generation/Judger 等正常提交会核验明确点名的旧版本、claim 唯一性与本次调用权限，并将该版本保存为来源卡的直接 refs，Recorder 再次核对直接引用。不沿全部引用扩张，不凭相似文本或缺失版本推断，也不附送旧复盘其他无关正文。可见范围、权限、固定版本及科研 scope 校验保持不变。

代码 manifest 证明文件身份，不独立授予权限：code/reference 的已提交产物绑定生产任务，不能由公开实验或其他正文路径覆盖私有生产者。无任务归属的兼容交接仅接受 Safeguard 的确切 candidate/reference 与 Coding 的 previous_candidate；默认源码交付也经过同一资产权限校验。

必需 Brief 的完整正文在 14000 字符内直接交付；Evidence 的投影视图通常在单页容量 12000 字符内直接交付。observations 中超过 7000 字符的纯数值数组或仅由这些数组组成的状态字典，用精确 read_card 字段入口和哈希交付，原件完整保留，按科研需要展开；逐项命名标量统计、含文字的表行和 scope 不作科学摘要。超过 1000 字符且完全相同的重复字典/列表保留首次正文，后续复用该字段的入口和哈希；所有不同统计值仍保留。含上述数组或重复结构投影的完整统计视图在 192000 字符内直接交付，避免退回对数百万参数或重复计数的强制分页。超出对应容量仍须完整读取。完整 ExperimentCard、Recorder 本批 Hypothesis/Review 及已授权的必需源码在 48000 字符内直接交付；更大正文沿用完整分页读取。Experiment 的 research_quantization_contract 仅在与其直接引用的当前 Brief 合同完全相同时替换为精确字段入口和哈希；实际 quantization_contract、不同合同及不可变原文保留。
