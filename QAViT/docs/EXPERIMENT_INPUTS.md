# 科研实验接口速查

Generation/Evolution 可在 H 正式提交前直接 request_experiment，多次实验后继续分析，也可不实验直接形成 H。程序检查后执行，结果回原任务，无常规 Judger 审批/解释。可选 research_note 保存当前认识、缺口和下一步判据；失败/预算拒绝不等于测量。需要新代码时使用 Coding 单实现与 Safeguard 科学审查。

使用唯一 request_experiment(experiment)，或 plan_l3 直接提交同一草案。所有研究角色均直接接收有效 Evidence 或失败反馈；已提交的最终 L3 计划需要修订时，程序创建同角色新计划任务并保留旧版本。无法继续完成规定协议时可仅提交 defer_reason，不伪称验证完成。

最小科学请求只需 question 与 measurements；controls 表达必要科学约束。prediction、decision_impact 可省略。程序保存原始需求、补齐可确定的模型/数据身份、资源、默认值与参数镜像；未知科学参数由研究者或 Coding 按职责决定，不编造预测或测量。

已有服务直接选择 execution.service 与必要参数；不额外安排 Coding 或模型审批。需要新代码时 execution 可省略，程序按 custom_experiment 提供上下文，Coding 完成单份实现及必要 execution_choices 后再绑定完整可执行记录。独立重复需要 replication.requested=true 和科学目的；缓存不算独立重复。

| service | 科学执行参数与能力 |
|---|---|
| profile（L1） | tool、arguments；接口参数见下表。查询结构不等于测得性能。 |
| tensor_probe（L1/L2） | site、method、parameters={bits,signed,granularity,axis,group_size}、calibration={count,offset?}；L2 另给 evaluation={split,count}。L1 只在校准输入测局部误差，L2 额外单点替换整网回放。 |
| quantize_evaluate（L2/L3） | calibration={count,offset?}、evaluations=[{kind,split,count}]、plan。现有方法直接使用数值计划；相同设置的 layer 可用 modules 数组代替单个 module，宿主逐位点展开。没有 plan 表示需新增 executor_v1 实现。 |
| evaluate_artifact（L2/L3） | model_dir、evaluations；实际加载标准 plan v2/package v1 或 plan v3/package v2 模型包并评测；不通用加载自定义包。 |
| custom_experiment（L1/L2/L3） | 新方法/统计的目的、所需结果和科学约束；Coding 完成单份源码与必要执行选择，再由 Safeguard 对照原始需求审查。 |

仅实际需要校准时声明 calibration；无图像的权重统计、能力查询和纯浮点对照不要求虚构校准。已声明的 calibration.count 必须为正整数或显式 null（从 offset 到末尾）。tensor_probe/quantize_evaluate 固定 calibration split。样本池容量不等于本次用量，见 data_receipt。L3 必须符合 Brief 的实际评测协议；不能用较小样本冒充完成。L1/L2 的局部干预、消融和浮点对照不继承最终候选的全网覆盖/位宽要求，整网评估不改变实际干预范围。自定义实现可使用新方法名、observer 和参数，无须给内置计划 parameter_bindings；实际选择内置执行器时仍受其能力约束。输出层级、任务性能、泛化及硬件加速分别判断。

| profile.tool | arguments |
|---|---|
| query_structure / query_execution_capabilities / query_ptq_plugins | {} |
| query_ptq_sites | prefix、offset、limit(1–32)，可选 include_outside_scope |
| profile_structure | modules([]为全部)、count、seed、offset |
| profile_distribution | modules、target(weight/activation)、views(tensor/token/channel数组)、tokens(all/cls/special/patch)、statistics(range/mean/std/rms/quantiles/zero_fraction数组)、histogram_bins(0–64)、top_k(1–8)、count、seed、offset |
| profile_hardware | modules、count、seed、offset、warmup(1–10)、repeats(1–20)、refresh |
| profile_sensitivity | modules、variants（name、method=rtn、weight/activation规格或null）、count、seed、offset |
| profile_plugin_sites | sites（module/kind）、count、seed、offset |
| profile_ptq_candidates | count、seed、offset；plans/grouped_plans/trial_id三选一；trial_id另需mode=refit_targets、mutable_targets、variants |
| profile_trial_diagnostics / profile_trial_ablation | trial_id、count、seed、offset；ablation另需variants（site/target）；execution.trial_evidence_refs绑定本运行成功search证据 |
| read_profile_evidence | evidence、section、offset、limit(1–32)，可选path数组 |
| query_method_library / read_method_document | 优先直接 read_asset(method_catalog 中的文档)，不用为读取已有文档申请数值实验 |

内置服务无需包装代码。新量化器选择 tensor_probe.requires_code=true；不再要求 validation、数学案例、interface_cases 或双实现比较容差。能力查询和位点目录不消耗校准样本，不受无关校准数量约束；实际校准和测量必须绑定并核验规定条件。代码接口见 CODE_INTERFACES.md，位点计划及方法语义见 EXECUTION.md、methods/。

内置计划先规范化唯一确定的表示，再校验完整数值合同：格式对应 bits，per_tensor 的空 axis/group_size，禁用目标的空 method，以及格式自身唯一规定的 scale 等可省略；冲突仍报具体字段。方法、非唯一量化选择和未知轴由科研角色决定；需要真实 shape 的检查在正常准备捕获后完成，不要求额外发起实验。自定义 make_executor 使用自己的表示，不能据内置枚举拒绝新算法。

资源由宿主按本运行允许池分配；新输入 allowed_gpus=auto 使用实际可见与获分配设备的交集，也可集中显式指定编号/UUID。无需猜选固定卡，默认一实验一卡。resources.gpu 可固定主卡；gpu_count 可选并按能力/空闲卡尽力启用，目前仅 server 原生非训练冻结包的纯分类评测分片，其余沿单卡。必需时延、显存指标或不兼容结果协议保留原单卡测量，不能删指标、减样本或降低量化覆盖来启用多卡。卡忙正常排队，不消耗尝试。启用 exclude_numeric_wait 的运行中，estimated_seconds 仍是单任务超时，不因超过智能体余额拒绝实验；重叠的推理/Coding/审核正常计时。实际范围和回退原因由回执记录。
