# 科研实验接口速查

使用唯一 request_experiment(experiment)，或 plan_l3 直接提交同一草案。

必填科学正文：level、question、prediction、measurements、controls、decision_impact、execution、resources.estimated_seconds。controls 说明基准和保持不变的条件；estimated_seconds 是数值 worker 的时间上限，代码准备与验证也消耗总预算。对照 budget_receipt.research_remaining_seconds 规划可完成的实验。

execution 必填 service、seed。backend 继承 Brief 已授权适配器；training 仅在 Brief 有明确布尔约束时可省略，否则显式提供 execution.training。resolved_choices 是宿主内部镜像，可省略；已提供镜像发生冲突仍拒绝。CPU额度、唯一已授权 GPU、服务接口、未申请重复的标记、计划研究模型范围由程序补齐。方法、位宽、signed、粒度/轴、样本数仍由科学设计明确选择。独立重复须给 replication.requested=true、purpose、configuration_difference。

| service | 科学执行参数与能力 |
|---|---|
| profile（L1） | tool、arguments；接口参数见下表。查询结构不等于测得性能。 |
| tensor_probe（L1/L2） | site、method、parameters={bits,signed,granularity,axis,group_size}、calibration={count,offset?}；L2 另给 evaluation={split,count}。L1 只在校准输入测局部误差，L2 额外单点替换整网回放。 |
| quantize_evaluate（L2/L3） | calibration={count,offset?}、evaluations=[{kind,split,count}]、plan。现有方法直接使用数值计划；相同设置的 layer 可用 modules 数组代替单个 module，宿主逐位点展开。没有 plan 表示需新增 executor_v1 实现。 |
| evaluate_artifact（L2/L3） | model_dir、evaluations；实际加载已冻结模型包并评测。 |
| custom_experiment（L1/L2/L3） | 新方法/统计的明确 algorithm、parameters、数据视图和公共输出结构；自动走独立 Coding/Reference/数值验证/Safeguard。 |

calibration.count 必须为正整数或显式 null（从 offset 到末尾）。tensor_probe/quantize_evaluate 固定 calibration split。样本池容量不等于本次用量，见 data_receipt。L3 必须符合 Brief 的实际评测协议；不能用较小样本冒充完成。输出层级、任务性能、泛化及硬件加速分别判断。

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

内置服务无需包装代码。新量化器需 tensor_probe.requires_code=true；新代码 validation 必填数学规格、cases、atol、rtol。各 case 有 seed；executor_v1 另有 calibration_count/evaluation_count；experiment_v1 另有 context 及非空 interface_cases（seed/shape/sites/counts，容量覆盖冻结数据视图）。更详细的实现合同在 CODE_INTERFACES.md，位点计划及方法语义在 EXECUTION.md、methods/。不要为适配夹具改变冻结算法或真实模型条件。
