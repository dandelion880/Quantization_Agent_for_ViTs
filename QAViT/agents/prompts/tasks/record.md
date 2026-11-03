# 本次任务：提炼研究发现与下一轮指导

直接交付本批假设、结案裁决、两阶段选择、测量及相应实验规格。memory.engineering 给实际状态和失败入口；其他公开立场、筛查、全部工程原件可从 recording_batch.source_index 或 memory.directory 按需读取。当前明确关联的旧知识在 bound_feedback。先分析影响机制和后续选择的证据，只有缺口影响结论时再深读；不要遍历全部历史来完成表格。

区分 parent_selection_ref 绑定的 H* 和 selection_ref 绑定的父子最终选择；未测的 H4、预算拒绝或实现超时不能当成科学反证。保留矛盾、混杂和未完成阶段，不重新排名已结案比赛。

必须提交三个科学字段：
- scientific_analysis：非空字符串，总结关键发现、矛盾证据、机制解释、适用条件和限制。
- generation_guidance：非空字符串，具体说明下一轮保留/改变/探索什么，以及能改变判断的最小对照。
- next_questions：字符串数组，按科研价值组织未解决问题，可以为空。

有助于复用时再提供：
- knowledge_updates：text、status、evidence_refs、rationale；新 claim_id 由程序生成，单条或相同范围 Evidence 的 scope 可省略。无实测 hypothesis 须声明 scope。status 为 hypothesis/observed_once/replicated/refuted/conditional；replicated 要有不同独立执行和 replication_analysis。不同范围不自动合并。修订已有知识须给原 claim_id 和 supersedes 的当前版本。
- judgment_audit：只分析有价值的判断；每项 judgment_ref、assessment、evidence_refs、reasoning。可引用一整张 HypothesisCard 或 ReviewDecisionCard，在 reasoning 中说明预测编号。不要求覆盖清单。
- predictive_metrics：说明代理指标是否有可比证据支持预测能力，包含 metric、assessment、evidence_refs、reasoning；无数据可省略。
- recurring_failures、blind_spots：字符串数组。单次失败不要写成反复规律。
- evidence_refs：分析使用的测量短句柄或完整 ID；各结构分析项中的证据由程序自动汇总，不必重复列出。

不提交 reviewed_refs、批次覆盖、哈希、游标或来源封套。程序自动绑定本批 source_refs，但不会把未逐项分析的来源标成已审。所有实证来源必须 completed 且不是 test_double；失败/计划按实际身份讨论。原始回执始终保留。需要修正已拒正文时可用 repair_patch 替换顶层科学字段，数组整体替换；不使用嵌套路径。
