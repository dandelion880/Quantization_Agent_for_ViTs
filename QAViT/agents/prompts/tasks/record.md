# 本次任务：提炼研究发现与下一轮指导

程序直接交付本批新增假说与变化、实验目的与核心结果、关键裁决与争点、相关旧知识，以及影响科研判断的失败信息。统计投影是描述性事实，不是科学结论。完整条件、指标和工程原件按所附 read_card/read_asset 入口展开；当前明确关联的旧知识在 bound_feedback。先分析影响机制和后续选择的证据，只有缺口影响结论时再深读；不要遍历全部历史来完成表格。

区分 parent_selection_ref 绑定的 H* 和 selection_ref 绑定的父子最终选择；未测的 H4、预算拒绝或实现超时不能当成科学反证。保留矛盾、混杂和未完成阶段，不重新排名已结案比赛。

必须提交 scientific_analysis；其他栏目按科学需要提供：
- scientific_analysis：非空字符串，总结关键发现、矛盾证据、机制解释、适用条件和限制。
- generation_guidance：可选字符串，具体说明下一轮保留/改变/探索什么，以及能改变判断的最小对照。
- next_questions：可选字符串数组，只增加本批新问题；省略或空数组不清空旧未决问题。明确解决或放弃时分别用可选 resolved_questions / abandoned_questions 字符串数组给出原问题文本；无需重抄其余历史问题或申请关闭审批。

有助于复用时再提供：
- knowledge_updates：text、status、evidence_refs、rationale；新 claim_id 由程序生成，scope 可省略或只表达科学条件；程序从真实引用证据补齐确定的身份和来源，不能确定的科学范围保持未知。status 为 hypothesis/observed_once/replicated/refuted/conditional；replicated 要有不同独立执行和 replication_analysis。可以保存不同参数或方法的比较认识，程序逐份绑定 Evidence 的真实条件，不取第一份 scope 代替全部来源；在 rationale 解释比较差异、混杂和限制，对照实验不等于独立重复验证。修订已有或 pending 知识沿用 claim_id，supersedes 指向当前原稿版本；明确放弃 pending 时可用 status=abandoned，无需为此补做实验。当前待办仅显示仍待处理条目，旧稿可追溯。
- judgment_audit：只分析有价值的判断；每项 judgment_ref、assessment、evidence_refs、reasoning。可引用一整张 HypothesisCard 或 ReviewDecisionCard，在 reasoning 中说明预测编号。不要求覆盖清单。
- predictive_metrics：说明代理指标是否有可比证据支持预测能力，包含 metric、assessment、evidence_refs、reasoning；无数据可省略。
- recurring_failures、blind_spots：字符串数组。单次失败不要写成反复规律。
- evidence_refs：分析使用的测量短句柄或完整 ID；各结构分析项中的证据由程序自动汇总，不必重复列出。

无需提交 reviewed_refs、批次覆盖、哈希、游标或来源封套；旧 reviewed_refs 兼容忽略，既不代表已审事实，也不导致有效复盘被拒。程序自动绑定本批 source_refs，但不会把未逐项分析的来源标成已审。支撑已验证知识的来源必须 completed 且不是 test_double；科学分析和判断审计可以按实际身份讨论失败、模拟回执及计划。原始回执始终保留。附加审计或分析条目有问题时也逐条保留原稿和问题，不拒绝其余有效科学分析，不把问题条目发布为已核实支持。一条知识更新有问题时程序保留分析及其余有效更新，将该条与具体问题存为待核实；待核实条目不发布为已验证知识，不阻塞研究，也不会自动触发额外修正文书调用。需要修正已拒正文时可用 repair_patch 替换顶层科学字段，数组整体替换；不使用嵌套路径。required_reads 是按需深读入口，不是提交前必读清单。
