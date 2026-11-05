# 本次任务：完成当前阶段的选择

使用 context.task 中的 selection_stage、candidate_refs、eligible、matches 和 direct_results。parent 阶段仅依据 H1-H2、H1-H3、H2-H3 选定加强对象 H*；选择持久化后交给 Evolution。final 阶段只在 H4 与已选 H* 中确定最终候选，matches 仅含 H4–H*；parent_selection_ref 是已固定的初始选择。禁止将四场合并为四候选全局排名，不补做 H4 与其它初始方案的比赛，不采用 Elo 或自创加权总分。

已有明确直接胜负时复用其优胜者；H4 不合格时保留合格 H*。初始全不合格时选 null，跳过演化。未决时必须说明暂定选择理由和证据缺口。最终选择供 L3、Recorder 和下一轮反馈使用，不能用初始 H* 选择冒充最终结果。

依据已结案的直接胜负及同分候选的已知交锋关系选择候选。eligible 非空时，从 groups_by_direct_results 的第一组选择；该组含多个候选时，说明组内暂定选择依据及研究目标相关性。局部未决不能抹除其他已确定的胜负，也不能用“探索价值”把明确劣势候选写成当前优胜者；探索建议写入 uncertainties 或后续问题。

每场最终裁决一经结案即固定，包括 tie/undetermined，不提交改判请求或重开比赛。后到证据可用于解释科学局限、未决组内的暂定选择和 Recorder 知识更新，不能修改 matches 或据此更换已持久化的 H*。

提交 ReviewDecisionCard payload：
- selection_stage、selected_hypothesis_ref 和父选择身份由宿主绑定，无须复制；显式提交时不得与当前任务冲突。
- selected_slot：从 eligible 中选一个；只有 eligible 为空时才为 null。
- provisional：可选 boolean。省略时程序依据直接结果中的循环、并列或未决补齐；这些条件存在时不能明确写 false。暂定性不等于可以忽略确定胜负。
- selection_basis：可选字符串，需要单独说明选择依据时填写；省略时程序原样采用 rationale 交给 Evolution，不要求重复同一论证。
- rationale：非空字符串，说明直接结果、必要的局部比较、选择依据及结论范围。
- evidence_refs：EvidenceCard 短句柄或完整 ID 数组，可省略或为 []。
- uncertainties：可选字符串数组，说明排名仍受哪些问题影响。

当不存在循环、并列或未决时，必须选直接结果第一组中的候选。不得把本轮选择直接宣布为通过完整 L3 评测的 incumbent。
