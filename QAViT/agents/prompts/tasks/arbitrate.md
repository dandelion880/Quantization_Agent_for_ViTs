# 本次任务：中立裁决当前配对

阅读当前配对双方已汇合的立场、相关假设、既往交锋和实际证据。以 ResearchBrief 目标及同一比较口径判断，不继承任何一方的辩护立场。

memory.cards 交付当前绑定对象和本场新返回结果的正文或必读入口；已公开的前序实验、统计和解释由 memory.directory/retrieve_memory 分页发现，再用 read_card/read_asset 读取。当前明确关联的问题和适用条件由 bound_feedback 或 required_reads 交付，不能只读目录就作证据判断。

先确定决定性争议，再判断现有证据是否足够。有必要且当前可完成的取证可以先请求；工具调用次数不等于交锋 exchange 次数。

outcome 只能为：
- A 或 B：当前目标与比较范围下，一方具有可说明的决定性优势。
- tie：已有可比较依据，在当前目标下没有决定性优劣；不能表述为已经证明统计等效。
- undetermined：关键证据缺失、条件不可比较或冲突未解决，目前不能可靠裁决。
- continue：仍有一个具体、可推进且可能改变裁决的争议，值得下一次交锋；不能仅为重复已有论点而继续。

可提前结束。exchange 达到 5 时必须从 A、B、tie、undetermined 中选择，不得 continue。ineligible 由资格审查和程序处理，本任务不能输出该值。

选择 undetermined 前明确评估是否仍有当前预算、权限与实现能力下可执行且有区分力的实验，或可推进的下一交锋。有实验路径先请求取证；不足五回合且有具体争议则 continue。无合理路径可提前未决结案，不为耗满回合重复讨论。第5回合可记录仍有后续争议，但不能开启第6回合。

最终裁决一经结案即固定，包括 tie/undetermined；后到证据可用于后续科研分析，不能改判。初始三场依次结案后才开始下一场，不对前场发起赛后复核。

提交 ReviewDecisionCard payload：
- outcome 使用上述合法值。
- undetermined 另需 continuation_assessment 对象：experiment_available、next_dispute_available 为布尔值，reason 说明路径可行性/区分力或无法继续的具体原因；不能声明路径可行却直接提前未决结案。
- decisive_reason 为非空字符串：胜负的决定性依据，或平局、未决、继续的具体原因。
- changed_judgments 为对象数组，每项含 before、after、reason、evidence_refs；说明本次证据或论证改变了什么科研判断，无改变时为 []，不能改写已结案比赛。其 evidence_refs 只放真实 EvidenceCard ID，并包含于顶层证据列表。
- next_questions 为字符串数组，写剩余问题；continue 时至少一项说明下一交锋具体要解决什么及什么结果会改变裁决。
- rationale 为非空字符串，概括比较目标、关键争议及证据如何支持本次结论。
- evidence_refs 为EvidenceCard 短句柄或完整 ID 数组，可为 []。
- uncertainties 为字符串数组，可为 []。

不要把双方共同符合的结果当成一方独有优势，也不要因缺少证据而默认 H3/H4 获胜或失败。
