# 本次任务：审查一份实验草案

审查 context.task.experiment_draft 中提供的 ExperimentCard 草案正文是否科学明确、可执行，并能对提出的问题提供有解释力的结果。草案尚未冻结，不虚构正式卡片 ID。

检查预测和观测量是否对应，比较对象与控制变量是否足以处理关键混杂；核对模型、数据划分、位宽、位点、训练选择及随机种子是否明确并符合 ResearchBrief。确认实际服务、参数和接口有依据，资源估计与当前预算相容。已有测量能够回答相同问题时，检查新执行是否有明确重复验证或不同条件的目的。

对照 context.data_receipt 区分可用数据与草案已选视图，核对来源及已校验/未知范围，不把池容量当作科学参数。耗时按 context.budget_receipt.research_remaining_seconds 判断，未结算估计与复盘/停机预留按回执口径解释。L3 的任务、split、count 和完成条件以 ResearchBrief.evaluation_protocol 为准，不附加固定双任务；缩减评测不能冒充协议要求的完整评测，批准也不代表获得 incumbent 资格。

需要 Coding 的实验，必须已有足够清楚的数学规格、输入输出、数值案例、容差与验证条件；后续实现仍需独立参考和 Safeguard。审批不是数值通过，也不是实验结果。

若草案含有阻塞性的缺陷，approved=false，并指出具体修正；只有无法从授权上下文确定的关键科学定义才使用 clarify。不得在审批时暗中改写草案、替换方法或补入会改变科学含义的默认值。L3 不设置 L2 性能阈值，仍遵守实际预算和运行配置。

提交 ReviewDecisionCard payload：
- approved 为 boolean。
- rationale 为非空字符串，说明科学目的、规格完整性、控制设计、能力与资源的检查结论；拒绝时列出具体阻塞问题。
- evidence_refs 为本次依据的EvidenceCard 短句柄或完整 ID 数组，可为 []；被审 ExperimentCard 不放在这里。
- uncertainties 为字符串数组，记录审批后仍需验证的内容。

不要通过 request_experiment 再次提交被审草案；本任务提交该草案的审批决定。批准只表示草案可以进入后续流程，不宣称已经执行或已经正确。
