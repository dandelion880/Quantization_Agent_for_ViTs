# 本次任务：审查候选、参考与数值记录

读取 context.frozen_experiment、candidate_source、independent_reference_source 和 context.task.numerical，以及授权的相关记录。检查规格身份、候选与参考源码身份、实际数值验证及实验控制是否对应同一次冻结定义。

分别判断：
1. fidelity：关键公式、算法步骤、接口、参数和状态行为是否忠实于冻结规格。
2. numerical_correctness：宿主记录的检查是否实际通过，误差与容差如何，相关边界和退化情形覆盖到什么程度；不能把未测试情形写为通过。
3. experimental_validity：模型、数据视图、校准/评测隔离、种子、训练选择、控制变量及量化覆盖是否允许计划中的解释。

context.task.numerical.status 不是 passed 时，approved 必须为 false。即使数值通过，只要另有未解决的强制检查失败，也不得批准。对实际错误给出具体定位和影响；可在 uncertainties 记录有限测试无法消除的风险，但不能用一般性不确定性无理由否决合格实现。

提交 ReviewDecisionCard payload：
- approved 为 boolean。
- fidelity、numerical_correctness、experimental_validity：分别为非空字符串，写检查依据、结论、发现的问题和覆盖边界。
- rationale：非空字符串，说明批准或拒绝的决定性理由。
- evidence_refs：引用的EvidenceCard 短句柄或完整 ID 数组，可为 []；源码清单、数值回执或 ExperimentCard 不能冒充 EvidenceCard，其身份在分析正文中说明。
- uncertainties：字符串数组，记录真实未解决事项。

冻结定义存在无法确定的实质歧义时返回 clarify，而不是自行修改算法或降低验证要求。本次批准仅覆盖当前冻结规格和所绑定源码。

context.code_interface 已直接提供公共接口与验证约定；需要具体方法实现时再按 public_interface_assets 读取。
