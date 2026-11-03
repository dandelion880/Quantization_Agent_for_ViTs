# 本次任务：形成一个 H4

## 输入与工作

阅读 context.task.parent_hypothesis 指定的 H*、parent_selection_ref 对应选择、selection_basis、critique_refs、evidence_refs 及可用 Meta-review；相关完整卡片在上下文或按引用读取。区分确定问题与未解决争议，保留 parent_provisional，不假设前三场都得出了胜负；H* 已持久化，本任务不能更换。

选择一个明确的 improvement_objective，说明它由哪项审查、辩论或证据引出。可以修复、简化、组合、处理混杂或探索新机制；不要求改善已经被实验验证。明确相对父方案的改变及由此产生的新预测。若组合多个组件，设计能识别各组件贡献和相互作用的验证。

提交一个 H4，不改写父假设，不把历史实测数值归于尚未执行的新方案。

## 提交正文

提交 HypothesisCard 的 payload，包含：

- title、research_question、mechanism：非空字符串，分别给出标题、问题和包含前提及替代解释的机制说明。
- intervention、implementation、validation_design：各为对象或字符串，分别明确算法改动、实现路径、可区分解释的验证方案与决策影响。
- predictions：非空数组，每项含非空字符串 statement、measurement、falsification；id 可省略由程序编号，说明相对父方案哪些预测改变、哪些仍保留。
- sources：非空对象数组，说明实际来源、核实状态和支撑内容。
- mechanism_relations：数组，每项含非空字符串 relation、target、explanation；说明继承、修改、互补或替代关系。
- feedback_used：对象数组；使用 Meta-review 时写其完整 metareview_ref 和具体 change，没有可用反馈时为 []。
- scope：对象，说明适用条件、未检验范围和与父方案范围的区别。
- improvement_objective：非空字符串，说明要解决的具体问题、预期改善及判断改善是否成立的办法。

H4 的改进仍未测量时明确写为预测。完整定义与验证路径已经具备即可提交，不为取得支持结果而无限延长本任务。
