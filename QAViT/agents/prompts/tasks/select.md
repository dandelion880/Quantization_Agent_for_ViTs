# 本次任务：综合证据，预选一项 L3 候选

比较 context.task.candidate_refs 中已经完成的 H1/H2/H3，结合各自实测 Evidence、实际条件、修订、失败及 Memory，选择最值得投入本轮正式 L3 的方案。通常三条假设均已完成；incomplete_slots 表示因局部失败未提交的槽位，不能虚构其结果，也不妨碍分析已有候选。

综合机制可信度、实验可比性、负结果、已知混杂、实现可行性和预算。L1/L2 是研究和预选依据，局部 NMSE、整网 KL 不代表 ImageNet Top-1 或 ADE20K mIoU。无需给出全序、语言辩论或让三条 H 全部做 L3；已有证据足够时直接选择。若缺一项能实质改变选择的测量，可自主 request_experiment，通过原 Coding→Safeguard→执行链取得 L1/L2 反馈后再决定。

提交 selected_slot（H1/H2/H3 中一个已提交槽位，或无可行方案时 null）和 rationale；以自然语言说明比较依据、重要反证、未解问题以及为何将预算投向该候选。实际引用测量时用 evidence_refs 保留可追溯来源；selection_basis、uncertainties 可按需补充。证据不足仍可保守选择 H1 基础方案并说明不确定性。选择不能伪称 L3 已完成或修改已有候选身份；需要实现适配的细节可在依据中交给后续 plan_l3 明确。

程序绑定完整候选版本、选择来源和“预选”状态，生成 ReviewDecisionCard；不计算旧胜负表，不新增审核。L3 计划由后续 Generation 任务提交。
