# 本次任务：综合候选证据并修订最终版本

横向比较 context.task.candidate_refs 中已提交的 H1/H2/H3，结合各自 Evidence、条件、修订、失败及 Memory，明确各候选的机制、重要反证、混杂与未解问题。保留 H1/H2 完整方案改良、H3 开放创新及独立 B；槽位不是质量等级。incomplete_slots 表示未提交的槽位，不得虚构其结果。

三条最终候选各自按当前冻结协议进入既定 L3，不按语言排名、成熟度或 L1/L2 代理指标淘汰其中两条。局部 NMSE、整网 KL 不代表 ImageNet Top-1 或 ADE20K mIoU。保留科学综合与必要取证：缺少能实质改变机制定义或修订判断的测量时，可 request_experiment 经原 Coding→Safeguard→执行链获取反馈；已有证据足够时直接提交，不设最低实验次数。

提交 rationale，以自然语言说明横向比较、修订判断、重要反证及各候选剩余不确定性；引用测量时使用 evidence_refs，selection_basis、uncertainties 可按需补充。程序固定 selected_slots 与 selected_hypothesis_refs 为全部已提交候选及其确切版本；无需选择唯一 selected_slot。旧 selected_slot 只兼容历史偏好，不控制 L3 名额。B 独立保存，不占候选槽位或要求重复验证。

改变科学干预、适用位点或机制定义时，先用 revise_candidate 提交 parent_ref、完整新正文及理由；程序发布不可变新版本并结束旧综合任务，再启动绑定所有当前版本的新综合。旧证据只作为修订依据，不能自动验证新版本。无需修订时保留原版本；工程适配交给对应 plan_l3/Coding。修订工具中断后宿主恢复原请求和原版本，不为恢复另发科学改动。

程序生成 ReviewDecisionCard，绑定完整候选版本、综合来源和待验证状态；不新增审核。后续 Generation 为各候选分别提交 L3 计划。失败、延期或未提交候选按实际状态保留，个别失败不阻断其余候选，也不能将部分完成描述为全轮验证完成。

H3 的修订沿同一 inspiration_id 延续，不要求换机制；暂停/切换只改变后续科研投入，不能取消本轮已提交最终候选的 L3。低分或未入库的成果仍供下一轮恢复。

横向综合各候选实际 topic_ref 与 method_id；轮中修订的影响范围见 topic_scope_changes。共同初始主题不要求相同父方案或结论。必要主题修订先 revise_topic 明确受影响槽位；已提交版本不回写，若改变候选再 revise_candidate。暂停主题不淘汰其本轮最终候选。

综合服务或输出修复额度耗尽时，宿主保存待综合的确切候选与待做 L3，不据失败发布科学结论；恢复仍沿同一任务、候选版本和原预算。
