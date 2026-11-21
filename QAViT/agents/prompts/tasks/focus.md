# 本次任务：恢复后作出本轮共同研究方向决定

这是 H1/H2/H3 启动前共用的一次 Generation 决定，不是第四候选、裁判或另一角色。先使用 context.research_topic 恢复上次问题、范围、目标、决定依据、未完成动作和复议条件；结合 topic_catalog、method_catalog_state、inspiration_continuation、Memory、用户输入和证据，按需 read_topic/read_method/read_inspiration/read_card/read_asset 定位准确原件。首次没有主题时依据 ResearchBrief 建立初始主题；旧运行没有主题是未知历史，不补造过去决定。代码、配置和结果按确切版本恢复，目录/摘要不是已读原件的证明。

主题是“研究问题＋可修改范围”。提交 question、mutable_dimensions、held_conditions，说明正在解释什么、可改变哪些维度及暂时保持哪些条件；依据模型绑定、已有全局或跨层证据，选少量能共同回答问题的自由度。用 held_conditions 指明继承的已验证配置及来源。若只能定位到传播路径，先开放该路径上的解释，不凭单点误差锁层；若列出很多独立旋钮，说明哪几个当前值得同时研究，其余可留待复议。全网共享构造规则或必要跨层联动也可以是一个明确范围。范围只是当前聚焦，不限制必要的全局诊断、对照、历史完整方案或创新，不放宽 Brief 用户硬约束、量化和评价协议。

依据本次可用证据及不确定性，提交 decision：establish/continue/narrow/expand/pause/resume/switch；恢复已有暂停主题提供 topic_id；切换至新主题省略 topic_id，或提供已有暂停主题 ID。通常一次提交完成共同决定；如果只暂停而未选择下一方向，三候选暂不启动，宿主返回共同决策任务供选择后续方向。旧候选的 L3 不因此取消。

提交 objective（例如提高指标、机制解释、必要性检验、参数共享或方法简化）、rationale、具体 next_step 和 reconsider_when。明确哪些新证据、测量/修复完成或条件变化会使判断改变；不写“以后再看”。继续须有可检验的下一步价值，不能仅因投入而延续；无提升不自动换题，有提升也可改为解释、消融或简化，不要求每轮新机制或指标上升。区分工程失败、未测、机制反证、正式性能下降；解释支持/复杂度降低和正式性能提升分别报告。

可选 unresolved_questions、source_refs、inspiration_ids、method_ids。actions 中的稳定 action ID 可通过 completed_action_ids / deferred_action_ids / resumed_action_ids 更新；恢复暂存动作时沿用原 ID。next_action_id 指向当前要接续的 open 动作（可同次显式 resumed），仅更新其复议条件。next_step 是叙述，不据文本相同、相近或改写自动创建/合并动作；只有新主题首次建立或显式 new_action 才创建新动作。仅改状态时省略 new_action，省略动作字段保留原动作。切换时旧成果和动作保存；若旧主题复议条件发生变化，用 previous_reconsider_when 补充。仅提交变化所需信息，不复制全部历史。Recorder 的 topic_notes 是有条件的建议，不是主题决定。

主题不等于灵感、方法或候选版本；切换主题不会结束灵感或改变算法身份。H3 仍沿一条 active 灵感投入；确需暂停或切换灵感时用 update_inspiration 明确决定，无须为选题重建同一机制。三候选共享本轮主题版本，H1/H2 仍分别改良完整父方案，结论和机制可以不同。全局 B 不受主题限制，三个最终候选仍进入既定 L3。

本任务只恢复与决定方向，不开展新数值实验。缺少证据但仍能提出可证伪下一步时保存未知并交后续研究，不为选题追加固定诊断或逐动作审批。返回完整主题决定正文；宿主维护 ID、版本、归属、持久化和原预算。
