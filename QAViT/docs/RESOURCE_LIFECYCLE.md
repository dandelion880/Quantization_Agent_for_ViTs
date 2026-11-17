# 资源调用、来源与沉淀

分类与唯一维护位置见 [knowledge/INDEX.md](../knowledge/INDEX.md)。这里说明真实调用链，不设额外阶段、引用要求或提交门禁。

## 按需知识与常驻上下文

`runtime/driver.py:request_body()` 调用 `agents/interaction.py:context()`，再由 `agents/roles.py:role_spec()` 拼接 common、当前角色和任务片段、提交视图与工具 Schema。ResearchBrief、当前任务、预算、必需短证据/反馈进入上下文；长正文给分页入口。角色不会自动读取全部 Prompt 或全部历史。

所有角色只获得 `context.knowledge_index.read_asset` 这一导航指针。Generation/Recorder 额外获得 Skill 短 INDEX；允许 request_experiment 的任务获得 method_catalog 与 EXPERIMENT_INPUTS 正文；generate/select/code/safeguard/record/plan_l3 获得 complete_solutions；Coding/Safeguard 获得 CODE_INTERFACES 正文。其余资源自主读取，目录不是阅读证明。

`authorized_assets()` 从 docs/ 与 knowledge/ 发现文本、合入 Brief 明确列出的 reference/ 来源和当前可见任务产物；`request_body()` 冻结路径/哈希及 Journal 序号。read_asset 使用同一授权快照，核验当前字节并返回 UTF-8 字符页、sha256、offset/next_offset。旧 skills/、methods/、solutions/ 别名不变，knowledge/ 别名与项目相对路径均可用。新增目录没有开放任意文件或同回合私有子任务。

续调沿现有工具结果保留阅读内容，重复正文压缩；research_note 之前的旧结果可转成短入口，必要时重读。导航/Skill/方法发现目录不可用时可继续原科研路径，read_asset 失败回到原任务。数值执行所需的配置、插件合同和会话快照仍须真实有效，不能借“可选知识”跳过数值核验。

## 宿主读取与同次请求去重

Journal 的 `read_session` 只在一次宿主操作内复用已校验原件：每次新操作重新核验，操作内文件 stat 改变也重新解析/校验；返回对象始终独立。`card(ref, field=(...))` 只复制所需字段。ACL 祖先每张卡片一次已验证读取，Task/Review 保留原判断需要的正文，其它卡片只取引用头；授权仍按原序号、父链、首个产物 owner、manifest 和文件哈希判断。它不是持久内容缓存或权限缓存。

队列仍由 Journal 的 task_snapshot 事件增量派生；一批新事件只复制各任务最终版本，完整历史事件/TaskCard不改写。默认 `snapshot()` 保留完整独立快照；仅需部分信息的宿主调用可使用 `snapshot(states=..., kinds=..., fields=...)`、`get_by_key` 或 `held_resources`，先选取再复制。任务提交、恢复、预算与调度语义不因字段投影改变。

Evidence 统计仍遍历全部条目并保留哈希、覆盖说明、极值与反例读取入口；只为判断显示大小时，编码达到阈值便可停止。按原角色交付时，同一请求已完整提供且内容/顺序完全相同的长 sample_ids/sample_selection 副本，可替换为原字段的 read_card、sha256 和 body_location.same_value_as；首份完整列表仍在本次正文或完整工具结果中。样本数、split/full_dataset、科研条件、负面观测和其它列表不因此省略。指针不证明原件已全部读完，也不增加必读要求。

## 方法、方案和跨运行范围

read_asset 是方法正文的普通入口，不触发实验。`MethodLibrary` 在数值 ProfileHost 初始化时冻结 methods/index.json 与方法全文到该会话 method_library.json；query_method_library/read_method_document 仍可用于旧数值调用及复现，查询并非额外 LLM，文档也不全量注入角色。快照与插件覆盖核验保持；`loop.require_method_documents` 是无执行效果的兼容字段。无调用方的历史 require_plan_documents 阅读检查已移除。

方法组件、完整方案种子、协议实测 Top-3 的边界见 [solutions/README.md](solutions/README.md)。跨运行自动共享目前仅限 `SolutionLibrary` 的同协议方案桶：研究运行在 EXP/solutions/<protocol_hash>/top3.json 保存最新项及 archive；非研究运行隔离在自己的 solutions/。新运行初始化读取桶并写入自己的 Journal 快照，已有运行依其快照读取，更新机会由轮末有效 L3 结果触发。它不是跨运行经验数据库。

运行内 `ResearchMemory` 从本运行 Journal 派生 L0 卡片事实、L1 进展/incumbent/未决问题、L2 有来源和 scope 的知识修订、L3 Metareview。`MemoryRetriever` 先按角色/序号/范围过滤，再按词法相关性排序；不扫描其它 EXP 运行，不做向量检索或自动合并。普通检索附带的 method_documents 只是静态文档入口，不是历史记忆。旧运行经验若需用于新研究，目前由维护者核验后明确整理到新 Brief、静态知识或可复用方案；不存在自动历史经验导入流水线。

## 测量、解释与来源

request_experiment → 原始 ExperimentCard → 必要时 Coding 单实现 → 宿主绑定 → Safeguard → 数值 worker → 程序 EvidenceCard → 原研究任务续调。已有内置服务沿其直接执行路径。Evidence 保留真实条件、执行身份、原始指标和产物；失败也是执行事实，不自动构成科学反证。Recorder 只综合，不生成测量。

程序归档来源和计数；Recorder 用 Metareview 提交科学分析及可选知识更新。L2 修订绑定可见来源/范围，不同条件不自动合并；待综合材料继续 pending。运行内 incumbent 与协议 Top-3 分别维护，前者是本研究状态，后者是完整方案复用库。Skill/方法正文、README、索引不会被运行角色自动改写。

可选来源示例（无需凑齐全部类型）：

```json
{"sources": [
  {"document": "methods/gptaq.md", "status": "已读项目说明，未核实论文全文", "supports": "机制定义"},
  {"ref": "C12", "supports": "本次观测；句柄须来自当前上下文"},
  {"source": "本卡理论推导", "status": "待验证"}
]}
```

`sources.ref` 仍按 Journal 可见性解析；`evidence_refs` 仍仅能引用真实 EvidenceCard。为兼容旧输出，只有 sources 中可解析为**当前授权静态文本**的 ref 路径会改存 document；保留 status/supports 等内容，读过哪些字节及其哈希由原 read_asset 回执追溯。显式 document 是作者的文档来源声明，不自动证明已读，也不赋予新权限。未知卡片、错误或私有 ref 仍沿既有可选条目问题记录处理，其余有效正文继续提交。文档不会加入 scientific_refs 或测量支持链。

旧提交中的 submission_issues 与不可变卡片不补写；兼容修正只作用于新提交。恢复仍执行原源码身份与显式干预规则；不能把静态知识编辑伪装成未改变源码的续跑。

## 同包双任务评测

冻结实验只含 classification 与 ade_frozen 两项、server 后端且无训练时，
未显式给出 gpu_count 的资源请求最多取两张空闲授权卡；显式 gpu_count=1 保持串行。
两项各自沿用原有单卡评测，母进程先完成一次 prepare/export/load 和 parity，
ADE 子进程从同一不可变包独立 load；ready/start 握手完成身份核验后，
分类在主卡与 ADE 在副卡同时启动原有评测。load 使用禁止读 batch 的 FrozenData，
不会再次校准或分片，握手等待同样受数值超时与进程清理管理。
自定义 make_executor.load 在子进程按原 Safeguard 批准源码导入，执行包文件、
plan、FP 和 Q 全状态身份校验与原有审计；只有实现显式设置 parallel_frozen_load=True
才启用独立重载并行；未声明时保留串行。不支持的后端/服务和卡不足均在结果中记录串行原因。

子进程继承 GPU lease，进程身份写入作业 processes，受同一 TaskRuntime 的
取消、数值超时与恢复清理管理；不额外预留或叠加子进程预算。失败保留已经完成
的任务原件和 dual_task_failure.json，整次实验仍失败。dual_task_receipt.json
记录同包 hash、设备和真实评测时间交集，实际并行以该回执而非配置判定。

## 事后知识更正与完整方案归档

运行内知识仍由Recorder综合。明确授权的维护可调用 `ResearchMemory.append_correction`，仅为既有claim追加有证据、scope与supersedes绑定的Metareview；author=program，标记maintenance_correction/operator/reason/model_called=false，不冒充模型复盘或正式测量。同一幂等键不能换正文。原卡片/提交不改，派生checkpoint推进，原正式summary/status不重写。L2返回当前版本；历史复盘短视图附knowledge_revisions，详细历史仍按需read_card。

SolutionLibrary的Top3仍只用于保留与B排序。available/catalog及冻结加载从本运行已提交的library/archive/considered事件发现完整历史方案；首次新运行可冻结同协议共享archive，后续读取不直接扫描共享桶以免泄露未来结果。排名外有效方案与未测种子分别标注，历史成绩不冒充新测量；configuration_asset与批准源码保持可追溯。


## 独立灵感库与跨轮接续

`InspirationLibrary` 复用本运行 Journal，以 `inspiration_updated` 保存稳定 ID 的简短版本记录，以 `inspiration_task` / `inspiration_candidate` 绑定科研任务和确切候选。它独立于 SolutionLibrary；允许只有想法、依据、进展、问题和下一步，没有代码、完整方案或正指标。`update_inspiration` 不产生实验、不预留数值预算；新条目默认 paused，Generation 用 status 与 reason 激活、暂停、恢复或切换。切换原条目和旧源码都保留，最多一条 active。更新工具与回执幂等提交，中断后复用同一请求；旧快照不能覆盖新进展。Recorder 只能修改已有进展、待办、引用与已吸收部分，不能改变机制或活跃状态。

H3 任务默认绑定当前 active ID；发布候选前已开展的研究请求及后续 L3 请求继承该 ID。旧调用未先建条目时，H3 提交可建立初始条目并关联同任务的请求；不自动导入旧运行。综合阶段针对某灵感取证可在实验草案提供 inspiration_id；不依靠实验描述猜归属。发布后的 H3 科学修订与 L3 始终沿其确切候选的灵感身份，后续切换不改写已有归属或取消三候选 L3。

恢复视图从原 task_snapshot、ExperimentCard、EvidenceCard 和代码 manifest 派生，提供原始请求/配置、source_sha256、源码快照版本、执行结果和失败引用，不重复保存完整方案或大型产物。下一轮 H3 context.inspiration_continuation 默认给最近三个请求和代码入口、未完成事项及完整 read_inspiration 入口；原件保持可读，暂停条目同样可读。Coding/Safeguard 沿已有 implementation_history 直接取得原源码，仍核验文件哈希与可见序号；每份代码结果的 parent_code_ids 记录宿主交付的父代码上下文，实际增量和撤销方式由 Coding explanation 说明，不代表程序已验证科学派生关系。既有冻结包继续通过 evidence_refs/solution_refs 的严格来源恢复，失败或未导出的包不伪造为可加载模型。

H1/H2 对库内完整父方案用 solution_derivation 保存 base_solution_id、changes、rollback；宿主绑定父记录的完整哈希和源码快照引用。该字段兼容旧卡片省略；库外父方案仍需在正文给出完整、可追溯定义。下一轮交付各槽位先前派生入口，原完整父方案和代码保留以供回退。可用灵感部分实际进入 H1/H2 后，以 absorbed_parts 的 part / target_ref 追加确切候选引用；部分吸收不暂停整个灵感，也不更换 H1/H2 的完整基础。弱保底不等于原样重交基线或每轮保证不降。

低分、未入 Top-3、失败不会自动改动灵感状态。Generation 根据实际进展和具体下一步决策，Recorder 记录，宿主负责存取与调度。H1/H2/H3 最终候选仍全部进入原 L3；B 独立保留；正式方案仍按有效双任务 L3 的 ADE20K 主指标及既有规则更新。灵感库目前仅在本运行内跨轮/重启恢复，无跨运行自动合并或源码资产迁移。源码缺失/哈希不符沿既有错误处理返回，不从文字合成“恢复成功”；修改宿主后的历史恢复仍需要显式 intervention。


## 动态研究主题与方法谱系

新轮次先排 `rNNN/focus`（Generation，原订阅串行锁与 BudgetLedger），在 H1/H2/H3 启动前恢复上次主题、依据、用户澄清、关联灵感/方法、关键证据、原代码配置和未完成动作。通常一次模型决定产生一个主题版本并绑定本轮三个槽位，不三路重复选题。若 Generation 仅暂停而未选择后续方向，则先回到共同选题，不自动启动暂停方向的候选；可一次 switch 同时保留旧方向并选择新方向。失败保留待重试状态，不由宿主填充“默认正确”选题。共同调用占用原 LLM/墙钟预算，不增加候选或数值预算。

主题条目是 question、mutable_dimensions、held_conditions、objective、rationale、next_step、reconsider_when 和短引用；主题不是算法正确性声明或数值执行约束。问题可由增益转为解释、必要性检验、消融、共享或简化；相关进展与正式指标分开，方案晋升规则不变。Generation 决定 establish/continue/narrow/expand/pause/resume/switch。next_step 是叙述；初次建立主题或显式 new_action 才创建稳定 action ID。completed_action_ids / deferred_action_ids / resumed_action_ids 由 Generation 精确更新状态，next_action_id 接续 open 动作并更新复议条件。暂停/恢复不按文本重复创建动作，完成后要做新的同文行动必须显式 new_action。原始版本和复议条件始终可读，省略不会清空旧动作或来源。

Journal 的 topic_version 以 topic-id@version 为身份、decision_ref 指向真实 ReviewDecisionCard；topic_round_binding 固定共同初始版本，topic_scope_binding 记录轮中影响槽位和理由。Generation 用 revise_topic 明确 affected_slots / impact，仅影响后续投入和新提交；冻结调用继续按原读取快照归属。已发布候选的 L3 严格继承该候选 topic_ref/method_id，不随当前主题切换。科学修改仍需 revise_candidate，新版本才获得新归属。主题状态不改变 InspirationLibrary.active；灵感切换仍须明确 update_inspiration。

method_candidate 仅补充算法身份、父 method_id（或 solution:<完整方案ID>）、变化说明与候选引用。同算法跨主题继续时 Generation 显式沿用 method_id；新算法/派生省略 ID、声明 method_lineage，由宿主分配新身份并链接父候选。宿主不凭同名、槽位或灵感推断算法相同。旧候选不回填虚构谱系；已有完整方案、HypothesisCard 修订及源码父链继续可读。request_history 复用原请求、代码 manifest、源码快照、配置和 Evidence 投影，不复制完整方案。

context.research_topic 默认交付摘要、最近三项请求/实现和最多八项未完成动作，完整版本由 read_topic 可达；method_catalog_state/read_method 给出沿革及准确实现入口。实际研究请求、最终计划、代码和 Evidence 经已有不可变引用链恢复；任何主题都可以读全局历史完整方案/B。主题不会新增实验的科学范围拒绝门禁，用户 Brief 的全局空间、硬约束、量化与评价协议仍是原执行链的权威。

Recorder 通过可选 Metareview.topic_notes（topic_ref、progress、reconsideration、source_refs）整理有条件的认识、动作进展和复议建议，只追加 topic_note，不改主题状态。Generation 下次恢复这些建议后独立判断。历史无主题状态保持显式未知；已启动的旧轮不插入事后“共同决定”，下一新轮再选题。恢复仍要求原源码变更 intervention；没有跨运行自动合并、资产迁移、语言辩论或宿主科学打分。

恢复读取以 source_snapshot 的 path、sha256、field 和 source_version 保留原件身份。`read_asset({asset: path, snapshot_field: field})` 返回授权成员目录；加 `member` 和 offset/limit 读取确切原文及成员 SHA256。验证完整归档哈希后仅开放执行实现、必要配置和方法/方案成员；评价规则、测试、角色 Prompt 不在恢复视图。原 research_run.json 和 intervention commit 保持完整归档，旧回执的过宽文件授权不继续生效。跨运行完整方案仍用原归档版本，不能以当前源码替代。

自动 topic_catalog、inspiration_library、method_catalog_state 各给最多八项短目录和分页入口；对应 read 工具省略身份即可浏览/按 query 查询，指定身份可读完整历史。当前主题/活跃灵感的未完成事项与复议条件继续交付，同一请求或实现正文在 context 内只保留一份，其他位置给出准确 body_location。Recorder 错误 topic_notes 逐项进入 submission_issues，原条目和原因保留；其余有效正文正常发布，不放宽身份核验。
