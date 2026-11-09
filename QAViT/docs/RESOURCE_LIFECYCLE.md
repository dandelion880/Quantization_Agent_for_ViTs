# 资源调用、来源与沉淀

分类与唯一维护位置见 [knowledge/INDEX.md](../knowledge/INDEX.md)。这里说明真实调用链，不设额外阶段、引用要求或提交门禁。

## 按需知识与常驻上下文

`runtime/driver.py:request_body()` 调用 `agents/interaction.py:context()`，再由 `agents/roles.py:role_spec()` 拼接 common、当前角色和任务片段、提交视图与工具 Schema。ResearchBrief、当前任务、预算、必需短证据/反馈进入上下文；长正文给分页入口。角色不会自动读取全部 Prompt 或全部历史。

所有角色只获得 `context.knowledge_index.read_asset` 这一导航指针。Generation/Evolution/Judger/Recorder 额外获得 Skill 短 INDEX；允许 request_experiment 的任务获得 method_catalog 与 EXPERIMENT_INPUTS 正文；generate/evolve/code/safeguard/record/plan_l3 获得 complete_solutions；Coding/Safeguard 获得 CODE_INTERFACES 正文。其余资源自主读取，目录不是阅读证明。

`authorized_assets()` 从 docs/ 与 knowledge/ 发现文本、合入 Brief 明确列出的 reference/ 来源和当前可见任务产物；`request_body()` 冻结路径/哈希及 Journal 序号。read_asset 使用同一授权快照，核验当前字节并返回 UTF-8 字符页、sha256、offset/next_offset。旧 skills/、methods/、solutions/ 别名不变，knowledge/ 别名与项目相对路径均可用。新增目录没有开放任意文件或同回合私有子任务。

续调沿现有工具结果保留阅读内容，重复正文压缩；research_note 之前的旧结果可转成短入口，必要时重读。导航/Skill/方法发现目录不可用时可继续原科研路径，read_asset 失败回到原任务。数值执行所需的配置、插件合同和会话快照仍须真实有效，不能借“可选知识”跳过数值核验。

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
