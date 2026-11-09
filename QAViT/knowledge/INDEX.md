# 科研资源导航与维护规则

这是静态科研知识的统一入口。当前资料保留原路径以兼容已保存的引用与快照；本目录不复制正文、不保存运行记忆。导航本身也按需读取，已知资源可直接打开。`read_asset` 支持项目相对路径、`knowledge/…` 及原有 `skills/…`、`methods/…`、`solutions/…` 别名；外部论文只开放 ResearchBrief 明确授权的 source_documents。

## 选择与维护入口

| 资源 | 权威来源与内容 | 发现、使用者及维护方式 |
|---|---|---|
| Prompt | [common](../agents/prompts/common.md)、七角色文件、tasks 任务片段；[roles.py](../agents/roles.py) 组装 | 每次角色调用常驻公共+本角色+本任务指令。维护者改原片段及 Prompt 哈希基线；不在 INDEX 重复定义角色规则 |
| Contract | ResearchBrief 冻结本次科研问题；[cards.py](../core/cards.py)、[execution_contract.py](../core/execution_contract.py)、[static_w4a4.py](../core/static_w4a4.py) 及数值契约执行边界 | 各角色接收相关合同；维护者同步实现、接口说明和回归。文档、Skill 或方法名称不能改变可执行边界 |
| Skill | [短目录](../docs/skills/INDEX.md) → SKILL.md → 各自 references | Generation/Evolution/Judger/Recorder 接收短目录，自选零到多个策略；正文按需 read_asset。维护者编辑 SKILL.md 的 name/description 与正文，再生成目录；不加新阶段或必做实验 |
| Method | [方法索引](../docs/methods/index.json) 与 [独立组件文档](../docs/methods/README.md) | 可申请实验的研究任务接收 context.method_catalog；retrieve_memory 的普通检索可附文档入口。参数、数学和适用条件由维护者编辑，执行能力仍以数值插件及合同为准 |
| Recipe / 完整方案种子 | [种子清单](../docs/solutions/seeds.json)、[方案层次说明](../docs/solutions/README.md)；执行构造在 tools/ptq_agent/execution/baselines.py | 静态种子可按需 read_asset；运行时从 context.complete_solutions 的 configuration_asset 读当前协议配置。维护者改种子/构造器与配套测试，不填入未经当前协议实测的成绩 |
| 来源 / references | [PTQ 来源索引](../docs/references/ptq_methods.json)、[模型资产来源](../configs/assets/deit_model_manifest.json)，以及项目 reference/ 原始论文 | 按需 read_asset 读取授权索引或文本；配置中的资产来源由宿主绑定，链接不扩大读取权限。文献工具保存下载原件与解析文本。维护者登记来源、版本、许可及验证边界；历史外部路径不是框架实现依赖 |
| Memory / Evidence / Recorder | 当前运行 EXP 内不可变 Journal、EvidenceCard 与 MetareviewCard；[范围及沉淀](../docs/RESOURCE_LIFECYCLE.md) | retrieve_memory/read_card 受角色、时间快照与立场隔离限制。程序归档测量，Recorder 综合知识修订；静态知识由人维护，不由 Recorder 自动改写 |
| Toolbox | [实验速查](../docs/EXPERIMENT_INPUTS.md)、[代码接口](../docs/CODE_INTERFACES.md)；实际工具定义在 agents/roles.py、tools/experiments.py、tools/profiling.py | 科研角色 request_experiment；Coding→Safeguard→数值执行→Evidence→原研究者续调。维护者同步实现/接口/测试；引用工具名不代表已执行 |

源码链接供开发者定位；它们不自动赋予角色源码读取权限。Coding/Safeguard 可读的公共实现接口仍由既有授权列表决定。没有自动交付某类目录的角色，需要时仍可用 read_asset 读允许的静态资料。

## 内容的唯一维护位置

- Skill 触发描述以各 SKILL.md 的 frontmatter 为准；`python -B QAViT/tests/audit_knowledge.py --write-skill-index` 只更新短目录的生成区域。运行时只读已生成的 INDEX，不扫描或加载全部正文。
- 方法标题、关键词、插件映射改 methods/index.json；原理正文改对应 Markdown。read_asset 是常规科研入口；数值 MethodLibrary 是会话冻结文档兼容接口，不是额外阅读要求，详见方法目录 README。
- 方案种子、协议 Top-3 与运行 Memory 各自维护，不能以文档编辑伪造实测成绩。跨运行复用的实际范围见 [资源生命周期](../docs/RESOURCE_LIFECYCLE.md)。
- 系统用法与接口写 docs；README、STRUCTURE 只提供概览和入口，不另建规则副本。变更调用边界时同步对应接口文档和测试；目录链接与生成内容由 `audit_knowledge.py` 做离线一致性检查，不增加运行时门禁。

来源可用可选 `sources.document` 记录 Skill/Method/Recipe 路径，`sources.ref` 留给 Journal 卡片，推导或文献用 `sources.source` 自然说明。实际阅读内容、范围和版本见 read_asset 回执；声明来源不能替代实验 Evidence。资源缺失或读取失败时，研究角色可保留未知并沿原路径继续。

当前结构见 [STRUCTURE](../../STRUCTURE.md)，历史记录与产物入口见 [HISTORY](../docs/HISTORY.md)。
