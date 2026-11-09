# 可编辑的 PTQ 方法库

本目录供人和 V2 角色直接通过 read_asset 阅读。`index.json` 是索引，每个 Markdown 是独立组件的说明；与完整方案库分开。下文实现路径均相对 QAViT；来源索引中的外部路径保留历史溯源意义，不是框架运行依赖。

V2 的 context.method_catalog 直接给出简短标题、适用问题和 `read_asset(asset="methods/<id>.md")` 入口；索引可直接读取 `read_asset(asset="methods/index.json")`。资料查阅不需要申请实验或进入 Coding/Safeguard。宿主按当前调用的授权快照绑定文件哈希。已有 Profiling `query_method_library` / `read_method_document` 仍用于数值会话中的文档查询兼容：query 是关键词过滤，空查询浏览全部；会话文本及 SHA256 保存于 method_library.json。它们不是科研角色查阅资料的前置步骤。

| 编辑目标 | 位置 |
|---|---|
| 修改原理、伪代码、适用条件、注意事项 | 对应 `<id>.md`，无需改数值代码 |
| 修改标题、关键词、论文归属、源码指引 | `index.json` |
| 新增执行算法 | `tools/ptq_agent/execution/methods.py` 的独立 target 接口及实际实现 |
| 新增合法格式/粒度、顺序和冲突 | `execution/plugin_contracts.py`、需要时 `sites.py` |
| 新增数值/模型适配证明 | 使用固定上游源码 oracle、真实位点 forward、覆盖/保护/重载检查；注明数据预算和适用范围 |
| 登记原文、固定 commit、许可 | [项目内来源索引](../references/ptq_methods.json) |

每张方法卡应写清：解决何种可测问题、何时不适用、数学/伪代码、独立参数、观察和执行顺序、互斥/并列关系、成本、源码位置、已验证和未验证的边界。记录**组件**的约束，避免写“某模型直接按这些层套此配方”的预制答案。

索引的 `plugins` 必须对应真实注册方法；启动检查保证每个注册插件有说明。只添加文档不会凭空开放执行能力。可执行性由 Python 契约和模型实测决定，文字不能放宽保护、量化范围、格式、显存或数据隔离。

例如新增独立激活方法时，先实现 `TargetMethod.fit/apply`，补 schema 和方法语义校验、导出参数校验/加载版本，再增加含 `activation` 映射的索引项和文档。不要把同一论文的其它方法硬编码进它的 fit/apply。全局白名单仍由任务配置决定。

`execution/recipes.py` 保留原论文配置参考；`execution/baselines.py` 为新 static W4A4 完整种子提供独立构造入口，见 `docs/solutions/`。Generation 可直接采用、适配或组合；不把完整方案混入 Memory 知识。AdaLog（A网格与bias预处理独立）、AdaRound、GPTAQ v1已注册；ERQ/IGQ仍只作为人工研究参考，尚未开放执行。

V2 科研流程与实验申请入口见 [主 README](../../README.md)。本目录只描述数值组件；局部诊断模型不能导出，正式模型的完整覆盖门槛保持生效。

无调用方的历史 `require_plan_documents()` 阅读检查已移除；`query_method_library` / `read_method_document` 及其会话快照保持。`loop.require_method_documents` 仅保留旧配置兼容字段，当前科研与数值计划执行路径不以阅读记录判定能否提交或执行。`required_reads` 表示按需深读入口，不是阅读确认。文档不能代替实际测量；科学选择由研究角色负责，数值合同、真实范围、来源和重载检查继续执行。

分类、权威来源与目录维护见 [统一导航](../../knowledge/INDEX.md)，会话快照和运行内外记忆边界见 [资源生命周期](../RESOURCE_LIFECYCLE.md)。
