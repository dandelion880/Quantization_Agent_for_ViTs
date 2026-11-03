# 可编辑的 PTQ 方法库

本目录供人和 V2 角色通过 Profiling 工具阅读。`index.json` 是索引，每个 Markdown 是独立组件的说明；不提供整模型候选或论文配方。下文实现路径均相对 QAViT；来源索引中的外部路径保留历史溯源意义，不是框架运行依赖。

V2 通过实验申请调用 `query_method_library(target, query)` 查索引，再按需 `read_method_document(method)`，或用 `methods:[...]` 一次完整读取1–8篇。query 是简单关键词过滤，兼容 `post_gelu` / `post-softmax` 等分隔写法，空字符串返回全部；未命中不会自动推荐方法，也不是语义搜索引擎。每个 Profiling 会话将实际文本和 SHA256 冻结在该会话目录的 `method_library.json`，人工修改从新会话起生效。

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

`execution/recipes.py` 保留为开发者离线数值回归资料；自主搜索不注册 `query_ptq_recipe`，不把既有 baseline 的完整方案放进任务或 memory。AdaLog（A网格与bias预处理独立）、AdaRound、GPTAQ v1已注册；ERQ/IGQ仍只作为人工研究参考，尚未开放执行。

V2 科研流程与实验申请入口见 [主 README](../../README.md)。本目录只描述数值组件；局部诊断模型不能导出，正式模型的完整覆盖门槛保持生效。

数值工具保留 `loop.require_method_documents=true` 时的计划校验：需要本会话的索引查询、所选插件对应说明及calibration类说明的成功读取记录，并校验冻结文档的hash。它不强制使用某个插件，也不把文档当实测evidence；是否经过此校验取决于实际工具调用路径，不是 V2 的全局科研门槛。资料读取不保证理解或精度提升，实际作用仍需自主试验检验。
