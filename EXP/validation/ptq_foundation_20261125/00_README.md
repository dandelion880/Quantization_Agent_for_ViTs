# QAViT 研究基础补充包

版本：2026-11-25，五个工作包文字补全版。

这批材料用于补充 QAViT 的科研基础。协议、方法正文和五个工作包的接入文字由本包提供，Codex 负责映射到现有组件、实现缺失能力并验证。文件可以拆分、合并或迁移；方法的数学含义、来源、原配方与适配变体的区别应保留。

上一版完成了用户要求的“协议确认＋方法拆解”，但还不足以单独承担五个工作包的全流程接入。本版增加 07—12 六份正文，补齐组件对应、角色文字、Skill 内容、任务与比较语义以及集成验证，并完善分类和交接说明。Codex 任务提示词按用户要求仅在会话的可编辑写作块中提供，不收入压缩包。

本包依据用户提供的《A survey on post-training quantization for Vision Transformers: Failure modes and deployment constraints》，并对照 QAViT(20261124-101904) 源码及可访问的原论文、作者实现编写。综述是检索目录。遇到它与原论文不一致的地方，以查明的原始来源为准，保留更正记录。

## 覆盖概况

综述的 83 个条目全部有明确去向：51 条进入 PTQ 方法拆解，5 条 PTQ 来源尚待补齐，1 条合并为同家族出版版本，1 条作为无训练算子旁证；14 条 QAT 排除，6 条非独立 PTQ 训练路线排除，5 条训练边界未决而暂不录入。这里按文献条目计数，包含既有能力补全与版本关系，不等于新增 51 个独立算法，也不等于全部配方已复现。

ERQ 的 2025/DUQ 后续版本另有方法正文。51 条拆解材料及该版本均读取了原方法正文，或在 DAQuant 的情况下以实际作者源码核定行为。原文/代码中仍有具体未闭合项，它们在各卡片内保留，不以占位默认值补齐。

## 先读什么

1. [07_FIVE_WORK_PACKAGES.md](07_FIVE_WORK_PACKAGES.md)：先确定 A—E 的具体范围、文字依据、完成条件及与下一阶段的边界。
2. [01_STATIC_W4A4_V2.md](../../../QAViT/knowledge/ptq/01_STATIC_W4A4_V2.md)：新协议草案，解释 static W4A4、离线优化、扩展方法和公平比较。
3. [03_BASELINE_AND_INTEGRATION.md](../../../QAViT/knowledge/ptq/03_BASELINE_AND_INTEGRATION.md)：现有能力、本包的增量和接入边界。
4. [SOURCE_COVERAGE.md](../../../QAViT/knowledge/ptq/SOURCE_COVERAGE.md)：按综述逐项查找方法及去向；source_coverage.json 为统一机器可读账本。
5. 各 methods_*.md 与 [existing_methods_delta.md](../../../QAViT/knowledge/ptq/existing_methods_delta.md)：完整原配方、可组合机制、研究工作流及已有方法的增量。
6. [08_METHOD_COMPONENT_MAP.md](../../../QAViT/knowledge/ptq/08_METHOD_COMPONENT_MAP.md)：51 个主条目及 ERQ 后续版本的组件/配方对应、共用数值能力及状态重估关系。
7. [metrics_registry.md](../../../QAViT/knowledge/ptq/metrics_registry.md) 与 [11_TASK_COMPARISON_AND_EVIDENCE.md](../../../QAViT/knowledge/ptq/11_TASK_COMPARISON_AND_EVIDENCE.md)：metric 定义、任务/data/head/reference、比较与排名、父方案和历史证据。
8. [04_RESEARCH_WORKFLOWS.md](../../../QAViT/knowledge/ptq/04_RESEARCH_WORKFLOWS.md) 与 [05_TAXONOMY_AND_TOPIC_DESIGN.md](../../../QAViT/knowledge/ptq/05_TAXONOMY_AND_TOPIC_DESIGN.md)：研究思路、八个可组合研究维度和主题设计。
9. [09_ROLE_TEXT_BLOCKS.md](09_ROLE_TEXT_BLOCKS.md) 与 [10_RESEARCH_SKILL_CONTENT.md](10_RESEARCH_SKILL_CONTENT.md)：与实际活跃角色、阶段和 Skill 体系对应的可接入正文。
10. [12_INTEGRATION_ACCEPTANCE.md](12_INTEGRATION_ACCEPTANCE.md) 与 [06_CODEX_HANDOFF.md](06_CODEX_HANDOFF.md)：真实低成本验证情境、证据和接入注意事项。

02_PROTOCOL_PROFILES.json 是便于接入的协议说明数据，不是当前 QAViT 可直接加载的配置。方法文稿中的新接口名称也是建议名称；可调用性以接入后的实际注册和验证为准。

## 本次交付的范围

本次会话完成文件层面的文字补充，没有修改运行中的宿主、提示词或数值实现，也没有执行算法复现。交给 Codex 的本轮任务是 A—D 的配套接入和 E 的必要低成本真实验证；更广泛的综合调优、多轮完整科研搜索及全量正式评测属于后续阶段。接入时可以修改文件组织，但不应要求 Codex 重新概括论文来代替本包正文。

本版覆盖上述五包的文字需求，不表示原始来源缺口已消失，也不表示实现已经完成。来源充足且符合当前软件后端与模型数学前提的能力应继续实现；确实缺少公式、参数、独立 PTQ 证据或特定后端的条目须分别记录具体限制，不能编造，也不能把整个方法库缩减为少数容易完成的条目。

QAT 方法严禁录入。这是本轮用户明确要求，覆盖此前对相邻训练研究的宽泛接入计划。QAT 不进入方法正文、完整配方、primitive、metric 或研究工作流；范围核对表只保留题名、排除原因和来源。混合论文只提取可独立核实的 PTQ 部分，类别无法确认的条目暂不录入。

每种录入的 PTQ 方法按三个层次吸收：

| 层次 | 内容 | 用途 |
| --- | --- | --- |
| 完整原配方 | 模型与任务、表示、目标、参数、步骤、校准与优化、精度例外 | 复现参照和完整研究起点 |
| 可组合机制 | 输入输出、数学操作、依赖、作用顺序、需要重估的状态 | 替换、组合、消融与新方法设计 |
| 研究工作流 | 问题如何被发现、作者怎样论证、适合开展什么判别实验 | 诊断、选题、实验设计和知识提炼 |

原论文的整网结果属于其模型、数据和配方。为 QAViT 提出的变体、跨任务用法或组合使用，均需要新的实验。本文没有执行这些方法，也不包含新测得的量化精度。

## 如何理解完成状态

来源状态、文稿状态和实现状态分开记录。

- 原文和作者代码可读：可以据已读材料写配方，仍需在本地核对数值与重现。
- 只有原文可读：可按论文实现，不能声称已对照作者代码。
- 原始来源已确认属于 PTQ、但只读到摘要等有限材料：只写可确认的机制和适用问题，逐项列出未核实内容。该条目不能冒充可直接复现的完整原配方；只有综述归类而无法确认训练边界的条目不录入方法内容。
- 论文身份或引用有冲突：保留综述条目并记录冲突；不依据错误题名、错链或想象补全算法。
- QAT：排除其方法内容。二值、结构和硬件论文同样先核实是否存在独立 PTQ 方法；依赖 QAT 的配方与机制不录入。

查明的缺口不等于算法无效；完整文稿也不等于 QAViT 已实现。来源账本中每个条目的去向都指向正文或具体缺口说明。

## 接入时的写作约定

面向人的正文使用明确问题、步骤、数学定义和条件。保留必要的公式、参数和反例，删除套话。不要把一个通用警告重复到每篇方法；公共约定在协议和本说明中维护。

方法来源、建议适配、实际实验结论分别落在相应位置。新条目加入方法索引时，尚未实现的接口不填入现有 plugins 字段。Prompt 只提供职责、当前协议、实际能力摘要和按需阅读入口，方法长文留在库内。

## 主要输入

- 综述：Journal of Systems Architecture 181 (2026), 104001，DOI：https://doi.org/10.1016/j.sysarc.2026.104001 。本包使用用户提供的完整 PDF。
- 源码基线：QAViT(20261124-101904).zip。已核对 core、agents/prompts、runtime、tools/ptq_agent、方法与 Skill 资料。
- 各方法原论文及代码：见正文末尾或条目内的稳定 URL、版本和阅读范围。

本包不捆绑第三方论文全文、模型权重、数据集或作者代码。

`PACKAGE_MANIFEST.json` 记录本包文件摘要、输入文件指纹和覆盖统计。它用于检查交付完整性，不是执行配置。
