# QAViT Scientific Skill Catalog（2026-11-16）

只在科研决策点选择相关 Skill。目录供研究者发现能力，不是执行顺序、审批门禁或必填清单。每条 Skill 的正文与 `references/` 均按需读取；没有合适 Skill 时继续正常研究。

<!-- SKILL_CATALOG_START -->
| Skill | 何时使用 / 何时不需要 |
|---|---|
| [adapt-quantization-method](adapt-quantization-method/SKILL.md) | 当已有量化瓶颈、成熟方法可迁移的原理，或一个值得形式化的理论构想， 需要把它变成可实现、可证伪的 PTQ 改良/新机制时使用。选择改变表示、码本、 误差补偿或算法结构，并处理它与完整方案的交互。单纯忠实复现、尚未定义的报错修复、 仅比较校准 metric 或分配 scale 搜索分别交其他流程；H3 不必先完成实证诊断。 |
| [design-discriminative-experiment](design-discriminative-experiment/SKILL.md) | 当两个机制能解释同一结果、候选改良与校准/预算等因素混杂、H3 需要可证伪原型， 局部代理改善却与下游或双任务冲突，或很难选择下一项实验时使用。设计能改变科研判断的受控测量及结果分支。 已确定的常规评测、单纯选择校准目标或安排 scale 搜索不需要本技能； 已有证据足够时可以直接决策，不以新实验数量为目标。 |
| [design-scale-calibration](design-scale-calibration/SKILL.md) | 在需要设计 PTQ scale/zero 候选、分组与共享方式、粗细搜索、联合更新、 校准顺序、lookahead 的搜索安排或有限重复校准时使用。 负责“允许选择什么、如何有效选择并安装”，与评价目标的数学定义分工。 若问题主要是引入新码本/等价变换、未知误差来源或已有结果的解释，应使用相应 Skill。 |
| [diagnose-ptq-failure](diagnose-ptq-failure/SKILL.md) | 当 ViT 量化精度异常、W/A 联合作用不明、局部误差与整网结果冲突， 或候选反复集中同一位点而缺少全局依据时使用。定位误差在哪里引入、如何传播， 并识别最值得干预的机制及剩余不确定性。已明确的工程报错、单纯挑选校准 metric、 已定位后的 scale 搜索和完整候选的最终评价分别交相应流程或其他 Skill。 |
| [evaluate-ptq-candidate](evaluate-ptq-candidate/SKILL.md) | 在需要评估 PTQ 局部干预或完整候选、解释正负结果与指标冲突、比较历史方案、 判断探索 oracle 的可迁移价值，或决定继续、修订、暂停、转入正式 L3 时使用。 区分测量有效性、机制证据、任务收益和迁移范围；将判断转成下一笔科研投入。 仅归档既定事实、直接执行已确定计划或排查单纯工程报错时无需使用。 |
| [learn-from-reference-solution](learn-from-reference-solution/SKILL.md) | 在 PTQ 失败原因不明、想知道现有表示能否获得好解，或已有轻量 STE、 高成本搜索、人工优良方案可供观察并转化为可部署规则时使用。 负责从参考解提取可检验结构、设计无需继承参考参数的新方法；不把参考解当最优上界。 普通现成方法迁移、例行校准和自动生成/沉淀 Skill 不属于本技能。 |
| [select-calibration-objective](select-calibration-objective/SKILL.md) | 在需要选择或修订 PTQ 校准 metric、重建目标、参考输入、token 权重或 lookahead， 或候选局部误差排序与真实后缀/任务效果不一致时使用。 负责定义“什么叫选得好”并检验排序可信度，不负责设计 scale 候选族、分组和执行顺序。 已有固定目标的普通执行、仅评价完成的结果或纯实现错误无需使用。 |
<!-- SKILL_CATALOG_END -->

读取方式：`read_asset({"asset":"skills/<Skill名称>/SKILL.md"})`；正文中的相对参考链接使用 `skills/<Skill名称>/references/<文件名>.md`。本目录随 context.skill_catalog 交付时无需再次读取；角色可按科研职责自行选择，不增读取权限限制。

正文中 `query_*` / `profile_*` / `read_profile_evidence` 是既有 `profile` 服务内的工具名，并非新的角色工具；科学请求沿 `request_experiment`，实际参数查 `context.experiment_interface`。方法说明直接从 `context.method_catalog` 的 `read_asset` 读取；完整方案用 `context.complete_solutions`（若本任务未交付则按需读 `solutions/README.md`）；历史证据沿 `retrieve_memory` / `read_card`。合成/单元数值证据使用语义名称，不与 Eval-L0 离线评价混称。

触发可组合，也可跳过；没有固定调用顺序。共同术语按需读 [科研约定](../../knowledge/research-skills/RESEARCH_CONVENTIONS.md)，完整案例见 [研究路线](../../knowledge/research-skills/research-routes/INDEX.md)，接口边界见 [源码兼容说明](../SCIENTIFIC_SKILLS_COMPATIBILITY.md)。本轮暂缓自主 Skill 沉淀。

科学输入以 ResearchBrief 和已绑定的 Evidence 为准。具体 Tool 参数、方法 ID、可访问完整方案 ID 与 Memory 引用由已有接口返回，避免 Skill 猜测路径或硬编码。

维护：表格从 SKILL.md 的 name/description 生成，勿单独改写触发描述；开发者运行 `python -B QAViT/tests/audit_knowledge.py --write-skill-index`。资源分类见 [统一导航](../../knowledge/INDEX.md)。
