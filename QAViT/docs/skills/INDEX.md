# QAViT Scientific Skill Catalog（v0.3）

只在科研决策点选择相关 Skill。目录供研究者发现能力，不是执行顺序、审批门禁或必填清单。每条 Skill 的正文与 `references/` 均按需读取；没有合适 Skill 时继续正常研究。

<!-- SKILL_CATALOG_START -->
| Skill | 何时使用 / 何时不需要 |
|---|---|
| [adapt-quantization-method](adapt-quantization-method/SKILL.md) | 当已有可信量化瓶颈、明确理论缺口，或希望构造有机制依据的 PTQ 假设时使用。 帮助从成熟方法和现有可执行方案中提炼可迁移原理，选择最小针对性改动， 或形成可证伪的新机制。若只需诊断错误或按原样执行方案，则不必调用。 |
| [assess-evidence-transfer](assess-evidence-transfer/SKILL.md) | 当 PTQ 候选已有局部、冻结前缀或初步任务证据，但可推广范围、反证或进一步 研究价值不明确时使用。判断是否补充小对照、升级独立整网评测、回退或暂停。 常规事实归档、已确定的直接执行和明确工程报错不需要本技能。 |
| [design-discriminative-experiment](design-discriminative-experiment/SKILL.md) | 当至少两种 PTQ 机制能解释同一结果、候选收益可能被输入前缀或校准等因素 混杂、或下一次实验昂贵而需要更有判别力的证据时使用。设计可证伪预测、 受控对照与最小真实测量。普通既定评测和证据已足够的情形不必使用。 |
| [diagnose-ptq-failure](diagnose-ptq-failure/SKILL.md) | 当 ViT PTQ 的整体精度或局部误差反常、不同量化位点或方案结果冲突， 而主导机制仍不清楚时使用。选择可信基线、W/A/耦合、分布、真实量化前缀 与传播诊断中的有用证据。已知只是工程报错或问题已定位时不必使用。 |
<!-- SKILL_CATALOG_END -->

读取方式：`read_asset({"asset":"skills/<Skill名称>/SKILL.md"})`；正文中的相对参考链接使用 `skills/<Skill名称>/references/<文件名>.md`。本目录随 context.skill_catalog 交付时无需再次读取；角色可按科研职责自行选择，不增读取权限限制。

正文中 `query_*` / `profile_*` / `read_profile_evidence` 是既有 `profile` 服务内的工具名，并非新的角色工具；科学请求沿 `request_experiment`，实际参数查 `context.experiment_interface`。方法说明直接从 `context.method_catalog` 的 `read_asset` 读取；完整方案用 `context.complete_solutions`（若本任务未交付则按需读 `solutions/README.md`）；历史证据沿 `retrieve_memory` / `read_card`。文中的 L0 指推导或极小检查，不新增实验级别或流程阶段。

触发可组合，也可跳过；没有固定调用顺序。

科学输入以 ResearchBrief 和已绑定的 Evidence 为准。具体 Tool 参数、方法 ID、Top-3 方案 ID 与 Memory 引用由已有接口返回，避免 Skill 猜测路径或硬编码。

维护：表格从 SKILL.md 的 name/description 生成，勿单独改写触发描述；开发者运行 `python -B QAViT/tests/audit_knowledge.py --write-skill-index`。资源分类见 [统一导航](../../knowledge/INDEX.md)。
