# QAViT Scientific Skill Catalog（v0.3）

只在科研决策点选择相关 Skill。目录供研究者发现能力，不是执行顺序、审批门禁或必填清单。每条 Skill 的正文与 `references/` 均按需读取；没有合适 Skill 时继续正常研究。

| Skill | 触发信号 | 适用角色 | 不适用 |
| --- | --- | --- | --- |
| [diagnose-ptq-failure](diagnose-ptq-failure/SKILL.md) | 量化结果异常、主导误差机制不清楚 | Generation、Evolution | 明确的工程错误或已有可信定位 |
| [design-discriminative-experiment](design-discriminative-experiment/SKILL.md) | 竞争解释或混杂因素影响结论，需要区分性证据 | Generation、Judger、Evolution | 已有实验计划的普通执行，或证据已足以决策 |
| [adapt-quantization-method](adapt-quantization-method/SKILL.md) | 需要将量化机制映射到已有方案的改进、迁移或开放创新 | Generation、Evolution | 仅仅定位错误或直接复现现成方案 |
| [assess-evidence-transfer](assess-evidence-transfer/SKILL.md) | 局部/中间指标已有结果，需判断推广范围或下一项科研投入 | Generation、Judger、Evolution；Recorder 可参考 | 常规事实记录或无需额外科研判断的既定评测 |

读取方式：`read_asset({"asset":"skills/<Skill名称>/SKILL.md"})`；正文中的相对参考链接使用 `skills/<Skill名称>/references/<文件名>.md`。本目录已经交付，无需再次读取。上述适用角色是建议，不增读取权限限制。

正文中 `query_*` / `profile_*` / `read_profile_evidence` 是既有 `profile` 服务内的工具名，并非新的角色工具；科学请求沿 `request_experiment`，实际参数查 `context.experiment_interface`。方法说明直接从 `context.method_catalog` 的 `read_asset` 读取；完整方案用 `context.complete_solutions`（若本任务未交付则按需读 `solutions/README.md`）；历史证据沿 `retrieve_memory` / `read_card`。文中的 L0 指推导或极小检查，不新增实验级别或流程阶段。

触发可组合，也可跳过；没有固定调用顺序。

科学输入以 ResearchBrief 和已绑定的 Evidence 为准。具体 Tool 参数、方法 ID、Top-3 方案 ID 与 Memory 引用由已有接口返回，避免 Skill 猜测路径或硬编码。
