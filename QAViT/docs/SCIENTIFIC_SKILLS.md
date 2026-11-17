# 按需科研 Skill

当前七项能力来自用户提供的 `qavit_skill_system_20261116.zip`：诊断、判别实验、机制改良、目标选择、尺度校准、参考解启发、候选评价。它们按当前科研决定区分，可选择零到多项，不是顺序阶段、审批层或必做实验。`evaluate-ptq-candidate` 替换旧 `assess-evidence-transfer`；旧正文留在 Git/原运行快照和本次备份，不再提供第二个同义入口。本轮暂缓自主 Skill 沉淀。

七份主文、16 份包内专项参考位于 `docs/skills/`；已验证的 `diagnose-ptq-failure/references/frozen-propagation.md` 保留作为额外 API 参考。共同约定、PTQ4DINO 和五篇论文路线放在 `knowledge/research-skills/`，来源见 [账本](../knowledge/research-skills/SOURCES.md)。研究建议不等于已有 API，实际边界见 [兼容说明](SCIENTIFIC_SKILLS_COMPATIBILITY.md)。

## 发现、读取和来源

`RoleInteraction.context()` 仅向 Generation/Recorder 交付短 `context.skill_catalog`。其他角色仍可从 `context.knowledge_index` 或已知路径按需读取；没有新增路由器、调用字段或权限开关。正文和长参考不会随目录常驻注入。目录缺失或读取失败仍可继续原科研任务。

已有 `read_asset` 参数示例：

```json
{"asset":"skills/design-scale-calibration/SKILL.md"}
```

相对 Markdown 链接以所在文档目录为基准，调用时使用解析后的项目路径或现有别名，例如 `skills/design-scale-calibration/references/candidates-and-groups.md`、`knowledge/research-skills/research-routes/ptq4dino.md`。重复结果由现有 `compact_results()` 去重；长文使用原分页回执。`memory.asset_locations` 只是交付元数据，不是新记忆层。

`authorized_assets()` 递归发现 docs/knowledge；原 `request_body()` 冻结可见路径与 SHA256，`host_tool()` 核验版本后读取。`runtime.identity.source_files()` 和运行源码快照覆盖所有新增正文/引用，恢复仍按原源码身份规则进行。已冻结旧运行不会被本次内容替换。可选 `sources.document` 引用文档，`sources.ref` 引用 Journal 卡片；读取或引用文档不等于取得测量证据。

## 与现有研究链协作

方法说明沿 `context.method_catalog` / `read_asset(methods/...)`；完整方案沿 `context.complete_solutions`，可见 archive 也是起点。历史事实沿受权限及时间边界限制的 `retrieve_memory/read_card`；跨运行完整方案与当前 Journal 的通用经验不混称。

保留 H1/H2 完整方案改良、H3 开放创新、独立保底 B、候选综合修订和冻结迁移。科学请求沿原 Coding→Safeguard→数值执行→Evidence→原研究者链，Recorder 综合认识。没有最低实验数、H3 配额、额外审查角色或 Skill 阅读证明。获准研究 STE 与最终独立 training-free 的来源和执行语义见兼容说明；全程禁训合同不会被 Skill 放宽。

## 维护与验证

触发描述仅维护各 SKILL.md 的 name/description，用 `python -B QAViT/tests/audit_knowledge.py --write-skill-index` 生成短目录，无参数检查引用与一致性。增加文档不会新增数值能力。

七项行为任务保存在 `tests/skill_evals/seven_skills_20261116/CASES.md`，评阅要点另存 `REVIEW_GUIDE.md`；两者不属于生产知识读取权限。测试允许零 Skill，不按是否命中特定名称评分。内容可读、工程回归、真实 LLM 决策、科研收益分别报告；文档检查和 CPU tiny 数值不能证明真实 DeiT/DINO/ADE 效果。

输入包哈希、原件、接入映射、实测验证和剩余缺口见 [本轮交付](../../EXP/validation/skill_system_integration_20261116_r1/REPORT.md)。原 ZIP 在完成后按用户要求删除，解压原件和来源账本保留。本轮不安装个人 Skill，不修改历史测量或冻结评分，不自动启动长期正式实验。
