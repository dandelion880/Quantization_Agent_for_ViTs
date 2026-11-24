# 按需科研 Skill

人工库覆盖诊断、判别实验、机制创新、目标、尺度、参考解和候选评价。每项以自己的 SKILL.md 描述触发条件，正文与 references 按需读取。静态案例、方法文档和工程资料经[科研资源导航](../knowledge/INDEX.md)查找。人工及 Recorder 版本使用相同的发现和选择入口。运行中的版本、来源、停用和恢复以[资源生命周期](RESOURCE_LIFECYCLE.md#recorder-流程版本)为准。

`context.skill_catalog` 提供相关描述。`read_asset({"asset":"skills/INDEX.md","find":"calibration"})` 定位完整目录中的匹配项；`skills/design-scale-calibration/SKILL.md` 展开步骤，正文相对链接在同一包中读取。工作材料可压缩先前阅读，原版本沿回执恢复。

[适配来源](../knowledge/research-skills/SOURCES.md)和[执行边界](SCIENTIFIC_SKILLS_COMPATIBILITY.md)分别保存研究建议的出处与现有数值能力。人工维护后可用 `python -B QAViT/tests/audit_knowledge.py --write-skill-index` 更新源码导航。数值评测使用当前 ResearchBrief 和既有实验链。
