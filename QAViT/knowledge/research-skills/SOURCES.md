# 七项科研 Skill 来源与版本

内容包：`qavit_skill_system_20261116.zip`，版本 2026-11-16，135583 字节，SHA256 `e5d6871fd3025788c05bb124330c0ff01b840d4019b23cf151f0ec224cf17da1`。包内 MANIFEST 的 41 个条目在安装前逐字节核对。原件（含 START_HERE、CODEX_TASK、兼容审查、设计依据和来源账本）及接入映射保存在 `EXP/validation/skill_system_integration_20261116_r1/source_package/` 与 `integration_map.json`；它们是开发来源记录，不扩大生产角色读取权限。

七份主文和 16 份参考安装到 docs/skills，共同约定和六条完整案例安装到 knowledge/research-skills，调整本地相对链接；额外保留现有冻结传播 API 参考。当前实现边界单独维护在 [兼容说明](../../docs/SCIENTIFIC_SKILLS_COMPATIBILITY.md)。旧 assess-evidence-transfer 被候选评价替换，历史原件不回写。

[论文来源](research-routes/PAPER_SOURCES.md) 保留包作者核对的版本及原文 URL；[PTQ4DINO 案例](research-routes/ptq4dino.md#来源与数值账本) 保留历史人工原件位置、日期和样本范围。本次接入未重新复现实验，也未独立重新核验这些外部论文/Library 原件。公开论证、包作者研究建议、假想情境与历史测量各按正文标注，不能当作本轮 Evidence。

所有安装内容进入原 source identity/hash 和源码快照机制；当前运行只读取其授权版本。本轮无自主 Skill 生成或晋升，评价样例与评阅要点只在 tests/skill_evals，不常驻交付研究角色。

## 2026-11-24 流程适配

以下固定版本用于核对思路。QAViT 使用自己的实现和表述，保留原包与论文署名；平台工具、依赖安装和审阅编排没有移植。

| 来源与具体文件 | 许可原件 | 本地适配 |
|---|---|---|
| [agentskills/docs/specification.mdx](https://github.com/agentskills/agentskills/blob/83021e2be457a9b6679148719c1f6074f0ad975e/docs/specification.mdx) | [Apache-2.0（仓库）](https://github.com/agentskills/agentskills/blob/83021e2be457a9b6679148719c1f6074f0ad975e/LICENSE) | 目录、必需字段、metadata 与按需 references |
| [scientific-agent-skills/skills/experimental-design/SKILL.md](https://github.com/K-Dense-AI/scientific-agent-skills/blob/c21d13b3cc2894680229ba9959a53281acb9200e/skills/experimental-design/SKILL.md) · [scientific-agent-skills/skills/statistical-analysis/SKILL.md](https://github.com/K-Dense-AI/scientific-agent-skills/blob/c21d13b3cc2894680229ba9959a53281acb9200e/skills/statistical-analysis/SKILL.md) | [MIT](https://github.com/K-Dense-AI/scientific-agent-skills/blob/c21d13b3cc2894680229ba9959a53281acb9200e/LICENSE.md) | experiment-patterns、ranking-validation 中的区组配对、采样单位与排序反转 |
| [AI-Research-SKILLs/21-research-ideation/creative-thinking-for-research/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/21-research-ideation/creative-thinking-for-research/SKILL.md) | [MIT](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/LICENSE) | intervention-design 中的结构映射和可证伪预测 |
| [knowledge-work-plugins/bio-research/skills/scientific-problem-selection/SKILL.md](https://github.com/anthropics/knowledge-work-plugins/blob/5e504ae11498f2b8daa30d86b7dab97b2ecd8092/bio-research/skills/scientific-problem-selection/SKILL.md) | [Apache-2.0](https://github.com/anthropics/knowledge-work-plugins/blob/5e504ae11498f2b8daa30d86b7dab97b2ecd8092/LICENSE) | intervention-design 中停滞问题的改述与分支决策 |
| [Auto-claude-code-research-in-sleep/skills/result-to-claim/SKILL.md](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/4ba94ae2a1c1fbbb14858750dd8764cabf6c28bd/skills/result-to-claim/SKILL.md) · [Auto-claude-code-research-in-sleep/skills/kill-argument/SKILL.md](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/4ba94ae2a1c1fbbb14858750dd8764cabf6c28bd/skills/kill-argument/SKILL.md) · [Auto-claude-code-research-in-sleep/skills/research-wiki/SKILL.md](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/4ba94ae2a1c1fbbb14858750dd8764cabf6c28bd/skills/research-wiki/SKILL.md) | [MIT](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/4ba94ae2a1c1fbbb14858750dd8764cabf6c28bd/LICENSE) | evidence-and-comparison 中的结论范围、替代解释与可修订来源 |
| [ShinkaEvolve/skills/shinka-inspect/SKILL.md](https://github.com/SakanaAI/ShinkaEvolve/blob/0e09d0d5869a9a9d8568a5eb10c88d00a3677404/skills/shinka-inspect/SKILL.md) | [Apache-2.0](https://github.com/SakanaAI/ShinkaEvolve/blob/0e09d0d5869a9a9d8568a5eb10c88d00a3677404/LICENSE) | 压缩交付中的父子、测量与实现入口 |
| [superpowers/skills/systematic-debugging/SKILL.md](https://github.com/obra/superpowers/blob/22f0b463e7a5e579904dd6211a520321035526f7/skills/systematic-debugging/SKILL.md) · [superpowers/skills/verification-before-completion/SKILL.md](https://github.com/obra/superpowers/blob/22f0b463e7a5e579904dd6211a520321035526f7/skills/verification-before-completion/SKILL.md) | [MIT](https://github.com/obra/superpowers/blob/22f0b463e7a5e579904dd6211a520321035526f7/LICENSE) | knowledge/coding/{debug,verify} 中的边界追因与修改相关验证 |
| [TensorRT-LLM/.claude/skills/perf-nsight-systems/SKILL.md](https://github.com/NVIDIA/TensorRT-LLM/blob/d33040bb81e886833e1677ae97691f21a8250a6a/.claude/skills/perf-nsight-systems/SKILL.md) | [Apache-2.0](https://github.com/NVIDIA/TensorRT-LLM/blob/d33040bb81e886833e1677ae97691f21a8250a6a/LICENSE) | debug 中依据实际时序分离等待、主机与设备开销 |
| [Alzubi 等，EvoSkill v1](https://arxiv.org/html/2603.02766v1) | CC BY 4.0 | 从真实经历提议新流程或修订；发布接现有 Recorder checkpoint。 |
