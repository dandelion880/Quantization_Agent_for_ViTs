# 历史记录与产物导航

当前实现从 [QAViT README](../README.md)、[资源导航](../knowledge/INDEX.md) 与 [STRUCTURE](../../STRUCTURE.md) 进入。历史报告记录当时版本、输入和验证范围；不因保留链接就视为当前能力或当前协议成绩。日志、原始卡片、数据、checkpoint 和源码快照保持原位，只读查阅；不要从历史 EXP 导入一次性框架代码作为当前运行依赖。

本表列出本轮整理时各独立目录的 REPORT.md。完整有效文件树在 [本轮盘点](../../EXP/validation/knowledge_resources_20261108_r1/PROJECT_RESOURCE_INVENTORY.md)，部分原件仅在本机，是否入库以 .gitignore 为准。当前伪代码见 [本轮 PSEUDOCODE](../../EXP/validation/host_context_20261110_r1/PSEUDOCODE.md)。

| 历史目录（相对 EXP） | 当时报告 |
|---|---|
| validation/host_context_20261110_r1 | [宿主性能与上下文简化：真实慢请求、等价/权限回归与 Eval-L0](../../EXP/validation/host_context_20261110_r1/REPORT.md) |
| implementation/qavit_v2_20261026 | [QAViT v2 实施与验证记录](../../EXP/implementation/qavit_v2_20261026/REPORT.md) |
| implementation/v2_structure_20261027 | [QAViT V2 结构整理与离线验证](../../EXP/implementation/v2_structure_20261027/REPORT.md) |
| validation/adaptive_finish_20261106_r1 | [QAViT 贯通收尾修复（2026-11-06）](../../EXP/validation/adaptive_finish_20261106_r1/REPORT.md) |
| validation/cli166_resume_20261031_r1 | [GPU、网络与 Codex 0.166.0 恢复检查](../../EXP/validation/cli166_resume_20261031_r1/REPORT.md) |
| validation/contract_interfaces_20261030_r1 | [contract_interfaces_20261030_r1](../../EXP/validation/contract_interfaces_20261030_r1/REPORT.md) |
| validation/controlled_run_20261105_r1 | [controlled_run_20261105_r1](../../EXP/validation/controlled_run_20261105_r1/REPORT.md) |
| validation/draft_preflight_20261029_r1 | [草案提交到冻结的局部修复](../../EXP/validation/draft_preflight_20261029_r1/REPORT.md) |
| validation/e2e_readiness_20261030_r1 | [最小修复与真实贯通启动准备](../../EXP/validation/e2e_readiness_20261030_r1/REPORT.md) |
| validation/end_to_end_repair_20261105_r1 | [QAViT 真实贯通修订报告](../../EXP/validation/end_to_end_repair_20261105_r1/REPORT.md) |
| validation/experiment_inputs_20261029_r1 | [实验输入整理：数据、预算与 Brief 评测范围](../../EXP/validation/experiment_inputs_20261029_r1/REPORT.md) |
| validation/generation_loop_20261104_r1 | [QAViT Generation 自主循环与上下文优化（2026-11-04）](../../EXP/validation/generation_loop_20261104_r1/REPORT.md) |
| validation/interface_simplification_20261031_r1 | [QAViT V2 最小接口收尾与 CPU 验证](../../EXP/validation/interface_simplification_20261031_r1/REPORT.md) |
| validation/knowledge_resources_20261108_r1 | [科研知识资源整理交付](../../EXP/validation/knowledge_resources_20261108_r1/REPORT.md) |
| validation/memfs_completion_20261030_r1 | [memfs_completion_20261030_r1](../../EXP/validation/memfs_completion_20261030_r1/REPORT.md) |
| validation/memfs_delivery_20261030_r1 | [memfs_delivery_20261030_r1](../../EXP/validation/memfs_delivery_20261030_r1/REPORT.md) |
| validation/memfs_followup_20261030_r1 | [memfs_followup_20261030_r1](../../EXP/validation/memfs_followup_20261030_r1/REPORT.md) |
| validation/memory_recorder_20261105_r1 | [Memory 与 Recorder 优化（2026-11-05）](../../EXP/validation/memory_recorder_20261105_r1/REPORT.md) |
| validation/parallel_debate_20261106_r1 | [三场初始辩论局部并行化](../../EXP/validation/parallel_debate_20261106_r1/REPORT.md) |
| validation/parallel_finish_20261106_r1 | [QAViT 贯通前收尾与准备验证](../../EXP/validation/parallel_finish_20261106_r1/REPORT.md) |
| validation/parent_evolution_20261029_r1 | [第二批：固定 H* 后仅比较父子](../../EXP/validation/parent_evolution_20261029_r1/REPORT.md) |
| validation/real_research_tiny_20261031_r1 | [CPU tiny ViT 真实科研验证与最小修复](../../EXP/validation/real_research_tiny_20261031_r1/REPORT.md) |
| validation/recorder_closure_20261030_r1 | [recorder_closure_20261030_r1](../../EXP/validation/recorder_closure_20261030_r1/REPORT.md) |
| validation/research_gates_20261105_r1 | [科研门禁集中清理（2026-11-05）](../../EXP/validation/research_gates_20261105_r1/REPORT.md) |
| validation/research_refactor_20261103_r1 | [QAViT 科研导向简化报告](../../EXP/validation/research_refactor_20261103_r1/REPORT.md) |
| validation/research_simplification_20261031_r1 | [QAViT V2 科研导向最小简化](../../EXP/validation/research_simplification_20261031_r1/REPORT.md) |
| validation/research_upgrade_20261107_r1 | [QAViT 最小科研能力升级（2026-11-07）](../../EXP/validation/research_upgrade_20261107_r1/REPORT.md) |
| validation/scientific_skills_20261108_r1 | [首批科研 Skill 接入与验证](../../EXP/validation/scientific_skills_20261108_r1/REPORT.md) |
| validation/serial_debate_memory_20261029_r1 | [第三批：按场推进、共享科研正文、结案固定](../../EXP/validation/serial_debate_memory_20261029_r1/REPORT.md) |
| validation/single_implementation_20261105_r1 | [两批科研链简化实施记录](../../EXP/validation/single_implementation_20261105_r1/REPORT.md) |
| validation/storage_recovery_20261031_r1 | [存储恢复记录](../../EXP/validation/storage_recovery_20261031_r1/REPORT.md) |
| validation/two_rounds_20261104_r2 | [QAViT 真实运行最终报告 — 2026-11-04](../../EXP/validation/two_rounds_20261104_r2/REPORT.md) |

其它入口：EXP/runs 为各运行事实与源码冻结，EXP/inputs 为具体研究配置，EXP/outputs 为历史整理输出，EXP/cache 为可重建临时数据。它们的目录名称不保证属于当前 V2 格式；恢复须遵守原身份/配置合同，不能批量迁移或重写成新格式。
