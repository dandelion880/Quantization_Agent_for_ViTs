# 按需科研 Skill

`skills/` 来自用户提供的 QAViT_scientific_skill_pack_v0_3.zip。四个 Skill 的正文及九份策略参考保留原文；初次仅调整 INDEX 的实际读取路径和接口说明；当前触发目录由原 frontmatter 生成，正文及参考仍不改写。它们是研究者可选择的静态策略资料，不是新阶段、审批层、工具或必做实验。来源文件哈希、接入差异及验证记录见 [本次交付](../../EXP/validation/scientific_skills_20261108_r1/REPORT.md)。

## 发现与读取

`RoleInteraction.context()` 对 Generation、Evolution、Judger、Recorder 交付短 `context.skill_catalog`（INDEX.md）。Coding、Safeguard、Supervisor 不自动接收目录；所有角色另可从 context.knowledge_index 按需打开统一导航；原有 docs 读取入口保留。目录缺失时省略，不影响正常研究。

研究者依据当前问题选择零到数个 Skill。例如：

```json
{"asset":"skills/diagnose-ptq-failure/SKILL.md"}
```

以上是已有 `read_asset` 的参数。正文中的相对参考链接以该 Skill 目录为基础，传入如 `skills/diagnose-ptq-failure/references/quantization-error-map.md`。没有新增路由模型、工具 Schema、Skill 调用字段或配置开关。阅读失败由既有工具回执返回原研究任务，仍可继续研究与提交。

初始上下文不包含四份正文或 references。已读内容随现有 tool_results 续调，重复结果沿 `compact_results()` 去重；INDEX 的哈希和交付位置也登记到原 `memory.asset_locations`，避免再读目录时重复注入。这里的 asset_locations 是上下文展示元数据，不是新 Memory 层。research_note 之前的历史读取仍服从既有压缩规则，必要时可以重读。

`authorized_assets()` 递归发现 docs 与 knowledge 文本；`request_body()` 冻结读取权限与哈希，`host_tool()` 核验后按字符分页返回。`runtime.identity.source_files()` 已自动覆盖这些文件，因此无需另设资产白名单或 Skill 版本协议。正常运行的源码变更/恢复规则仍适用。

## 与既有资源协作

| 资源 | 实际入口 | 职责 |
|---|---|---|
| 方法组件 | `context.method_catalog`、`read_asset(methods/...)`；Memory 检索也可返回文档入口 | 数学定义、参数、适用条件与数值实现 |
| 完整方案 | `context.complete_solutions` 的 `configuration_asset`；`solutions/README.md` | 当前协议的完整可复用起点与实测/待测状态 |
| 历史事实与经验 | `retrieve_memory`、`read_card` | 受可见性限制的历史对象、知识修订与证据范围 |
| Toolbox | `context.services`、`context.experiment_interface`、`request_experiment` | 实际执行测量；Skill 中 profile/query 名称是 profile 服务的内部工具 |
| Evidence / Recorder | 原实验回执、EvidenceCard、MetareviewCard | 原始观测与科研综合分开；经验由现有 Recorder/Memory 沉淀 |

Skill 不自动改写方法库、Top-3 或 Memory，也不能把建议、文献、例子或局部代理指标当成当前实测。H1/H2/H3、固定 H*→H4、并行 Judger、Coding→Safeguard→执行、Recorder、Supervisor、科研合同及预算完全沿用原实现。

## 维护与验证

维护者在对应 SKILL.md 更新 name/description 与策略正文，细节放本条 references。用 `python -B QAViT/tests/audit_knowledge.py --write-skill-index` 更新 INDEX 的生成表格；无参数运行做一致性检查。触发描述只维护一处，运行时不扫描四份正文；增加 Markdown 不会新增可执行方法。统一分类与维护规则见 [资源导航](../knowledge/INDEX.md)。评估案例保存在 `tests/skill_evals/ROUTING_AND_REPLAY.md`，不进入运行时目录；它是测试设计，不能当作已完成实验。

`tests/test_scientific_skills.py` 验证目录、链接、身份、实际读取、续调去重、无 Skill 路径和失败回退。真实烟测、完整回归、未验证范围、项目资源盘点及回退依据统一保存在本次 EXP 交付目录。移除本次目录交付及提示词增量即可撤回接入；按备份逐项撤回，不能用 git reset 覆盖此前已有改动。

文档来源用可选 sources.document，Journal 卡片用 sources.ref；历史路径误填 ref 的兼容处理及读过/实测边界见 [资源生命周期](RESOURCE_LIFECYCLE.md)。
