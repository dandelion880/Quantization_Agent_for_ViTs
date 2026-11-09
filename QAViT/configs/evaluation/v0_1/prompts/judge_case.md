# QAViT Eval-L0 · 单任务评分输入模板

将以下标记替换为程序已冻结、可追溯的数据。`rubric` 插入 `rubrics/*.json` 中与 `role/purpose` 相匹配的完整部分。此文件**只定义评分消息布局，不创建、选择或生成测试任务**。

## 1. 评价身份

```json
<case_identity: {case_id, run_id, task_id, task_key, role, purpose, slot_if_any, evaluated_task_state, input_version_refs, evidence_cutoff, rubric_id}>
```

说明：本评分针对哪个冻结角色任务、哪些来源版本、截至何时可见。若是 `rank`，程序应先排除 `author=program` 的输出；不能让评分器把程序裁决当成 LLM 的科研成果。

## 2. 原始研究问题与任务指令

```
<ResearchBrief / quantization_contract / 该角色对应的原始任务定义和必要的工作要求>
```

如有独立 `original_request`（Coding/Safeguard）或已冻结父假说 H*（Evolution），应明确交付其真实版本。

## 3. 当时可见的科研上下文

```
<subject_visible_context：真实送达本次任务的关键上下文、可见知识与来源；必要时包括与当前判断直接相关的 read_card/read_asset 工具结果>
```

绝不把后续轮次的科研结论、另一立场未汇合的私有发言或赛后真实精度混入本块。对长上下文可引用冻结资产而非武断裁剪；无法提供重要片段时在覆盖标记中声明，评分器须允许 Unknown。

## 4. 此次角色任务的完整可观察行动

```
<task_trajectory：在同一 task_id 下的所有可恢复模型调用及可见响应；关键工具参数/回执/实验结果；重试与修复；最终提交的卡片/代码/决策原件>
```

按事件顺序展示；将 LLM 自主选择与宿主自动绑定/执行行为区分。只把真实提交与真实回执视为已发生动作。专门的程序成本汇总不应让评分器按 Token 多寡给科研质量分。

## 5. 相关证据索引

```json
<source_index: [{source_id, source_type, immutable_ref, locator_or_path, availability, summary_or_verbatim_excerpt, scientific_scope}]>
```

来源必须真实授权且本次评分可读取。若只给摘要/节选，明确 `availability=partial`，不可让评分器假装已审完整材料。对于实测，提供必要的 Experiment/Evidence 身份、数据/模型/位点/范围/状态；不得混淆缓存、synthetic/test_double、真实完成测量。

## 6. 评价 Rubric

```json
<selected_role_purpose_rubric: 原样插入完整 criterion IDs、问题、inspect 与 0/1/2 anchors；如有 Generation slot overlay 也插入>
```

所有评分项按原始顺序返回。无需生成不存在的任务类别。

## 7. 可用证据缺失与评价边界

```json
<coverage_notes: {missing_or_truncated_sources, known_harness_or_environment_errors, unavailable_validation, source_cutoff_integrity}>
```

如独立数值结果在角色结束后出现，只可在**另行标记的** `posthoc_observations` 中用于报告，不能通过本模板影响当时的行为质量评分。

## 8. 评分输出要求

严格按照 `contracts/judge_result.schema.json` 返回 JSON；不得提出新科研请求、修改原轨迹或额外启动数值实验。依据 Rubric 给每项分数、理由、来源指针与置信度。程序之后负责覆盖率、归一化、成本、汇总与人工审查所需的原件索引。
