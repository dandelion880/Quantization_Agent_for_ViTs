---
name: allocate-ptq-precision
description: >-
  各层/token/窗口敏感度不均，固定预算下需要精度分配，或动态分组/格式选择的收益与成本不清楚。
---

## 在真实预算下研究静态或动态分配

**触发。** 各层/token/窗口敏感度不均，固定预算下需要精度分配，或动态分组/格式选择的收益与成本不清楚。

**前提与范围。** 当前 profile 允许相应自由度，并定义共同任务目标和资源度量。固定主线下可使用敏感度作诊断，但不能暗中提高某些边的 bit。

**步骤。**

1. 定义决策变量：离线 layer bit、格式、固定组，或输入相关 scale/group/bit 路由。动态 scale、动态分组和动态 bit 是不同算法。
2. 取得与本问题匹配的质量—成本响应，记录敏感度所用父状态、数据和准备成本；方法中的全数据统计不能藏在少量校准标签后。
3. 选适当求解器，如贪心、ILP 或 Pareto；保留原算法和约束。预算用实际权重/激活规模、元数据、例外、分支和必要重排计算。
4. 每次更新后重算真实预算及完整配置。LampQ 论文的异规模层一加一减不必然保持参数量加权预算，不能仅据文字声明通过；自行修复应标变体。
5. 对动态策略先区分输入依赖价值与更多组数/更高平均精度，再计入运行统计、路由和 packing。缺 kernel 时先保留算法证据，部署收益另测。

**结果解释与下一步。** 静态分组已稳定有效，研究更简单规则；跨输入敏感位置变化大，才进一步研究动态；平均 bit 看似相同但真实成本不同，先修比较口径；理论 BOPs 降低却无速度，转部署路径。

**案例草案。** window-based 模型上，比较 MixA-Q 独立 PTQ 分支与相同资源条件的统一激活精度。窗口路由不能直接移到全局 attention 后仍沿用原法证据；若需要迁移，先定义分组与共享权重规则。

**复用边界。** AMP/MPTQ 共用家族，不能作为两份独立验证；TokenDynamic/OptimalFormat 原配方尚未闭合，不能启用“已复现”执行；向量码本索引 bit 不等于 scalar WbAb。混精度与动态原法保持各自 profile。

**来源。** `04_RESEARCH_WORKFLOWS.md` §7、10；[AMP](../../methods/recipes/amp-vit.md#amp-vit)、[LampQ](../../methods/recipes/lampq.md#lampq)、[MixA-Q PTQ](../../methods/recipes/mixa-q.md#mixa-q)、[IGQ](../../methods/recipes/igq_vit.md#igq_vit)；`research_corrections_coverage.md` 的预算口径。
