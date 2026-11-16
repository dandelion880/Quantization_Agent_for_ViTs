---
name: diagnose-ptq-failure
description: >-
  当 ViT PTQ 的整体精度或局部误差反常、不同量化位点或方案结果冲突，
  或连续方案集中于同一位点而缺少全局依据时使用。从完整冻结方案区分局部新增、
  上游累积与下游传播，以冻结干预寻找可改善位置和校准背景。已知只是工程报错时不必使用。
---

# 定位量化失效

**核心成果**：从可追溯异常提炼一个有证据支持的主机制、一个有意义的替代解释，以及最值得进一步观测的量化边或误差传播位置。允许“不足以定位”。

## 快速判断

1. 从当前任务与历史 Evidence 还原同条件症状：模型、任务、W/A 位宽、目标量化位置、校准/搜索/独立集、比较对象、FP/量化输入前缀；不把跨数据/不同协议数据当成配对对照。
2. **可疑实现或评测**：如果 FP 基线、算子等价性、位宽/码点、静态冻结、覆盖或样本身份存在具体疑点，查看 [fidelity-triage.md](references/fidelity-triage.md)。已验证可信的基线不必再从头审计。
3. **不清楚主要误差来源**：如有必要，比较同协议的 FP、W-only、A-only、W+A；优先选能区分 W、A 或交互效应的测量。参见 [quantization-error-map.md](references/quantization-error-map.md)。只看到 W-only 良好，不能直接断言 A 是唯一根因。
4. **已发现特殊分布或局部异常**：依据 token/channel、softmax/GELU/LN、异常值能量和输出传播选择测量。先辨认异常是否实质贡献量化误差，再考虑干预；参见 [distribution-and-propagation.md](references/distribution-and-propagation.md)。
5. **方案反复集中于首层 Q 等单点**：先判断是否缺少完整方案的全局轮廓。可选一组有代表性的 block 边界，或一次低样本全网边界观测；用同输入 FP/Q 配对和冻结干预区分新增误差与上游累积。按需读取 [frozen-propagation.md](references/frozen-propagation.md)，它包含可直接交给 Coding 的现有 custom API 示例。最大局部 MSE 不是根因排序，FP 旁路空间也不等于可实现的 W4A4 收益。
6. 查已有结果能否排除某些解释。若两种机制仍能解释数据，优先转用 `design-discriminative-experiment` 制定一个小干预，而不是无条件跑更多直方图。

## 工具和知识的选择

- 位置不明：`query_structure`、`query_ptq_sites`、`profile_structure`。
- 分布异常：`profile_distribution`、`profile_plugin_sites`；注意 token 类别与 channel 粒度。
- 简单 W/A 线索：`profile_sensitivity`；其 RTN/QDQ 局部代理不等于复杂候选的真实任务表现。
- 当前/历史完整方案：通过顶层 `evidence_refs` 或 `solution_refs` 声明来源，使用 `context["load_frozen"]` 加载，并调用 `ptq_agent.execution.propagation.profile_frozen_propagation`。返回来源回执、真实样本、分解交叉项、CLS/patch 与干预传播；加载失败只属工程未执行。
- 已有候选或冻结 trial：`profile_trial_diagnostics`、`profile_trial_ablation`；根据 Evidence 的真实上下文解释旁路结果。
- 先按需读取相关 Evidence/Memory、方法手册，而非全面扫描。缺少工具时可给已有 Coding/执行系统提出最小科学测量目标，不需要新工具 API。

**关键边界**：分布统计是关联证据，不是因果；有意旁路某量化器是局部诊断，不是完整 static W4A4；结构信息不是精度证据。ResearchBrief 的合法合同由已有系统负责。

**典型情境**：W4A32 接近 FP、W4A4 明显退化且尚无同协议 A-only/交互证据。选择相关对照与真实量化前缀诊断，不直接把“激活异常值”写成结论。

**不触发**：已定位 `Coding` 的 shape 错误，正在执行既定修复。

**频率与范围**：全网轮廓不是每轮门禁；已有足够证据时只测未决边界。位点和校准策略由 Generation 依据不确定性与反例选择，不强制换位点、枚举指标或耗尽固定次数。H1/H2 继续改良已有完整方案，H3 可探索新的机制，独立 B 不占研究槽。

**退出**：下一项研究决策已经清楚、实现问题已归类，或证据不足且继续 profiling 不值成本。自然语言给出症状、候选机制、证据边界与下步建议；不新增固定输出格式。
