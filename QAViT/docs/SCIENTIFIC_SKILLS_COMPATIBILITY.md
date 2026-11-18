# 科研 Skill 与实际执行能力

按 2026-11-18 当前源码核对。此页是按需接口说明，不是新流程或授权。包内研究建议可以超出现有实现；接口名相似不代表数学对象相同。来源与接入记录见 [Skill 说明](SCIENTIFIC_SKILLS.md)。

## 研究参考与最终方法

沿现有 ResearchBrief、实验 `question/controls`、模型输入 `execution.training`（宿主规范化后为 `execution.resolved_choices.training`；后者仅为历史兼容输入与内部冻结字段） 和 Coding→Safeguard→执行→Evidence 链表达目的和自由度，不增加专用角色或必填研究阶段。

- `hard_constraints.training=false` 或 `training_allowed=false` 禁止整个运行中的优化器更新，不能由 Skill、实验声明或“仅研究”文字豁免。只读梯度本身不被训练 guard 禁止；若 Brief 要求 forward-only，仍须遵守更严格语义。
- 只有 Brief 已明确允许参考优化时，研究实验才可声明 `training=true`。例如新 Brief 的 `hard_constraints.training_allowed=true` 表示允许而非要求训练；最终 training-free 目标、允许更新的 scale-only 等范围在 Brief 的目标/搜索空间/评测协议正文写清。不能把 `hard_constraints.training=true` 当成“可选许可”，它表示每次执行都要求 true。**不修改已有冻结 Brief。**
- 原始请求用 `controls` 写明参考用途、Q0/FP 来源、可更新对象、固定的权重/head、数据和隔离要求。`custom_experiment/experiment_v1` 可提供实现；没有现成 STE oracle API。宿主提供副本 `candidate_model`、只读原模型 `reference_model`，已有 `load_frozen(ref)` 按绑定来源另行加载。
- 参考结果在 `observations`/产物中标为 oracle-assisted，并记录优化前来源、实际优化对象和过程。宿主已有 `provenance.training_declared`、`gradient_audit`、`baseline_state_sha256`、`model_unchanged`、`execution_facts.frozen_loads`；这些分别证明声明、观测到的优化器调用和源状态，**不能单靠 optimizer_steps=0 证明独立 training-free**，也不替代对手写更新的源码审查。
- 最终构造另行声明 `training=false`，从参考优化前状态按规则和许可 calibration 数据生成 scale、zero、组成员及候选选择，不读取 oracle 的实例参数、成员或候选标签。规则可以受参考启发，参考实例答案不能成为隐藏输入。已有请求来源绑定、Coding/Safeguard 和执行回执用于复核，不新增自动“独立性裁判”；当前宿主没有通用语义证明器。
- 最终完整包沿原导出/重载和 L3 协议，ADE 迁移同一冻结状态，不重新拟合量化器或 head。引用优化参考的测量不自动赋予其进入最终包的资格。

## 已有能力及边界

| 能力 | 实际支持 | 不能据此假定 |
|---|---|---|
| 原生校准调度 | `plan.execution.calibration_unit=operator/block`、层序、多 pass、`calibration_context=current/float` | block 调度仍按位点拟合，不自动联合重建 |
| `search_frozen_scales` | native v3 冻结模型、显式候选；每 site/target 一个正标量倍率，相对本次调用初始状态；整网单 Tensor 或指定 block 输出 | 任意逐元素组、独立成员映射、zero 直接搜索、任意 lookahead、STE；多次调用仍各用自己的初始状态 |
| W 重编码 | RTN/uniform；换尺度后重编码真实权重 | GPTQ/GPTAQ 补偿结果不保证保留；改用 RTN 必须报告方法身份变化 |
| 冻结搜索背景 | `context=quantized/float`；float 关闭干预位点集合外的量化，集合内保留已启用 W/A；选定后恢复完整背景 | 不等于原生 `calibration_context`，也不把同位点未调整的另一目标变成 FP |
| 目标 | mse/cosine；分类 logits 的 teacher_kl、task_loss、top1_error；初始冻结 logits CE 梯度平方 `hessian_diag_ce` | 不是完整 Hessian；原生位点梯度又是另一对象；同起点 teacher KL 梯度可能近零 |
| token | 绑定 block 的 profile、冻结搜索和传播共用 CLS/special/patch 切片，核对实际网格 | 按绑定布局区分额外 token；空组不能用于拟合，任意多流或重排布局需适配 |
| 传播 | 配对 FP/Q 边界、同 Q 输入 FP block 重放、交叉项、冻结替换与 W/A 旁路 | 仅覆盖坐标对齐和实际 forward；FP 替换收益不是合法 W4A4 可达保证，helper 默认观测绑定的分类前向；ADE 冻结评价使用独立任务适配层，自定义 dense 输出需验证其前向 |

参数与状态细节见 [尺度选择](methods/scale_selection.md)、[校准顺序](methods/calibration_order.md)、[冻结传播 API](skills/diagnose-ptq-failure/references/frozen-propagation.md) 和 [代码接口](CODE_INTERFACES.md)。无接口的设计沿原 Coding 扩展，不能把上述建议直接塞入标量倍率参数。相同整数 code 不保证相同解码输出；当前 frozen search 枚举全部显式候选，没有按 code 去重优化。

## 来源、执行及记忆

H1/H2 可从可见 archive、Top-3 或完整待测种子改良，B 独立保留；待测种子不是 B。完整历史方案可跨运行复用，但通用 ResearchMemory 仍基于当前 Journal，不自动注入任意旧 run 的 L2。来源缺失/歧义回送 Generation；实现错误回 Coding；资源/宿主失败保留工程状态，均不算机制反证。

H1/H2/H3 是交错推进的三路任务；当前 driver 对 non-judger LLM 调用互斥，数值工作可按资源并发。Skill 接入不改调度。本轮不做自主 Skill 沉淀。开发回放和 rubric 留在测试/离线评价范围，不成为生产阅读材料、实验配额或提交门禁。
