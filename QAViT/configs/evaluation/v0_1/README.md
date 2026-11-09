# QAViT Eval-L0 · LLM Rubrics v0.1

**状态：待接入的独立评分规范；不包含评价程序、测试任务、测试样本或科研源码修改。**

适配依据：2026-11-09 提供的 `QAViT(7).zip`，以及 `knowledge_resources_20261108_r1.zip` 中的资源整理报告。评分目标是评估 *LLM 在当时信息与授权条件下的科研决策质量*，并附带客观运行指标；不是自动淘汰候选、审核新的研究请求或单凭文本分数宣称量化算法有效。

## 目录

| 文件 | 用途 |
|---|---|
| `PRINCIPLES.md` | 总体原则、评分口径、可靠性边界、定量汇总 |
| `rubrics/generation.json` | Generation：H1/H2/H3 机制、可证伪、实验与经验利用 |
| `rubrics/judger.json` | Judger：screen/position/arbitrate/rank/plan_l3/interpret 六个独立子 Rubric |
| `rubrics/evolution.json` | Evolution：从唯一 H* 及反馈形成 H4 |
| `rubrics/recorder.json` | Recorder：事实、证据范围、知识修订和下一轮指导 |
| `rubrics/coding.json` | Coding：单实现的科学忠实性与可执行性 |
| `rubrics/safeguard.json` | Safeguard：关键审查、误拒、漏放、可操作反馈 |
| `prompts/judge_system.md` | 独立 LLM Judge 系统指令 |
| `prompts/judge_case.md` | 单个角色任务的评分输入模板（不是测试任务生成器） |
| `contracts/judge_result.schema.json` | Judge 输出结构；离线校验，不触及研究提交合同 |
| `INTEGRATION.md` | 真实数据对接、模型路由、来源与计量规则 |
| `VALIDATION.md` | Codex 接入后应做的离线测试清单，不包含预制科研案例 |
| `CODEX_PROMPT.md` | 用户可以直接交给 Codex 的实施提示词 |
| `SOURCES.md` | Anthropic 官方来源及采用的原则 |

## 与现有任务的精确对应

源码 `agents/tasks.py` 中共有 12 个 purpose，映射到七种角色名。其中 **六类目前承担常规 LLM 任务**：

| role | purpose | 是否纳入本版 |
|---|---|---|
| generation | generate | 是；区分 H1/H2/H3 |
| judger | screen、position、arbitrate、rank、plan_l3、interpret | 是；仅对真正有 LLM 调用的任务评分，interpret 按历史存在情况评价 |
| evolution | evolve | 是；H4 必须绑定本轮已持久化 H* |
| recorder | record | 是 |
| coding | code | 是；只在需要新实现时出现 |
| safeguard | safeguard | 是；只在需要审查新实现时出现 |
| supervisor | supervise | **当前正常工作流以程序调度为主**；仅保留未来 LLM 评估扩展位，本版不评分。`experiment_request` 即使 `role=supervisor` 也不等于 supervisor LLM |

`rank` 可由程序按确定的直接胜负表自动提交；`author=program` / `selection_source=existing_verdicts_and_screening` 时不作为 Judger LLM 产出。`interpret` 不是现行普通实验回传的必要阶段，不能因缺席计零分。

## 使用方式

1. 评价程序只读历史的科研任务、调用轨迹、卡片和执行回执，并确定 `role`、`purpose`、冻结时点和可用证据。
2. 程序选择恰好一个匹配的 Rubric，原样注入 `judge_system.md` 和 `judge_case.md` 所述评价上下文；建议初始评分模型 `gpt-6.1-sol / medium`，与生产 Judger **分离**。
3. LLM Judge 只给每项 `0/1/2` 或 `not_applicable/unknown`、理由、原始来源指针、置信度、重要风险标记；**分数归一化和成本统计由程序完成**。
4. 对科研质量和成本分别报告，不合并为一个带权收益分。不对未见的研究任务、未验证的论文结论或缺失的 GPU 评测填充结果。
5. 人类专家将来自行选取案例校准，本包**不提供抽样接口或案例生成程序**。

请先阅读 `PRINCIPLES.md` 和 `INTEGRATION.md`，不可只复制 Rubric 名称而忽略证据边界。所有文件均为离线评价资产，不应注入 Generation/Judger 等被评价者的正常科研提示词。
