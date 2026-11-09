# 与真实 QAViT 的离线接入约定

本文件是给 Codex 的对接说明，**不授权修改现有科研阶段、研究 Prompt、原有测试合同、自动续调、预算或资料读取边界**。实现应以接入时源码为准做最小、可恢复、完整的独立评测扩展，不以本说明猜测生产逻辑。

## 1. 目录与隔离

建议：

- `QAViT/docs/evaluation/`：从本包复制 `PRINCIPLES.md`、`rubrics/*.json`、`prompts/*.md`、`contracts/*.json`、来源与接入说明。作为评分权威静态资源，不再在其他目录复制正文。
- `QAViT/tools/evaluation/`：只读数据归并、指标统计、独立 LLM Judge 客户端、分数验证与报告的离线程序及独立 CLI；生产流程不 import 也不调度它。
- `QAViT/tests/evaluation/`：评价模块的测试和无需 GPU 的局部夹具；不要新增生产门禁。
- `EXP/evaluations/<eval_id>/`：独立结果（原任务 ID、评分原始输出、聚合分、成本、错误/缺失标记、报告）。历史科研 EXP 不变。

若项目当前已有更合适的同职责离线目录，以最新 AGENTS.md/导航原则调整，保持“执行代码、静态知识、测试、运行产物”分离，勿为目录形式重构核心。未来如需新增评价资源导航，更新相应入口，但**不能让评分 Rubric 进入被评价者的科研常驻上下文**。

## 2. 与实际字段的对接

当前相关实现：

| 真实来源 | 用途与注意事项 |
|---|---|
| `agents/tasks.py` | purpose→role 的唯一任务表，判断适用 Rubric |
| `runtime/task_queue.py` | `task_id/key/kind/inputs.purpose/state/attempt/step/parent_task/dependencies/outputs`、等待与重试；`kind=llm` 才是角色 LLM 任务 |
| `runtime/model_router.py` | `jobs/<execution_id>/llm_audit.json` 的 request、requested/actual、response、elapsed_seconds、usage、error 等 |
| `runtime/task_worker.py` / `runtime/task_runtime.py` | `request.json`、`result.json`、故障和资源回执；各模型失败尝试仍可计量 |
| `core/journal.py` / `core/cards.py` | 不可变 TaskCard、Hypothesis/Review/Metareview/Experiment/Evidence 源卡片及引用/事件 |
| `agents/interaction.py` | 实际交付上下文与工具行动、读卡/读资产、有效提交/自动绑定；不能凭菜单权限假定模型实际读过文档 |
| `core/workflow.py` | H1/H2/H3、三场独立对辩、H*、H4–H*、final、rank 的程序自动提交路径 |
| `core/experiment_requests.py` | 原始实验要求、Code→Safeguard→Execution 关系，修订、缓存与 Evidence 回传 |
| `tests/audit_run.py` | 已有只读运行审计的统计基础，可复用其成功模式而不把结构校验当科研评分 |

原子调用通过 `execution_id` 与任务 `task_id` 关联，任务内部多次调用、`step`/`attempt` 以及工具子任务需按因果依赖串联。直接按角色扫 `llm_audit` 而不读取 task purpose，会把不同 Judger 任务与失败重试混合。`role=supervisor` 也可能只是 **experiment_request 的程序身份**，不代表 supervisor LLM。

任务的科学质量应基于**实际提交**（HypothesisCard、ReviewDecisionCard、MetareviewCard、完整 code artifact、plan_l3 草案/defer）和本任务可见轨迹。尚未提交但含有有价值的部分行动可以分项评价；因宿主中断导致证据不足用 unknown。不得把完整运行 L3 结果重复归功于每个参与 LLM。

## 3. 可见上下文与资料权限

为每个评价样本保存至少：

1. 冻结身份：`run_id / task_id / purpose / role / slot / selection_stage / exchange`（存在时）；生成时的源码/Prompt/研究协议身份和原始输入版本。
2. 该角色实际收到的系统/任务 Prompt 与科研上下文，以及其后每个 step 得到的工具反馈和可见材料。只收集**该角色在当前时间点被授权并实际交付的资料**。
3. 实际输出/提交的不可变卡片 ID、代码 artifact、候选版本与实验/证据 ID；每次回应/工具批次及失败回执。
4. 源地址索引 `source_id → immutable_ref + locator + availability`。如因规模原因摘要化，记录未读/未传/截断范围，防止评价器将目录或摘要误认原文。
5. 每条关键科学证据的类型（论文理论、统计原件、实验方案、真实/合成/test_double 结果）、数据/模型/位点/指标/分母/执行状态；程序可从原卡重建而非信任自由文本。

同一 exchange 的 `position` A/B 在生成时不能读取彼此未汇合立场。Eval-L0 回顾时可以验证双方提交，但不能给某一方因未响应当时不可见的对方发言而扣分。H4 只能拥有当时已持久化的 H* 与可见反馈。晚到证据和结果应放到单独的 posthoc 分析通道。

## 4. 先程序提取客观指标，再 LLM Judge

程序指标可直接报告：

- `model_calls_started / completed / failed / retries / repairs`；按 purpose/role/任务/尝试统计，不重复计入一份科研产出。
- `input_tokens / output_tokens / cached_input_tokens / reasoning_output_tokens` 与缺失用量数；保留 CLI 原单位和统计语义。不要把 reasoning output 叠加两次。
- 模型请求的 `requested` 与 `actual`；CLI 没有实际回显时真实模型/effort 为 `unknown`，不以请求值替代。
- 调用历时、预检及锁等候、队列等待、任务墙钟和独立数值等待，能区分的才分别报告；不能把单次 `elapsed_seconds` 当纯推理延迟，也不能将并发耗时简单累加成端到端时长。
- 按工具种类统计调用、失败、结果状态和缓存；**不按使用数多少给科研质量自动加减分**。
- 现有实验的真实完成/失败/中断、校准/评价范围与测量来源、代码审查和修复结果。
- Judge 自身的 token/time/call/失败在独立评价开销账本中报告，不混入被评价 QAViT 的科研成本。

记录 `data_coverage`：缺少哪些调用日志、实测、原始上下文、来源，评价器无法评分哪些 criterion。绝不能以缺失日志=零消耗、缺失数值=零精度、未调用工具=低质量替代。

## 5. 分数验证和报告

独立 Judge 使用 `gpt-6.1-sol / medium` 暂定配置。可以复用经过核验的 Codex CLI 传输/错误处理封装，但必须避免沿用 QAViT 科研 role 配置、生产配额与任务队列的副作用；不在生产 Judger 中新增一种 purpose。评分客户端不得被研究卡片中的指令诱导执行工具、修改文件、访问其它运行或启动实验。

由于初版评分模型可能与某些被评角色同系列，**它不是独立的客观科学真值**。尽量在科研质量评分材料里隐藏被评模型名与推理档位（保留任务目的和科研条件），而在独立指标报告中恢复身份；后续由用户抽样进行人工校准，有明显偏差再改 Judge，而不是立即使用多 Judge 投票掩盖系统误差。

每次评分：

1. 载入唯一适用的 JSON Rubric，组装 Judge System + Case Prompt + 冻结资料；质量 Judge 尽可能不暴露被评价者模型名称。
2. Judge 输出 `judge_result.schema.json`；程序进一步核对 `criterion_id` 集合恰好与该 Rubric 相同、status/score 匹配、所有 `source_id` 确实在来源索引内且 locator 可核查。校验失败只使**评价任务**失败/有限重试/留待人工，不影响科研任务。
3. `scored` 求归一化 Q；`unknown/not_applicable` 分列，输出每项理由、来源证据、置信度和需人工复核的重点。
4. 按 `role × purpose`（Generation 再按 slot）聚合：样本量、得分中位数/均值、有效覆盖、Unknown、错误类型、成本分布、原始案例链接。当前没有人工真值时不声称精确误拒/漏放率。
5. 报告质量与成本的并列关系，不设一个“综合模型战斗力分数”，不使用模糊的硬性通过阈值。

离线源码引用和科研数据可能包含环境路径/账户信息，写评测报告前仅向评测器传递必要字段、做安全脱敏；保留来源映射供授权环境复核。脱敏不能改变科学含义，无法核实则 unknown。

## 6. 后续人工校准与 L1 边界

当前不开发人工抽样工具，不自动生成“代表性测试任务”或固定金标准。可预留人工复核可读的 `case_id/criterion_id/source pointer/理由/置信度`，用户以后自己决定样本及专家分数。

后续 Eval-L1 才用冻结上下文的配对 A/B、多次回放尝试衡量不同 Prompt/模型的独立贡献；Eval-L2 才有额外局部数值验证；Eval-L3 才比较完整科研收益。**Eval-L0 的离线分数不是因果实验**。
