# Eval-L0 接入与验证交付

已独立接入用户 v0.1 评分包。生产角色、Prompt、研究流程、实验合同、预算/GPU、Memory 和恢复逻辑均未改。相对本轮开始时的 252 份既有 QAViT 文件，只有 README 导航发生修改；用户此前未提交修改原样保留。所有规范、运行、评分及校验来源见 [结构化验收](validation_summary.json)。

## 实现

- 15 份上传原文逐字节保存到 `QAViT/configs/evaluation/v0_1/`，含六类、11 purpose、43 项评分标准。这里避开生产 docs/knowledge 自动授权范围；`provenance.json` 保留 ZIP 和各文件 SHA256，规则正文没有维护第二份。
- `QAViT/tools/evaluation/` 提供独立 rules / prepare / judge / report / verify。直接读取提交链、卡片、全部调用和回执，冻结原字节，按任务归并 step/attempt/修复，保留 parent_task、origin_task、previous_request_task、依赖、产物和证据。
- 只以实际发送请求构造当时可见资料；不补读目录中的原文，不混入未来结果或私有对方立场。自己的原响应、宿主绑定提交和完整源码分列；既有实验观察独立列为 posthoc。
- 程序统计 Token/缓存/推理子字段、调用/重试、工具请求与执行、状态区间/数值等待及成本缺口；科学 Q、有效证据覆盖、Unknown/N/A、理由/来源/置信度独立呈现。真实模型未回显保持 null，缺用量不记零。
- Judge 固定请求 `gpt-6.1-sol / medium`，独立锁/超时/尝试上限和费用账本。复用已核验纯订阅传输，隔离 worker 不初始化生产 runtime/paths，不使用科研预算、队列、工具、GPU 或生产订阅锁。供应商账户额度仍可能共享。
- `QAViT/tests/evaluation/` 是 CPU 接口/故障夹具；没有专家抽样、科研样本生成器或新科研门禁。操作文档见 [EVAL_L0.md](../../../QAViT/docs/EVAL_L0.md)，流程见 [PSEUDOCODE.md](PSEUDOCODE.md)。

## 实际测试

| 检查 | 实际结果 | 原始记录 |
|---|---|---|
| 评分模块 CPU 测试 | 52 项通过，6.073s | [unit_delivery.log](unit_delivery.log) |
| 冻结源码后的全仓 CPU 回归 | 601 项通过，730.857s | [regression_frozen_final.log](regression_frozen_final.log) |
| 源码身份保护失败项定向复跑 | 1 项通过 | [regression_targeted_final.log](regression_targeted_final.log) |
| 目录、语法和递归源码审计 | passed，177 Python 文件，281 份身份文件 | [structure_audit.json](structure_audit.json) |
| 原历史保护 | 11,217 份读取文件 SHA256 均未变 | [isolation_audit.json](isolation_audit.json) |
| Judge 幂等恢复 | 再次执行返回 already_validated，无新增调用 | [judge_idempotence.log](judge_idempotence.log) |
| 真实超长输入边界 | 934,544 字符，超过默认 600,000；未截断、未调用、未评分 | [long_input_boundary.log](long_input_boundary.log) |

CPU 测试覆盖：六角色/11 purpose/三个 Generation slot、parent/final rank、程序身份排除、H4 冻结父方案、合法未决/defer、Schema/ID/状态/分值、伪造引用/错误 locator/摘录、Unknown/N/A 分母、完整代码/缺失代码、同任务多次调用/修复/等待/子任务、L3 origin_task、缺日志与失联、活动请求版本绑定、真实/synthetic/test_double/缓存、Token 不重算、定价缺口、并发区间、来源改写、提示注入数据边界、工具禁用与环境/导入隔离、限次失败/额度停止/中断恢复、原件及报告不可覆盖。

首轮全仓 599 项有 1 项在新增评价文件期间触发既有源码身份变化保护；未改恢复规则。冻结全部 QAViT 文件后定向和完整复跑通过。初期模板占位符的全角冒号问题，以及 Message 数据类补入 tool_call_id=null 的历史序列化差异均已修正；早期失败日志/冻结目录保留，不冒充最终验证。

## 真实历史与 Judge 烟测

最新完整提取：[real_history_20261109_r4](../real_history_20261109_r4/manifest.json)，来自已有 `deitb_training_free_static_w4a4_20261106_r1`。105 个角色任务、362 次 LLM 执行尝试、359 次确认启动、352 份完整响应；10 次尝试缺 Token 用量。原科研已知输入 21,205,762 / 输出 342,765 Token，缓存输入 1,384,576、推理输出 94,429 均为独立子字段，不再次相加。18 个既有实验任务为 8 completed / 10 failed，8 个既有 Evidence；本轮未启动数值实验。完整工具请求、任务/证据关系见 [run.json](../real_history_20261109_r4/run.json)，[当前报告](../real_history_20261109_r4/reports/report-0002.md)。

| 角色 | 真实任务数 |
|---|---:|
| generation | 6 |
| judger | 38 |
| coding | 34 |
| safeguard | 25 |
| evolution | 1 |
| recorder | 1 |

真实评分仅进行以下两例，各一次调用成功，CLI 0.167.0 实际确认工具 host 关闭，Schema/全部 criterion/来源指针/摘录验证通过。每个案例保留在评分当时的冻结版本目录，未把分数搬到不同 case_sha256 上。初版 Judge 评分不等于专家科学真值。

| purpose | 冻结版本 / case | Judge Q | scored/applicable/total | 输入 / 输出 Token | 累计 elapsed 秒 |
|---|---|---:|---:|---:|---:|
| screen | real_history_20261109_r2 / case-74f9e55addd6a45fd15adf68 | 100.0 | 3/3/3 | 50002 / 721 | 21.850 |
| code | real_history_20261109_r3 / case-868026e14ff22591faeef0bd | 90.0 | 5/5/5 | 50958 / 1424 | 37.432 |

[screen 逐项报告](../real_history_20261109_r2/reports/report-0002.md)；[Coding 逐项报告](../real_history_20261109_r3/reports/report-0003.md)。两次请求均为 gpt-6.1-sol / medium，实际 model/effort 均未由 CLI 回显，保留 Unknown；货币成本也未知。其余案例没有伪造评分；r4 的 plan_l3 在保留完整资料后明确报告 input_limit。

## 使用与边界

```bash
python -B QAViT/tools/evaluation rules
python -B QAViT/tools/evaluation prepare EXP/runs/<已有运行> --output EXP/evaluations/<新eval_id>
python -B QAViT/tools/evaluation judge EXP/evaluations/<eval_id> --case <case_id或task_id>
python -B QAViT/tools/evaluation report EXP/evaluations/<eval_id>
python -B QAViT/tools/evaluation verify EXP/evaluations/<eval_id>
```

真实 smoke 只覆盖 screen 和一次 Coding 修订；其他 purpose 的真实 Judge 语义尚未校准，interpret 在本历史中不存在。完整上下文超窗、独立数值正确性、人工金标准、误拒/漏放率、模型/Prompt 因果贡献与代表性科研收益均未验证。人工抽样、任务集和后续 Eval-L1/L2/L3 留待用户安排。源码新增继续纳入原递归身份，旧科研运行恢复仍需原显式干预，评价入口不自动续跑。

原 ZIP 状态：**deleted**。原始 SHA256：`dd85ee57de5b84abfafffd6c31742884db1a5f2c8e72beea9816ab085fe52329`。最终删除回执见 [archive_removal.json](archive_removal.json)；只删除本次上传 ZIP，接入资源、原科研文件、全部测试/评分产物保留。
