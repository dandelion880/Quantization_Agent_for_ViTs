# 正式完结后的 B 时点交付修复

本修复在三轮正式实验结束后进行，不是正式运行 intervention，不计入正式 Agent 预算或科研产出。正式 run、Memory、summary、manifest 与已冻结源码快照均未回写；没有恢复 run、额外调用 QAViT 模型或启动数值实验；外部 Codex 工程会话单独计入外部 Token 账。当前工作源码已经不同于正式冻结 core `6f5db3ec908f9e44ae2616b1e2f8ad6a8d2468b3f7ceffbcbe5c1f4a87ad2dec`，不得声称正式结果由修复后源码产生。

根因是有效 L3 在 Recorder 前已按原规则更新方案库；原 `fallback_B` 正确返回当前保留结果，但缺少晋升前时点，Recorder R2/R3 将其与新候选同一身份误解为“候选与先前 B 相同”。原历史 Memory 中的错误表述仍保留，没有将事后修复冒充当时理解。

最小修改仅补只读交付和提示：

- `context.B_transition.before_l3` 根据确切 selection_ref 找到产生该决定的预选任务，以任务首次持久提交 sequence 重建方案库和 incumbent；兼容无任务的历史决定卡、缺决定的 round_started，并明确缺少可核实锚点的 unknown。
- `after_l3` 固定到原 recording_batch.through_sequence，附方案库事件来源和本轮实际 solution_considerations。恢复、延期复盘及预算收尾不会把后轮 B 混成本批 B。原 `fallback_B` 保持当前读取快照含义。
- Recorder 提示明确候选可能已晋升，相同的候选/晋升后 B 身份不能推出没有新候选；同时保留历史比较不等于同期受控因果实验的限制。未修改排名、科学选择、评测协议或 Memory 更新规则。

最终专项回归 **20 项通过，15.439 秒**，包含 4 项新增时点测试、原 B 保留/来源回归、Recorder 收尾调度及提示词基线。覆盖首次无有效 B、晋升、恢复后出现下一轮 B、失败/不完整 L3、预算收尾、无决定/无锚点及无 L3。初次 42 项也通过，其中 imported TestCase 被 unittest 重复发现；修正测试导入后使用 20 项最终集合，不合并计数。原日志均保留。

只读历史核验见 `historical_context_check.json`：正式 Journal sequence **9149**、原提交 head 保持；14,204 个原运行文件的大小与 mtime 全清单无变化，research_run/summary/checkpoint/status 内容 hash 无变化。重建的实际身份如下：

| 轮次 | 预选前 B 快照 | 本批晋升后 B 快照 | 不同模型身份 |
|---|---:|---:|---|
| R1 | sequence 602 | sequence 4477 | 是 |
| R2 | sequence 5318 | sequence 6829 | 是 |
| R3 | sequence 7697 | sequence 9139 | 是 |

R2 的 `14e4fd… → c1c5ac…`、R3 的 `c1c5ac… → 6e5761…` 与独立 frozen payload 比较及科学 Memory 诊断一致。完整 model_ref、Evidence、指标及来源保存在 JSON 中。这是工程时点恢复和提示交付验证，没有重新调用 Recorder，不能声称历史 L2 已修正或新提示词的真实模型效果已验证。
