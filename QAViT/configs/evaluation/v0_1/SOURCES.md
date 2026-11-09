# 主要参考来源

以下为 Anthropic 原始来源；QAViT 的具体 `0/1/2` 锚点和角色映射是本包依据真实源码设计的**本项目方案**，不是 Anthropic 官方标准表。

1. Anthropic, **Demystifying evals for AI agents**：https://www.anthropic.com/research/demystifying-evals-for-ai-agents
   - task/trial/grader/transcript 的区分；程序、模型、人类三类 grader；研究 Agent 的证据扎根；逐项部分得分；Unknown；专家校准；避免过于刻板地检查固定行动路径；检查完整轨迹与评价器本身。
2. Anthropic, **How we built our multi-agent research system**：https://www.anthropic.com/research/multi-agent-research-system
   - 多 Agent 有不同有效研究路径；以科学结果与合理过程评估；早期少量真实样本；Rubric-based LLM Judge；人工审查；观测性与工具效率；大模型评分的一致性必须实测。
3. Anthropic, **Writing effective tools for AI agents—using AI agents**：https://www.anthropic.com/research/writing-tools-for-agents
   - 工具与测试案例应对应真实工作场景；记录调用次数、延迟、Token 和工具错误；避免验证器对格式或固定策略过拟合。

本包另依照用户 2026-11-08 至 2026-11-09 对 QAViT 五维评价与 Eval-L0–L3 的讨论；当前真实文件来源为 `QAViT(7).zip`（`agents/tasks.py`、全部角色与任务 Prompt、`core/workflow.py`、`core/experiment_requests.py`、`core/cards.py`、`runtime/model_router.py`、`runtime/task_queue.py`、`tests/audit_run.py`）和 `knowledge_resources_20261108_r1.zip`（`REPORT.md`、`PROJECT_RESOURCE_INVENTORY.md`、`PSEUDOCODE.md`）。
