# QAViT

QAViT（Quantization Agent for ViTs）是面向视觉 Transformer 量化研究的多智能体框架，结合假设生成、讨论与演化、量化实验和记忆反馈，支持可追溯的迭代研究。

LLM 负责科研推理，程序负责数值工具执行、预算与状态管理、证据归档。工具覆盖量化行为分析、校准、量化、评测及模型导出与重载；原始测量与科研解释分别保存。

- `QAViT/`：核心代码、角色提示词、工具、配置、测试和文档。
- `EXP/`：实验配置、运行记录与结果；Git 仅跟踪明确列入白名单的轻量报告和摘要。
- `reference/`：本地论文与参考资料，不随仓库分发。

使用前，请按[环境说明](QAViT/docs/ENVIRONMENT.md)准备依赖、模型服务登录和实验资产。运行依赖安装目录、模型权重、数据集及完整数据清单不包含在 Git 仓库中。

将[研究配置模板](QAViT/configs/research.json)复制到新的 `EXP/` 实验目录，填写研究目标、模型与数据、实验协议、资源和预算；模板不能直接运行。以下以新的 `my_research` 实验为例，从项目根目录执行：

```bash
python -B QAViT/run.py --help
python -B QAViT/run.py start --config EXP/my_research/input.json --name my_research
python -B QAViT/run.py status EXP/runs/my_research
python -B QAViT/run.py report EXP/runs/my_research
```

进一步阅读：[实现与开发入口](QAViT/README.md) · [科研资源导航](QAViT/knowledge/INDEX.md) · [目录与模块说明](STRUCTURE.md) · [历史验证导航](QAViT/docs/HISTORY.md) · [当前伪代码](EXP/validation/knowledge_resources_20261108_r1/PSEUDOCODE.md)。
