# 数值与工具库

此目录只包含 QAViT V2 使用的数值实现和工具支持；`ptq_agent` 是运行期数值导入命名空间。没有 Agent、Orchestrator、旧 Memory、LLM provider 或选优工作流。

先读 `profiling/host.py`（纯工具会话）、`registry.py`（协议/缓存）、`execution/profile_tools.py`（完整初始化），再定位具体 `execution/` 算法。
`QuantizationExecutor` 实现量化、审计、导出和重载；`RTNExecutor` 保留基础 RTN 及数值计划 v2/v3 加载。它们不作科研决定。
`configuration.py` 仅解析数值后端和资产，角色模型设置在 `configs/models.json`。

量化和评测公式保持原实现。`quantization/integer.py` 的 CPU NumPy QDQ 仍供局部 Profiling 使用；测试用小 ViT 在 `QAViT/tests/fixtures.py`。
本库保留了调用链需要的计划展开、分组、证据分页、独立任务程序和模型位点绑定。被删除的是旧科研决策框架，不是这些工具能力。

原始方法来源与适用限制见 `QAViT/docs/methods/`；第三方模型源码、运行依赖和许可证见 `QAViT/docs/ENVIRONMENT.md`。本项目未为没有授权记录的来源自行添加许可证声明。
