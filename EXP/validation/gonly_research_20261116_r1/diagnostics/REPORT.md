# 冻结方案全局传播诊断与 Skill 初版

新增 `ptq_agent.execution.propagation.profile_frozen_propagation`，经现有 custom 实验路径使用。模型由既有 `context['load_frozen']` 加载并核验；本模块不另开路径猜测或重校准入口。支持指定任意数量的实际 Transformer tensor→tensor block，配对捕获 FP/Q 输入和出口，重放 FP block 于实际 Q 输入，分开统计局部新增、上游映射和出口总误差。保留 signed cross term，避免错误相加 MSE。可明确 CLS/patch 布局，保留命名任务表征，选用真实标签分类 KL/交叉熵/Top1。

局部干预支持冻结 FP block 替换和精确 W/A 量化边旁路；其他数值状态保持冻结，比较下游边界与最终输出，成功或异常都清理 hook/恢复开关。FP 替换是诊断恢复空间，不是可实现的 static W4A4 收益。仅允许 calibration/search，真实数据视图、ID、count 和状态哈希留存。任意复杂 task head 可由已审核的 forward_fn 执行，本模块没有自动 ADE/head 支持。

增强已有 diagnose-ptq-failure Skill，新增 frozen-propagation.md 参考，已生成 Skill 目录、同步 knowledge 导航和接口链接；Coding/Safeguard 可读公开源码。没有增加阶段、实验次数、全网扫描或换位点门禁。

## 验证

`python -B -m unittest QAViT.tests.test_frozen_propagation -v`：6 项覆盖实际 tiny ViT AdaLog W4A4 导出与独立重载、分解代数、下游干预、零误差对照、异常清理、final 拒绝、宿主真实样本观察，以及 Generation 目录发现/read_asset 正文读取。最终回执见 validation.log。

另跑数值公开源码与知识模块回归，共 18 项中 17 项通过，1 项在并行开发期间被源码身份保护拒绝恢复，保留 regression.log；该检查没有弱化，需要父任务固定源码后的集成重跑。最初读取测试误用了大于 schema 上限的 limit，按真实 12000 上限修正；未扩大工具限制。Skill quick_validate 与 audit_knowledge 均通过。

小规模数值入口 `replay.py`：seed23、2 张 synthetic search 图像、32×32、24维、2块 ViT，AdaLog frozen 重载。真实 PyTorch 运算，明确不是 ImageNet/ADE 或科学泛化证据。结果在 numerical_result.json；摘要 summary.json。

| 边界 | 本块新增 MSE | 上游映射 MSE | 出口总 MSE |
|---|---:|---:|---:|
| blocks.0 | 1.1018e-5 | 3.7376e-3 | 3.7198e-3 |
| blocks.1 | 1.0660e-5 | 3.7095e-3 | 3.7717e-3 |

基准 logit MSE 3.9859e-4；FP 替换 block0/1 分别降低 2.6231e-4 / 1.9711e-4，首层 Q 旁路降低量为 **0**（有效负结果）。出口误差包含上游累积和抵消，不能据最大总 MSE 直接归因；不同干预对下游的边际影响可为下一步测量选点提供依据。上述结果只验证诊断能力，不替真实模型选择研究位点。

## 边界

未验证真实模型的研究收益、真实 Generation 采用 Skill 后的科学决策效果、GPU 路径或 ADE paired head 表征。当前 block replay 要求一次前向每边界恰访问一次，单 tensor 输入/输出且残差坐标对齐；其它结构通过 custom 明确适配。未启动正式贯通实验。
