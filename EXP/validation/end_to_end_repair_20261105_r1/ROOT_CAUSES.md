# 根因与受影响入口（实现前调用链核对）

| 已确认根因 | 当前调用链与同类路径 | 修订边界 |
|---|---|---|
| 续调依赖waiting父研究任务 | workflow.advance(plan_l3完成→L3 request)，return_code_clarifications，wait_children_or_fail；Generation/Evolution/Judger及程序创建请求 | 新的科研续调任务/版本承接结果与修订，不改写已完成记录 |
| 数值完成依赖interpret | progress_request正常/缓存命中→interpret，l3_outcome，driver.summary，run.py摘要 | Evidence成功独立于解释与Recorder；失败解释不得重跑有效测量 |
| 证据交付依赖直接任务refs | interaction.required_refs/deliver；假说、裁决、实验需求引用Evidence，含plan_l3、Coding/Safeguard | 明确科学支持边的有界投影；快照ACL、去重、精确原件保留 |
| 补字段与完整Schema次序 | expand_plan/preflight、check_plan、Coding build_plan、prepare | 单一确定性规范化，显式冲突保留；科学选择不猜 |
| 声明v2/v3而worker固定v3 | quantization_service direct/plugin、export/load、evaluate_artifact；prepare前validate_plan空shape | 复用RTNExecutor适配，捕获后仍验证真实shape；自定义加载边界明确 |
| 未列模块一律误报能力越权 | inspect_source→write_artifact/verify_artifact；quantizer/executor/experiment接口 | 精选纯数值/结构标准库支持；精确拒绝模块/操作，保留隔离 |
| 文档旧路径/状态语义 | methods/README、EXECUTION、CLI、Prompt与回归断言 | 与真实调用关系一致，不按遗留名称恢复门禁 |

本清单区分确定缺陷与能力边界：自定义包通用加载不作承诺；两轮真实LLM与完整正式数据贯通仍未验证。本轮只在已授权小规模配置内运行低成本验证，不自动扩展L3数据或预算。保留上一轮Memory/Recorder实现与历史产物。
