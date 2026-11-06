# 三场初始辩论局部并行化

本轮只调整确定性调度与初始场次自动反馈交付，未重跑昂贵科研贯通实验。基线为 Git `bd9ddab`，开始时工作区干净；已阅读项目 AGENTS.md、当前实现/测试及最近 `controlled_run_20261105_r1` 的报告和根因记录。该历史运行经过 H*、Evolution、父子挑战及 Recorder，最终 L3 正式延期，未把它当作完整数值验证成功。

修改范围：

- `core/workflow.py`：三场每次均可推进，三份正式结案事件全部存在后才选择 H*；同场使用独立资源键。
- `runtime/driver.py`：Judger 三并发、其余角色合计单并发、LLM 总上限及外部 worker 总上限；配对按最近启动时间公平服务；恢复旧未完成配对任务时只更新互斥资源键。
- `runtime/model_router.py`：移除 Judger 全程订阅单锁，复用文件锁提供三个项目级 Judger 槽，其他角色保留原单槽；逐次预检/隔离不变，补槽位及结束时间审计。
- `runtime/configuration.py`、`configs/research.json`：默认总 LLM 并发 4，允许 1–4；实验并发固定 1。历史显式配置 1 继续遵守，不改写旧配置。
- `runtime/task_worker.py`：CPU/GPU 数值实验共用项目级 `numeric.lock`，保留原 GPU 锁、执行合同及哈希检查。
- `agents/interaction.py`：仅初始三场的自动关联反馈排除本轮其它场 position/verdict；按需 Memory、任务提交快照、同回合 A/B 隔离保持原样。H4 挑战仍继承原有父方案反馈。
- `position.md`、`arbitrate.md`：各改一处串行说明；同步两项提示词基线、执行文档及 STRUCTURE。
- 更新原按场测试并新增 `test_parallel_runtime.py`、`test_parallel_requests.py`，扩展反馈边界测试。

默认上限：Judger 3（同配对同时 1）；Coding、Safeguard、Generation、Evolution、Recorder 等其他角色彼此合计 1；LLM 合计 4；真实数值实验 1；含文献的异步 worker 合计 5。LLM 总配置较低时相应收紧。项目文件槽使其它运行也不能突破订阅 3+1 和数值 1 的边界。

每场自动获得 Brief、共同原则/角色/任务提示词、对应两份候选和 Screening，后续回合使用本场持久 refs 与明确采用 Evidence；独立任务/尝试/目录/父子请求键和 call_id 绑定工具反馈。Coding 忙时原 experiment_request 正常入队，仍是单份 Coding→Safeguard→数值执行→Evidence 返回原申请者。不同请求可分阶段流水，等待任务不占用 LLM 槽。没有新增审批、强制读取、记忆冻结、实验次数限制或科研判断规则。

H* 仍按原直接胜负逻辑选择；循环、平局、未决使用原 Judger 综合与不确定性处理。H* 固定之后的 Evolution、H4 唯一父子挑战、L3、Recorder、跨轮流程均沿用原实现。预算仍按全局墙钟/活动段计时；每次尝试独立记录耗时、usage、错误与重试，不叠加并发 worker 耗时。

验证证据：

- 可控延迟测试使用真实 TaskRuntime 子进程，断言三 worker 同时执行区间交集大于 0.4 秒；恢复接收一次、重复 dispatch 无新尝试。
- 真实跨进程锁测试证明订阅槽最大 3 Judger+1 其他角色；跨两个 run 的数值 worker 存在时间重叠但实际 work 区间不重叠。延迟函数替代 LLM/数值计算，属于基础设施模拟验证。
- 三请求集成使用真实宿主队列、脚本化角色和 Evidence：Coding 忙时全部接受；Coding/数值始终单并发，不同阶段流水；相同工具 call_id 在不同任务下仍准确返回原申请者，恢复后不重复任务。
- 六种初始结案排列、不同回合进度、已有两胜仍等待第三份结案、多轮 Evidence/spec/裁决继承、Memory 按需读取及初始/H4 反馈边界均有回归。
- 最小真实 CLI 验证：[REAL_CLI_RESULT.json](REAL_CLI_RESULT.json)。3 个独立持久 worker 分别占用 Judger 0/1/2 槽，同一原订阅、`gpt-6.1-sol/medium`，三次成功，模型请求区间共同重叠 **5.5465 秒**，整个 probe **10.8481 秒**；平台报告 input_tokens=14085、output_tokens=120。每次独立预检，账号为 ChatGPT 订阅，credits=false；无 API、账户或付费切换，全部 probe 进程已停止。这是传输验证，不产生科研 Evidence。

真实 CLI 使用机器上保留且项目已经审核的 `0.166.0`，在 probe 配置显式指定路径；默认 fallback 已升级为未审核 `0.167.0`，保留版本保护会正确拒绝。真实验证源码身份为 `a1ee747b…`；之后仅收窄初始场自动反馈过滤及补其测试，所有 runtime 文件与最终版逐文件一致。完整身份分别见 `CLI_SOURCE_IDENTITY.json`、`FINAL_SOURCE_IDENTITY.json`。

最终冻结源码完整回归 **477 项全部通过，622.530 秒**，见 `DELIVERY.json` 与 `regression.log`；最终结构审计通过（152 个 Python 文件、214 项源码身份文件），见 `structure_audit_final.json`。定向验证包含 runtime 6 项、排队集成 1 项及反馈/H4 边界测试，均通过；全部再次包含于最终全量。早期完整回归因复核收窄 H4 反馈过滤而主动中断，保留在 `regression_pre_final_interrupted.log`，不计作通过。全部历史科研实验及结果未改写。

剩余边界：CLI 证明本地真实请求同时在途并成功，后端未回报内部推理线程时间及实际 model/effort，审计保留未报告状态；不能据此宣称完整科研运行加速比例。后续真实运行需指定已审核 CLI，或另行完成新版本审核。显式串行历史配置仍为 1。本轮没有新的模型数值精度结论。
