# GPU 调度与真实 subset 验证

实现：`runtime/gpu_resources.py` 与 `runtime/task_runtime.py` 在实际可见性、外部占用和 OS 租约基础上，先为未获得主卡的候选保留资源，再分配单候选内部双任务或已有分类分片所需余卡。候选身份沿确切 Hypothesis 引用和父请求追溯；尚在计划/Coding 阶段的候选也参与预留。无可用 GPU 的 CUDA 请求保持排队，不领取任务、不扣尝试、不创建预算预留。`tools/experiments.py` 将无卡状态交给该调度处理。

已验证：

- 0–6 卡的模拟硬件身份结合真实 OS 租约测试；3 候选分配依次为 `[]`、`[1]`、`[1,1]`、`[1,1,1]`、`[2,1,1]`、`[2,2,1]`、`[2,2,2]`。外部占卡、租约冲突、Coding 待执行、失败释放及恢复绑定均覆盖。
- `gpu_priority_regression.log`：6 项新调度/既有并行请求测试通过。`scheduler_existing_regression.log`：51 项资源、预算、双任务、进程与超时测试通过。
- 真实 5×V100 环境中授权卡 2/3/4，3 个独立候选分别校准、冻结并执行 ImageNet 与 ADE subset，均生成独立有效 Evidence，实际重叠 76.648 秒，运行中恢复 Workflow 后归属保持不变。
- 复用 H1 同一个冻结包在卡 2/3 双任务并行复测，实际测量重叠 6.080 秒；分类样本、计数、raw error sums，以及分割样本、混淆矩阵、指标均与串行结果精确相等。冻结包身份校验通过；暂停恢复不重置预算，活跃时间 7.368 秒、排除纯数值等待 96.540 秒。已停止所有验证进程、GPU 显存释放。

范围：真实验证使用 DeiT-B，RTN W8A8 仅作用于 `blocks.0.mlp.fc1/fc2`，每候选 2 张校准图、3 张 ImageNet search 图、1 张 ADE validation 图。未执行完整 W4A4 L3，也未启动正式科研轮次或新的科学候选评比。

保留的失败记录：r1 因并行修改 5 个测试文件触发已有源码身份保护，恢复被拒绝，worker 已终止。r2 的三个候选均成功，但额外复测实际只获得一张卡并正常串行回退，验证脚本要求两卡的断言失败；随后独立空闲两卡复测成功。未将失败脚本或部分结果改写为全成功。

完整路径与证据：`scheduler_validation_summary.json`、`gpu_candidates_r2_report.json`、`gpu_same_frozen_parallel_r1_report.json`；可执行脚本为同目录 `gpu_candidates_smoke.py` 与 `gpu_same_frozen_parallel.py`。

最终补充：候选身份追溯兼容历史请求的 `origin_task`，避免将旧 plan 与其请求重复计入 GPU 预留；旧无 GPU 即拒绝的接口测试已更新为排队语义。最终 `scheduler_final_regression.log` 的 34 项测试全部通过，包含新旧候选身份、无卡排队、纯数值等待预算、GPU 租约及请求接口。该补充未改变任何数值实现；真实 GPU 证据对应其报告内冻结源码版本。
