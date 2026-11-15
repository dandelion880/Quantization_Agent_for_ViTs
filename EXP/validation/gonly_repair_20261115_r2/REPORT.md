# 证据身份、启动恢复与真实双任务工程验证

本目录为本轮准备验证的独立新状态，全部验证于 2026-11-15 UTC 执行；不计正式科研轮次或 L3。历史运行、原 Evidence、模型包与数据均未改写。

- 证据根因：列表只识别标量臂标签，精确叶节点漏传 arm_id 等身份；显式 arms 容器的无标签列表或含数字字典键仍可能合并。修复保留记录身份（结构化标签用完整 hash）、显式容器的原键/索引及列表 arms 的 key_facts，不猜臂含义。证据交付和投影专项共 15 项通过（0.665 秒，含 3 项新增）。
- 恢复根因：旧启动意图只比较部分请求字段，环境从当前 body 而非最终持久 request 构建。修复比较完整正文（deadline 单独限制）、保留旧请求、改变/过期时写独立新请求，并绑定实际线程/GPU环境。真实 OS 进程注入“写请求与预留后、领取前中断”，验证原件复用、新正文实际执行、过期意图恢复、旧 GPU UUID 清理、确切回执 hash 及一次预算结算。新增 3 项加原数值 deadline 4 项共 7 项通过（2.260 秒）。日志分别见 evidence.log / request_recovery.log。
- 与双任务故障分工共同修正：deadline 从实际 worker request 显式传递到 evaluate_services 及双任务/分片子任务，不读取同目录遗留 request.json。数值超时与活跃预算保持独立。

新真实 GPU 验证入口为 gpu/check_frozen.py。只加载先前独立工程 minmax W4A4 校准 1 张包，不调用 prepare；新测量 classification search 64 张、ADE validation 1 张，在新输出下分别运行双卡及单卡。质量指标、样本顺序、整数正确数与 FP/Q 混淆矩阵全部逐值相同；运行布局的 latency/显存单列。双任务实际评测交集 **12.353678 秒**，dual/serial 服务墙钟分别 67.781054 / 44.003173 秒（含加载，不能用此小样本声称吞吐提升）。

故意在两个新作业保留已过期 request.json，另写 request-resumed.json 并显式传实际 deadline；双任务子请求及完成回执均与新请求 deadline 精确相同。再向内存父模型添加非持久 buffer，真实独立 GPU 子模型以 model_digest 身份不符拒绝；错误为 RemoteWorkerError，原包 hash 不变，未发布科学 Evidence。所有原件见 gpu/GPU_RESULTS.json、dual/、serial/、identity_failure/。

最终 gpu/cleanup.json 确认两个登记子进程退出、GPU 0/1 租约重新取得并释放、5 张 GPU 均为 0 MiB。该小样本只验证工程路径，不能替代完整 ImageNet 50k＋ADE20K 2k 的正式协议，也不是新增方法收益。
