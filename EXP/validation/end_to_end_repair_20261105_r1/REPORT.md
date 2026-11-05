# QAViT 真实贯通修订报告

已完成七类已确认根因及实际同类入口修复。保留程序 Supervisor、Generation/Evolution 自主循环、三场初始比较、固定 H*、H4 单独挑战、最终选择、L3、单份 Coding→程序绑定→Safeguard→执行，以及 Journal、四层 Memory 和轮末 Recorder。**具备进入下一次真实贯通验证的代码与预检条件；本轮没有完成真实 LLM 两轮科研。**

最终源码身份：`90065dd8ce99d0a352c6d08ff55b7b30657b592928b468eba00daea7627577b3`。本轮起点快照和差异保存在 before/、[changes_delivery.patch](changes_delivery.patch)；历史运行和报告未改写。实现前[根因清单](ROOT_CAUSES.md)及更新后的[伪代码](PSEUDOCODE.md)可直接对照。

| 根因 | 实际修复及覆盖 |
|---|---|
| 请求只有等待父任务才能续调 | Generation/Evolution/Judger 直接接收结果、失败、预算拒绝和科学澄清；最终 L3 由原 Judger 的新 plan_l3/revisionN 与新请求承接。旧计划、需求和失败版本保留；支持明确延期 defer_reason。恢复不会重开已被新计划或实现修订承接的旧失败。 |
| 解释状态绑死数值结果 | 全服务、正常/缓存/恢复路径有效 Evidence 即返回；取消不再必要的未启动历史 interpret，在途解释单独结算。解释/Recorder 失败不撤销数值验证、不重复实验。CLI 与报告分别给出流程、数值、解释和复盘状态。 |
| 只交付支持编号 | 从当前 H/Review/Request 的明确支持引用，统一交付有界观测、真实条件、状态/限制及精确入口，覆盖筛查、讨论、选择、演化、plan_l3、Coding/Safeguard；去重、快照、分支隔离仍生效，不递归展开旧实验。 |
| 表示字段早于规范化被 Schema 拒绝 | 直接计划、Coding 内置计划、执行绑定、grouped/patch 和 profiling 统一 normalize_plan，再完整校验。唯一的 bits、空 axis/group_size、禁用目标空 method、格式固有 scale 等由程序补齐；显式矛盾报具体字段，真正科学选择不猜。 |
| 版本和 shape 支持不一致 | RTNExecutor 适配 plan v2/package v1 与 plan v3/package v2；覆盖直接/Coding、导出、同次重载、独立 evaluate_artifact。正常准备阶段捕获真实 shape，随后严格复核；动态 shape 读取不冒充校准。 |
| 导入检查混淆正常库和执行能力 | 明确支持 contextlib/operator/bisect 等数值标准库；按模块/操作报告拒绝原因，区分 re.compile 与内置 compile。保留禁止的系统能力、真实 worker 写入边界及源码批准绑定。 |
| 旧说明和测试依赖 | 同步 Prompt、Schema、服务说明、接口/方法手册及测试；方法通过 method_catalog→read_asset 直接读取。require_method_documents 仅兼容旧配置，required_reads 是按需入口。 |

没有恢复 Reference、双实现、阅读确认、知识晋级、性能准入或科研必填栏目。真实位宽、数据范围、基准不变性、导出重载、执行身份、资产哈希与角色可见性检查保留；可选字段和 pending 不阻塞研究。上一轮 L1 事实完整性、问题增量、默认可用检索、比较知识逐证据条件、pending 修订及 Recorder 失败延期通过回归保留。

下一版信息流为：当前科研对象及明确支持证据→原角色自主研究/请求→规范化与执行（新代码经单实现链）→有效 Evidence 或具体失败→原等待任务/新计划版本→增量 Recorder→L1 当前状态与相关知识→下一轮。解释不再位于数值回执返回的必经路径。

实际验证：

- [最终完整回归](full_regression_delivery.log)：425 项通过，345.706 秒；[结构审计](structure_delivery.json)通过，146 个 Python 文件、208 个快照文件。包含 ACL/快照、缓存/独立重复、冲突校验、可选字段与 Memory/Recorder、正常标准库及真实 Landlock 写入保护。
- [两轮结构轨迹](two_round_structure_delivery/scripted_structure_result.json)：12.786 秒。真实调度/Journal/Memory，脚本角色和注入的协议回执；覆盖 H1/H2/H3→H*→H4→最终选择→L3 修订/恢复→Recorder→第二轮。**没有运行模型或 worker，夹具验证不是科研测量。**
- [真实隔离 CPU worker](plan_workers_delivery/plan_worker_validation.json)：4 个 worker，总 17.011 秒；v2/v3 各准备、导出重载和独立模型包评测，两次缓存命中未增加执行。随机 tiny ViT、合成数据，每次 calibration 2/search 1；两版本 MSE 均为 2.261051592331661e-06，独立重载相同。非真实数据精度或完整 L3 结论。
- 初轮全套发现 4 处仍假定强制 interpret 的旧测试和 2 处禁用 activation 却保留非空 method 的矛盾夹具；仅修正测试语义，未恢复旧依赖或放宽冲突检查。初轮失败日志和中断记录保留。

开销证据：

- 静态调用链：成功最终 L3 及同类原先无等待父任务的请求，不再额外要求一次 interpret 模型调用；历史恢复同样不启动不必要的旧解释。不是实测 token/延迟降幅。
- [同一大证据夹具](handoff_delivery/evidence_handoff_metrics.json)，当前实现关闭/开启支持交付：上下文 19,995→28,454 字符，必要 Evidence 正文 0→1；得到已展示观测的最低额外读取往返由 1→0（静态分析）。完整观测 213,210 字符，复用已有投影展示 5,903 字符，原件仍可按 site/token/metric 精读。
- 7 次热组装中位数 138.4→187.2 ms：主动交付增加了本地组装工作和首轮正文，不能声称所有指标下降。该比较不是旧版全源码重放，冷启动有导入/缓存效应。
- 本轮真实 LLM 调用 0；实际 token、模型延迟、Recorder 耗时与真实两轮总成本未测量，不用字符数折算。

能力边界与下一次启动：

[只读预检](readiness_delivery.json)确认配置、模型档位/订阅状态、资产和 GPU 可用；[文件核验](asset_readability_delivery.json)确认既定 calibration32/search1000 的 1032 张图像哈希，无 final 图像读取或评测。旧运行的 6 小时预算已过期，未重置或续借。

准备了 [NEXT_RUN_INPUT.json](NEXT_RUN_INPUT.json)，沿用 DeiT-B static W4A4、49W/97A、calibration32、classification/search/1000、最多两轮、21600 秒和原资源上限，禁止 final，不晋升 incumbent；仅使用新运行名并更新单实现流程说明。**启动仍需用户授权新的运行及预算**，本轮没有额外需要决定的科学参数。获授权后入口为 `python -B QAViT/run.py start --config EXP/validation/end_to_end_repair_20261105_r1/NEXT_RUN_INPUT.json`。

仍未验证真实 LLM 的多轮科研选择、真实 DeiT 候选完整 L3 和 Recorder 对第二轮实际研究行为的影响；不得以本轮测试数量、脚本轨迹或 tiny worker 代替。独立 evaluate_artifact 只加载标准包；自定义 executor 包由对应已批准 loader 在原执行链加载，experiment_v1 的宿主标准 L3 评测要求标准包，不承诺任意格式。跨 run 自动记忆和异步 Recorder 未新增。
