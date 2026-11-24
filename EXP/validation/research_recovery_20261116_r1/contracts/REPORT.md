# H3 来源契约、恢复与组件回放

已修复导致原三轮 H3 五次未测量的来源声明/错误路由问题。没有恢复旧正式运行，没有修改旧卡片、Memory、checkpoint或冻结包。新结果仅属本次工程验证。

## 实现

- ExperimentCard 公开 schema 显式给出 evidence_refs/solution_refs。prepare_request 把 controls 内明确的同名列表合并去重到顶层，保留 controls 和 original_request；不解析自然语言或任意路径来扩展来源。
- Evidence 与 solution 均绑定产生原实验时的数值源码快照及原批准 loader。数值源码 hash 相同可使用当前数值 runtime，发生变更则从原快照隔离加载。当前 Journal 中的来源不会因 resume 后宿主升级静默使用新实现。
- load_frozen 可用确切 Evidence、solution ID或已绑定 model_ref；唯一完整包时可省略参数。重复同包引用优先保留历史源码绑定；多个不同包不擅自选择，未知路径不授予加载权限。包/loader/hash与禁止重新校准保护保留。
- 缺来源/选择歧义发出 RequestRevisionRequired，保留 requested_source、available_sources、recovery；跨子进程及异常包装仍保留结构化原因。原申请任务收到 request_revision_required，Generation可修订请求；旧失败/代码保持，不消耗 Coding 修订。数值状态为 not_validated、mechanism_measurement_completed=false，明确失败前可能有局部计算，不谎称整个插件从未执行。
- 已选来源/不可变 loader失败归 host，OOM/磁盘/描述符耗尽归 resource；当前批准实现错误继续走 Coding→Safeguard。Eval-L0错误分层同步识别新origin。加载错误不能用作机制反证。
- 修复旧 preflight 测试夹具，使其真实经过 G-only generation→selection 创建当前候选绑定，再检查计划局部修复；未放松运行时版本检查。

## 原H3真实历史包回放

`replay_h3.py` 引用原 R3 request `ExperimentCard-04be4a545bf24c4eb8ae88fff237ac27@1` 和原 `606389…` 实现，来源为R2完整方案 `c1c5ac…`。新日志在本目录，原运行只读。校准/搜索明确缩为2/1，未假称原32/64实验。

本环境GPU不可用：NVML Unknown Error，torch CUDA可用=false、device_count=0。CPU回放规范化来源后确实进入原PVQueryExecutor.load，约85秒后被严格audit拒绝。独立只记录audit差异的再次回放约83秒同样拒绝，所有结果保留。

`load_audit_differences.json` 显示仅12个MLP fc2的effective_bias SHA与原CUDA导出不同；其它审计项一致。effective_bias由设备矩阵计算重新产生，历史包只保存哈希，不能在无GPU条件下恢复并证明原CUDA字节。本次没有放宽哈希审计、伪造原包或将其标为机制反证。原完整DeiT H3数值验证仍待可用GPU。

## 实际tiny组件机制测量

`replay_h3_tiny.py` 生成并完整export/load真实PyTorch两block tiny ViT static W4A4 AdaLog包，再调用原H3的实际码本读取、空间/打乱配对、联合枚举、lambda拟合、PV误差分解与lambda_one控制函数；仅适配明确的模型/来源标识、位点和2cal/1search样本数。数据是合成图像，不能冒充真实DeiT或ImageNet科研结果。

第一次回放暴露生产构造器插入的量化子模块仍training=True。已由校准实现配套修复build_model末尾递归eval/freeze并增强audit；失败保留在h3_tiny_first_failure。之后机制测量成功；附加诊断最初因夹具无标签被准确拒绝，随后关闭任务分类计分，只交付表征误差，没有伪造标签。

最终 `h3_tiny_result.json` / `h3_tiny_measurement.json`：

- 两个位点各5次同前向P/V捕获与原生PV输出逐值一致，max discrepancy均0；原模型和码本签名不变。
- 六个head实际拟合lambda为 `[0.93205953,1.08318901,1.07456720]`、`[0.88965523,0.69863021,0.82550335]`，均只使用两张cal。
- 五臂native/nearest/spatial_pair/shuffled_pair/lambda_one完成真实测量。该tiny近均匀概率背景下所有search成对差为0，lambda_one不同标量数为0；这支持执行及等价控制，不支持联合编码收益，也不能反证DeiT机制。
- 同一冻结tiny包另生成 `h3_tiny_propagation.json`：两block配对边界分解和block0 FP冻结干预，cal2输出relative MSE基线为0.0214866695；没有校准/参数更新、没有最终任务计分。

## 工程回归

最终运行入口：`python -B -m unittest QAViT.tests.test_source_recovery QAViT.tests.test_code_repair QAViT.tests.test_dual_task_evaluation QAViT.tests.test_experiment_preflight`，结果以 `tests_final.log` 为准。覆盖归一/别名/歧义/原source时代/包装异常/恢复不重复Coding、实际子进程request/resource错误回执、旧Coding修复和计划预检。

初期并行编辑引起核心身份拒绝，及旧无绑定L3测试漂移，均保留在tests_initial/tests_targeted日志；它们不是成功结果。未运行真实LLM科研调用或正式下一轮贯通实验。

### 最终复核补修

51项首次全过后，独立复核发现 Evidence 来源绑定误从 Card 封套读取 sequence；实际 sequence 位于 Journal 索引。原测试 mock 返回了非真实封套，掩盖该错误。现已删除该mock，使用真实 Journal 提交两张冻结 Experiment、两张 Evidence及中间源码 intervention，实际物化原始/后续两个来源缓存，并验证重开 Journal 后绑定一致。生产代码从 `journal.cards[experiment_ref]['sequence']` 读取，同时先验证已提交卡片字节。修复后整个 `test_source_recovery` **6项通过，5.703秒**，见 `source_journal_final.log`；这些测试与前51项有重叠，不相加声称57项独立覆盖。最终整体回归以主交付集成报告为准。

轻量复现包含 `replay_implementation.py`（从原H3实现作已声明2/1样本适配的代码），使tiny入口不依赖先运行耗时的DeiT失败回放。该文件属于本次EXP证据，不作为框架实现依赖。最终集成还捕获了sequence修复时误改historical_source同名表达式的NameError，根代理已精确恢复metadata访问并重新验证；中间失败日志保留于主报告目录。
