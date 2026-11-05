# 科研门禁集中清理（2026-11-05）

本轮沿模型提交、请求预检、执行条件绑定、Safeguard、worker、归档及原研究者续调清理同类阻断，延续已有单实现流程；未启动长时间完整科研轮次。

## 已落地的规则调整

- **探索与最终要求分离。** L1/L2 按本次显式干预条件执行，允许局部量化、消融和浮点对照后观察整网。最终候选的覆盖、位宽及正式协议在既有 L3 阶段检查。观察范围不生成全网干预事实；自身显式条件冲突仍拒绝。
- **能力检查跟随实际执行器。** 内置方法、observer、计划枚举仅限制内置执行器。经 Coding/Safeguard 的自定义方法和参数无需强制 parameter_bindings；没有结构科学约束的自定义计划可使用自身表示。明确结构约束仍在真实模型目录就绪后核对，不在模型尚未加载时归责研究者。
- **数据要求跟随真实操作。** 能力查询、权重统计、纯 FP 对照和不依赖图像的原生权重拟合不强制校准。原生 prepare/fit_model 也取消这些场景的隐藏校准图像读取。实际数据依赖算法继续绑定和核验数据视图；回执只记录真实读取的样本，显式声明校准却未读取不能冒充成功。
- **预检只在具备条件时进行。** 已提供字段的结构、参数镜像、服务层级、资源/授权及明确 L3 冲突在原研究者提交阶段反馈；不先固定坏请求再让 Coding 无效修订。缺失的实现选择仍可后补，包含同长度对象数组中的缺失字段；不改原值、不改原始请求，不提前要求模型目录。
- **梯度与参数更新分开。** 只读 autograd.grad/backward 正常执行并审计，不再按调用文件白名单放行。training-free 的 optimizer.step 禁令与源模型状态保护保留。
- **提交只要求科学核心。** feedback_used、预测/来源数组、附加论辩/复盘栏目、逐位点 reason/evidence 可省略；空列表、纯 FP 计划合法。裁决保留 rationale，不要求重复写 decisive_reason；选择已有理由和已知 provisional 状态由程序补齐。显式身份、实质决定、证据权限、H* 及最终选择关系仍核对。
- **失败粒度缩小。** 可选分析逐项保留有效内容，坏条目原文和具体问题进入 submission_issues；Recorder 知识问题保留 pending_knowledge_updates，不发布为验证知识。历史分页未读完不阻塞提交，交付记录仍如实保存。实现异常按实际源码调用栈归属修订，包含 NameError/UnboundLocalError；宿主异常保留工程原因，源码与批准重新绑定。

没有新增审批角色、宽松模式或例外白名单。Generation/Evolution 自主实验、辩论、H*、H4、最终选择、L3 和 Recorder 触发机制保持；提示词、模型可见 Schema、持久记录和接口文档同步调整。

## 验证记录

最终全量 `unittest discover`：**368 项通过，286.266 秒**，见 [regression_clean.log](regression_clean.log)。覆盖局部干预后整网观察、无图像统计、新方法/observer/参数、真实导出重载、只读梯度、NameError 同请求修订、缺省科研字段、可选条目局部问题、历史未读完提交、Recorder 范围补齐及固定科研流程。关键反例包含本次显式条件冲突、L3 最终规格冲突、内置未知能力、优化器更新、虚报校准、源码/批准失配及证据权限。

初轮全量的 5 处失败是维护旧“必须读完”政策的测试，已同步改为新预期；第二轮全量发现新可选 Schema 与实际工具校验器的表达不一致，以及局部探针归一化顺序问题，均修复后进入上述最终全量。没有靠保留旧门禁让旧断言通过。结构审计、CLI 入口和 `git diff --check` 也通过。

真实链初次尝试 `real_chain/` 在执行前暴露输入结构预检过晚：研究者将位宽放在 `quantization_contract.weight`，原请求固定后 Coding 无法删除该错误字段，因而出现无效修订。主动中止该验证，保留全部原件，未产生 Evidence 或测量。已将已知结构/冲突检查移到原提交阶段，并测试修复发生在同一研究者任务中；未要求提前冻结完整规格。

新的 [real_chain_r2/service_validation.json](real_chain_r2/service_validation.json) **passed**，使用原科学请求、既有 DeiT-B 预训练权重、GPU 0 / CPU 4 线程及已授权订阅模型。真实 Generation 再次遇到同一字段错误时，在原任务收到准确反馈、读接口并修正后继续；错误草稿未创建实验请求或 Coding 子任务。

最终完成：**Generation → 单份 Coding 源码 → Safeguard → 真实 worker / Evidence → 原 Generation 引用证据并提交 H1**。只有 1 份源码、1 次实际数值执行、1 条 real Evidence，没有 Reference 或双实现验证任务。

| 实测项 | 结果 |
|---|---:|
| 权重切片 | blocks.0.attn.proj.weight[0:16, :]，形状 16×768 |
| 参数 | bits=3、alpha=0.9、rows=16 |
| 自定义方法 / observer | clipped_maxabs_research_v1 / weight_absmax_clip_v1 |
| 元素数 | 12,288 |
| 共享 scale | 0.06789171695709229 |
| NMSE | 0.2623293995857239 |
| 固定目标梯度 L2 | 0.000341186037985608 |
| autograd.grad / backward / optimizer.step | 1 / 0 / 0 |
| 图像读取 / 模型前向 | 0 / 0 |
| 源模型状态 | model_unchanged=true |

后置只读核对见 [real_validation_checks.json](real_validation_checks.json)：批准绑定当前源码、原请求和执行卡；执行与最终回归的核心 SHA 完全一致；算法参数由程序镜像进入 execution.parameters；没有 parameter_bindings 或虚构 calibration；原 Generation 明确引用返回 Evidence。其解释指出梯度是同一残差的派生量，现有实验不支持裁剪优于未裁剪、整网精度或硬件收益。

NameError / UnboundLocalError 等恢复、无效自定义导出返回以及只读梯度/优化器更新反例，由实际 CPU worker 或数值接口回归验证，角色使用明确测试夹具；真实 DeiT 链未注入执行失败。没有将 CPU 回归当作真实科研结果。

## 能力边界

内置数值服务仍只执行其实现过的方法和评测适配器；自定义导出包使用同一批准实现的加载接口并由宿主实际重载、前向对比，不能凭自报重载成功。独立 evaluate_artifact 仍只直接支持宿主已有包格式。有明确位点/位宽要求的执行需要可核验描述；未知自定义计划不被提升成已经完成量化的事实。CPU 小模型回归只证明工程行为，不冒充真实任务精度。本轮不评估整网精度、完整数据集、硬件收益或完整 L3 科研结论。

最终核心 SHA：`84d3e5f0018da0a99ee544d607a4cbdd98235db734b506c9e061e18d9ea5434b`；真实交付源码 SHA：`b30b749ff192e845a8922809f084c062df7d4013b67286bd71ffe36eb86d6af6`。细节入口见 [DELIVERY.json](DELIVERY.json)，当前流程见 [PSEUDOCODE.md](PSEUDOCODE.md)。
