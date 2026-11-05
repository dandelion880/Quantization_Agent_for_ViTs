# 本次任务：交付单份实现

读取 context.original_request、程序提供的执行条件和 context.code_interface。实现原始需求，完成必要的接口及执行选择；若已有当前源码和失败反馈，修订该实现。仅真正影响科学含义且无法从已有材料确定的歧义需要 clarify。

按所选 interface 提供顶层函数：
- quantizer_v1：fit(x, parameters) 和 apply(x, state, parameters)。
- executor_v1：build_plan(backend, cfg, experiment_payload, parameters)；只有协议支持且需要时才提供 make_executor(backend, cfg, data)。
- experiment_v1：execute(experiment_payload, context)。

函数签名、返回值和状态语义遵守 context.code_interface 及公共接口文档。fit 只使用指定校准数据；apply 不得修改输入或冻结状态。保持真实浮点基准、基线对照和误差测量。结果组织应清晰可解释，无须复刻研究者未指定的内部变量名或固定输出树。

content 返回对象，source 是唯一必需字段：
- interface：可省略，程序采用当前已绑定的受支持接口；显式提交时须与该接口一致。
- source：完整 Python 源码字符串，包含所需导入、函数和辅助定义，不使用补丁、Markdown 包装或省略号。
- explanation：可选，按需说明关键算法、统计位点、张量轴和状态行为，以及具体执行选择的依据。
- execution_choices：需要补充时提供执行参数对象，只含必要选择；已知模型、数据、源码哈希、资源和来源由程序绑定。

程序保存源码与不可变原始需求，补齐执行记录后交 Safeguard 审查，再执行真实实验。无需交付独立参考代码或双实现比较夹具。公共接口已经直接提供，需要具体实现细节时再读取 public_interface_assets。
