# 本次任务：交付单份实现

读取 context.original_request、程序提供的执行条件和 context.code_interface。实现原始需求，完成必要的接口及执行选择；若已有当前源码和失败反馈，修订该实现。仅真正影响科学含义且无法从已有材料确定的歧义需要 clarify。

按所选 interface 提供顶层函数：
- quantizer_v1：fit(x, parameters) 和 apply(x, state, parameters)。
- executor_v1：build_plan(backend, cfg, experiment_payload, parameters)；只有协议支持且需要时才提供 make_executor(backend, cfg, data)。
- experiment_v1：execute(experiment_payload, context)。

函数签名、返回值和状态语义遵守 context.code_interface 及公共接口文档。fit 只使用指定校准数据；apply 不得修改输入或冻结状态。保持真实浮点基准、基线对照和误差测量。结果组织应清晰可解释，无须复刻研究者未指定的内部变量名或固定输出树。

content 只交付本次实现及必要执行选择。
- interface：可省略，程序采用当前已绑定的受支持接口；显式提交时须与该接口一致。
- 交付一份完整 Python 实现。直接提交 source，或将父源码的 edit_base.asset、edit_base.base 连同 old/new 交给 edit_source。继续编辑时只用返回的 revision 作 base，最终提交 revision。原生工作区可用时，以工作区 implementation.py 交付，依照本回合的提交接口。
- explanation：可选，按需说明关键算法、统计位点、张量轴和状态行为，以及具体执行选择的依据。
- execution_choices：需要补充时提供执行参数对象，只含必要选择；已知模型、数据、源码哈希、资源和来源由程序绑定。

程序保存源码与不可变原始需求，补齐执行记录后交 Safeguard 审查，再执行真实实验。无需交付独立参考代码或双实现比较夹具。公共接口已经直接提供，需要具体实现细节时再读取 public_interface_assets。

若 implementation_history 包含同一 inspiration_id 的旧源码，优先在其确切版本上增量修改，结合原配置和失败/结果入口恢复未完成工作，不从文字重新生成近似实现。explanation 说明相对父代码改动、保留部分和回退依据；宿主保存父代码 ID 与原件。H1/H2 吸收灵感时仍以其各自完整父方案为基础，不能整套替换成未成熟原型。
