# 本次任务：交付候选实现源码

读取 context.frozen_experiment 与 interface_reference；只实现冻结规格指定的接口和算法。若必要数学语义在授权资料中仍不明确，返回具体 clarify；不改变语义的实现选择由你完成。

按 execution.interface 实现所需顶层函数：
- quantizer_v1：fit(x, parameters) 和 apply(x, state, parameters)。
- executor_v1：build_plan(backend, cfg, experiment_payload, parameters)；只有需要且协议支持时才提供 make_executor(backend, cfg, data)。
- experiment_v1：execute(experiment_payload, context)。

函数签名、返回值和状态语义以已读接口文档及冻结规格为准。fit 只使用指定校准数据；apply 不得修改输入或冻结状态。不得为迎合测试引入未声明的特例、调整公式、改变数据或放宽容差。依赖必须在当前宿主允许范围内；缺少能力时指出具体依赖，不能自建文件、进程或网络入口。

content 返回一个对象，含且完整填写：
- interface：精确等于 execution.interface 的接口名字符串，不附函数签名或解释。
- source：完整 Python 源码字符串，包含所需导入、函数和辅助定义；不是补丁、Markdown 代码块、文件路径或省略号。
- explanation：非空字符串，说明关键公式和实现的对应、张量轴与状态行为、依赖及已知覆盖边界。

不返回 ReviewDecisionCard 或 ExperimentCard。宿主会保存源码、执行验证；没有执行反馈时，不宣称已通过语法、数值或任务测试。

context.code_interface 已直接提供公共接口与验证约定；需要具体方法实现时再按 public_interface_assets 读取。
