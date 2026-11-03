# 本次任务：独立参考实现

这是 purpose=reference，交付参考源码，不是审查报告或 phase=reference_design 卡片。

仅依据 context.frozen_experiment、已授权 ResearchBrief 和 interface_reference 实现冻结算法。不得读取或查找候选源码、候选结果、候选审查或相关运行日志；也不借用其他候选实现来拟合预期输出。

优先采用清楚、直接、易于逐项核对的数学实现。保持与候选应遵守的同一公式、码本、作用轴、边界规则、状态语义和接口；独立实现不意味着可以选择另一种科学定义。缺少会改变数学含义的定义时具体澄清，不擅自补齐。

按 execution.interface 提供必需顶层函数：quantizer_v1 的 fit/apply、executor_v1 的 build_plan，或 experiment_v1 的 execute；确切签名和返回值遵守授权接口文档。只使用当前允许的依赖与宿主上下文。

content 返回一个对象：
- interface：精确的冻结接口名字符串。
- source：完整、可由宿主执行的 Python 源码字符串，包含所需导入、函数和辅助定义，无 Markdown 包装或省略。
- explanation：非空字符串，说明从冻结数学规格到参考实现的对应、边界处理及有限验证范围。

小规模案例用于核查已定义行为，不反向决定公式。源码的保存、分离执行与数值比较由宿主进行；没有执行反馈时不宣称已运行或已通过。

context.code_interface 已直接提供公共接口与验证约定；需要具体方法实现时再按 public_interface_assets 读取。
