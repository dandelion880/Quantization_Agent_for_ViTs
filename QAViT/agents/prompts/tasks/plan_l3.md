# 本次任务：制定本次 L3 实验计划

依据 context.task.selection_ref 对应的 final 选择、hypothesis_ref 对应的完整最终候选、选择理由、相关证据及 ResearchBrief，提交一份实验需求草案。不要把 Evolution 前的 H* 选择当成最终候选。宿主会接收并安排该计划；不要再通过 request_experiment 重复提交同一份最终 L3 计划。

L3 不以某项 L2 性能达标为前提。根据当前实际预算、资源与评测协议设计可执行计划；估计应有依据，不能虚报耗时以通过门禁。若关键执行定义缺失，具体指出，不能用虚构路径或占位参数凑成计划。

使用 context.data_receipt 核对可用数据池、split、样本范围、身份摘要及来源；可用池大小不等于本实验选定数量，未知项不能猜测。资源估计对照 context.budget_receipt.research_remaining_seconds；remaining_seconds/available_seconds 是硬截止余额，复盘和停机预留、未结算估计的含义见回执。

按 ResearchBrief.evaluation_protocol 明确的任务、数据范围和完成条件制定计划；有 evaluations 时保持其 kind、split、count 等明确要求，不自动增加分类、分割或完整数据评测。协议允许的子集 L3 只支持其实际范围；协议要求完整评测时，不能因时间不足缩减后仍称完成。记录规定量化覆盖与例外、模型导出及重载身份、数据视图和指标口径；已有合格冻结模型包可按协议复用。小样本搜索、单点回放和历史指标不能冒充协议尚未完成的评测。

按 payload_contract.schema_location 指向的同一实验草案 Schema 提交，level=L3。常规执行参数已经在 context.experiment_interface；明确研究问题、所需测量和科学约束；有明确预测及决策影响时再提供。宿主生成固定字段与执行镜像。已有方法直接给 plan，已有模型包用 evaluate_artifact；真正新增能力才经 Coding 交付单份实现，程序补齐执行记录并交 Safeguard 对照原始需求审查。无需提交内部变量名、固定输出树或测试夹具。

提交的是待执行计划，不是实验结果。L3 流程完成、协议范围内的有效验证、完整数据评测及 incumbent 资格分别判断；incumbent 仍须通过后续程序检查，Brief 禁止晋升时不得声称可晋升。


若 context.task.request_feedback 存在，这是原计划执行反馈后的新版本任务。依据所附原始需求、真实失败/拒绝或 Coding/Safeguard 问题修订草案，旧计划与结果保持不变。优先用已有证据解决科学澄清；实现失败、预算拒绝或缓存均不等于假说反驳，缓存不算独立重复。缺少只有用户才能决定的信息时才使用 clarify。若本轮现有条件确实无法完成规定协议，可提交 {"defer_reason":"具体原因及未完成项"}，保留未完成状态并进入正常复盘；不得虚构数据、缩减必需协议或重复相同失败。

有效 Evidence 保存并通过规定数值协议后即返回；不会为返回实验结果另行强制调用 interpret。已有原研究步骤和轮末 Recorder 负责解释；解释或复盘失败不撤销已成立的数值验证。
