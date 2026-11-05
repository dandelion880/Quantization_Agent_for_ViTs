# 本次任务：审查当前唯一实现

读取 context.original_request、context.frozen_experiment、candidate_source 和程序交付的执行条件。直接对照原始需求核对统计位点、量化位宽、比较条件、数据使用、指标口径和基准不变性。审查只覆盖当前源码和执行条件，修改后程序会重新绑定及安排审查。

提交科学判断：
- approved：boolean，明确批准或拒绝。
- rationale：非空字符串，给出决定性依据；拒绝时定位具体问题和影响。
- fidelity、experimental_validity：可选，必要时展开算法忠实性和实验有效性。
- uncertainties：可选字符串数组，记录真实未解决事项。
- evidence_refs：可选，引用实际 EvidenceCard；没有实测时省略，不把源码或计划冒充证据。

不要求固定说明模板、双实现数值记录或独立参考源码。不能把尚未运行的检查写为通过；程序负责接口、身份和真实结果的确定性检查。原始科学定义存在无法消除的实质歧义时才返回 clarify。接口资料已经直接提供，需要具体细节时再读取 public_interface_assets。
