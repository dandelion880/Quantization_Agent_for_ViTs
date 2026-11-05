# 方法与证据分级

方法资料随本项目交付在 `QAViT/docs/methods/`（工具别名`methods/<id>.md`），目录索引`docs/methods/index.json`。实际数值库在`QAViT/tools/ptq_agent/`，使用 `ptq_agent` 数值工具命名空间；不调用旧智能体项目。读取时返回实际hash；注册名称不能当验证证据。来源审查记录在`EXP/implementation/`，逐项工具验证及正式搜索分别在EXP/validation和EXP/runs。工具入口已调用不等于全部参数/方法组合均验证。

| 类别 | 存在实际实现的组件 | 已有证据与边界 |
|---|---|---|
| W | rtn、gptq、uniform_full、uniform_affine、uniform_minmax、adaround、gptaq | 旧数值/位点/重载测试；其中AdaRound通过梯度更新alpha，属于参数更新，是否允许由冻结实验决定。历史完整Agent候选不等于所有组件被采用。 |
| A | rtn、uniform_full、uniform_affine、uniform_minmax、twin_softmax、log2、log_sqrt2、twin_gelu、asymmetric_gelu、ptf、adalog | 旧真实语义位点与reference实现。AdaLog是独立base×scale适配，不是全套FPCS；log2/log_sqrt2/AdaLog的旧参考编码有额外下溢零（最多17值），必须按冻结实验检查实际码本，不能仅凭位宽标签判断；原profiling数值能力保留。 |
| 前处理 | smoothquant、repq_ln、adalog_bias | 明确边仿射与等价性gate；不能把别的实验的整网精度迁移到当前覆盖。 |
| 算子 | integer_layernorm、integer_softmax | 浮点模拟整数参考，不是真实部署kernel。 |
| 校准 | minmax/search/manual/percentile/coordinate；mse/cosine/hessian；并行或顺序 | 保留完整原参数。只读梯度诊断可属于 training-free，真实零梯度也应如实记录；内置 Hessian 加权搜索需要非零信号，信号全零时报告该算法不适用，不暗换搜索目标。 |
| 仅资料或未接入本版 | 旋转、ERQ、全参数QAT、QPEFT及其它库外想法 | 名称并不等于本版有实现或实测；按研究需要提案并标未知，不自动扩展范围。 |

方法卡：rtn、uniform_full、uniform_affine、uniform_minmax、gptq、gptaq、adaround、smoothquant、repq_ln、adalog_bias、twin_softmax、log2、log_sqrt2、twin_gelu、asymmetric_gelu、ptf、adalog、integer_layernorm、integer_softmax、scale_selection、calibration_order。

历史结果的位宽、覆盖和方法变体必须回到原运行事实核对，不能作为新 V2 运行的测量结论。具体算法取舍由当前角色与冻结实验决定；本说明不提供固定整网答案。
