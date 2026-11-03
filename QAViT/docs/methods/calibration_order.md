# 顺序、诊断与组合

来源：执行合同。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/pipeline.py:fit_model; plugin_contracts.py:validate_composition`。

预处理始终在W/A前。w_then_a：A观察冻结W；a_then_w：W观察冻结A；joint：同算子W/A由同一参考拟合。within_stage=parallel维持该阶段参考输入，sequential按实际拓扑启用已拟合算子，后续统计含前缀量化误差。

低位时前缀误差和observer协议可能显著改变后层scale，应作为实验变量；不要默认顺序无关。coordinate要求joint；quantized_prefix重参数化及progressive算子各有额外约束，读取对应卡。

按需profile分布→查独立组件条件→对不确定选项做小范围profile_ptq_candidates→compose_experiment提交。局部diagnostic只读calibration，允许部分位点；它不会成为正式候选、不能代替全模型覆盖与search/final评价。

写方案时说明实测问题、采用/不采用组件的原因、可检验预期；失败后检查具体数值变化，剩余一次就是最后候选。方法不是越多越好；不要把“以后再试”当作当前修改。

没有有限预算下必达论文精度的数学保证。来源配方、整体baseline不是自动搜索动作；由LLM根据独立组件和测量组成方案。


诊断参数：`plans=[plan1, plan2, ...]` 的每个plan是一次独立实验；`plan.layers`中的位点在同一次实验里同时启用。比较同一位点的两个方法，应写两个各含该位点的plan，不能在一个plan里重复该位点。普通 `profile_sensitivity` 只实现RTN，variant.name只作标签，修改名字不改变实际算法；其它方法须走 `profile_ptq_candidates` 的显式method字段。

## 如何形成可检验的搜索步骤

1. 用真实 catalog 的语义选代表位点，分开查看权重、普通输入、LN 后、GELU 后和概率分布。不能用权重统计推断激活；失败的调用不算已有证据。
2. 根据统计提出独立假设，例如尾部截断、非对称网格、跨通道变换或前缀观察。查相关独立卡，比较2–4个只改变一个因素的局部方案。`profile_ptq_candidates(grouped_plans=[...])`与`compose_experiment`使用同一分组语法，诊断可只含部分位点。
3. 局部筛查后再组成完整方案。若同时改多个因素，应记录各自支持证据；小样本MSE或校准Top-1不保证整模型Top-1。
4. 完整候选失败时，`profile_trial_diagnostics(trial_id,count,seed,offset)`可在校准子集上检查该冻结候选的全部激活边，返回失真和负支置零率。参数不会重新拟合；高局部误差只是定位线索，不是唯一根因证明。

分组的 `selector.semantics` 使用实际目录值（例如`post_layernorm`、`post_gelu`），不根据层名猜测。`preprocess_groups`只展开你明确选择的方法和参数；SmoothQuant可选一般Linear输入边：直接LN前驱填真实norm，其它边填norm=null；RepQ-LN仍只接受真实LN→Linear边。它不自动选择重参数化。各组必须互斥；正式提交仍检查全覆盖、保护、格式、顺序和实测证据。
