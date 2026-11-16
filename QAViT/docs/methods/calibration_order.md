# 顺序、诊断与组合

来源：执行合同。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/pipeline.py:fit_model; plugin_contracts.py:validate_composition`。

预处理始终在W/A前。w_then_a：A观察冻结W；a_then_w：W观察冻结A；joint：同算子W/A由同一参考拟合。within_stage=parallel维持该阶段参考输入，sequential按实际拓扑启用已拟合算子，后续统计含前缀量化误差。

低位时前缀误差和observer协议可能显著改变后层scale，应作为实验变量；不要默认顺序无关。coordinate要求joint；quantized_prefix重参数化及progressive算子各有额外约束，读取对应卡。

按需profile分布→查独立组件条件→对不确定选项做小范围profile_ptq_candidates→compose_experiment提交。局部diagnostic只读calibration，允许部分位点；它不会成为正式候选、不能代替全模型覆盖与search/final评价。

写方案时说明实测问题、采用/不采用组件的原因、可检验预期；失败后检查具体数值变化和剩余预算，由 Generation 决定后续对照。方法不是越多越好；不要把“以后再试”当作当前修改。

没有有限预算下必达论文精度的数学保证。来源配方、整体baseline不是自动搜索动作；由LLM根据独立组件和测量组成方案。


诊断参数：`plans=[plan1, plan2, ...]` 的每个plan是一次独立实验；`plan.layers`中的位点在同一次实验里同时启用。比较同一位点的两个方法，应写两个各含该位点的plan，不能在一个plan里重复该位点。普通 `profile_sensitivity` 只实现RTN，variant.name只作标签，修改名字不改变实际算法；其它方法须走 `profile_ptq_candidates` 的显式method字段。

## 如何形成可检验的搜索步骤

1. 用真实 catalog 的语义选代表位点，分开查看权重、普通输入、LN 后、GELU 后和概率分布。不能用权重统计推断激活；失败的调用不算已有证据。
2. 根据统计提出独立假设，例如尾部截断、非对称网格、跨通道变换或前缀观察。查相关独立卡，比较2–4个只改变一个因素的局部方案。`profile_ptq_candidates(grouped_plans=[...])`与`compose_experiment`使用同一分组语法，诊断可只含部分位点。
3. 局部筛查后再组成完整方案。若同时改多个因素，应记录各自支持证据；小样本MSE或校准Top-1不保证整模型Top-1。
4. 完整候选失败时，`profile_trial_diagnostics(trial_id,count,seed,offset)`可在校准子集上检查该冻结候选的全部激活边，返回失真和负支置零率。参数不会重新拟合；高局部误差只是定位线索，不是唯一根因证明。

分组的 `selector.semantics` 使用实际目录值（例如`post_layernorm`、`post_gelu`），不根据层名猜测。`preprocess_groups`只展开你明确选择的方法和参数；SmoothQuant可选一般Linear输入边：直接LN前驱填真实norm，其它边填norm=null；RepQ-LN仍只接受真实LN→Linear边。它不自动选择重参数化。各组必须互斥；正式提交仍检查全覆盖、保护、格式、顺序和实测证据。


## 可执行的校准调度

完整 v3 `plan.execution` 支持以下选择。省略新增字段保持旧语义；`within_stage=parallel` 指共同观察背景，不是GPU并行。

| 字段 | 实际执行 |
|---|---|
| `order=w_then_a/a_then_w/joint` | W后A、A后W，或同算子共同参考；block模式将该阶段顺序应用到每个block内 |
| `calibration_unit=operator`（默认） | 按实际forward观察到的算子拓扑调度；W/A分阶段时阶段遍历全部算子 |
| `calibration_unit=block, block_modules=["blocks.0",...]` | 将明确路径的真实连续子图组成block；在block内完成阶段，再启用该block并前进；路径以实际forward核验；block外位点作为独立单元 |
| `layer_order=forward/reverse` | 按实际拓扑或其逆序安排算子/block；不以JSON列举顺序代替拓扑 |
| `reestimate_passes=N`（默认1） | 再次捕获实际数据并重拟合；第二pass起保留当前其它目标状态，关闭当前被拟合目标，避免重复QDQ；不是复用旧缓存或只改计数 |
| `calibration_context=current/float` | 保留当前已启用目标，或在每次拟合时临时关闭全部Q；结束恢复并安装新参数 |

`block_modules` 不得嵌套且须含实际位点。整数复合算子及`observer_batch_size`的渐进冻结目前不与block/多pass/float背景组合，明确拒绝；GPTAQ差输入补偿与`quantized_prefix`观察要求current背景。没有强制某种顺序较优。

block调度本身仍沿用位点/算子拟合目标；真正 **block输出重建选参** 另由[尺度选择API](scale_selection.md#完整冻结背景中的block整网目标)完成，不能把两者混称。回执逐目标记录pass、block、实际启用W/A、样本ID、候选分数与选中参数；完整导出包保留所有参数。原生minmax/search/coordinate与自定义外层搜索均不读取final选参。
