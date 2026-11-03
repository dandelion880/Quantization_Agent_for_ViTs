# FQ minmax signed / unsigned observer

来源：FQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/grids.py:fit/apply`。

保留 FQ 原网格：signed 的尺度为 absmax/((2^b-1)/2)，unsigned 为 (max-min)/(2^b-1)，zero point 在编码范围内截断；量化时使用 round(x/scale+zero)。与 RTN、RepQ affine 的端点/舍入不同。

这是独立基础网格，不自动安装 PTF、LN 或 softmax 算子。通道形状来自实际profile；W支持输出通道，A支持实测非batch轴。

minmax 使用整个指定池。execution.observer_batch_size 属于调度设置，当前仅允许 joint/parallel 的 minmax/PTF/log2 组合；不要在其它策略上填这个字段。

适合基线和兼容 FQ 尺度来源。不同 observer 协议不应混称原方法。

