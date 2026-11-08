# Unsigned affine + 可选 percentile observer

来源：RepQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/grids.py:fit/affine_qdq`。

UINT 表示编码无符号，不代表解码值不能为负。伪代码：s=(xmax-xmin)/(2^b-1); z=round(-xmin/s); y=(clip(round(x/s)+z,0,2^b-1)-z)*s。原 observer 不把 z 强行裁剪到编码范围。

均匀网格适合非中心对称分布，可与 repq_ln 的共享尺度独立组合；不是 repq_ln 的内置强制后端。W通常沿输出轴分尺度，A可tensor或实测通道轴。

calibration.strategy=percentile 时，在显式 percentiles 中以张量 QDQ MSE 选候选，例：[0.999,0.9999,0.99999]。minmax、manual、search 也独立可用；当前不支持 coordinate。

percentile 减少离群值影响但可能截断任务重要尾部，应结合局部/任务误差。对非常窄的GELU负支，affine不一定解决所有问题，可比较其它网格；对概率长尾也不是专用log的替代保证。


退化常量通道采用明确数值扩展：仅当hi==lo时把区间扩到零；全零通道用机器epsilon尺度。非退化通道保持原affine公式。这是有记录的工程适配，避免per-channel正常常量通道导致整网准备失败。
