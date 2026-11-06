# 完整 signed uniform 网格

来源：PTQ4ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/grids.py:fit/apply`。

用于 signed 整数，不等于 RTN：Q=2^(b-1)，s=absmax(x)/(Q-0.5)，y=clip(round(x/s),-Q,Q-1)*s。低于32位，支持当前契约内位宽。

W 可 per_tensor/per_channel(axis=0)/per_block(axis=0,group_size=连续输出行数)；A 可 per_tensor 或经真实形状验证的非batch通道轴，包括 attention head 轴1。合法计划可在 prepare 从宿主允许的数据视图观察 shape 后严格验证，不要求提交前额外测量；轴必须由研究者明确选择。不是任意分块 reshape。

可单用 minmax 或独立 search；coordinate 要求 joint/static，且同算子全部目标使用兼容的 uniform_full/twin 网格。per-channel W 搜索和 per-tensor W 是不同粒度。

本组件不包括 twin 激活、Hessian metric 或预处理，不能仅选本方法就宣称采用整篇论文。低位时仅换网格端点可能几乎没有改善。

