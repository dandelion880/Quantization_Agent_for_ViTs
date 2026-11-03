# GPTQ 权重误差补偿

来源：GPTQ。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:gptq; calibration.py:gram`。

只作用 Linear 权重，不能用作 A 方法或 Conv2d 方法。需输入校准数据；H=2*X^T X 的累积、阻尼与 Cholesky 逆分解后，逐列量化并向未处理列传播误差。

伪代码：err=(w-q)/diag(Hinv); W_remaining -= err*Hinv_row。原始 minmax 量化器不接受额外 percentile/search/coordinate。

支持契约内整数位宽；W per_channel(axis=0) 或静态输入组 per_group(axis=1,group_size=合法整除输入维数)。options 可 blocksize、damping、actorder。未填时执行源码默认值；默认 minmax calibration。

可和任意合法 A 方法分开；a_then_w 的统计基于已冻结 A 输入，w_then_a 的 A 基于量化 W；这是真实不同协议。外层 sequential 与 GPTQ 内部列补偿不同。

成本/风险：O(in_features²) Gram 和分解工作区；低位 A 的严重失真不能仅凭 W 的GPTQ修好。无自动优化器/微调。

