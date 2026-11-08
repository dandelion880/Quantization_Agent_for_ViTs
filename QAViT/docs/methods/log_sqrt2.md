# 概率 log√2 网格

来源：RepQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:activation_qdq; grids.py:percentile_log`。

非负概率使用更密的 log√2 指数；以2的幂与奇偶指数补偿表达，推理仍为浮点reference。UINT4/6/8、per_tensor、static。

伪代码：e=round(-2*log2(x/s)); y=2^-ceil(e/2)*(1+(e mod2)*(sqrt(2)-1))*s；超出指数范围的值下溢为0。需要真实概率分布，不能用在任意有负激活上。

可用 minmax 或独立 percentile observer（显式候选按张量MSE选），也可用合法 search/manual；不支持当前coordinate。

该网格可单用，无须先做repq_ln；与其它概率量化器互斥，和其它边的GELU网格/权重方法可并列。更密的指数也缩小动态覆盖，需要校准评价。


static W4A4 适配：显式 `options.reserve_zero=true` 将指数有效域从0..15缩为0..14，把下溢映射到零（AdaLog shifted 域为下溢端点），每组最多16值。默认 false 保留历史公式；此变体必须重新校准、记录，不能沿用论文或旧包成绩。冻结选项参与导出/重载。
