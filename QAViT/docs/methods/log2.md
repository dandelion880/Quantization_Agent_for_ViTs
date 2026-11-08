# 概率 log2 网格

来源：FQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:activation_qdq`。

只接已确认概率边；UINT4/6/8、per_tensor、static，不能量化有负值GELU。

伪代码：e=round(-log2(x/s)); 对允许指数范围外下溢置零，y=s*2^-e。码本为非均匀值，不是均匀UINT整数算术。

共享 s 默认概率尺度，冻结后不会按每次输入更新。可按合法策略比较 scale；不等同整数 softmax 的多项式/阈值，integer_softmax 是独立算子。

与twin_softmax/log_sqrt2/普通概率量化是替代关系。若精度需要，比较 AV 输出误差和小概率置零情况，不能仅按概率元素MSE断定最终准确率。


static W4A4 适配：显式 `options.reserve_zero=true` 将指数有效域从0..15缩为0..14，把下溢映射到零（AdaLog shifted 域为下溢端点），每组最多16值。默认 false 保留历史公式；此变体必须重新校准、记录，不能沿用论文或旧包成绩。冻结选项参与导出/重载。
