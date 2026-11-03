# 概率双均匀区间

来源：PTQ4ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:fit_activation/activation_qdq`。

只接 post_softmax 概率边，UINT4/6/8、per_tensor、symmetric=false，axis/group_size=null，static。

核心：分别量化 [0,split] 与较大概率区间，低区间步长 split/(2^(b-1)-1)，高区间步长 1/(2^(b-1)-1)，按原函数组合。每张量共享 split，保留小概率分辨率。

minmax 是固定初始 split（默认2^-8），并没有优化阈值。options.split 可显式设；manual 的 scale_values 对应 split。search/coordinate 若不指定 factors/factor_grid，程序比较20个2^-i阈值。显式 factors 是对初始 split 的倍率，不能产生split>1。

局部评分是 AV 输出；V 的浮点/量化状态影响最佳 split。coordinate 可用 peer_precision=quantized/float 明确选择。不能与同边log、RTN或LIS叠加。

依据：大量接近0概率且极少大值，检查 AV 输出误差。不能只依据QKV权重分布选择；需要概率实测。

