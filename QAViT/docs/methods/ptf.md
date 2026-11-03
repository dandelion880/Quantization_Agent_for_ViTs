# Power-of-Two Factor 通道尺度

来源：FQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:fit_activation`。

用共享base和逐通道2^k因子描述尺度。根据minmax选base，在有限k候选内用观测张量误差选择通道指数；不是任意per-channel浮点尺度。

UINT4/6/8、per_channel、真实最后feature轴、static，options.ptf_k控制指数上限。原位点为pre_layernorm；post_layernorm是明确适配变体，不等同原方法。

伪代码：s_c=base*2^k_c; k_c=argmin_k error(QDQ(x_c,s_c))。可手动调整共享base；不能直接给任意独立channel scale破坏幂关系。

PTF是量化器，integer_layernorm另行显式声明尺度边；使用PTF不自动改变LN运算。原文多批observer只在最后批评分指数，running minmax覆盖全池；省略observer_batch_size是单池变体。

与RepQ-LN当前方案级互斥；与同LN链SmoothQuant互斥。需要测量实际通道差异和形状；若用户要求最粗粒度，需解释通道因子的收益及开销。

