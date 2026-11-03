# GELU 正负双区间

来源：PTQ4ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:fit_activation/activation_qdq`。

只接真实 post-GELU 边，通常可直接在语义为 post_gelu 的 fc2 Linear 输入选择 activation_method=twin_gelu；不要再同时给 .act::post_gelu 安装第二量化器。

INT4/6/8、per_tensor、static。负区间用精确GELU下界约-0.16997124与固定负尺度；正区间使用单独的正scale。两支分别round/clip再合并。

普通对称minmax若半步长超过负支幅度，负值会全部归零，即使负值占多数；双区间针对这种不对称。minmax只是正尺度初始化，可用search或joint/static的coordinate在fc2输出上选正尺度。scale调整默认不改负尺度；options.negative_factor>0明确改变负区间，属于可记录变体。

与同一fc2输入的SmoothQuant/其它仿射互斥，因为变换后不再是原GELU分布；可与其它边预处理并列。

需要验证正尾部裁剪、负支误差、fc2局部输出以及整模型Top-1；不是单独启用就必然恢复论文精度。

