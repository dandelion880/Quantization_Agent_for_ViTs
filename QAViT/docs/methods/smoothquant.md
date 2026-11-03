# 激活/权重等价缩放

来源：SmoothQuant。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:smoothquant; pipeline.py:preprocess`。

独立前处理，必须在任何W/A量化前。在目标Linear输入边执行 X'=X/s、W'=W*s，s_c=amax(X_c)^alpha/amax(W_:c)^(1-alpha)，下限保护保持有限正scale。norm是已确认直接前驱LN或null；项目用显式边变换，不宣称融合。

alpha∈[0,1]迁移激活/权重量化困难；并非越大越好。激活离群明显但W可承受放大时值得测；需同时看W/A误差。只量化某目标不代表缩放另一侧无代价。

对Linear输入使用 `profile_distribution(target=activation, views=[tensor,channel])` 检查通道范围差异，并核对相应权重列的范围。整体直方图不能替代通道统计；局部比较应同时包含变换后W/A，避免只观察A改善而忽略W恶化。

preprocess条目字段 method/module/norm/evidence，可选alpha；不接受额外calibration字段。先FP等价检查再拟合网格。

同一Linear边不能叠加repq_ln；不能在变换后的fc2输入使用原post-GELU twin/asymmetric；与同LN链PTF互斥。可与其它边特殊网格/GPTQ并列。

来源为LLM量化方法思想；在视觉模型的效果必须实测，不把来源论文LLM指标移植为ViT保证。

