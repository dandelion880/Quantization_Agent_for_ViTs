# 整数 LayerNorm 参考算子

来源：FQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/operators.py:integer_layernorm`。

独立算子，读取已冻结input_site/output_site尺度，不拥有observer，不自动安装PTF或其它方法。

input/output必须是同一LN直接两边（最终norm→CLS→head仅已核对图允许）；输入per_tensor或PTF关系，输出UINT affine网格。execution为joint/parallel/static；operators条目为method/module/input_site/output_site。

保留原整数近似和无eps分支，方差退化明确失败。不能简单以普通FP LayerNorm替代后宣称整数LN。

当前是浮点张量模拟整数公式，没有真实integer kernel或速度收益。仅目标需要更完整算子量化且收益/误差有据时选，不因方法存在而强加。

