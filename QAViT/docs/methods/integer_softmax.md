# Log-Int-Softmax 参考算子

来源：FQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/operators.py:integer_softmax`。

独立softmax替换：原多项式/整数近似与log_round阈值，与普通log2 QDQ不是同一计算。

需同attention的attention_logits输入尺度、post_softmax输出log2网格。输入per_tensor；输出minmax；joint/parallel/static。整数算子引用两条已声明边，仍逐边校验位宽和保护。

operator_calibration=progressive在输入尺度就绪后让LIS影响后续observer；observer_batch_size多批路径仅最后批启用，之前保持FP。float是显式变体。

不能在同概率边再叠twin/log_sqrt2。当前无真实整数kernel，目标若仅W/A误差不一定需要启用。

适配边界：原多项式的常数项 floor((1/0.35815147)/logit_scale²) 必须大于0。较粗的4位logits可能违反这一条件并产生0/0，程序明确拒绝，不加eps、不升位宽。本轮真实DeiT两图诊断观察到这种失败；组件可注册并不保证所有低位scale可用。

