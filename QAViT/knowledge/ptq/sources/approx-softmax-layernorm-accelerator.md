## 无训练算子旁证：Approximation-based Softmax / LayerNorm Accelerator

<a id="approx-softmax-layernorm-accelerator"></a>

**身份与范围。** 综述 [95]，真实论文为 [Hardware Accelerator for Approximation-Based Softmax and Layer Normalization in Transformers](https://www.mdpi.com/2079-9292/14/12/2337)，§4–§6。已读全文。原文应用实验是 **BERT/GPT-2**，没有 ViT PTQ baseline 验证；不把它登记成 ViT 量化算法。

**第一层：原方案。** 使用分段线性逼近指数、Newton–Raphson 迭代处理 LN 平方根，在共享固定点数据通路上切换 Softmax/LN 模式。离线预计算分段系数，不训练网络权重；实验改变区间宽度和迭代次数来观察误差。硬件为 VCU128/VU37P FPGA，50 MHz；该运行频率和 host/传输方式属于结果条件。权重/激活 W4A4 不是本论文定义的验证对象。

**第二层：旁证组件。** 可吸收分段系数表、迭代截断、算子误差—资源曲线与共享算子数据流描述。仅作为 operator 研究参考，不制造新的“成熟 PTQ 方法名”。作者 RTL/代码未核实。

**第三层：研究 workflow。** 原文评估近似强度对 NLP 任务的影响。QAViT 可据此设计先算子、后整网的实验：在相同输入分布上比较误差与延迟，再进入 ViT 验证；验证之前只能称算子旁证。其 NLP 分类/生成结果不能成为 DINOv3/ADE 成功证据。

