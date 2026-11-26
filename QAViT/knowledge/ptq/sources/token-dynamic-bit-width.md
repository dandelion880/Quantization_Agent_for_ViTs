## Token-based Dynamic Bit-width Assignment

<a id="token-dynamic-bit-width"></a>

**证据边界。** 综述 [42]；[Pattern Recognition 正式页面](https://www.sciencedirect.com/science/article/pii/S0031320325009306)，DOI [10.1016/j.patcog.2025.112269](https://doi.org/10.1016/j.patcog.2025.112269)，[作者实验室目录](https://cvlab.yonsei.ac.kr/publications/)。已核实身份、出版社摘要/引言及章节说明；全文方法公式、实验超参数与作者代码未取得。正式卷期为 2026 年 3 月。以下是**已证实机制及待补配方**，尚不能作为可直接执行的成熟 baseline。

**第一层：原配方已知部分。** 属 PTQ。输入实例改变 token 的动态范围，因此推理时按 token 分配 bit（含 class token）。为避免逐 token 显式计算全部量化误差，以 quantization interval 近似误差用于分配；另用移位式 quantizer 处理 post-GELU 正负不对称。已知这两者分别位于 §3.2、§3.3；校准数、候选 bit 集、阈值求解、误差近似式和完整位宽覆盖尚未核实，不能补写默认值。

**第二层：可登记的研究组件。** 仅登记“interval-based token error proxy”与“input-conditioned bit allocator”的概念及来源，状态为 source-incomplete，不能启用执行；现有 dynamic scale ≠ dynamic bit allocation。移位 GELU 也不能直接映射到当前 asymmetric GELU 名称，需对照完整码字定义。

**第三层：后续 workflow。** 原摘要声明分类、检测、实例分割验证，但尚未读取完整表格。QAViT 的下一项工作应是补齐 §3/§4.1 与实现，然后以固定 token-wise 位宽为对照，检验 interval 代理的排序质量及 runtime overhead。该验证方案是适配计划，不是作者已验证的 DINOv3 结论。

**协议。** 完整算法应归 token-dynamic。当前只保留来源与研究待办，不列为已验证可执行配方。

