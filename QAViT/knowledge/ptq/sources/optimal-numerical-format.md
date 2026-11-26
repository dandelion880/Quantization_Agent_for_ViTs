## Finding Optimal Numerical Format

<a id="optimal-numerical-format"></a>

**证据边界。** 综述 [87]；[IEEE DOI](https://doi.org/10.1109/ICASSP49357.2023.10096798)、[作者页面](https://superdocker.github.io/publications/)、[IEEE SPS 报告摘要](https://rc.signalprocessingsociety.org/conferences/icassp-2023/spsicassp23vid1432)。已核对题名、作者和官方摘要；尚未取得原文正文及官方代码。不是“搜索更多 bit”这一简单变体。

**第一层：原配方已知部分。** 作者提出按每个矩阵乘法的数据特点选择 fixed-point 或 floating-point 数值格式，服务于 sub-8-bit PTQ。模型无需 QAT。尚缺候选 exponent/mantissa 分配、scale 求解、格式切换成本、校准预算及精确 W/A 覆盖，不能宣称 FP4、FP6 或某套标准格式已经按原法验证。

**第二层：可登记的研究组件。** 需要数值格式描述、可表示数集合和算子级格式选择器；codebook 必须同时表示总位数、指数/尾数、特殊值、rounding 与饱和行为。QAViT 底层已有多格式容器，只能作为实现基础；没有证据说明已包含本文选择算法。

**第三层：后续 workflow。** 补齐原论文后，先在相同总 bit 和相同矩阵输入下比较不同格式的表示误差，再将候选带入任务与设备验证。格式选择可以成为新的研究维度，但当前材料仅够形成有出处的候选问题，不能生成“原法复现完成”的结果。

**协议。** 进入 mixed-format 的来源待办，完整 recipe 尚未就绪。若协议只要求固定总 4 bit 而不强制整数，未来可探索固定 4 bit 格式；仍需明确算子支持与导出格式。

