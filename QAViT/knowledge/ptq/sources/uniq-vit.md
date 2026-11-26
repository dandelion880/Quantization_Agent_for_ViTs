## 7. UniQ-ViT（Neurocomputing 2026；2025 在线）

<a id="uniq-vit"></a>

**身份与来源。** 综述 [25]；[出版 DOI](https://doi.org/10.1016/j.neucom.2025.132072)、[作者机构条目](https://research.birmingham.ac.uk/en/publications/uniq-vit-optimization-driven-uniform-quantization-for-vision-tran/)、[作者仓库](https://github.com/Dexter-Yu/UniQ-ViT)。在线日期2025-11-07，正式卷663出版2026-01-28。本次实际访问仓库仅见 README 与一个 commit，尚无算法实现；出版全文下载返回403，因此不以综述补齐公式。

**可确认 paper-recipe。** 作者明确称 PTQ，包含 AQO 与 SRO：AQO 先建立局部量化范围，再分块联合细化量化参数和权重；SRO 以两阶段解耦优化处理 post-LN 通道差异，目标是全套 uniform quantizer。作者报告分类、检测及实例分割实验。初始化范围目标、每阶段 loss、优化哪些权重/scale/zp、迭代/样本预算、SRO 变换方程和真实位宽例外，本次均没有可引用的全文或源码确认。

**primitive 与工作流。** 当前只登记 “range_init→block_refine” 和 “post-LN scale-reparameterization schedule” 的机制身份与来源，**不提供以 UniQ 名义执行的伪完整算法**。RepQ-LN、coordinate search 和块优化已有/拟有基础可以承接将来原配方，但不能将它们随意拼合后命名 UniQ。下一查验位置明确为正式文 §方法/AQO/SRO、实验实现细节，以及仓库后续 release 的初始化与 reconstruction 入口；补齐后再填写参数表。

**接入与 profile。** 全均匀、推理固定参数的目标与 static W4A4 相容，但最终覆盖范围、原配方样本预算和训练边界必须从全文确认。本次状态为“作者声明 PTQ，原配方待获取”，不能放进可执行成熟基线排行榜；保留来源卡和待核字段，防止 Codex 用其他方法代写。这里的不足是证据可用性，不是因为使用梯度而拒收。

