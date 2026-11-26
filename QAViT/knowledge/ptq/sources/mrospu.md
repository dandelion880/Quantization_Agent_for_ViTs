## 14. MROSPU（PRCV 2025 会议；2026 正式出版）

<a id="mrospu"></a>

**来源。** 综述[75]，[Springer原始章节](https://link.springer.com/chapter/10.1007/978-981-95-5693-9_9)，正式在线2026-01-21，pp123–137。原文标题 *Model Reconstruction Optimization and Scale Perturbation Update for ViTs Low-Bit Post-Training Quantization*。本次可读范围为出版社摘要/元数据，全文需订阅；没有确认作者官方代码。

**可确认范围。** 摘要明确PTQ并说明避免端到端重训练：先逐块计算FP/Q输出loss并进行梯度重构，随后对attention block的量化scale施加扰动，用遗传算法选新scale。这支持“有限校准重构＋黑盒scale搜索”两段机制身份。未读到精确loss、重构优化变量、数据/标签量、遗传种群/交叉/变异规则、fitness、训练迭代和位宽例外，不能从Evol-Q、QDrop拼凑一份所谓原配方。

**primitive 与workflow。** 当前条目仅保留独立可核查的两阶段依赖：重构后的checkpoint是scale搜索的父状态，搜索结束必须再固定quantizer参数。QAViT现有候选搜索和新重构执行器将来可承接，但本文不定义MROSPU专属默认超参，也不产生执行preset。若先研究“重构后再进化搜索”，请使用QAViT自定义实验名并以已验证模块组成，不把结果归到该论文。

**接入状态。** 下一查验应取得出版社全文的方法、算法和实验实现节，并验证是否有与标题对应的作者release。作者已明确PTQ身份，但完整recipe未核实；仅列来源与待获取配方，暂不进入“成熟可执行基线”。其目标与最终staticW4A4原则上相容，仍需核实实际位宽、参数和优化预算；此处不作部署效果断言。

