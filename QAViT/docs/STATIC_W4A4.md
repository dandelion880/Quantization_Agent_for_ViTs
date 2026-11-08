# Static W4A4 协议

新 ResearchBrief 使用 `quantization_contract: {"protocol":"static_w4a4/1"}`。旧运行的冻结记录保持原义。模型主体 Linear/Conv 主权重 W4、输入 A4；Attention Q/K/V 和用于 PV 的 Softmax 输出 P、用于 FC2 的 post-GELU 均 A4。每个固定量化组最多16码字，零计入；参数在推理前冻结。

允许 LN、Softmax、GELU 内部高精度，允许 gamma/beta、bias、CLS token、位置编码、Residual 加法和矩阵累加高精度；进入上述计算边界须 A4。head、neck 可 FP，实际选择进入 coverage.task_modules。粒度、对称性、量化器、校准、重参数化与补偿是搜索空间；[-8,7]与[-7,7]都合法。内置插件的能力限制不是全局算法禁令，自主实现可走 Coding。

L1/L2 按显式本次范围执行，允许局部/浮点消融。L3 的 plan 根据真实模型目录检查边界，同一语义输入边不重复 QDQ；检查已冻结参数、原生码本容量、导出重载和评测前后状态。自定义 executor 的 `audit(model,plan)` 返回 `static_w4a4={"frozen":true,"boundaries":{"<site_id>/weight或activation":{"bits":4,"max_codewords":16,"includes_zero":true}}}`，以实际每组码本最大容量计算，不能用样本 unique 计数冒充。宿主核对实际边界集合与容量、独立重载和状态不变；码本实现由现有 Safeguard 科学审查。无需额外 LLM 审核。

原 log2/log_sqrt2/AdaLog 可产生16个非零值再加下溢零。严格四位使用显式 reserve_zero 变体或自主合法码本；不静默更改旧实验。程序不按 prose、插件名或符号风格筛查科研主张。

RepQ 原始非整数 mean(z) 在双端饱和时也可能形成17码字，宿主按冻结参数核验，不能用观测样本中未出现端点作为合规证明。现有完整RepQ种子显式在变换后以affine percentile拟合整数零点；非规定边界的内部高精度位点不要求4 bit。
