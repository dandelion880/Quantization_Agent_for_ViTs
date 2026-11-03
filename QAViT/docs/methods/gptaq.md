# GPTAQ：原始 FP 对齐及输入差异补偿

来源：arXiv:2504.02692v1（GPTAQ；后续版本改名GPTQv2，不能混用版本）。官方Intelligent-Computing-Lab-Panda/GPTAQ固定commit `d6bd2e41e9abc8f22a76d6ea789c450b6c143ced`，`vit_quant/gptaq_utils.py`和`quant_utils.py`。本地已验证上游位于LowRank_W4A4_Vision_Research/third_party/gptaq_reference；新接口 `execution/gptaq.py` 适配只读输入和统一packed编码。

合法：Linear W，INT/UINT，per_channel(axis=0)，minmax校准；必须a_then_w/sequential，以已冻结A及逐层量化前缀取得真实量化输入。options为blocksize(128)、damping(.01)、actorder(false)、gptaq_clip(false，可显式启用官方80点L2.4网格裁剪)。同一W只能选择GPTAQ或GPTQ/AdaRound/其它舍入方法之一。

同一图像完整token对齐两条流：Xfp来自原始只读FP模型，Xq来自实际量化候选；不是以已量化前缀的浮点输出作teacher。累积H=2XqXq^T/N及dXXT=2(Xfp-Xq)Xq^T/N。上三角Cholesky逆U后，P=.25*triu(dXXT@U.T,1)@U。每列更新Wremaining -= error*Urow - w*Prow，跨block也包含更新后Wblock@Pcross。仅打开quantized_prefix或测量误差不等于本方法。

网格忠于此版本官方WeightQuantizer：signed用amax/(2^(b-1)-1)，unsigned用零包含区间；与旧GPTQ中点网格有别。AdaLog bias依赖W时先把统计输入还原成有效中心输入，避免重复补偿。保留原始FP输出做对齐验证，记录dXXT范数。

适配：逐图流式统计，输入A是项目独立的静态插件，不复用原仓库动态per-token A或整模型配方；当前不开放未经核对的Conv/输入组GPTAQ。空间O(in_features²)多个矩阵，代价高于GPTQ；FP32并关闭此计算的TF32。W/A、前缀、输入变换或calibration数据改变都使统计缓存失效。量化编码保存重载不重新校准；仅reference QDQ。
