## DFQ-ViT / QuaSC：E2H synthesis 与 activation correction

<a id="dfq-vit"></a>
<a id="e2h-activation-correction"></a>

**身份合并。** 综述 [78] 对应 [Data-Free Quantization for Vision Transformers without Fine-tuning](https://arxiv.org/abs/2507.14481)，[80] 对应 [Data-Free Quantization of Vision Transformers via Easy-to-Hard Synthesis and Activation Correction](https://doi.org/10.1145/3771936)。后者出版社摘要将方法称为 **QuaSC**，作者与机制均对应前者路线。应保留两个文献版本记录，暂按同一方法家族吸收，不能凭综述两行创造两个独立 baseline。以下可复现设置以已经读到的 arXiv 版本 §3.2–3.4、§4.1 为准；ACM 最终版全文尚未完整取得，因此不把预印本设置自动写成期刊定稿设置。

### 第一层：原配方

先生成合成图，后校准量化器并估计激活修正量。E2H 通过渐变裁剪难度，从较大的全局区域转向较小的局部区域，平衡全局与局部语义。生成期间更新图片像素，教师权重保持固定。Activation Correction Module 从合成输入统计全精度与量化模型中间激活差的均值，得到修正矩阵；校准和推理时在选定层加回该矩阵，以减轻分布偏移。它不是天然可折叠的权重重参数化，额外状态与加法应进入部署描述。

预印本使用 **16 张图、500 次合成迭代**；Adam，DeiT 学习率 0.25、Swin 0.2，裁剪范围下限 0.08、上限 1。权重为对称均匀量化，激活为非对称均匀量化；实验覆盖 W3A8、W4A8、W6A8、W8A8。不存在已核实的“同样设置成功 W4A4”结论。没有核实到作者公开实现，当前为论文级配方。

### 第二层：可复用组件

拆成难度调度器、合成图生成器和激活偏差估计/应用三个组件。E2H 只改变校准源；ACM 会改变推理图及状态格式。ACM 的平均维度、插入间隔、张量布局应成为显式配置，不能把完整 token×channel 修正矩阵默默简化为 per-channel bias 后仍叫原法。若采用简化或折叠版本，应登记为新变体并验证等价条件。

### 第三层：研究 workflow

原文有 E2H 与 ACM 的组件消融。QAViT 可先复用同一合成 seed 做“无课程/有课程 × 无修正/有修正”的小型因子实验；分别测合成目标、真实任务、修正张量大小与推理开销。再比较只在分类分辨率估计 ACM 和在 dense 任务形状上重新估计 ACM，检验空间依赖。对 DINOv3/ADE 的动态裁剪或多尺度输入，不能假定原矩阵自动通用。

**差距与协议。** 数据路径需要课程及生成记录；量化导出需要修正状态、shape 约束及 reload 一致性。E2H 可作为固定 4 bit 主线的输入来源；完整 ACM 只有在其额外推理状态、精度和成本被协议接纳时才进入该轮比较。否则保留 data-free 加补偿扩展 profile。两篇文献共享家族 ID，分别记录版本证据。


## 接入依据与来源边界

**基线组件** 合成数据接口基础、配对激活误差、常量状态。

**配方阶段与依赖** 补 E2H 裁剪课程、图片生成及激活差均值矩阵校正；与 e2h-activation-correction 共一家族。

**部署条件** 原 W3/4/6/8A8；E2H 可作主线来源，ACM 的位置状态/加法/表示按实际终图归属。

**尚需核对** ACM 不能直接简化为通道 bias；ACM 出版社最终版全文/官方代码未全核，A4 与变形状支持为新验证。
