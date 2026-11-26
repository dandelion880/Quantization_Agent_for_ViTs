## HC-PTQ

<a id="hc-ptq"></a>

**定位与来源。** 综述 [39]，ICCV 2025 workshop，[HC-PTQ: Poincaré-Based Hyperbolic Clustering for Data-Free Quantization of Vision Transformers](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/papers/Mineo_HC-PTQ_Poincare-Based_Hyperbolic_Clustering_for_Data-Free_Quantization_of_Vision_Transformers_ICCVW_2025_paper.pdf)，§3.1–3.5、§4、Tables 1–5。已读原文；没有核实到作者代码。HC-PTQ 与 AHCPTQ 是不同方法。

### 第一层：原配方

不需要校准图像。将选定权重矩阵的行看作向量，经原点指数映射嵌入 Poincaré ball；映回切空间做 K-means，再将中心映回双曲空间。依据双曲距离为权重行分配最近中心，重建时再映回欧氏空间。可选择只量化 MHA、只量化 MLP 或两者联合；聚类更新的是码本，未进行任务训练。

该方法为**权重向量码本量化**。b bit 是码本索引预算，不能直接等同逐标量 WbAb。原文存储式为 `K×d×32 + N×b`，包含 FP32 码本。验证模型为 ViT-B/16/224；MHA-only、MLP-only、联合覆盖分别为约 32.7%、65.4%、98.1% 参数。联合 4 bit 设置的分类表现明显下降；原文还将结合激活量化作为后续方向。因此它是已发表的探索路线，不应以“成熟 W4A4 PTQ”身份排入主线。

### 第二层：可复用组件

有用的新增对象是**向量码本及真实存储成本模型**：索引位数、向量维度、码本精度、模块覆盖、解码路径需共同描述。曲率/映射、聚类初始化、停止准则等参数只有原文或作者实现明确支持时才能赋默认值。现有均匀/对数量化容器不能被视为已实现这类向量量化。

### 第三层：研究 workflow

原文比较三个模块覆盖方式与不同索引位宽。QAViT 可在权重码本扩展 profile 下首先比较欧氏行聚类与双曲距离分配，使用同一码本规模、相同模块覆盖和总存储预算，检验收益来自哪里。需同时测分类、ADE、实际文件大小及解码/算子成本。完整 W4A4 适配还需要另加激活量化，它是组合实验，不是原法复现。

**差距与协议。** 需要码本 artifact 和成本核算，而非增加一个 `bit=4` 别名。优先级低于已有 scalar PTQ 的成熟补完；保留权重码本研究资格，按作者代码缺失、单架构和低 bit 退化限制标注证据成熟度。


## 接入依据与来源边界

**基线组件** 权重访问、任务评价与通用成本记录；非现有 scalar quantizer。

**配方阶段与依赖** 补向量码本、双曲/切空间映射与聚类、索引序列化和真实存储计算。

**部署条件** b 位是权重行向量索引，另存 FP32 码本；归独立 weight-codebook 比较，不是 WbAb。

**尚需核对** 原未量化 A、低位联合覆盖明显退化、官方代码未核；不能称成熟全 W4A4 或用索引位数代替标量位宽。
