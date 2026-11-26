## SARDFQ

<a id="sardfq"></a>

**定位与来源。** 综述 [85]，ICCV 2025，[Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Semantic_Alignment_and_Reinforcement_for_Data-Free_Quantization_of_Vision_Transformers_ICCV_2025_paper.html)，§3、§4.1、组件消融；[作者代码](https://github.com/zysxmu/SARDFQ)。早期预印本 [arXiv:2412.16553](https://arxiv.org/abs/2412.16553) 名为 SPDFQ。已读预印本和正式 ICCV 正文的 §3.4–4.1，核对两阶段的优化对象与预算，并核对公开生成与量化入口；正式版的细节优先级高于早期版本。

### 第一层：原配方

从高斯噪声生成校准图，利用随机的结构化 attention prior 对齐生成图的注意力；对局部区域单独优化，使图像具有多种语义，再以软标签表达多语义目标。随后采用有限合成集进行**逐块 PTQ 重构**，仅更新当前块的量化模型参数，使其输出接近浮点教师输出。这里存在模型参数优化，但原流程是局部校准重构，不是完整数据集的 QAT 训练。

已核实版本的基础设置为 32 张图、1000 次合成迭代，生成 Adam 的 β 为 (0.5,0.9)，Swin 学习率 0.25、其他模型 0.2；量化阶段 Adam 的 β 为 (0.9,0.999)，学习率 4e-5，weight decay 0，100 iterations，配合 cosine 调度。W 为 per-channel，A 为 per-layer，量化所有矩阵乘法；主要低精度结果包含 W4A4、W5A5、W6A6。原文针对不同模型设置不同 prior 系数，不能省略成“一个万能超参数”。

### 第二层：可复用组件

组件包括 attention-prior 生成与匹配、局部图像区域更新、软目标及逐块重构的数据接口。QAViT 已有 attention 输出等误差目标；新增重点是**让校准数据来源与重构器解耦**，以便把相同生成图交给不同已实现量化器，而不将生成方法与量化 backend 的贡献混在一起。需保存原图、局部区域、软目标三者的对应关系。

### 第三层：研究 workflow

原文用统一量化学习阶段比较各生成方法，并消融 prior、局部语义、软标签。QAViT 也应先固定局部重构算法与预算，比较生成源；之后固定生成源比较现有重构器。原论文分类结果不是 DINOv3 双任务验证，新增 dense 语义目标时必须另记变体，不能沿用“原论文已验证”标签。可检验 32/更多合成图的收益、合成预算与重构预算如何分配，但新增预算点应明确是适配实验。

**差距与协议。** 有限校准下的逐块 PTQ 优化符合新的主线边界；应在 capability 中分别声明像素更新、量化权重更新与其预算。当前欠缺合成来源与教师/图像/软目标绑定，不能把已有 block 调度名称当作 SARDFQ 重构的完成证据。官方代码存在生成和量化入口，落地时应锁定 commit、核对 README 示例与论文设置差异。


## 接入依据与来源边界

**基线组件** attention 观测、图片入口、有限重构数值基础。

**配方阶段与依赖** 补 attention prior、局部语义图像更新、软目标；以生成数据驱动独立有限块重构器。

**部署条件** 原含 W4A4，可按实际覆盖进入主线；有限校准更新当前块量化模型参数（含权重），不等于无参数更新。

**尚需核对** 按正式 SARDFQ 锁版本、模型 prior 系数和有效命令；不能将早期 SPDFQ 或库中同名入口混用。
