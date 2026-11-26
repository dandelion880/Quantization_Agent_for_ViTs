## 2. APQ-ViT（ACM MM 2022）

<a id="apq-vit"></a>

**身份与来源。** *Towards Accurate Post-Training Quantization for Vision Transformer*，综述 [17]。[原文](https://arxiv.org/pdf/2303.14341) §3.2–3.4、Algorithm 1、§4.1；[正式 DOI](https://doi.org/10.1145/3503161.3547826)。其 arXiv 上传于 2023，不能据此改写会议年份。官方代码本次未确认。

**paper-recipe。** BBC 将候选 scale 对单层的影响观察到整块输出，再按任务梯度平方加权。令块误差 \(e=\operatorname{vec}(\hat O-O)\)，\(g=\nabla_O L_{\rm task}\)，基础近似为 \(\sum_i g_i^2 e_i^2\)。Bottom-elimination 再依据误差绝对值分位数屏蔽较小误差。缓存浮点块输出及梯度后，逐块、块内逆层序交替搜索两个矩阵输入的 scale。Softmax 使用 MPQ：非负均匀编码 \(q=\operatorname{clip}(\operatorname{round}(A/s),0,2^b-1)\)，\(s=\max(A)/(2^b-1)\)；这不是对数量化，意在保护较大的注意力值。

作者 ImageNet 使用 32 张图，COCO 使用 1 张图；分类 scale 范围 [0,1.2]（scale=0 的候选如何跳过或截断需实现显式处理），检测 [0.5,1.2]，100 个候选、3 轮交替，bottom-elimination 超参 \(\gamma=10\%\)。这里存在原文内部歧义：§3.2 文字说去掉小误差，但 Eq.(8) 印出的比较方向与此不一致，且式中误差绝对值位置不够一致。文档不能替作者静默修正；首次实现应保留 “literal-eq8” 与 “text-intended-top-error” 两个明确变体，并用原论文曲线/作者反馈裁定。MPQ 的 max 是原实现使用何种 batch/head/reduction 轴，本次也没有代码证据，必须通过 recipe 参数显式定义。

**primitive。** 新增的差量是“块级候选响应＋误差分位 mask＋任务梯度权重”的组合，而非新增一个名叫 hessian 的标量。敏感度缓存与候选响应分开：模型/head、参考精度、输入来源改变后重估 \(g\)；候选改变后重算 mask，不能把基线 mask 冻结当原配方。

**workflow 与接入。** 基于现有 frozen scorer 的同候选评估，拆开 block observation、gradient weighting、bottom mask、MPQ 四个因素做对照，记录 metric 改善是否转成 heldout 任务增益。作者验证 ViT/DeiT/Swin 分类与检测；本方法可作为 static W4A4 配方，但本次标记为原文已读、公式歧义待闭合，不能直接宣称与作者数值相符。


## 接入依据与来源边界

**基线组件** 梯度平方评分、位点/块观测、尺度搜索。

**配方阶段与依赖** 补 BBC 块级响应、逆层序交替搜索、候选相关 bottom mask、MPQ 的均匀 Softmax 网格。

**部署条件** 静态四位可接入；MPQ 的 max 归约轴及参数冻结行为要明确。

**尚需核对** Eq.(8) 与文字的 mask 方向不一致，max 轴/零 scale 候选处理待闭合；不得静默修正成唯一原法。
