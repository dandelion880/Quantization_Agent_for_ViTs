## 21. QwT（CVPR 2025）

<a id="qwt"></a>

**身份与来源。** *Quantization without Tears*，综述 [15]。[会议原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Quantization_without_Tears_CVPR_2025_paper.pdf)，[作者修正版](https://cs.nju.edu.cn/_upload/tpl/00/ed/237/template237/paper/CVPR2025_QwT_Corrected.pdf) §3；[官方代码](https://github.com/wujx2001/QwT)，实际核查 QwT-cls-RepQ-ViT/qwt_vit_and_deit.py 的 generate_compensation_model 与 linear_regression。

**paper-recipe。** 对量化块增加并行线性补偿 \(C(X_q)=W_cX_q+b_c\)。关键参考语义是：浮点块与量化块接受**同一个当前量化前缀输入** \(X_q\)，拟合 \(E=B_f(X_q)-B_q(X_q)\)，即 QAViT 的 local_fp，而非直接拟合完整浮点前缀输出。令 \(\bar X=[X_q;\mathbf1^\top]\in\mathbb R^{(d+1)\times M}\)，闭式解
\[
[W_c\ b_c]=E\bar X^\top(\bar X\bar X^\top)^\dagger,\qquad
Y=B_q(X_q)+W_cX_q+b_c.
\]
这里 \(\dagger\) 表示数值最小二乘；具体 inverse/lstsq 与数值条件按实现锁定。官方按块顺序拟合，加入补偿后产生下一块输入；以 \(R^2>0\) 接受拟合，否则用零补偿。每次拟合都对应当前状态，不能把未补偿前缀缓存一直沿用。作者 512 张 ImageNet，补偿矩阵保存 FP16；CNN 采用分组 1×1 卷积方案。

论文讨论累计误差，但其上述实际拟合参考不直接等于 full-FP teacher；逐层改进可能减少后续误差，不能据讨论文字修改算法含义。零分支允许拒绝不可靠局部拟合，也不提供 heldout 任务保证。

**primitive。** affine_residual_fit 输入 current block 的 paired local outputs 与 qprefix input，输出 matrix/bias、拟合秩、条件数、\(R^2\) 和增量运行成本。若增加 ridge、低秩约束或改为 fp_model target，需独立 variant ID，均不能默认称原 QwT。量化器/重构权重变化后必须重拟合该块和其下游。

**workflow 与接入。** QAViT 可用现有 paired hooks 采集，增加 streaming covariance/least-squares 与 adapter candidate；先验证 local_fp 和 fp_model 两种目标分别改善什么，再比较 BiasComp、QwT-v2。QwT 的固定 W4A4 backbone 外保留 FP16 补偿矩阵，属于扩展 profile；总参数、矩阵乘、输入存活期及 memory traffic 要计入，不作为纯 W4A4 等成本基线。作者有 TensorRT/RTX3090 测量，可做原平台证据；DINO/目标后端需重新测。状态：原文、修正版与关键作者代码已核；未复现。


## 接入依据与来源边界

**基线组件** paired hooks、块边界、传播诊断。

**配方阶段与依赖** 补同一当前 Q-prefix 输入上的 local FP/Q 残差矩阵最小二乘、R² 接受规则、逐块补偿与刷新。

**部署条件** 原保留 FP16 矩阵补偿分支，属于额外计算；计矩阵/输入存活/带宽，非纯 W4A4 同成本。

**尚需核对** 现 local_fp_parent 在改上游候选时不是所需 local_fp_candidate；ridge/低秩/full-FP target 都是另名变体。
