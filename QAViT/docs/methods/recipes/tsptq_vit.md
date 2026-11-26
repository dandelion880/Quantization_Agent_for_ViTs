## 6. TSPTQ-ViT：两个“双尺度”分别解决数值分布与通道异常

<a id="tsptq_vit"></a>

**证据。** 综述 [20]；[原论文 PDF](https://arxiv.org/pdf/2305.12901)，ICASSP 2023，§3.1–3.2、图 2–3、§4；已读全文，未核实官方实现。

### 原配方

V-2SF 对 post-Softmax/post-GELU 分两种数值区间：小值保留较低有效位，大值保留较高有效位，两个 scale 由移位关联。Softmax 的 region bit 在总位宽内；GELU 低幅区还要占一个符号位，高幅区利用其大值为正的性质省去符号。原文采用 Softmax 的尺度移位差 4、GELU 的差 3。不能在目标 4bit 的 payload 之外再免费增加 region/sign bit。[原文 §3.1、图 2](https://arxiv.org/pdf/2305.12901)

O-2SF 作用于 LayerNorm 输入：根据各通道绝对最大值做 K-means，识别异常通道和普通通道，为两组分别选择 scale；最后将二者约束为 power-of-two 比例，以便移位对齐。Hessian-guided 目标与 W scale 交替搜索。原文校准 32 张、搜索 3 轮、普通候选 100、最终移位候选 6；主要实验为 6bit/8bit，包含首尾层并使用整数 Softmax 近似。[原文 §3.2–4](https://arxiv.org/pdf/2305.12901)

### primitive 与顺序

建议 `rep.bit_window_two_scale` 输出包含 region 的完整码字及解码表；`group.static_outlier_channels` 输出固定通道标签；`search.shift_aligned_scales` 在有限移位集合内搜索可对齐的尺度对。它与已有 PTF 的差别是两组共享两个尺度，不是每通道各自从多个指数中选择；与 PTQ4ViT twin 的差别是实际 bit-window 结构，不能只复用名称。

已查看原始 PDF 图 2：6bit Softmax 从 9bit 初始整数出发，低区阈值为 \(2^5\)，保留 5 个低位；高区保留 5 个高位并舍入首个截断位，另加 1 个 region bit。GELU 的低幅阈值为 \(2^4\)，低区是 region+sign+4 位，高区是 region+5 位正数。8bit 对应阈值分别为 \(2^7\)、\(2^6\)。解码应逐位核对恢复移位，不把 payload 当作同一线性整数。

原文文字写出的 \(s_l=2^{-m}s_s\) 与“保留高位后恢复较大间隔”的步长方向存在符号含义疑点；本包把它列为公式核对项，不静默照抄成错误解码。先固定 token 语义和通道范围统计，再分组、交替搜索 W/A、最后约束尺度比例。W 或输入边变化后应重新评估分组，而不是沿用过期最大值。

### 研究工作流与当前差距

QAViT 有 twin/PTF/Hessian 相关零件，但没有 TSPTQ 的完整编码与两组通道搜索。拟议 `study.metadata_vs_resolution` 比较每通道 1bit 组标签与多指数元数据的成本，同时观察小值区和大值区各自的误差。static W4A4 可以作为固定分组与固定两尺度的新适配。由于 region/sign 也占用总位宽，应先做码本容量分析，再运行完整任务；原文较高位宽验证不能替代这种论证。


## 接入依据与来源边界

**基线组件** twin/PTF 基础、通道统计、Hessian 代理与尺度搜索。

**配方阶段与依赖** 补 V-2SF 的区域/符号码字、O-2SF 的通道 K-means 与 PoT 尺度比；保持交替搜索原顺序。

**部署条件** 固定两组/两尺度可静态；原 6/8 bit 的字段常量不能直接搬到四位。

**尚需核对** 尺度符号方向存在局部歧义；官方代码未核。必须从总 bit 内分配 region/sign，不能加免费标签。
