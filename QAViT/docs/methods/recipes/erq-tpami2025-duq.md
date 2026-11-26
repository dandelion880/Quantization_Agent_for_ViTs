## 6. ERQ 扩展版与 DUQ（TPAMI 2025）

<a id="erq-tpami2025-duq"></a>

**身份与来源。** *Towards Accurate Post-Training Quantization of Vision Transformers via Error Reduction*，[v2 原文](https://arxiv.org/html/2407.06794v2) §4.2.1–4.2.3、Algorithm 1–2、§5.1；[DOI](https://doi.org/10.1109/TPAMI.2025.3528042)、[官方代码](https://github.com/zysxmu/ERQ)。已读 test_quant_expand.py、quant/quantizer.py。DUQ 是此版本新增的重要差量。

**paper-recipe。** LN 重参数化把不同通道尺度转入 W，可能产生集中在部分输入列的离群值。DUQ 为输入列选一个静态集合 \(\mathcal O\)：逐输出行识别上下尾部离群权重，统计各列出现次数，取频次最高的列；对 \(\mathcal O\) 与其补集分别使用 b-bit uniform 量化器，两个分区都为4bit，不是离群值 FP 旁路。随后仍执行 Aqer→DUQ/Wqer、舍入翻转与 ridge 剩余权重修正。

论文阈值写行内1/99百分位，\(|\mathcal O|=.05d_{\rm out}\)；当前代码的 best_top_number=x.shape[0]//20 也按输出行数决定“选多少输入列”，并对正权重99分位、负权重1分位分别计算阈值。因此不能擅自改成 .05d_in 并称原 recipe。应记录 output-based-count；如果改成输入通道比例，是可研究的新变体。全正/全负行、空集合、选中列数超过可用列等边界原代码未充分防御，移植时需要显式处理并列为实现差异。

核实超参：k=1、T=20；ImageNet32张、COCO1张、DIV2K8张（48×48 patch）；常规 ViT/DeiT-S/B/Swin λ1=λ2=1e4，DeiT-T为1e3，检测/SR另有值，不能用一个默认覆盖全任务。官方求解使用未归一化矩阵和 args.coe 的映射，与论文 expectation 记法需核对；bias 用增广常数列参与拟合且不按权重一样舍入。post-Softmax 为 log√2，其编码容量应另核查。

**primitive。** static_tail_column_partition 输出列索引、两组 quantizer 参数与重排映射，依赖 LN/scale 重参数化后的 W。分区确定后再做权重修正；若 W 变化后重分区，量化器和二阶矩切片随之重建。额外成本包括两组 scale/zp、索引或静态重排与分组 GEMM，不等于零成本。

**workflow 与接入。** 在 ERQ 2024 上增量比较 DUQ on/off、原输出维度计数/输入维度比例、保持分区/重估分区，评价 heldout 任务和真实导出图。固定分组、固定参数可归 static W4A4；与 per-output-channel 单量化器比较时同时报 metadata、算子拆分和延迟。作者有分类、检测/实例分割和 SR 结果，不直接证明 ADE20K/DINOv3。状态：原文与关键源码已核，无本地执行结果。


## 接入依据与来源边界

**基线组件** 上一行 ERQ 底座、固定分组/量化存储基础。

**配方阶段与依赖** 先完成重参数化与 Aqer，再在该 W 状态上确定 DUQ 尾部列分区，接 Wqer 舍入与剩余权重修正；独立记录 T=20 和代码求解尺度。

**部署条件** 两个分区均真四位且成员固定时可主线；另计索引、scale 与分组算子。

**尚需核对** 选列数原按 d_out 计算；空/单符号行边界、sum/mean λ、log√2 容量要核。是 [37] 变体，不是第 84 条文献。
