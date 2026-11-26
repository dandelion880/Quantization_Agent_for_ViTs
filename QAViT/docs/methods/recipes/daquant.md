## 12. DAQuant：域对齐是校准优化目标，不是校准图像选择器

<a id="daquant"></a>

**证据。** 综述 [30]；[论文 DOI](https://doi.org/10.1016/j.patcog.2025.112182)，Pattern Recognition 171（2026）112182；[出版社条目](https://www.sciencedirect.com/science/article/pii/S003132032500843X)。此次未取得可读的论文全文，但已取得并核读[官方 PR26-DAQuant 仓库](https://github.com/iLearn-Lab/PR26-DAQuant) main 的 19 个源码文件，包括 `DAQuant.py`、`quantizer.py`、`int_vit_layer.py`、`datautils.py` 与 README。以下细节凡来自源码均明确按“官方代码快照”而非论文式号记录。

### 官方代码可重建的原配方

代码把 block 逐个转成量化模块，用分布统计初始化平滑 scale/shift，再优化 DGA、LWC、LAC、WRC 参数组。原始模型参数冻结，但 WRC 是与 W 同形状的可学习 compensation，不应称为完全不更新权重的无优化方法；它在量化前加入 W，最后可合入量化 W。`generate_act_scale_shift.py` 的统计包括绝对激活的通道标准差和通道均值；不要只套用 SmoothQuant 的 absmax 初始化。[统计源码](https://github.com/iLearn-Lab/PR26-DAQuant/blob/main/generate_act_scale_shift.py)、[量化源码](https://github.com/iLearn-Lab/PR26-DAQuant/blob/main/quantize/quantizer.py)

基础块损失为源域 FP 与量化输出的 SmoothL1。启用域对齐后，代码再用源域 quant_out 的 log_softmax 与目标域 FP block output 的 softmax 计算 KL，取绝对值乘权重；softmax 沿特征轴 dim=2。这是特征分布对齐，不是标签分类 KL，也不意味着源/目标 batch 中的样本语义天然配对。[`DAQuant.py`](https://github.com/iLearn-Lab/PR26-DAQuant/blob/main/quantize/DAQuant.py)

README 给出 20 epoch 量化、10 epoch 域适配示例，默认 32 样本；AdamW 参数组学习率在 `main.py` 为 DGA 5e-3、LWC 1e-2、LAC 1e-4、WRC 1e-5，域对齐示例权重 1.5。普通数据 loader 随机抽样/打乱，不是领域感知的样本子集优化器。公开任务为分类/域适配，本次未找到语义分割的原文或官方实验，不能向用户承诺已有跨分割验证。

### primitive、运行时与当前差距

建议 `init.distribution_scale_shift`、`opt.learned_clip_and_weight_correction`、`loss.domain_feature_alignment` 三个 primitive 分离。输入必须同时携带 source/target 的样本 ID、域标签、teacher 输出和 batch 配对规则。已有 SmoothQuant/GPTAQ 可作为消融，但不能视为 DAQuant 原样实现。

官方 quantizer 每次 forward 重新求范围。CLI 名称为 per_token，但 `lac=True` 时沿 dim=1 求极值，对 B×N×C 张量实际上是跨 token 的通道范围；不能仅看参数名归类。static 版本须实现明确冻结观察器及固定分组后重新优化/评估。README 的 AutoGPTQ 实际部署命令是 W4A16，不是原生 W4A4 加速证明。[参数入口](https://github.com/iLearn-Lab/PR26-DAQuant/blob/main/main.py)、[部署说明](https://github.com/iLearn-Lab/PR26-DAQuant)

### 研究工作流

拟议 `study.domain_calibration_transfer` 分离三件事：校准样本来自哪个域、怎样选样本、损失是否含域对齐。先固定子集比较损失，再固定损失比较选样方法，防止两种变量共同改变而无法解释。对用户双任务，应检验同一领域校准是否改善分类同时损害密集预测，并用既有双任务指标做 Pareto 报告。没有 target 数据时，不能假装执行了原来的域对齐配方。DINO 结构属于需要单独验证的新适配。


## 接入依据与来源边界

**基线组件** SmoothQuant 类变换、捕获、均匀网格、任务数据入口。

**配方阶段与依赖** 补分布 scale/shift 初始化、DGA/LWC/LAC/WRC 参数组、源/目标域特征对齐及 block 调度；WRC 合并。

**部署条件** 所核代码 forward 重估范围，属动态；冻结观察器的版本需独立验证。W4A16 部署不是 W4A4。

**尚需核对** 正文未取得、已有官方源码级配方；不冒称论文全文已核。随机 loader 不是选样算法，未见分割验证。
