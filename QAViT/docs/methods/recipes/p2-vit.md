## P2-ViT

<a id="p2-vit"></a>

**证据。** 综述 [43]；[Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer](https://arxiv.org/abs/2405.19915)，§III-C–D、Algorithm 1、§V-A/B；[官方仓库](https://github.com/shihuihong214/P2-ViT)。已读算法、混精度设置和公开 README，未复现硬件或完整搜索链。

### 第一层：原配方

在预训练模型上求浮点 scale，然后依据量化后的输入/输出误差选择相邻 PoT exponent；候选包含 `floor(log2 s)-1` 到 `ceil(log2 s)+1`。不是简单把 scale 舍入到最近二次幂。对 LN 后激活采用 PoT-aware smoothing，将通道变化按二次幂迁移到权重；残差输出使用相应粒度以控制误差。等价变换和搜索是离线操作，无任务训练。重定标比例化为位移，以服务硬件实现。

使用 **100 张 ImageNet 图**校准，验证 ViT-B、DeiT-T/S/B。原主要 W4、W8 比较不等于所有激活均为 A4；应从具体表格读取 W/A/attention 位宽。完整 mixed 分支只搜索 W∈{4,8}，先 Hessian/Pareto 粗分配，再遗传搜索细化：population 25，mutation/crossover 数量 10、概率 0.5，20 iterations。该混精度分支与固定 W/A 的 PoT scale 消融应分开登记。

### 第二层：primitive

PoT exponent 候选器、量化误差驱动的 scale 选择、PoT 平滑、位移重定标、coarse-to-fine 分配组成可独立调用的组件。QAViT 已有 PTF 与 SmoothQuant；P2 所补的是全链路 scale 的约束及重定标路径，而不是又一个 LN PTF 名称。误差目标需写明来自 X、XW 或权重本身，不能统一替换成 scale 的标量距离。

### 第三层：workflow

原文比较直接舍入 PoT、误差自适应舍入及 smoothing，并展示硬件。QAViT 可先在相同 bit 下做 FP scale、nearest PoT、adaptive PoT 三组，再测试 smoothing；最后才加入位宽搜索。测量 scale 元数据大小、位移次数和实际算子延迟；专用 FPGA 的数据流收益单列，不外推成 GPU 上的算法加速。

**协议与差距。** PoT scale 可以作为固定 4 bit 主线中的可选表示约束；完整 4/8 分配另归 mixed。需要支持 scale 格式、shift 范围与 accumulator 的导出语义。公开仓库有量化入口，但 README 不能证明完整 mixed 搜索和 RTL 已一并公开，代码证据分项记录。


## 接入依据与来源边界

**基线组件** PTF/PoT、SmoothQuant、梯度与量化基础。

**配方阶段与依赖** 补误差驱动 PoT 候选、PoT smoothing/requantization；完整法还需 Hessian/Pareto 与遗传 4/8 W 分配。

**部署条件** 固定 PoT 组件可主线适配；完整 mixed 与 FPGA 配置单列，W4 不等于 A4。

**尚需核对** 官方量化入口可核，完整搜索/RTL 是否公开未闭合；专用硬件收益不能继承到参考 QDQ。
