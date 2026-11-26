## PracticalEdge：受范围约束的整数非线性核

<a id="practicaledge"></a>

**真实身份与证据。** 综述 [26]，准确题名为 [Practical Edge Kernels for Integer-Only Vision Transformers Under Post-training Quantization](https://proceedings.mlsys.org/paper_files/paper/2023/file/12f429641e5c63a9ca7fd1c5c4804d32-Paper-mlsys2023.pdf)，MLSys 2023，Zining Zhang、Bingsheng He、Zhenjie Zhang。已读 §3 算法、§4 设置和部署结果；没有核实作者完整公开 kernel 仓库。

### 第一层：原配方

固定预训练模型，以校准数据选择整数算子的范围和移位参数。Softmax 先做指数范围缩减与多项式近似，再通过受范围约束的重定标和左移处理分母/除法，兼顾截断、溢出和小概率信息。LayerNorm 关注均值、方差、平方根计算中的 int32 范围；采用 uint16 输入以保留足够精度。GELU 也采用受控整数近似。这里更新的是校准参数与算子实现，无权重训练。

原文使用 **100 张 ImageNet 训练图**。Softmax 输入 uint8，其 `m=12`，逐层 `M` 用 output-oriented 或 loss-based 方法选择；LN 输入 uint16、`M=15`；GELU 输入 uint8；内部计算 int32。因而它不是全链路 W4A4 的原证据。部署为 Apple A13/M1 ARM CPU、NEON 内核与 Torch-Lite；论文分别报告 kernel 和端到端延迟，不能将约 5× kernel 加速改写成全模型 5×。

### 第二层：primitive

吸收整数范围分析、截断比例、定点除法与移位选择、独立算子误差、算子实际数据类型描述。现有 QAViT 的 fake quant 表示不足以验证这些问题。可复用现有 NMSE/误差评价，但增加整数溢出、截断、除零以及实际 accumulator dtype 的观测；这些是部署诊断，不应替代最终任务指标。

### 第三层：workflow

原文通过 uint8/uint16 LN、不同移位选择及单算子/整网延迟消融，说明范围错误会造成巨大精度下降。QAViT 应先把相同输入交给浮点参考、模拟整数公式、真实整数 kernel，核对数值，再比较整网 task loss。将其用于固定 A4 需重新推导范围与精度，是低 bit 适配实验。原法先进入 integer-operator 扩展 profile，作为“精度退化是否来自算子整数化”的可触发诊断。

**接入缺口。** 需有 operator implementation 与 quantizer representation 的分离、导出时实际 dtype/scale/shift 的持久化、reload 后数值一致性以及指定设备的 profiling。现有 PyTorch 计时不能证明 NEON 实现已经完成。


## 接入依据与来源边界

**基线组件** integer operator 接口、冻结 scale 状态和 profiling。

**配方阶段与依赖** 补范围受控整数 Softmax/LN/GELU、shift/dtype/溢出语义及真实 ARM 内核适配。

**部署条件** 原 Softmax/GELU uint8、LN uint16、内部 int32；它是整数算子证据，非全 A4。

**尚需核对** 公开 kernel release 未核；ARM NEON 算子收益不能写成任意平台端到端加速。
