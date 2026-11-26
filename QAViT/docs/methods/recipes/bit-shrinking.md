## 12. Bit-Shrinking（CVPR 2023）

<a id="bit-shrinking"></a>

**来源。** 综述[33]，[CVPR原文](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf) §3、Algorithm1–2、§4。未确认可用官方代码。

**paper-recipe。** 作者明确是小校准数据的PTQ：逐块更新权重，由高bit向目标bit自适应过渡，避免一次注入过大量化噪声。sharpness指相同权重下量化扰动带来的loss增量，区别于权重本身偏离预训练模型的重构误差。先以32张校准图测目标bit的总sharpness \(S_b\)，从8bit开始，选择下一连续bit \(b'\) 使相邻sharpness增量不超过阈值；通过二分匹配阈值，再对当前bit的块输出重构优化，直到目标整数bit。

核实设置为 \(\tau=.04S_b\)，二分容差为总sharpness的.01；每个bit阶段3200次，Adam初始lr5e-6、ReduceLROnPlateau。高bit的scale从目标scale按 \(s_{b'}=s_b2^{b-b'}\)得到。若目标W2A4，先W/A同时降到4，再只降W到2。原文所谓连续bit是校准中量化分辨率的连续控制，不要求推理支持分数bit。最终再在目标bit重构一次。本文没有从可访问材料确认整套数据抽样数量与所有首尾精度例外，不能只据“用32张测sharpness”推断总校准预算32张。

**primitive。** quant_noise_sharpness、continuous_precision_schedule、adaptive_bisection_step。保存每块实际bit序列、搜索次数、阈值和优化步数；同目标bit时不同块可能走不同路径，总预算不是固定3200次。该调度可以驱动一个有限校准块优化器，不能拆成“任意全数据训练的降bit方案”后仍列入本PTQ库。

**workflow 与接入。** 比较直接4bit、固定8→6→4、按sharpness自适应，既对齐原配方也对齐总F/B预算，检验收益来自轨迹还是多花计算。主论文ViT重点为W6A6/W8A8，低到4bit的突出证据主要CNN；ViT W4A4属于需验证迁移。最终固定4bit符合主协议；优化中渐变bit无需放进动态推理组。状态：原文算法与调度超参已读，官方代码与完整精度/数据配置待核。


## 接入依据与来源边界

**基线组件** 局部误差、权重访问、最终网格表示。

**配方阶段与依赖** 补量化 sharpness、连续分辨率、二分选下一 bit、逐阶段局部权重重构及最终目标 bit 重构。

**部署条件** 仅构造阶段变 bit，终态固定可主线；不是运行时动态位宽。

**尚需核对** 整套数据量、首尾例外和官方代码未闭合；ViT 原主要 6/8 bit，A4 是适配而非已验证结论。
