# 从成熟量化工作中抽取“发现—机制—方案—验证”研究策略

本文件不是论文方法手册，也不是必选模块清单。**迁移的是解决一类误差的思考方式，而非机械拷贝模块。** 具体公式、模型适配、可用算子和实现要求到项目 `docs/methods/` 或原论文再查。每一行的“何时失败”与“如何检验”同样重要。

| 研究路径 | 观察与假说 | 论文采用的关键思路 | 适合 QAViT 尝试的策略 | 最小反证或边界 |
| --- | --- | --- | --- | --- |
| **PTQ4ViT：先识别特殊分布** | ViT 中 Softmax/GELU 相关激活的分布特性使简单均匀量化失效 | 针对相应激活设计 twin uniform quantization，并用 Hessian-guided metric 筛选尺度 | 当误差集中于特殊非线性输出时，先分析码点利用率和损失敏感性，再考虑分段网格与更合适的标定目标 | 改善是否真的来自网格，而非增加搜索候选/校准预算？异常边的收益能否传至下游？ |
| **RepQ-ViT：校准表示与推理表示解耦** | LayerNorm 后通道幅度差异大；Softmax 后呈幂律分布 | 校准时采用更合适的通道尺度/对数式量化，再进行 scale reparameterization，使推理形式可执行 | 对“共享尺度吞没小通道”先验证通道级差异，再探究是否存在可证明或可数值验证的等价吸收路径 | 特定模块中有无可吸收的相邻运算？分支、残差、bias、量化器与 LayerScale 是否使等价失效？不能只验证校准态 |
| **GPTAQ：重新审视量化器所见输入与所追目标** | 逐层量化后的真实输入已偏离 FP 输入，独立层内重构不能覆盖累计失配 | 在量化前缀输入下校准，仍拟合完整 FP 模型对应层输出（asymmetric calibration） | 当局部重构优秀而长程传播差时，明确区分拟合输入、优化目标、冻结前缀状态，做输入×目标的小型对照 | 量化前缀下是否真正改进下游？结果是否依赖校准集和顺序？“异步参考”不等于自动因果改善 |
| **ERQ：权重和激活误差不能完全独立处理** | W/A 联合量化具有耦合，某一端的局部改善可能被另一端抵消 | 先做 activation quantization error reduction（包含 ridge-regression 型权重调整），再通过 weight quantization error reduction 处理权重误差 | 若 W-only、A-only 与 joint 差异明显，尝试针对耦合误差的阶段补偿，并检验先后顺序 | 仅额外引入更多自由度是否就能带来同等收益？优化一端后另一端误差是否重新扩大？最终权重是否都正确量化？ |
| **GPTQ：权重重构是成熟基线，但不能替代 W/A 机制判断** | 权重舍入误差可利用局部二阶近似和逐次补偿减轻 | 局部权重量化及误差补偿作为可复用机制 | 以完整已执行方案库中的 GPTQ 类候选作可比起点，分离 W4A32 的成功与 W4A4 的额外 A4 瓶颈 | W4A32 好不推出 W4A4 好；单换权重算法未必解决激活分布/输入前缀问题 |

## 怎样真正借鉴这些论文的科研过程

1. **从具体失败转为机制陈述**：不要停在“block2 差”。明确“哪个量化边、在哪些 token/channel、因为何种压缩表示丢失信息”。
2. **针对机制提出新自由度或新目标**：是换码点布局、重新分配 scale、改变校准输入/目标，还是消除误差耦合？要指明新自由度怎样改变预测。
3. **选能够排除替代解释的实验**：例如搜索预算相同的 shared-scale vs per-channel；相同候选的 FP-prefix vs quantized-prefix；相同参数量的 scale-only vs compensation。
4. **在真实部署约束下闭合方案**：校准时可用 FP 或辅助表示；最终 static W4A4 推理保留的权重/激活量化、码点数、FP 例外以及算子范围必须符合当次 ResearchBrief。
5. **把论文的强结论降格为当前模型的可证伪假设**：DINOv3、DeiT、Swin、分类与冻结分割头之间的差异，不允许用论文原实验替代本模型实验。

## 用于开放创新的组合原则

允许跨路径组合，但要解释交互。例如：先用 RepQ 式通道表示改善网格，再以 GPTAQ 式真实量化前缀做非对称标定；其科学增量不是“两个论文名称相加”，而是一个明确的命题：**重新分配尺度后，后续量化前缀的误差传播是否仍然限制性能？** 至少保留只做第一种、只做第二种以及组合这三者中真正必要的对照。若预计高成本，可先选最可能改变结论的一项。

另一个开放例子是非均匀码本或周期函数变换，但必须说明：有限码点怎样落地、逆映射/函数算子留在哪个精度路径、对异常值和常见值的误差怎样权衡。数学上值域有界本身不等于低比特量化误差小。

## 原始文献（查定义及证据时优先）

- PTQ4ViT (ECCV 2022): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720190.pdf
- RepQ-ViT (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.html
- GPTAQ (ICML 2025): https://proceedings.mlr.press/v267/li25dn.html
- ERQ (ICML 2024): https://proceedings.mlr.press/v235/zhong24a.html
- GPTQ (ICLR 2023): https://arxiv.org/abs/2210.17323

