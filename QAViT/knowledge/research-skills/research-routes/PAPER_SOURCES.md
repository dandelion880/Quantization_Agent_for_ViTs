# 量化论文路线案例：来源与使用边界

核对日期：2026-11-16。

这些文件用于学习研究决策，不是论文全文替代品，也不是原作者实验室历史的复原。每个案例把少量可核对的公开事实集中在第 1 节，后续给出本包作者为 QAViT 编写的研究设计。数学式使用清晰的教学记号；只有明确标出的对应式才归属于论文。案例中的“假想”片段没有实测数值。

## 原始来源与核对位置

| 文件 | 原始来源 | 已核对的定位 |
| --- | --- | --- |
| [ptq4vit.md](ptq4vit.md) | [arXiv:2111.12293v3](https://arxiv.org/abs/2111.12293v3)，[PDF](https://arxiv.org/pdf/2111.12293) | §3.2、Fig. 2–4；§3.3 Eq. 7、Fig. 5；§3.4 Algorithm 1；§4.2–4.3、Table 3 |
| [repq-vit.md](repq-vit.md) | [arXiv:2212.08254v2](https://arxiv.org/abs/2212.08254v2)，[PDF](https://arxiv.org/pdf/2212.08254)，[ICCV 2023 页面](https://openaccess.thecvf.com/content/ICCV2023/html/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.html) | Algorithm 1；§3.2 Eq. 13–17；§3.3；Fig. 2–3；§4.4 Table 3–4 |
| [gptaq.md](gptaq.md) | [ICML 2025 / PMLR](https://proceedings.mlr.press/v267/li25dn.html)，[arXiv:2504.02692v3](https://arxiv.org/abs/2504.02692v3)，[PDF](https://arxiv.org/pdf/2504.02692) | Fig. 1；§4 Eq. 4；§4.1–4.2、Algorithm 1；§5.1–5.2；§5.5 |
| [erq.md](erq.md) | [ICML 2024 / PMLR](https://proceedings.mlr.press/v235/zhong24a.html)，[正式论文 PDF](https://openreview.net/pdf?id=jKUWlgra9b) | §3 Eq. 4；§4.1 Eq. 5–8；§4.2；Table 4；§6 |
| [adalog.md](adalog.md) | [ECCV 2024 / ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3969_ECCV_2024_paper.php)，[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03969.pdf) | §3.2 Eq. 5–10；§3.3 Eq. 11–15；§3.4 Algorithm 1；§4.1；Table 4–5 |

RepQ-ViT 的细节依据作者 arXiv v2 原文核对；ERQ 采用 ICML 2024 会议版本。复用路线时应保留这些版本边界，后续扩展版的新增方法与实验不能自动记到原版本中。

## 使用案例时的证据习惯

- 引用已发表结果时，重新核对模型 checkpoint、数据、位宽、激活静态/动态、粒度及任务。论文的相同缩写不保证协议相同。
- 对研究策略可以做类比迁移；对具体超参数、性能数值和硬件收益需要独立依据。
- 读到一个方法并不等于当前工具已经实现该方法。源码中的插件名称也不足以证明公式和校准语义一致。
- 来源支持的机制与本包提出的实验建议应分别记录。后者经过真实实验后才能成为 QAViT 自己的证据。
- 案例里的浮点撤回、W-only 或 A-only 只是可选诊断；最终候选继续遵守项目指定的 static W4A4 与 training-free 约束。
- 研究结论应允许“当前证据不支持”“仅适用于这些位点”“需要在另一模型再验证”等范围表述。不要把失败的工具请求当作科学反证。

## 内容选择原则

PTQ4ViT 用于学习指标与表示的交互；RepQ-ViT 用于学习稳定维度和等价变换；GPTAQ 用于学习输入—目标来源与传播；ERQ 用于学习耦合误差与顺序补偿；AdaLog 用于学习如何有依据地扩展搜索空间。它们互补，但不是固定串行流程，也不要求每个研究问题都引用全部五篇。
