# 无真实校准数据的方法：合成数据、校准与权重码本

本文只吸收能够确认属于 PTQ 的路线。合成图片时对**图片像素**求梯度，不等于 QAT；用有限校准集做局部量化重构，也不自动等于 QAT。反之，数据来自合成图并不能把量化学生的完整训练变成 PTQ。PSAQ-ViT v2、MimiQ 等筛查结果放在覆盖账本，不向方法库录入其配方、组件或 workflow。

文中“原论文证据”表示论文做过的实验；“QAViT 适配实验”表示后续拟做的工作。本次没有在 QAViT 上运行这些算法。固定 4 bit 推理协议可以接纳离线搜索与有限校准优化，但原文只验证 W4A8 的方法不能直接标成已验证 W4A4。

<a id="psaq-vit"></a>

- [PSAQ-ViT](../../docs/methods/recipes/psaq-vit.md)

<a id="clamp-vit"></a>

- [CLAMP-ViT](../../docs/methods/recipes/clamp-vit.md)

<a id="dfq-vit"></a>
<a id="e2h-activation-correction"></a>

- [DFQ-ViT / QuaSC：E2H synthesis 与 activation correction](../../docs/methods/recipes/dfq-vit.md)

<a id="sardfq"></a>

- [SARDFQ](../../docs/methods/recipes/sardfq.md)

<a id="hc-ptq"></a>

- [HC-PTQ](../../docs/methods/recipes/hc-ptq.md)

## 接入现有 QAViT 时的共同落点

现有 `tools/ptq_agent/backends/image_data.py` 与 `tools/data_receipt.py` 应承接合成数据的可重放来源；`execution/calibration.py`、`calibration_search.py` 和 `frozen_trials.py` 承接有限校准预算与已有误差评价。建议让生成器交付标准样本及 metadata，量化器消费相同接口。主线搜索无需默认启用 data-free：当真实校准不可用、样本数敏感或需要检验数据覆盖不足时，已有研究技能才触发这些路线。总成本应包含合成、教师 forward/backward、搜索与重构，才能判断真实科研收益。
