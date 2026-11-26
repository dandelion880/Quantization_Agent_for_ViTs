# 已有方法的配方与研究语义补全

本文件处理综述中的 PTQ4ViT [16] 与 RepQ-ViT [19]。QAViT 已有相关数值组件，因此它们不计作全新的算法实现。需要补全的是原始配方、作者实现差异、组件组合顺序和研究用途。本文未执行原法复现。

<a id="ptq4vit"></a>

- [PTQ4ViT：twin 表示与 Hessian-guided 搜索是一套配方的不同部分](../../docs/methods/recipes/ptq4vit.md)

<a id="repq_vit"></a>

- [RepQ-ViT：先建立准确的细粒度网格，再转换部署表示](../../docs/methods/recipes/repq_vit.md)

## 与其他既有基础的关系

RTN、GPTQ、GPTAQ、AdaRound、SmoothQuant、PTF、log2 及 bias transform 在当前 QAViT 中已有不同程度的数值支持。新方法卡中的“复用”指这些组件确实存在，不代表相应论文完整原法已复现；更不代表它们在新任务、新位宽上已获验证。FQ-ViT、AdaLog 及其当前差距已在表示与分布文稿中补写；其它作为共用基础的公式和适用条件沿现有文档维护。
