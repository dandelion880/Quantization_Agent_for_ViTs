## 13. RegCache：先重塑 token 结构，再给底层 PTQ 校准

<a id="regcache"></a>

**证据。** 综述 [32]；[原文 v5](https://arxiv.org/html/2510.04547v5)，2026-07-08 版本，记录为 ECCV 2026；已读 §3–4、§5、附录 E/H/L/N；[官方仓库](https://github.com/spbob0418/RegCache) main 已取得，核读 README、候选提取/搜索脚本及对应模型入口。原文有多模型、多 PTQ 基线实验，但不是用户 DINOv3 的既有实证。

### 原配方与实际数据成本

先逐层量化定位敏感层，再从参考图像取该层及之前若干层的大范数 token，平均候选对应的 KV，缓存成可重复插入的 register 前缀；最后在敏感块在线删除残余的大范数 patch token。原文从 ImageNet 训练集 **50,000 张**选 top-100 候选 token，100 不是校准图像数。前缀副本搜索 1–15 个，默认删除数搜索到 10；基线 PTQ 本身的校准集另计。[原文 §4.1–4.3、附录 E](https://arxiv.org/html/2510.04547v5)

主配方按参考任务精度选择配置，附录 L 提供无标签输出 MSE 选择。两者应分别登记，不能把有标签搜索的成本隐入“training-free”。附录 E.2 的基线均为 per-tensor dynamic，删除对象也是根据当前输入的 token 范数选择。对 DINOv2，作者发现仅在选定层插入可能优于一路插到末层，因此插入范围不能写死为所有后续层。

### primitive 与依赖

`structure.register_candidate_bank` 输入参考池、敏感层与原始 token 布局，输出带图像/层/位置来源的候选；`structure.cached_kv_prefix` 输出固定缓存及插入范围；`structure.outlier_token_policy` 输出在线删除索引与恢复映射。前缀结构稳定后再运行底层 A/W 校准。每条 attention mask、位置编码、special token 和残差路径都要引用实际布局，不能把 prefix 当作普通 patch。

QAViT 已有 CLS/special/patch 结构描述及传播指标，可直接用于对齐；缺的是缓存 KV 插入、可变 token 结构及密集预测恢复。原文 ADE20K 的 W8A8 SigLIP-B/16 用周围 patch embedding 均值插值恢复删除位置，不能将其结果移植成 4bit DINO 语义分割保证。[原文附录 H](https://arxiv.org/html/2510.04547v5)

### 研究工作流与配置归属

原配方纳入 `ptq_dynamic_structure`。若改为固定删除位置或仅前缀不删除，均是新适配，不能使用原方法的结果。拟议 `study.register_mechanism_applicability` 先测试目标模型是否已经有 register、异常 token 是否跨图像稳定、出现在哪层，再决定候选搜集预算。原文脚注明确 DINOv3 是已包含 register 的例外，并无 DINOv3 实验；已有 register 也不能自动推出完全没有 outlier，应以实测决定是否值得接入。

这项工作的研究价值是将异常值视为模型结构功能的结果，而不只调整 quantizer。`metric.prefix_outlier_suppression` 应同时记录异常范围、被删除 token 的语义损失及恢复误差；分类改进不等于密集预测改进。参考池、敏感层搜索、前缀/删除搜索与底层校准分别计费，便于与小样本 PTQ 公平比较。


## 接入依据与来源边界

**基线组件** CLS/special/patch 布局、逐层误差与传播诊断。

**配方阶段与依赖** 补候选 token/KV bank、缓存前缀插入、当前输入 token 删除与 dense 恢复；结构定后重校准底座。

**部署条件** 原动态 PTQ 加在线删除，属动态/结构扩展；只前缀或固定删除是独立变体。

**尚需核对** 候选来自 5 万图、top-100 是 token；DINOv3 自带 register 且无原文实证，先验不能替代适用性验证。
