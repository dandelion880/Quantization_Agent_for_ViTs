## 9. PFCR-ViT（Neural Networks 2025）

<a id="pfcr-vit"></a>

**来源。** 综述[76]，[原文](https://arxiv.org/html/2412.14633v1) §3.2–3.3、Algorithm 1、§4.1.2；[正式论文](https://www.sciencedirect.com/science/article/pii/S089360802500437X)。本次未确认作者公开实现，不能把同名推荐系统仓库当 PFCR-ViT 代码。

**paper-recipe。** PFCR 将含shortcut的 MHSA 与含shortcut的 MLP 作为最细单元，两个单元合为一个 block，再以二叉组合逐步扩大重构范围。每级继承上一级已优化权重，最小化当前组合单元 FP/Q 输出平方差。它不是同时给多层 loss 加权，而是先细后粗地更新重构单元。POS 另分两阶段：先 FP W+A低位宽，只用 g=0、1 两级；然后 W/A 都低位宽，用全部允许粒度。学习率与迭代随级别调整为 \(lr_g=lr_0(1-.2g)\)、\(I_g=I_0(1+.2g)\)。

对 L 个 block，共2L个最细单元；按原文 Eq.(15) 确定最大粒度，非2幂时省掉不能覆盖的最粗一级，不能粗暴扩为“整网重构”。每个组合单元 \(R_g\) 的目标为 \(\|R_g(X_g)-\hat R_g(X_g)\|_F^2\)，更新范围局限于该组合内的参数。原文强调权重的继承和优化，未提供足够代码以确认每一种 scale/zp 是否与 W 同时学习，不能替其扩大 optimizer 参数集合。

原配方64张初始化量化参数、1024张 ImageNet重构，W channel uniform，A layer uniform，post-Softmax log2；\(lr_0=4e-5\)，3/4/6bit的\(I_0=800/300/100\)。COCO用1张、batch1、lr6e-7、500次；采用Adam和cosine。有限校准上的多阶段优化属于作者 PTQ 范围。

**primitive。** reconstruction_partition_tree、fine_to_coarse_schedule、activation_first_stage。图必须识别shortcut与stage/downsample边界，记录每个stage真实覆盖；每次父级重构前用当时量化前缀重新采集输入，不可默默复用另一阶段的 Q 缓存。对于非标准DINO模型，通过metadata表达单元依赖，不硬编码12层/固定MHSA类。

**workflow 与接入。** 对照固定单block、只扩大单元、PFCR顺序、POS开关，按总F/B次数和峰值内存记录开销；重构loss改善只作为局部证据，最终用两任务分数与泛化差评判。全recipe最终staticW4A4可行；需审计log2码表零值编码与首尾覆盖。作者有ViT/DeiT/Swin、ImageNet与COCO实验；状态：原文/Algorithm已核，细粒度optimizer参数与实际公开实现待确认。


## 接入依据与来源边界

**基线组件** 块观测、实际 shortcut/stage 元数据、原子量化基础。

**配方阶段与依赖** 补重构二叉分区树、先细后粗继承、POS A-first/W+A 两阶段、层级预算调度。

**部署条件** 终态可静态；非 2 幂 block 数按原规则截层，不能擅自扩大到整网。

**尚需核对** 作者代码及精确 optimizer 参数集合未核；log 零码和首尾覆盖待审计，不擅自加入可学习 scale/zp。
