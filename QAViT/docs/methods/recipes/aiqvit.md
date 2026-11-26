## 3. AIQViT：低秩修正合入量化权重，区间由校准学习

<a id="aiqvit"></a>

**证据。** 综述 [63]；[原文 v1](https://arxiv.org/html/2502.04628v1)，Method 的 Architecture-Informed Low-Rank Compensation、Dynamic Focusing Quantizer、Optimization Strategy，及实验设置；[AAAI 2025 正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/33939/36094)。[作者主页](https://longguangwang.github.io/)列出论文，本次没有核实到可下载的官方完整实现；因此算法可依据全文实现，初始化、优化器等未明确细节不能借用他人复现冒充作者设置。

### 原配方

AIQ 的 DFQ 使用**均匀量化**，不是对数域量化。它学习各层 post-Softmax 的关注区间 \([b_1,b_2]\)，区间内部均匀分配码字，两端按边界规则处理。名称中的 dynamic 说明区间在优化中调整，原文不足以证明需要运行时按输入重估区间。低秩修正的关键计算是 \(Q_A(X)Q_W(W_0+BA)\)：冻结原始 \(W_0\)，学习低秩 A/B，再量化合并权重，不应改写成独立 FP16 LoRA 推理旁路。[原文方法部分、式 (9) 与 DFQ 定义](https://arxiv.org/html/2502.04628v1)

完整配方还包含 rank 搜索与课程调度。候选 rank 为 `{10,20,50,100,150}`，先做可微架构搜索，再选择 rank 并做块重构；按重构误差定义样本难度，从较易样本逐渐扩展。论文的 ImageNet 校准集为 1,024、batch 24，COCO 为 256、batch 1，点云为 512、batch 32；rank 搜索 2,000 次，校准 6,000 次。此预算不能与 32 图的纯参数搜索相混。论文已有分类、检测、实例分割及点云任务验证，但不等于已经验证用户的 DINO 双任务。

原文课程覆盖比例为 \(\lambda(t)=\min(1,0.5+0.5t/T)\)，从一半较易样本扩展到全部；NAS 中架构权重与修正参数的优化集分工应保留。拟议比较还需固定每次更新的样本数，防止把课程收益与实际见到的数据量混淆。

### 可组合 primitive

`weight.mergeable_low_rank_correction` 接收 W、校准输入和浮点目标，输出修正后的单个 W 及优化轨迹；发布前必须 merge、重新量化、丢弃 A/B 推理分支。`rep.learned_focus_interval` 输出固定边界与 声明位宽下的码字映射。`search.rank_allocation` 单独消耗搜索预算，`schedule.easy_to_hard_reconstruction` 单独决定哪些校准样本参与当前更新。这些应可消融，避免只能调用不可解释的“AIQ 总开关”。

实现 DFQ 时需锁定：低于 \(b_1\) 的整数码、该码实际解码值、上界饱和、边界梯度及零值是否专门保留。整数码为零不意味着解码后必为数值零；若另加一个零值而不减少其他等级，会突破真 4bit。低秩候选的初始化、NAS 的 train/validation 划分和 drop-path 应按原方法章节补齐；源码未取得的部分在 manifest 中列明确字段，不以猜测补默认值。

### 研究工作流与当前差距

QAViT 的 RepQ-LN 和普通均匀量化可复用，缺低秩修正训练、rank 搜索、关注区间和课程调度。拟议实验先固定 rank 与搜索预算，区分“增加可优化自由度”与“自动选择 rank”的贡献，再研究关注区间保留的 attention 质量是否服务于分类和密集预测两种目标。课程调度属于优化中的样本安排，不能登记为已经证明更好的通用校准集选择算法。冻结区间并量化合并 W 后可进入 static W4A4。


## 接入依据与来源边界

**基线组件** 均匀网格、RepQ 类变换、捕获与误差评价。

**配方阶段与依赖** 补 Q(W0+BA) 的可合并低秩修正、均匀关注区间学习、rank 搜索、easy-to-hard 样本调度。

**部署条件** 区间冻结、BA 合入且重新量化 W 后可主线；保留 FP 低秩分支则是另一配置。

**尚需核对** 官方完整代码未核；初始化、NAS 数据分工、边界/零码处理须按原文补齐。不能写成 log quantizer。
