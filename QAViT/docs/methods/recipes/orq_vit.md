## 9. ORQ-ViT：逐 token 位置的量化轴改变，不是保留 FP 异常值

<a id="orq_vit"></a>

**证据。** 综述 [29]；[出版社原文 DOI](https://doi.org/10.1016/j.sysarc.2025.103530)，Journal of Systems Architecture 168（2025）103530；[开放获取 PDF](https://www.sciencedirect.com/science/article/pii/S1383762125002024/pdfft?md5=955bdcf612c5b14cf27069d3f126746e&pid=1-s2.0-S1383762125002024-main.pdf)，已读 §3.1–3.4、Algorithm 1–2、§4.1–4.3。未核实作者官方代码。

### 原配方

将 \(A\in\mathbb R^{P\times C}\) 按 token 行分开，使少数异常通道中的大值变成每行的稀疏尾部；对各行选择 clipping 范围，异常值参与饱和量化而非 FP 旁路。逐行 A scale 与逐输出通道 W scale 位于矩阵乘约简维之外，因此可将整数点积与输出重标定分离。V 作为第二个乘数时采用对应通道轴，post-Softmax 延用 RepQ 类量化，LN/Softmax 算子仍为浮点。

Algorithm 1 用首批数据在 `{99.9,99.99,99.999}` 分位候选中选最小量化误差范围，随后只优化 scale。分类：1,024 图、batch 64、Adam 初始率 4e-5 加 cosine、W4A4 为 300 次，W6A6 为 60 次；COCO：1 图、batch 1、50 次。检测中只在 MatMul 子模块使用其逐行机制，因为其他位置行数随输入变化。[原文 §3–4](https://doi.org/10.1016/j.sysarc.2025.103530)

### primitive 与当前协议差距

建议 `observer.row_percentile_select` 输出以固定 token 位置索引的 s/z；`opt.scale_only_reconstruction` 仅更新声明的 scale，首尾层按 layer、主体按 block；`layout.fixed_token_scale` 记录 CLS/special/patch、分辨率、window 规则与尺度长度。QAViT 已有动态 per_token，但当前 contracts 拒绝这种位置专属静态尺度，这才是关键扩展点。“per-token 一律动态”的旧判断应修正。

static profile 不应在推理时偷偷重估分位数，也不应把不同 crop/resolution 的同一序号视为同一语义。实现应在布局不匹配时拒绝加载或选择明确的重新校准配置。原文 first-batch 初始化、后续随机 mini-batch 更新的两个数据阶段分别留日志，方便排查第一批偶然性的影响。固定 scale 下的 FP LN 并不妨碍主 matmul 的低位 PTQ，但不能据此称整个计算图为 integer-only。

### 研究工作流

拟议 `study.axis_vs_outlier` 对同一激活，比较 tensor、channel、固定 token 位置和动态 token 四种轴语义，固定总码字与额外元数据成本。`metric.code_occupancy` 统计有效码字利用、饱和比例和各 token 类型覆盖，配合现有输出 NMSE。利用已有双任务视图，专门检查位置尺度是否在分类 crop 上有效、在密集预测 resize/window 上失效；这比重复观察“有异常值”更能指导适配。原文 W4A4 可作为复现目标，用户 DINO 双任务仍属于新验证。

另外应固定首批初始化样本，单独改变后续重构批次，再反向固定重构样本而改变首批。此配对设计能判断收益来自量化轴本身还是首批恰好覆盖了有利位置分布，避免把随机初始化方差误当方法提升。


## 接入依据与来源边界

**基线组件** percentile、捕获、token 布局、动态 per_token 表示基础。

**配方阶段与依赖** 补固定位置逐行尺度、first-batch 初始化、scale-only 重构及 V 的正确轴；扩展静态位置契约。

**部署条件** 原尺度可静态；必须绑定分辨率/window/token 布局，检测原版只在适用 MatMul 用逐行机制。

**尚需核对** 现 contracts 禁止静态 per_token，不能直接加载原法；官方代码未核，跨布局共享是新实验。
