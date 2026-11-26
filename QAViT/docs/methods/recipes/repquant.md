## 8. RepQuant：在量化空间学边界，再重参数化与重构 W

<a id="repquant"></a>

**证据。** 综述 [23]；[原文 v1](https://arxiv.org/html/2402.05628v1)，§III-B–E、Algorithm 1、§IV-A、Table IX–X。已读方法全文与设置；本次没有核实到独立的 RepQuant 官方完整仓库，不能将 [RepQ-ViT 仓库](https://github.com/zkkli/RepQ-ViT)等同为 RepQuant 已发布实现。

### 原配方

RepQuant 在 RepQ 基础上增加逐通道双边 clipping。对每个 post-LN 通道的最大/最小值，分别乘两个 sigmoid 收缩因子，最小化通道激活量化前后的平方误差。随后用已有 RepQ 的等价关系把细粒度 A 网格转成部署层级的共享网格，变更 LN affine 和相邻 W/bias，再对变更后的 W 执行 GPTQ。原文 Algorithm 1 的层序遍历把 A 与 W 连在一起，避免先量化 W、随后变换 W 又使其网格失效。[原文 §III-D–E](https://arxiv.org/html/2402.05628v1)

已确认的 ViT 校准规模为 1,024，双边 clipping 使用 Adam 100 次、学习率 0.01；LLM 的 128 样本设置不是 ViT 设置。Softmax 的 log√2 表示与 log2 移位/奇偶系数实现另有等价关系，不能用两个不等价的 codebook 任意替换。论文已有 W4A4 验证，适合进入静态优化 PTQ，而非按旧“禁止任何学习”条件排除。

### primitive 与复用

建议 `observer.learned_dual_clip` 输出每通道上下界、s/z、clipping 比例及优化轨迹；`transform.repq_shared_grid` 直接复用现有 RepQ-LN，但允许输入已学得的 channel_params；`schedule.activation_then_weight` 确保新 W/bias、输入缓存与 GPTQ 曲率来自同一版本。QAViT 的 `repq_ln(..., channel_params=...)` 已具备可接入口，这是补全工作，不需要重写 RepQ 数学。

GPTAQ 的目标传播与 RepQuant 原版的 GPTQ 不同。二者可以组成新的对照，但“RepQuant+GPTAQ”必须独立命名，不能自动继承 RepQuant 原结果。学习 clipping 的参数初始化、STE 处理、共享 zero-point 是否非整数，以及 reparam 后的实际零点语义均需与原式对齐；如果真整数格式强制改成整数 zero-point，要单列其误差。

### 研究工作流与当前差距

新增价值是 `study.clip_in_quantized_space`：对比浮点空间平滑与量化空间边界优化，在相同校准集、搜索/梯度预算和最终 W 方法下分析 A/W 的联合变化。还应做“先 A 后 W”和“先 W 后 A”的可解释顺序实验，记录失效缓存而不是仅报最终分数。校准误差之外增加 held-out clipping 率，用已有双任务指标检测某任务的稀有激活是否被选择性削弱。支持 bits 参数本身不代表所有位宽已经获得成熟结果。

建议保存每次通道边界更新的饱和率与损失，而不只保存最终上下界。这能区分优化确实收敛、sigmoid 卡在极端区间，以及少数异常校准图主导边界三种情况。


## 接入依据与来源边界

**基线组件** repq_ln 的 channel_params 接口、GPTQ、log_sqrt2。

**配方阶段与依赖** 学习双边 clipping→A 网格→等价重参数化→重新收集 W 统计→GPTQ；显式限定优化变量。

**部署条件** 有限校准后冻结可主线；继承的 zero/log 码态仍须检查，GPTAQ 替换 GPTQ 另记变体。

**尚需核对** 独立官方完整代码未核；clipping 初始化/STE 与零点语义需按所选来源闭合，不能只套现有 RepQ。
