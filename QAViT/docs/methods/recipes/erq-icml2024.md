## 5. ERQ（ICML 2024）

<a id="erq-icml2024"></a>

**身份与来源。** 综述 [37]。[ICML 正式版](https://proceedings.mlr.press/v235/zhong24a.html)、[2024 原文](https://arxiv.org/html/2407.06794v1)。当前 QAViT 已有 knowledge/research-skills/research-routes/erq.md 研究卡；差距是执行实现和版本扩展，不是缺文献名。

**paper-recipe。** 分 Aqer 与 Wqer。Aqer 先使用 post-LN channel scale 初始化及等价重参数化，再把激活量化误差吸收到浮点权重中。写 \(X_q=X+\Delta X\)，冻结量化器后解
\[
D^*=\arg\min_D M^{-1}\|(W+D)X_q-WX\|_F^2+\lambda_1\|D\|_F^2
=-W\,E[\Delta x x_q^\top](E[x_qx_q^\top]+\lambda_1 I)^{-1}.
\]
更新 \(W\leftarrow W+D^*\)，接着 Wqer 对未量化输入列反复二分：先量化一半，优化其向上/向下舍入，再用剩余 FP 列吸收残差。若当前已量化部分的误差为 \(\Delta W_s\)，则
\[
D_r^*=-\Delta W_s E[x_sx_r^\top](E[x_rx_r^\top]+\lambda_2 I)^{-1}.
\]
把 \(W_r\leftarrow W_r+D_r^*\) 后继续处理，直到所有目标权重硬量化。推理不存在这些临时 FP 剩余列。

舍入 refinement 可用二阶矩 \(C=E[xx^\top]\) 和 \(J(\delta w)=\delta w C\delta w^\top\) 评价；其导数 \(2\delta w C\) 仅为闭式代理的导数，不需要整网反向。RTN 初始化后，在梯度与误差同号的元素中挑 top-k 翻转舍入；新代理变大即回滚/停止，否则继续，最多 T 次。这里 \(C=\mu\mu^\top+\Sigma\) 的等式可以直接由二阶矩推导，计算时不必制造“实测激活严格高斯”的前提。

ICML 2024 原配方为 ImageNet32张、COCO1张，k=1、最大舍入迭代 T=100；分类 ViT/DeiT-S/B/Swin 的 λ1=λ2=1e4，DeiT-T为1e3，COCO为1e5。2025扩展版的 T=20 不应倒写到2024配方。原文个别式子存在误差符号或交叉矩阵转置不一致；本文按统一误差定义直接由最小二乘求导，实际实现应验证输出残差下降，而非混用相反符号。

**primitive。** activation_ridge_fit、rounding_flip_search、remaining_weight_ridge_fit。统计既可流式累积也可缓存；必须明确 sum/mean 归一化，若从 \(XX^\top/M+\lambda I\) 改为 \(XX^\top+\lambda I\)，对应 \(\lambda\) 相差 M。推荐 solve/Cholesky，不将显式 inverse 当成接口要求。每次变换 W 或激活量化器后，更新对应参考和重建输入，不能把 Aqer/Wqer 任意交换。

**workflow 与接入。** 现有 GPTQ/GPTAQ/AdaRound 提供同类权重候选，ERQ 的新增价值是显式拆出 A 误差补偿和部分量化残差补偿。对照应测：Aqer 浮点收益、Wqer 后保留率、heldout 重构、传播误差及两任务结果；共享同一 FP teacher 和覆盖配置。作者原配方以少量校准数据完成 ViT 分类/检测。最终 static W4A4 适配成立；求解中 FP 参数合法。2025 DUQ 不应倒写进 ICML 2024 卡。状态：原文已核，现有卡可复用，执行尚待接入。


## 接入依据与来源边界

**基线组件** 已有 ERQ 研究卡、二阶矩/GPTQ/GPTAQ 基础、RepQ 变换。

**配方阶段与依赖** 补 Aqer ridge→二分 Wqer→舍入翻转→剩余 W ridge；统一误差符号、矩阵轴与统计归一化。

**部署条件** 临时 FP 待量化列不留到推理，最终可主线；保持 2024 配方身份。

**尚需核对** 没有原生 ERQ 执行注册；2024 的 T=100、λ 和符号不能混入 2025。闭式下降不证明任务提升。
