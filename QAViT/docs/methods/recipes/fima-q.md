## 4. FIMA-Q（CVPR 2025）

<a id="fima-q"></a>

**身份与来源。** 综述 [34]。[原文](https://arxiv.org/html/2506.11543v1) §3.2–3.3、Algorithm 1、§4.1；[官方代码](https://github.com/ShiheWang/FIMA-Q)。已读 configs/4bit/best.py、utils/block_recon.py 与调用入口。

**paper-recipe。** 基础仍为 QDrop/AdaRound 块重构，新增对 teacher KL 曲率的 DPLR 估计。设块输出扰动 \(d\in\mathbb R^a\)，\(g=\nabla_d L_{\rm KL}\)，局部近似 \(g\simeq Fd\)。论文对角项使用 \(g_i/d_i\)；rank-one 用 \(gg^\top/(g^\top d)\)；进一步收集 \(k\) 个扰动及梯度，\(D,G\in\mathbb R^{a\times k}\)，由约束 \(FD\simeq G\) 构造
\[
F_k=G(D^\top D)^{-1}D^\top,\quad
L_{\rm DPLR}(e)=\alpha e^\top F_ke+(1-\alpha)\sum_i(g_i/d_i)e_i^2 .
\]
只计算 \(e^\top G\)、小矩阵求解及 \(D^\top e\)，避免存储 \(a\times a\)。rank 从 1 逐渐增加，每间隔 x 次重算校准平均扰动/梯度，直到 r。论文 k>1 放宽对称约束，不能把该近似无条件称为 PSD 真 Fisher。重构更新 AdaRound 参数与 A scale。论文使用 ImageNet1024/COCO256、rank15；COCO 由 RPN 分类输出 KL 提供敏感度，优化 backbone+FPN，RPN/ROI 主要校准。

**官方代码必须单列。** 当前 W4A4 配置 rank5、温度20、128张初始化、1024张优化、batch32、20000迭代，Adam 舍入默认 lr .001、A lr \(4\times10^{-5}\)。patch embedding 输入 A8，token/head 分组开关开启。代码收集 \(|d|,|g|\) 的样本均值，计算 \(B=(DD^\top)^{-1}\)（此处存储为 \(k\times a\)），再使用
\[
L_1=(|e|G^\top)B(|e|G^\top)^\top,\quad
L_2=\operatorname{mean}(|e|^2\operatorname{mean}_k|G|),
\]
各自除首次/更新时 loss，p1=p2=1。它并非论文公式的逐字实现；源码中小矩阵直接 inv，没有本次确认的阻尼回退。不要把这两种损失混写为一个“FIMA metric”。

**primitive。** 分别登记 fim_secant_paper、fima_official_abs_normalized、progressive_curvature_cache；缓存记录样本、扰动生成方式、秩、temperature、正则/数值截断、model/head 指纹。安全数值变体可使用 pinv/阻尼，但属于 QAViT variant，必须记录实际有效秩与改动，不能静默替换。

**workflow 与接入。** 复用现有 frozen 多观察点/双任务管线；新增 prepare 阶段建立只读梯度缓存，forward-only scorer 读取缓存。重点比较对角、低秩、组合以及实际 selected regret，不以“更像 Hessian”代替任务证据。完整算法属于有限校准 PTQ，主协议允许；固定4bit可迁移，作者 A8/分组细节另存原 recipe。状态：数学、算法与官方 W4A4 实现均已核；代码/论文差异明确，尚未运行。


## 接入依据与来源边界

**基线组件** 梯度捕获、AdaRound、冻结多观察点评分。

**配方阶段与依赖** 补渐进扰动/梯度缓存、DPLR 小矩阵求解、QDrop 重构；论文 secant 与官方 abs-normalized 分开。

**部署条件** 有限校准可主线；作者 A8 例外和 token/head 分组另存实际配置。

**尚需核对** rank、归一化及损失公式 paper/code 不同；直接 inv 的退化回退若修改需标变体，不能称精确 PSD Fisher。
