## 3. APHQ-ViT（CVPR 2025）

<a id="aphq-vit"></a>

**身份与来源。** 综述 [35]。[原文](https://arxiv.org/html/2504.02508v1) §3.2–3.3、§4.1，[官方代码](https://github.com/GoatWu/APHQ-ViT)。本次实际阅读 configs/4bit/best.py、utils/block_recon.py、utils/mlp_recon.py，而非仅 README。

**paper-recipe。** APH 以任务蒸馏损失估计块输出重要性。分类用 teacher KL，检测由分类 KL 与框回归 smooth-L1 组成。对同一浮点输出 \(z\in\mathbb R^a\) 加减 \(\delta\mathbf1\)，\(\delta=10^{-6}\)，两次反向得到 \(g^+,g^-\)；在作者对角假设下估计 \(h_i=(g_i^+-g_i^-)/(2\delta)\)，沿校准样本平均为 \(\bar h_i\)，以 \(\sum_i\bar h_i e_i^2\) 重构。因为扰动是全元素同方向，在一般非对角 Hessian 下它实际测得 \(H\mathbf1\)，不能称无假设的 exact diagonal。

完整方法先做 MR，再做 QDrop 重构。MR 将 MLP 的 GELU 替成 ReLU，保留原 GELU MLP 为 teacher；令 \(u=\operatorname{ReLU}(\mathrm{FC1}(X))\)，同时比较 \(\mathrm{FC2}(u)\) 和 \(\mathrm{FC2}(\min(u,q_p))\) 与原 MLP 输出，损失 \(L_{\rm direct}+2L_{\rm clamp}\)，二者均 APH 加权。direct 分支保障超阈值元素仍有梯度。然后优化 AdaRound 舍入与激活 scale，最终关闭 QDrop、硬化舍入。原文 1024 张 ImageNet/256 张 COCO、batch32，两阶段各 20000 迭代；量化重构 activation lr \(4\times10^{-5}\)、舍入 lr \(10^{-3}\)，原文 \(p=.99\)，W channel uniform、A uniform 包括 attention map。

**官方代码差异。** 当前 4bit/best.py 为初始化 128 张、优化 1024 张、temperature20、pct=.98、patch embedding 输入 A8、attention matmul head 分组开启。APH 实现使用 \(|g^+-g^-|\) 后样本平均，省去公共 \(2\delta\) 系数；MR 的 Adam 默认 lr \(4\times10^{-5}\)。这些字段应登记为 official-code recipe，不能拿论文概述覆盖。

**primitive。** 分为 aph_prepare、aph_weighted_error、gelu_to_relu_mlp_reconstruct、qdrop_reconstruct。只采用 APH metric 不等于复现 APHQ。MR 改变函数与分布，必须在其后重新校准、重新构建输入缓存及敏感度缓存，不能直接沿用原 GELU 量化器。公式版与 abs-code 版 APH 独立标识；FP32/FP64 有限差分稳定性也要记录。

**workflow 与接入。** 在相同候选/预算下比较现有 site 梯度平方、APH、普通 MSE；再比较仅 APH 与 APH+MR，区分 metric 收益和架构变化收益。DINOv3 多头可分别生成分类、分割敏感度，再按预先声明任务权重融合，这是 QAViT 实验而非原文结果。最终固定 W4A4 可行；原代码 A8 例外须保留作者 profile，同时另建覆盖一致的 W4A4 迁移 profile。状态：原文及关键官方实现已核；没有执行复现。


## 接入依据与来源边界

**基线组件** 只读梯度捕获、AdaRound、teacher KL、结构观测。

**配方阶段与依赖** 补 APH 双扰动准备、加权损失、GELU→ReLU MLP 恢复、QDrop 块重构；MR 后刷新量化与敏感度。

**部署条件** 离线恢复后可静态；原代码 patch 输入 A8 单列。结构变化另记成本，不自动排除主线。

**尚需核对** 全方向差分一般是 H1 非精确对角；paper 与 abs-code、p 值不同。只加 APH metric 不等于整法。
