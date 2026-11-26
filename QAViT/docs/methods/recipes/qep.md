## 16. QEP（NeurIPS 2025）

<a id="qep"></a>

**身份与来源。** *Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization*，综述 [99]。[正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/df2034a516cbd617a96492cc476276c9-Paper-Conference.pdf) §5、§6、附录 B/C；[官方代码](https://github.com/FujitsuResearch/qep) src/gptq.py、src/llama.py。正式版本仍以 LLM weight-only 为主要实证范围，不能因被 ViT 综述引用就记为已验证 ViT W4A4。

**paper-recipe。** 分别跑浮点前缀和已量化前缀，得到 \(X_f,X_q\in\mathbb R^{d_{\rm in}\times M}\)，令 \(\Delta X=X_f-X_q\)，目标从同输入误差改为 \(\|WX_f-\hat W X_q\|_F^2\)。先解连续权重修正，再交给原来的 RTN/GPTQ/AWQ/QuIP 量化器：
\[
H=X_qX_q^\top,\qquad
W^*(\alpha)=W+\alpha W\Delta X X_q^\top(H+\lambda I)^{-1}.
\]
无阻尼、满秩、\(\alpha=1\) 时这是该局部最小二乘的解；加入阻尼和 shrinkage 后是受控修正。实现应解线性方程而非显式逆矩阵。随后以 \(X_q\) 统计调用原量化器，传播更新后的量化输出并进入下一层，不能一次收集全网络 \(X_q\) 后跨多层长期复用。

正式实验默认 \(\alpha=.5\)，Llama-2-70B MLP 层设 0；修正统计来自 C4 的 128 段、每段 2048 tokens，量化基础方法保留自己的校准来源（GPTQ/QuIP 为 C4，AWQ 为 Pile）。附录 C.1 与当前命令行默认 QEP damping 均为 \(1.0\,\mathrm{mean}(\operatorname{diag}H)\)，不同于 GPTQ 的 .01 默认值。论文对 \(\alpha\) 与正则化的联系是平均谱效应的对应，不能把所有 \(\alpha\) 实现成一个普通标量 ridge 后声称逐矩阵相等。

**primitive。** 接口为 paired-input-moments → weight-correction → chosen-weight-quantizer，输出修正权重、秩/条件数诊断和实际量化结果；可将偏置并入常数输入列，但这是需要明确版本化的扩展。原 QEP 不包含激活量化器训练。QAViT 既有 GPTAQ 和传播配对数据，应先检查目标和矩阵项的数学重合，再提供 QEP 的 shrinkage/damping/原配方配置，而非复制一套同义算法。

**workflow 与接入。** 拟比较 GPTQ、现有 GPTAQ、QEP+GPTQ，以及不同 \(\alpha\) 在前缀误差、local_fp 误差和双任务 heldout 表现上的影响。若把 \(X_q\) 扩成包含 A4 的量化输入，属于 ViT W4A4 迁移；最终 \(W^*\) 必须再次量化为 W4 才进入主 profile。闭式局部拟合和论文条件定理均不保证实际近似量化器的端到端任务收益。状态：正式原文及关键官方代码已读；LLM 作者证据充分，ViT 迁移未实测。


## 接入依据与来源边界

**基线组件** GPTQ/GPTAQ、配对输入、二阶矩和传播诊断。

**配方阶段与依赖** 补全矩阵连续 W 修正、独立 α/damping、然后调用选定 W 量化器；逐层刷新 Q-prefix。

**部署条件** 原为 LLM weight-only；把输入含 A4 误差用于 ViT 后重新量化 W4，是主线迁移变体。

**尚需核对** 不能声称已验证 ViT W4A4；与 GPTAQ 三角逐列补偿不等价，QEP 原 damping 不能用 GPTQ 默认替代。
