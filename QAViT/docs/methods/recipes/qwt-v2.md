## 22. QwT-v2（2025 预印本）

<a id="qwt-v2"></a>

**身份与来源。** 综述 [44]。[原文](https://arxiv.org/html/2505.20932v1) §3.3–3.4、§4。前序 QwT 仓库可作家族入口，本次没有确认独立 v2 可运行官方实现；不得把 QwT v1 的矩阵补偿代码标成 v2。

**paper-recipe。** 对**每个量化 Linear/Conv 输出**做每输出通道 affine 修正，不只在 Transformer block 末端加一个 affine。把模块输出展平成 \(Y_q,Y_f\in\mathbb R^{M\times d_{\rm out}}\)，两者来自同一当前输入；对每个通道求
\[
\alpha_c=\frac{\sum_m(Y_{q,mc}-\bar Y_{q,c})(Y_{f,mc}-\bar Y_{f,c})}
{\sum_m(Y_{q,mc}-\bar Y_{q,c})^2},\qquad
\beta_c=\bar Y_{f,c}-\alpha_c\bar Y_{q,c},
\]
并输出 \(Y'_q=Y_q\operatorname{diag}(\alpha)+\mathbf1\beta^\top\)。512张 ImageNet；无需梯度。零方差时分母处理本次未见完整数值实现，QAViT 应明确采用常数预测或正则，不静默写死任意 epsilon 后声称原配方。

部署方案利用量化线性层的输出重定标。忽略 zero-point 只为解释，原 integer accumulator 为 \(u_c\)，输出单位为 \(s_xs_{w,c}\)。修正后为
\[
\alpha_cs_xs_{w,c}\left(u_c+\frac{\beta_c}{\alpha_cs_xs_{w,c}}\right).
\]
因此把 \(\alpha_c\) 合入输出 multiplier，把 \(\beta_c/(\alpha_cs_xs_{w,c})\) 舍入进 accumulator bias。实际有输入/权重 zero-point、整数舍入、乘数定点化和 output saturation 时需完整保留其误差。若 \(\alpha_c=0\) 或负值，目标后端是否支持对应乘数与 bias 处理必须验证，不能直接除零或假定正数。

**primitive。** channel_affine_fit 只需每通道 count、sum_yq、sum_yf、sum_yq2、sum_yq_yf；另设 affine_fold_integer_output，输出编译后图与额外舍入误差。固定补偿依赖所拟合的 quantizer、模块输入和 scale；上游变化后顺序重估。

**workflow 与接入。** 在相同底座比较 BiasComp/full QwT/v2，对每通道报告拟合残差、heldout 外推、负/零 multiplier 比例；再比较浮点 affine 与真正融合后的整数结果及延迟。论文提供潜在融合方式，未给可直接引用的实测端到端后端收益。浮点 affine 实现属于补偿扩展 profile；确认 multiplier/bias 已融合且最终覆盖满足主协议时可作为固定 W4A4 变体。状态：原文方法与实验设定已读；独立代码与实际后端融合尚未核实。


## 接入依据与来源边界

**基线组件** Linear/Conv 输出观测、每通道统计、bias/scale 状态。

**配方阶段与依赖** 补每个 Linear/Conv 输出的 affine 拟合；再实现 multiplier 与 accumulator bias 的真实整数融合。

**部署条件** 浮点 affine 路径先作补偿配置；确认融合、四位覆盖及终图后才可主线。

**尚需核对** 独立 v2 代码/后端融合未核；零方差、零/负 α、整数舍入须定义，不能引用“零开销”作实测结论。
