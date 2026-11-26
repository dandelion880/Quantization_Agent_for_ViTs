# AdaLog：独立的可搜索对数网格

来源：ECCV 2024 AdaLog，arXiv:2407.12951v1；官方 GoatWu/AdaLog，固定 commit `75a57d2515521bb1711159ea229d1a37665f3fed`，`quantizers/logarithm.py`。数值实现：`execution/adalog.py`；统一搜索：`pipeline.py:fit_target`。

合法位点是实际 post-softmax 或精确 GELU 输出/对应 Linear 输入；静态 UINT4/6/8、per_tensor，axis/group_size=null。并非均匀 UINT 网格。参数 base=2^(q/37)，固定 options.log_base_q（1..256，默认37）或在 activation_calibration.strategy=search 下显式 options.log_base_candidates。二者互斥；scale 因子仍用既有 factors/factor_grid，可用 mse/cosine/真实 hessian 梯度评分。先读实际分布及方法比较再决定候选，宿主不选择 q 或方法。

原式：code=round(-log2(clamp(x/scale,1e-15,1))*37/q)。推理用 table1=floor(code*q/37)、table2=round(2^(-mod(code*q,37)/37)*(2^(b+1)-2))/(2^(b+1)-2)，输出=2^-table1*table2*scale，code>=2^b 时置零。LUT 舍入忠于官方 inference，不用连续训练公式替代。零掩码与2^b个指数槽的原文表示不是已实现的b位原生整数kernel，不宣称存储/速度收益。

GELU 独立使用时先加负界0.16997124254703522，QDQ后减回；若显式选择独立 adalog_bias 组件，平移已由输入边完成，这里不再重复平移。与同边其它 A 网格互斥；不能叠加改变 GELU 语义的 SmoothQuant/RepQ。bias组件不是强制条件。

适配：沿用现有重建评分器显式枚举 base×scale，不声称实现原论文完整 FPCS 搜索、全模型配方或论文精度。成本为样本数×候选base数×scale数的局部输出重建，缓存逐位点释放。输入、W、bias变换、数据或尺度改变均使旧评分失效；最终参数和LUT导出/重载严格核对。

static W4A4 适配：显式 `options.reserve_zero=true` 将指数有效域从0..15缩为0..14，把下溢映射到零（AdaLog shifted 域为下溢端点），每组最多16值。默认 false 保留历史公式；此变体必须重新校准、记录，不能沿用论文或旧包成绩。冻结选项参与导出/重载。

## 1. AdaLog：把表示能力与搜索能力分开补齐

<a id="adalog"></a>

**证据。** 综述 [62]；[AdaLog 原文 v1](https://arxiv.org/html/2407.12951v1)，ECCV 2024，重点为 §3.2–3.4、Algorithm 1、§4.1；[官方仓库](https://github.com/GoatWu/AdaLog)已取得 master 分支的源码快照，核读 `quantizers/logarithm.py`、`quant_layers/linear.py`、`configs/4bit.py` 与 `test_quant.py`。没有将浮动分支冒充已锁定 commit。

### 原配方与应保留的两个复现 profile

正值激活采用可调底数的对数量化：先将 \(-\log_b(x/s)\) 舍入、截断为整数指数，再用 \(s b^{-q}\) 解码。底数控制相邻量化等级的比例，scale 控制范围，两者不能用一次独立的 scale 搜索代替。post-GELU 先做正向平移，线性层 bias 用**实际量化并解码后的权重**抵消平移；这是一项等价变换，不是拟合残差的补偿器。原文还把指数分解为整数移位与余数 LUT，以固定分母近似 \(\log_2 b\)。

FPCS 的核心是联合候选的逐层细化：在初始参数网格评估输出误差，保留若干好区域，在这些区域内再细分，而不是每轮只围绕唯一最优点缩小区间。接入应保存每轮候选、保留索引、局部网格和评估次数，才能区分表示收益与额外搜索预算。[原文 §3.2–3.4](https://arxiv.org/html/2407.12951v1)

论文设置与公开代码不能混写：论文 §4.1 的主校准集为 32 张，搜索示例使用候选规模 128、细化轮数 4，LUT 近似分母为 37；[官方 4bit 配置](https://github.com/GoatWu/AdaLog/blob/master/configs/4bit.py) 为 `eq_n=128`、`search_round=3`、`steps=6`、`calib_size=32`、`optim_size=1024`，且 `qconv_a_bit=8`。代码里 activation FPCS 有 base/scale 网格和 beam width，权重 FPCS 又有独立默认值。`test_quant.py` 的额外优化开关启用重构，应建立“纯校准”和“含重构”两份配置，不能共享论文名而隐藏成本。

代码的 GELU 正向偏移常量是 `0.16997124254703522`。activation FPCS 的一个实际入口采用 base 8 × scale 16 的初始网格、保留宽度 32、6 次细化；这些是该入口参数，不能误传给所有 W/A 搜索器。缓存必须包含是哪一类入口及其覆盖位点。

### 可组合 primitive 与顺序

建议 `rep.adaptive_log_codebook` 只负责编码/解码；`search.fpcs_joint` 接受参数空间、候选生成器、目标函数及预算，输出固定参数与完整搜索轨迹；`transform.positive_shift_fold` 接受偏移、原始 bias、最终解码 W，输出补偿 bias；`rep.log_lut_decode` 独立承担部署近似。先确定最终 W 或在 W 更新后重算补偿，随后冻结 A 和 LUT。后续 GPTQ/GPTAQ 重新量化 W 时必须使旧 bias 补偿失效。

[官方 log quantizer](https://github.com/GoatWu/AdaLog/blob/master/quantizers/logarithm.py) 会用一个额外 mask 把超出指数范围的值置零：4bit 标签可能包含 16 个正值再加零。QAViT 旧参考实现已记录相同风险。static W4A4 应显式提供总共 16 个码字的版本，例如预留零码；这会改变原配方，应另命名并报告偏差，不能把 mask 当作免费存储。训练近似解码与推理 LUT 解码也要分别保存误差。

### 研究工作流与当前差距

QAViT 已有 AdaLog 表示与等价 shift，缺的是完整联合搜索、论文/代码预算及 LUT 审计。拟议 workflow `study.representation_vs_search` 在相同前向评估预算下比较固定 log2、log√2、AdaLog 网格和 FPCS；原文的“自适应底数有益”不能直接推导“FPCS 在任何预算下都更好”。新增 `metric.log_underflow_mass` 统计被置零的 attention 概率质量，配合已有层输出 NMSE，区分大量小数值被舍弃与少量大 attention 边被破坏。更低位宽的探索按原论文验证范围和另行声明的实验问题开展，不作为本次静态四位协议的隐含要求。


## 接入依据与来源边界

**基线组件** AdaLog base×scale、adalog_bias、log/twin 数值与候选评分。

**配方阶段与依赖** 补多区域保留的完整 FPCS、搜索轨迹、LUT 解码；补偿绑定最终解码 W，纯校准与重构配方分开。

**部署条件** 固定码本可主线；原代码首层 A8 和额外下溢零需保留作者配置或另建合规版本。

**尚需核对** 论文与代码细化次数、入口默认值不同；不能把现有 base×scale 搜索称 FPCS，LUT 与参考解码仍需对齐。
