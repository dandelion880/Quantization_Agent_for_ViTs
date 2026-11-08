# AdaLog：独立的可搜索对数网格

来源：ECCV 2024 AdaLog，arXiv:2407.12951v1；官方 GoatWu/AdaLog，固定 commit `75a57d2515521bb1711159ea229d1a37665f3fed`，`quantizers/logarithm.py`。数值实现：`execution/adalog.py`；统一搜索：`pipeline.py:fit_target`。

合法位点是实际 post-softmax 或精确 GELU 输出/对应 Linear 输入；静态 UINT4/6/8、per_tensor，axis/group_size=null。并非均匀 UINT 网格。参数 base=2^(q/37)，固定 options.log_base_q（1..256，默认37）或在 activation_calibration.strategy=search 下显式 options.log_base_candidates。二者互斥；scale 因子仍用既有 factors/factor_grid，可用 mse/cosine/真实 hessian 梯度评分。先读实际分布及方法比较再决定候选，宿主不选择 q 或方法。

原式：code=round(-log2(clamp(x/scale,1e-15,1))*37/q)。推理用 table1=floor(code*q/37)、table2=round(2^(-mod(code*q,37)/37)*(2^(b+1)-2))/(2^(b+1)-2)，输出=2^-table1*table2*scale，code>=2^b 时置零。LUT 舍入忠于官方 inference，不用连续训练公式替代。零掩码与2^b个指数槽的原文表示不是已实现的b位原生整数kernel，不宣称存储/速度收益。

GELU 独立使用时先加负界0.16997124254703522，QDQ后减回；若显式选择独立 adalog_bias 组件，平移已由输入边完成，这里不再重复平移。与同边其它 A 网格互斥；不能叠加改变 GELU 语义的 SmoothQuant/RepQ。bias组件不是强制条件。

适配：沿用现有重建评分器显式枚举 base×scale，不声称实现原论文完整 FPCS 搜索、全模型配方或论文精度。成本为样本数×候选base数×scale数的局部输出重建，缓存逐位点释放。输入、W、bias变换、数据或尺度改变均使旧评分失效；最终参数和LUT导出/重载严格核对。

static W4A4 适配：显式 `options.reserve_zero=true` 将指数有效域从0..15缩为0..14，把下溢映射到零（AdaLog shifted 域为下溢端点），每组最多16值。默认 false 保留历史公式；此变体必须重新校准、记录，不能沿用论文或旧包成绩。冻结选项参与导出/重载。
