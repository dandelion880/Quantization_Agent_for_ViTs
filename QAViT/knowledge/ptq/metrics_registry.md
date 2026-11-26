# QAViT 指标注册与实验执行说明

> 上传材料的待接入事项和原文预算按 2026-11-25 出处保留。实际数值入口、适配差异与证据见 [实现账目](implementation_status.json)。

核验日期：2026-11-25。本文是供 Codex 接入的文字规格与注册契约，不是已经实现的 API。原文方法详见 methods_reconstruction_metrics.md；所有新增目标仍服务于 PTQ，有限校准优化允许，QAT 不进入方法库。

## 1. 升级目标：把 metric 变成可检验的研究对象

QAViT 已经能同时看分类、分割、多个中间层，并在相同候选池比较多种加权目标。新的任务是扩展“能观察什么、怎样估计、是否会选对候选、花了多少资源”，而非把已有 MSE/KL 重新包装成新功能。每个 metric 都需要能追溯其观察位置、参考、数据、归约和计算预算；复杂敏感度 metric 是否值得使用，最终由独立数据上的选择质量和任务收益判断。

当前源码可复用部分如下。路径相对 QAViT 根目录。

| 当前能力 | 复用位置 | 具体新增差量 |
| --- | --- | --- |
| 流式 SSE/能量/点积、NMSE/cosine、结构检查 | tools/ptq_agent/execution/metrics.py | 添加研究 metric 扩展；不修改现有 max_per_output_relative_mse_v1 的结构一致性判据 |
| 同候选、同图像/窗口、FP64 分块计算 | tools/ptq_agent/execution/frozen_trials.py | 新 observer、缓存敏感度 scorer、完整 checkpoint/adapter 候选 |
| 多层 head_features、分类/分割 logits、任务加权 | 同上 | score identity、预声明归一化、Pareto/selected regret；现有功能不重写 |
| local_added/upstream_mapped/total/cross、FP block 替换 | tools/ptq_agent/execution/propagation.py | 归一化交叉项、干预收益、组合收益保留和置信区间 |
| minmax/search/manual/percentile/coordinate | execution/calibration.py 及 calibration_search.py | 新目标与可微重构调度；校准方法已有 |
| 站点梯度、logits 交叉熵 Hessian 近似 | 当前 calibration/profile 实现 | 独立 APH/FIMA prepare，不将两种既有信号混名为一个 Hessian |
| GPTQ/GPTAQ 统计与误差补偿 | execution/plugin_math.py、execution/gptaq.py | QEP/ERQ 的版本化配置、差异对照，避免重复算法 |
| 候选覆盖、selector、组合收益的研究思路 | knowledge/research-skills/research-routes/ptq4dino.md | 升为标准结果字段与可复用分析，而非宣称此前没有相关思路 |

## 2. 一个 metric 的身份至少包含什么

建议注册项与运行实例分离：注册项定义数学函数；运行实例定义它对哪一个模型、head、数据和候选生效。相同数学函数换参考、token 范围或归约，不能共用一个不带语义的标量名字。

| 字段 | 说明 |
| --- | --- |
| id / version / family | 稳定英文 ID；数学定义或归约改变需换版本 |
| role | diagnostic / candidate_score / reconstruction_objective / task_evaluation / cost；一个函数可有多种角色，但逐次运行需明确 |
| observation | module path、输入/输出、attention score/probability/AV、block/head_feature/logit；附真实 shape |
| reference | fp_model、local_fp_parent、local_fp_candidate、teacher_checkpoint 或 target_labels |
| input_state / suffix_state | FP/量化前缀及其指纹；任务敏感度所用 FP/Q 后缀及 lookahead 深度 |
| axes / token_scope | batch/token/channel/head/query/key/spatial 的轴映射；CLS/register/patch、有效 pixel mask |
| reduction | 先沿哪些轴归约，按图/窗口/元素/层如何加权；不能只写 mean |
| direction / unit | lower/higher/diagnostic，absolute/relative/dB/percentage-point；是否可直接相加 |
| requirements | labels、teacher forward、backward、perturbation、缓存统计及额外显存 |
| estimator | temperature、Hessian/Fisher 近似、扰动方向与幅度、秩、damping、采样 seed |
| invalidation | 哪些模型、量化、输入、任务变更使缓存失效 |
| evidence | 原文 URL/章节、作者代码路径/commit、本地实现状态与验证范围 |
| numerical_policy | 空集合、零能量、常数向量、ties、有效秩、非有限值的实际处理 |
| result_payload | scalar 之外的 count、分布、每原图结果、估计状态和失败原因 |

**两个 local_fp 不能混用。** 当前 frozen_trials.py 使用固定 parent 输入运行 FP block，记为 local_fp_parent；候选输出来自 candidate 全网。如果候选改了更早层，输入已变，所得差异不是纯同输入局部误差。QwT 拟合和严格局部分解需 local_fp_candidate：对 candidate 当前输入同时运行 FP/Q block。单站点候选且上游固定时二者一致。现有行为保留兼容，新增明确名字，不静默改语义。

**归约与数据单位。** 默认候选比较沿用当前 frozen 规则：同一个原图内先平均其窗口，再对原图等权；多层按预声明权重平均。这样不会让大图因窗口多而获得额外权重。另可提供 pooled-energy NMSE，但必须作为不同 reduction；不能将均值比与比值均值混用。分割正式 mIoU 必须先还原/融合原图预测再累计混淆矩阵，不能按重叠窗口当独立图像。

## 3. 分布、码表与量化器诊断

这些指标回答“量化器是否适合这一分布、所谓4bit到底编码了什么”。它们是机制诊断，不预设与准确率单调相关。

| 注册 ID | 定义与输出 | 依赖及解释 |
| --- | --- | --- |
| distribution_summary | min/max、mean/std、选定分位数、偏度、峰度；分别按 tensor/channel/token/head | 指定总体或样本矩；常数张量的标准化高阶矩设 undefined；大 tensor 可用有误差界的 sketch 并记采样率 |
| tail_mass_energy | 阈值外元素比例与能量比例 \(\sum x^2 1_{\{\lvert x\rvert >t\}}/\sum x^2\) | 阈值可为固定值、MAD/IQR 倍数或校准分位数；阈值必须由 calibration 定义，再测 heldout |
| axis_scale_dispersion | 各通道/token 的 std、RMS、maxabs 或指定分位尺度的分布及 max/median 比 | post-LN 通道与 post-GELU token 分开；分母零时返回状态，不任意用 epsilon 掩盖 |
| quantizer_clipping | lower/upper clipping 比例、被截断能量、max clipping distance | 根据量化前真实可表示域计算；饱和到边界 code 不一定发生 clipping |
| zero_point_overflow | 初始化/导出 zero-point 超出合法整数域的比例，clamp 前后可表示区间变化 | Q-HyViT 专项诊断；zp 应有整数域与目标 backend 定义 |
| quantizer_underflow_zero | 非零输入映射到零的比例、输入零比例、正负尾部 underflow | 对 log/非均匀码表使用实际 decode；与 clipping 分开 |
| codebook_usage | code histogram、active code 数、entropy、最大 code 比例、空桶率 | 按整数 code 而非近似浮点去重；另记录理论编码容量与实际不同 decode 值数 |
| distribution_shift | 固定 calibration bin 的 heldout histogram JS divergence，均值/尺度漂移 | JS 两端同 bin 与 smoothing；它是输入分布漂移，不是任意 feature 上的 task KL |
| reference_input_crossmoment | \(H=E[x_qx_q^\top]\)、\(C=E[(x_f-x_q)x_q^\top]\) 的范数、谱/有效秩、条件数代理 | 用于 ERQ/QEP/GPTAQ；完整矩阵可按需要生成，不为每层强制保存 |

4bit 的合法载荷最多16种 bit pattern；“16个正数级别再额外加入0”可能需要第17个符号。当前部分 log2/log√2/AdaLog 历史实现存在需要检查的 underflow-zero 路径，接入时验证实际 pack/decode 与特殊码约定。登记 actual_code_count、packed_bits、special_code_policy；不能只从配置里的 bits=4 推断硬件已经4bit。

分布统计需保留取样位置。重参数化把 A outlier 转到 W 后，A 的峰度下降并不代表整体问题解决：同时记录 W 的尾部、分组额外 scale/索引和最终 W4A4 任务变化。ERQ-DUQ 的列频次可作为 tail-column-frequency 专项输出，而非“通用离群分数”。

## 4. 局部重构与注意力结构

令参考与候选张量展平为 \(y,\hat y\in\mathbb R^n\)，\(e=\hat y-y\)。下列式子是统一实现定义；原论文采用不同归约时，登记 recipe variant。

| 注册 ID | 定义/轴 | 状态 |
| --- | --- | --- |
| mse | \(\lVert e\rVert^2/n\) | 已有，复用 |
| nmse | \(\lVert e\rVert^2/\lVert y\rVert^2\) | 已有；记录 pooled 或 per-image |
| sqnr_power_db | \(10\log_{10}(\lVert y\rVert^2/\lVert e\rVert^2)\) | 标准功率比口径；输入/误差零值按数学极限/undefined报告，不静默加epsilon |
| cosine_error | \(1-y^\top\hat y/(\lVert y\rVert\lVert \hat y\rVert)\) | 已有；常数零范数按既有 policy |
| teacher_kl | \(\sum_c p_{f,c}\log(p_{f,c}/p_{q,c})\)，按 image/pixel 的类别向量 | 已有；方向为 teacher→candidate；temperature 扩展须换实例 |
| pearson_error | \(1-\operatorname{corr}(y,\hat y)\)，先按声明轴去均值 | PTQ-ViT 代理；常数参考 undefined |
| attention_pairwise_hinge | 固定 query 内 \( [\theta-(\hat a_p-\hat a_q)\operatorname{sign}(a_p-a_q)]_+\) 平均 | 来自 PTQ-ViT；输入是 pre-Softmax scores；pairs 采样与 tie 规则必须给出 |
| attention_rank_agreement | 排序相同的有效 key-pair 比例；可另报 Kendall tau-b | 诊断提案，不冒称等于原 hinge；ties 对称处理 |
| attention_topk_overlap | \(\lvert \mathrm{TopK}(a)\cap\mathrm{TopK}(\hat a)\rvert /k\)，按 head/query | 稀疏注意力保留诊断；k、特殊 token/mask 固定 |
| attention_av_error | \(\lVert \hat A\hat V-AV\rVert_F^2/\lVert AV\rVert_F^2\) | 同 head/value shape；可拆只A、只V、联合，区分相互作用 |
| attention_row_mass | \(\sum_k\hat A_{qk}-1\)、row entropy 变化、被清零注意力质量 | 量化注意力不一定归一到1；是否 renormalize 是算法变体 |
| residual_bias_profile | \(E[e]\) 按保留 channel/token/head 坐标的均值、RMS与最大绝对值 | 为 BiasComp/QwT-v2 提供诊断；显式 reduce_axes |
| residual_variance_profile | \(\operatorname{Var}(e)\)、与输入/参考输出的 covariance | 检查常数补偿能修多少，以及是否需要 affine |
| nuclear_norm_proxy | \(\lVert Y\rVert_*\lVert e\rVert_F^2\) 或作者原指定形式 | PTQ-ViT 混精敏感度；是代理，不称 exact Hessian |
| linear_fit_diagnostics | SSE、\(R^2\)、有效秩、条件数、heldout residual；无拟合均值时的 \(R^2\) 基准要声明 | QwT/闭式补偿；同样本局部 SSE 保证与 heldout 泛化分开 |

SQNR 统一采用功率比的 10log10 定义。IPTQ-ViT 原文 Eq.(18) 写成 20log 的期望功率比，若按其文字复现，需另名 iptq_sqnr_paper；不得覆盖标准 sqnr_power_db。来源：[IPTQ-ViT](https://arxiv.org/html/2511.15369)，§4.5；本文件不把该合成分数当部署实测。

attention_av_error 的对照要固定其余网络。若计算 \(\hat A V\) 时 V 来自另一路不同前缀，它并非只改变 attention quantizer 的因果干预。attention_rank 与 top-k 只能说明结构保持，不能替代 task quality；均匀概率区域的 tie、被 masked 的 \(-\infty\) scores 需在有效 key 集合中处理。

可选补充 Gram/CKA、特征子空间 principal angles。只在需要研究表示几何或跨层迁移时使用，并声明 centering、采样与尺度不变性；默认基线不必计算整套高开销矩阵指标。

## 5. 任务敏感度与曲率：必须分版本的指标

### 5.1 已有两个信号先分清

**site_gradient_squared。** 对选定标量任务损失 \(L\)，在 site 的特征 \(z\in\mathbb R^a\) 得到 \(g=\nabla_z L\)，加权误差 \(a^{-1}\sum_i g_i^2 e_i^2\)。这是一类经验对角重要性近似；梯度是对哪个任务、标签或 teacher 概率计算的必须记录。不同 batch 先求梯度均值再平方，与先平方再平均，不是同一个估计量。

**logit_ce_hessian_diag。** 对 logits \(u\in\mathbb R^C\)，softmax \(p\) 的 CE Hessian 为 \(\operatorname{diag}(p)-pp^\top\)，对角是 \(p_c(1-p_c)\)。如果误差 \(e\) 是 logits，可以用对角目标 \(\sum_c p_c(1-p_c)e_c^2\)，也可用完整 logit 二次型
\[
e^\top H e=\sum_c p_c e_c^2-\left(\sum_c p_c e_c\right)^2.
\]
后一式无需存 \(C\times C\) 矩阵。它不能直接被当成中间 block 的 Hessian；中间特征还需要经过 suffix 的 Jacobian。QAViT 原有 hessian_diag_ce 及站点梯度分别保留，不合并命名。

**退化检查。** 当 teacher 与 student 恰好相同、soft KL 对同一 FP 路径求梯度，一阶梯度可为零；不能自动换成 hard pseudo-label 然后仍称同一 paper recipe。prepare 输出 grad norm、零比例、任务/温度和 teacher/student 状态。确需 hard pseudo-label 或人工扰动，应独立 ID 和数据预算。

### 5.2 APQ、APH、FIMA 的注册差量

| ID | 核心定义 | prepare / candidate_score 分工 |
| --- | --- | --- |
| apq_masked_gradient_error | \(\sum_i g_i^2e_i^2m_i(e,\gamma)\) | \(g\) 可缓存；mask 随候选误差重算。原 Eq.(8) 与文字方向有歧义，literal/text-intended 两 variant 独立 |
| aph_signed_directional | \((g(z+\delta\mathbf1)-g(z-\delta\mathbf1))/(2\delta)\) 的样本均值 | 2次 suffix backward/批；实为 \(H\mathbf1\)，只有相应假设下近似对角 |
| aph_official_abs | \(E[\lvert g^+-g^-\rvert ]\)，score 为 \(\sum_i h_i e_i^2\) | APHQ 当前源码版；省略公共 \(2\delta\) 且取 abs，不与 signed 版混用 |
| fim_secant_paper | \(F_k=G(D^\top D)^{-1}D^\top\)，另含论文 diagonal \(g_i/d_i\) | \(D,G\in\mathbb R^{a\times k}\)；无需显式 \(a^2\) 矩阵，缓存模型/扰动/秩 |
| fima_official_abs_normalized | \(L_1=(\lvert e\rvert G^\top)(DD^\top)^{-1}(\lvert e\rvert G^\top)^\top\)，\(L_2=\mathrm{mean}(\lvert e\rvert ^2\mathrm{mean}_k\lvert G\rvert )\)，按初始 loss 归一后加权 | 此处代码存 \(D,G\in\mathbb R^{k\times a}\)，先 abs 再样本均值；rank5 为当前4bit配置 |
| curvature_secant_residual | \(\lVert FD-G\rVert_F/(\lVert G\rVert_F+\epsilon)\)、有效秩、方向覆盖 | 估计器自诊断；小残差不等于任务排序一定准确 |
| task_prediction_kl | 固定 FP 或 Q suffix 下，当前块扰动对 teacher logits 的 KL | 复用 teacher_kl；新增的是 differentiable suffix 或 frozen intervention 观察路径 |
| contrastive_teacher_logits | 固定 batch/negative-bank 的 InfoNCE | CPT-V/Evol-Q 目标；L2归一、temperature、负样本、batch与常数倍率全入身份 |

APH 默认 \(\delta=10^{-6}\) 的数值稳定性依赖 dtype；报告正负 perturbation 后实际差值是否被表示。FIMA 的除以 \(d_i\)、小矩阵 inverse、非对称/负二次型都可能产生退化；若实现 pinv、ridge、abs、clip-to-positive，显式另建数值 variant。不要声称 paper approximation 是 exact/总为 PSD Fisher。registered_objective 同时记 raw_components 与 normalization_constants，避免在不同候选上重新归一而改变排序含义。

### 5.3 如何进入现有 frozen 管线

增加只读 **prepare_sensitivity** 阶段，在临时 probe 模型对选定 view/head 运行 FP/Q suffix，得到 detached 统计；随后现有 inference_mode scorer 只读取统计。这个分离避免把 backward 偷塞进已有 frozen API。完整 FIMA/APHQ 重构还需单独 optimizer executor，不能仅添加 scorer 就宣称方法实现。

敏感度缓存按其实际依赖绑定模型与 head 指纹、site、FP/Q 前后缀、teacher、数据/变换、token layout、temperature、perturbation seed/rank/dtype。原始未温度化 logits 与温度化概率/梯度分别缓存；只改温度或组合损失权重不必重跑未受影响的 teacher 前向。若定义的是固定参考 metric，候选变化后缓存可继续使用，前提是仍说明它是该固定参考的敏感度；若 recipe 要求每阶段重新估计，则按阶段刷新。MR、重参数化、上游补偿、token/分辨率变化通常会改变这些依赖。

双任务可分别准备 \(h^{cls}\)、\(h^{seg}\)，保留原始任务损失尺度，再按预声明权重形成组合。先测每个任务的 candidate ranking 和 cross-task transfer；不因“联合目标更复杂”就假定其更有科研价值。

## 6. 传播、交互与补偿收益

已有 propagation.py 的代数是研究基础。对同一样本与当前第 \(l\) 块，令
\[
y_f=F_l(x_f),\quad y_c=F_l(x_q),\quad y_q=Q_l(x_q);
\quad e_l=y_q-y_c,\quad e_u=y_c-y_f.
\]
则总误差 \(e=e_l+e_u\)，
\[
\|e\|^2=\|e_l\|^2+\|e_u\|^2+2\langle e_l,e_u\rangle.
\]
SSE 不能省略交叉项后相加。统计按同一 token mask、原图/窗口/层定义归约，避免比较不同能量单位。

| 注册 ID | 定义 | 用途 |
| --- | --- | --- |
| propagation_decomposition | input_total/output_total/local_added/upstream_mapped 和 signed cross | 已有，保持语义与原始 SSE |
| propagation_cross_alignment | \(\langle e_l,e_u\rangle/(\lVert e_l\rVert\lVert e_u\rVert)\) 与 signed cross/total-energy | 解释误差抵消或放大；任一误差零时 cosine undefined |
| propagation_amplification | \(\lVert F_l(x_q)-F_l(x_f)\rVert/\lVert x_q-x_f\rVert\) | 有限扰动增益，不称全局 Lipschitz 常数；输入差太小时返回 unstable 状态 |
| depth_error_profile | 各层 raw/normalized total/local/upstream 曲线、峰值及位置 | 深度漂移与局部注入区分，layer width/能量同时报 |
| intervention_task_gain | 替换某 FP block/关闭某量化器后 \(U_{\rm intervene}-U_{\rm base}\) | 已有干预可复用；这是给定其余模型的条件效应，不等于可部署修复收益 |
| factorial_interaction | \(I=U_{AB}-U_A-U_B+U_0\) | 研究两个 primitive 的协同/冲突；同数据、同底座、同预算 |
| stage_gain_retention | 在同一 heldout metric 上比较阶段改善在后续 W4/组合后保留多少 | 保存每阶段 raw gain；分母接近0时不只报比率 |
| weight_activation_cross_error | \(W\Delta X+\Delta WX+\Delta W\Delta X\) 各项/交叉项 | ERQ耦合诊断；W-only/A-only 不直接相加成 W4A4 |

收益保留可报告 \((L_0-L_{\rm final})/(L_0-L_{\rm intermediate})\)，但其参考、数据、误差定义必须相同，且中间配置仅为诊断。比例超过1、为负都有可能；不能截成[0,1]抹掉机制。端到端任务收益在固定部署 profile 上另外报告，避免靠中间 FP 权重得出最终低位宽已改进的结论。

## 7. 候选排序与选取质量

metric 的科研作用是帮助在有限预算内找到好配置。不能只看它与自身优化 loss 的一致性。预先固定候选集合 \(C\)，用 proxy 数据得到 \(m(c)\)，在独立诊断集得到任务 utility \(U(c)\)，令 \(\hat c=\arg\min_{c\in C}m(c)\)。

\[
\mathrm{selected\_regret}(m;C)=
\max_{c\in C} U(c)-U(\hat c).
\]

这是**给定候选池**的选择后悔值，不是相对全局最优方法。若按 loss 衡量 utility，就用 \(L(\hat c)-\min_{c\in C}L(c)\)。Top-1/mIoU 使用百分点，KL/MSE 使用各自单位；不要把它们直接相减。

| 注册 ID | 定义与保存内容 | 研究含义 |
| --- | --- | --- |
| candidate_rank_correlation | Spearman、Kendall tau-b；proxy越小对应utility越高，先统一符号 | 单调排序质量；候选少时精度有限 |
| pairwise_selection_accuracy | 候选对的优劣符号一致比例，ties预声明，附任务差异大小 | 可另报只对有实际差异的候选对，阈值需预先声明 |
| topk_candidate_recall | proxy前k中包含多少task前k；同时保留命中task最优的布尔量 | 看初筛能力，而非只看全排序 |
| selected_regret | 上式；保存 selected_id、pool_best_id、raw utilities、tie policy | 直接衡量选错损失 |
| candidate_pool_coverage | pool_best−baseline；以及不同搜索器候选集合并集 | 区分候选没找到好方案和 metric 选错 |
| combination_retention | 单独最优primitive装配后收益、与逐步重选/重校准之差 | 组合失效不应归咎于单站点 scorer |
| cross_task_selection_transfer | 用cls选、seg选、联合选的候选，在两任务上的utility向量/各自regret | 区分共享表征修复与单头投机 |
| calibration_selection_stability | 改 calibration子集/seed后胜者频率、rank一致性和最终utility分布 | 记录原图ID；固定head/transform以定位数据原因 |

若各候选任务差异小于评测不确定性，不宜武断判定“selector失败”。报告 paired uncertainty，并可将统计上近似的候选视为 practical tie；阈值不能依据最想保留的候选临时选择。离散 Top-1 容易 ties，多用连续 NLL/KL 辅助解释，但正式主结论仍按预声明任务指标。

多任务采用原始向量 \((\mathrm{Top1},\mathrm{mIoU})\) 和 Pareto 集为默认展示；需要一个选择 scalar 时，提前声明权重和归一化参考。不要对每一批候选各自 min-max 归一后比较不同实验，因为加入一个候选就会改变其他候选的分数。可用固定 baseline 的损失尺度或预先设定任务容忍度，仍需同时报未归一化结果。

现有 frozen 的 per_image/per_window_layers 输出足够实现大部分 selector 后处理。新增核心是独立 heldout 绑定、完整 candidate state 支持及结果字段，不需要另造一套前向执行器。当前 search split 在代码中是确认声明候选，调整为可用的 selection split 时需与新协议一起升级，不能绕过旧 API 含义。

## 8. 任务质量、数据选择与泛化

**分类。** 注册 top1、top5、NLL、teacher agreement、预测 margin drift；confidence/ECE 可作补充，固定 bins、是否等宽/等频与温度策略。数据必须标注标签来自真值、teacher pseudo-label 还是无标签；teacher agreement 不能替代真实准确率。基线 FP 使用相同 checkpoint/head/预处理，报告 FP值、量化值和绝对下降。

**语义分割。** 注册 dataset_miou、per_class_iou、pixel_accuracy；先对每张原图合并窗口、处理 ignore_index，再累计 \(C\times C\) confusion matrix。对类 \(c\)，\(\mathrm{IoU}_c=TP_c/(TP_c+FP_c+FN_c)\)，缺失类别是否排除按数据集标准。mIoU是类均值，不是每图IoU均值。需要研究边缘退化时增加 boundary F-score/Boundary IoU，明确边界宽度随像素或图像比例的规则。多尺度/翻转、插值方式、head和特征层固定记入任务profile。

**校准集选择本身也是研究变量。** 可以比较随机、类别均衡、有代表性的 feature 覆盖、困难样本、离群分位覆盖等有限校准集合；这些是 QAViT 研究工作流，不属于本文件任一方法的原配方。标签均衡有额外标签预算；teacher-feature聚类有 teacher forward/聚类开销；困难样本若依赖评测集任务结果，会消耗诊断/selection预算。每种策略保存 unique image IDs、解码/增强指纹、抽样 seed、预算和覆盖统计。

建议分开四种数据用途：calibration-fit 拟合量化参数；selection 选候选/超参/checkpoint；heldout-diagnostic 检验代理和机制；final-evaluation 给最终任务结果。资源不足时并非必须构造四套新数据，可在 training split 内重用并如实记录自适应查询次数；但 final-evaluation 参与选模后就不再是未触碰最终评测。研究模型跨任务时还需区分来自ImageNet与ADE等不同校准域的迁移矩阵。

## 9. 不确定性与复现性

注册 paired_metric_ci、paired_task_delta_ci、seed_variability、selection_frequency、bootstrap_regret。默认重采样单位是**原始图像**，让同图的多个窗口、token和pixels共同被采入；不能把它们当独立样本制造狭窄置信区间。候选与baseline使用同一批重采样ID，比较差值而不是两个不配对区间。

分类可对逐图 correct/NLL 差值做 paired bootstrap。分割保存每图 confusion matrix，重采样后先求和，再重算各类IoU和mIoU；不能 bootstrap 每图mIoU均值来声称dataset mIoU置信区间。排名/selected regret可对proxy数据和heldout数据分别重采样，重做选择，评估“选择过程”本身的不稳定性；只固定已选候选bootstrap任务误差回答的是较窄问题。

数据seed、优化seed与系统非确定性分开报告；同条件尽量使用相同数据ID、初始化和随机流。小规模筛选可先单seed，只有临界结论、预期小收益或主要baseline再增加重复；不强制所有新metric全模型多seed。报告重复数、均值/中位数、spread和完整运行ID，失败运行与数值退化不能悄悄删除。

参考统计方法来源为 Efron 的 bootstrap 原始论文（1979，[DOI](https://doi.org/10.1214/aos/1176344552)）；此处 paired 原图重采样和分割confusion重算是结合QAViT数据结构的实验设计，不是某PTQ论文已经使用该流程的声称。

## 10. 校准成本、存储与真实部署

资源账本分三部分，避免把搜索付出的成本与最终模型推理成本混为一谈。

| 注册 ID | 至少保存的字段 | 比较方式 |
| --- | --- | --- |
| calibration_data_budget | unique images、views/windows、tokens/pixels、label counts、总样本呈现次数 | 32张重复优化20000次仍是32个unique图，但计算不等于32次前向 |
| calibration_compute_budget | teacher/student forward数、backward数、梯度probe数、候选数、优化步、线性求解/分解数量 | APH两次扰动/FIMA刷新/进化完整前向都计入；另报walltime、device、precision、峰值显存 |
| research_search_budget | 方法/超参/seed trial数，selection标签查询，失败/中断成本，缓存构建与复用 | 报单次recipe与找到最终recipe的总成本，两者都有价值 |
| model_storage_bytes | packed weights、A参数、scale/zp、索引/重排、bias、补偿、header/alignment | 用实际序列化字节和部署常驻字节；理论bit数另外列 |
| activation_runtime_bytes | activation峰值、workspace、缓存/输入存活、attention中间量与adapter buffers | 最终实际shape、batch和窗口策略 |
| runtime_latency_profile | 冷启动、warmup后p50/p90/p99、吞吐、batch、输入shape、并发、精确backend/device版本 | 固定测量范围；同步GPU计时；真实kernel与QDQ模拟分开 |
| operator_precision_coverage | 按operator和参数/MAC/时间分别统计W/A/accumulator/非线性精度，FP fallback及Q/DQ边界 | “95%层量化”不足以解释主要耗时；不同分母并列 |
| exported_numeric_delta | 导出整数/融合图与reference-QDQ的输出差、任务差 | 特别验证QwT-v2 multiplier/bias融合、整数溢出、scale边界 |
| end_to_end_energy | 有测量设备时记录每样本能耗、采样方法和测量窗口 | 无真实测量就留空，不能由FLOPs或bit数推造 |

矩阵W4A4固定并不自动意味着 LN/Softmax/accumulator 均为INT4。按最终图逐项声明精度；动态分组、混合bit、FP补偿矩阵、位置偏置等扩展profile仍可完整研究，但分别比较资源。QwT 的矩阵补偿需计FP16参数、额外GEMM和输入buffer；QwT-v2仅在真正融合并验证后才能声称不新增独立affine kernel。理论 BOPs 不等于延迟，某些16码以外的编码也不能据“平均4bit”宣称标准INT4内核适用。

预算比较同时保留两种视角：**原recipe** 评估作者方法在其本来设计下的效果；**资源匹配** 在相近unique data、F/B计算或walltime预算下比较。不能把需要1024张优化的方法强行缩到32张，再以失败证明方法无效；也不能把高成本重构与廉价minmax只按最后精度排一张没有成本的表。

## 11. 建议的接入层次与研究工作流

### 11.1 先接能共用的数学与执行器

1. **Observer/statistics。** 扩充attention、保留轴误差、codebook和流式二阶矩；沿用已有图像/窗口身份。
2. **Metric prepare/score。** prepare产 detached sensitivity cache；score消费固定候选或完整adapter state。FIMA/APH分别保存paper与official-code版本。
3. **PTQ optimizer。** 新增有限校准块重构、scope/参数可变范围、QDrop、阶段重估和硬化；score与optimizer共享数学函数但不混淆运行角色。
4. **Selector analyzer。** 从同候选分数与独立task结果产生ranking、regret、Pareto、bootstrap。
5. **Cost/export observer。** 连接真实模型字节、算子覆盖和backend测量，不从recipe名字自动生成性能结论。

这些是代码组织建议，不要求一次运行全模块。当前只读执行链不应直接解除 freeze 后用于优化；增加明确optimizer入口，完成拟合后生成冻结候选，继续复用现有scorer和导出校验。

### 11.2 四个能产生论文级证据的实验模板

**代理是否有用。** 固定底座、候选池和数据；比较 MSE/NMSE/site-grad/APH/FIMA/任务KL；报告prepare+score成本、排序和selected regret。先在机制相关位点看信号，再扩到全网。若更复杂代理只降低训练loss但不改善选取，保留该失败结果和可能原因。

**方法收益来自哪里。** 作者原recipe复现后，拆 primitive 做增量/必要交互对照，例如 APHQ 的MR与APH、ERQ的Aqer/Wqer/DUQ、PD的全局目标与DC、QwT目标reference与补偿形状。共享初始化与覆盖，记录再校准是否发生，防止把一次重校准收益误记为新增机制。

**数据与任务是否稳定。** 同一量化配方在不同calibration seed、分类/分割来源和规模上拟合，再测两任务；记录胜者变动与误差传播。所有头、图像几何和token范围在每次实验中显式绑定。

**数值收益是否转为部署收益。** 从同一已冻结模型导出真实backend，检验编码容量、覆盖和融合前后数值差，再测真实latency/memory。比较static W4A4与补偿/动态/混精扩展的Pareto边界，而非用W4A4标题掩盖额外成本。

### 11.3 证据层级

每条结果单独标记 paper_reported、official_recipe_reproduced、qavit_unified_protocol；这是证据来源，不是高低评价。官方代码行为与论文不一致时附variant。本文只是 specification，所有新增metric的 executable/reproduced 状态初始均为 false；已有实现的状态沿用当前QAViT证据，不因注册而扩大。

## 12. 关键原文索引

- PTQ-ViT attention ranking/敏感度：[NeurIPS2021 PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/ec8956637a99787bd197eacd77acce5e-Paper.pdf)，§3。
- APQ-ViT bottom-elimination：[arXiv](https://arxiv.org/pdf/2303.14341)，§3.2。
- APHQ APH/MR：[arXiv](https://arxiv.org/html/2504.02508v1)，§3；[作者代码](https://github.com/GoatWu/APHQ-ViT)。
- FIMA DPLR：[arXiv](https://arxiv.org/html/2506.11543v1)，§3；[作者代码](https://github.com/ShiheWang/FIMA-Q)。
- ERQ2024：[ICML](https://proceedings.mlr.press/v235/zhong24a.html)，§4；ERQ-DUQ：[v2](https://arxiv.org/html/2407.06794v2)，§4。
- QEP：[NeurIPS2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/df2034a516cbd617a96492cc476276c9-Paper-Conference.pdf)，§5、Appendix B/C。
- PD-Quant：[CVPR2023 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.pdf)，§3。
- Evol-Q：[作者修订稿](https://arxiv.org/pdf/2308.10814)，§3；[作者代码](https://github.com/enyac-group/evol-q)。
- BiasComp：[原文](https://arxiv.org/pdf/2404.01892)，§3、§5.1.3。
- QwT：[CVPR2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Quantization_without_Tears_CVPR_2025_paper.pdf)，§3；QwT-v2：[原文](https://arxiv.org/html/2505.20932v1)，§3.3–3.4。

这些引用支撑对应原方法；上述统一归约、selection regret分析、数据分工和成本账本是QAViT的研究规格，不冒称由一篇方法论文完整提出。
