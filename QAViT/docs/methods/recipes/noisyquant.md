## 10. NoisyQuant：固定偏移改变网格相位，补偿依赖最终 W

<a id="noisyquant"></a>

**证据。** 综述 [28]；[原文 v2](https://arxiv.org/html/2211.16056v2)，CVPR 2023，§3.2–3.3、§5.1 与校准规模附录；[官方仓库](https://github.com/kriskrisliu/NoisyQuant) main 已核读 `fast_quant.py`、`run.sh`、README。论文与代码快照存在明确差异，应提供两个复现配置。

### 原配方及代码差异

为每层采样一次固定通道偏移 N，对 \(X+N\) 做 A 量化，同时改 bias：

\[
Y=\widehat W Q_A(X+N)+b-\widehat W N.
\]

偏移不是推理时重新抽取的随机噪声；补偿使用量化解码 W。它改变输入在量化网格中的相位，以缓解重尾激活的量化误差。论文先确定基础量化器，再以经验量化误差选噪声幅度，N 与补偿 bias 以 INT16 保存。主实验为 W6A6/W8A8，分类及 DETR 校准各 1,024 图。[原文 §3.3、§5.1](https://arxiv.org/html/2211.16056v2)

[公开 `fast_quant.py`](https://github.com/kriskrisliu/NoisyQuant/blob/main/fast_quant.py) 实际用了 `randn_like` 生成方向，继而搜索 mean/range，并以 Linear 输出 MSE 选择候选；与论文“Uniform 样本、激活误差目标”的叙述不同。代码 mean 候选索引为 -200…199，range 为 0…1999，不能把这些数当成论文统一默认。代码用先量化再减 N 的数值形式实现效果；部署规格应将减 N 折叠至输出 bias 并验证等价。

### primitive 与顺序

`transform.fixed_noise_shift` 接受已冻结基础网格、随机种子和搜索预算，输出 N、选中幅度/均值与误差轨迹；`transform.fold_shift_with_decoded_weight` 与已有 AdaLog bias 的折叠基础设施共享，但 N 是逐通道搜索所得，不能套用 GELU 的固定正向偏移。优化 W、替换 GPTQ/GPTAQ、改变 W scale 都必须重算补偿。

static W4A4 可研究的是主矩阵乘 W/A 都为真 4bit，同时如实保留 INT16 bias/N 的元数据及输入加法；若运行图保留浮点残差矩阵支路，则另属扩展配置。应核验偏移是否把有用值推到饱和边界，不能引用论文 6bit 的精度作为 4bit 保证。

### 研究工作流与当前差距

新增 `study.quantization_phase_shift` 将等效零点移动、固定随机通道偏移、GELU 常量 shift 做配对比较，固定所有其他 clipping 与 W 方法。多种子结果用于区分随机方向的幸运效应，训练/校准之外保留样本检查泛化。建议 `metric.shift_compensation_residual` 在关闭 A 量化时检查折叠前后应近似相同，再开启量化测收益；这能快速定位使用原始 W 而非最终解码 W 的实现错误。它是具体的等价性验证扩充，复用现有 gate，不新建重复的宏观诊断系统。

还应比较同样预算下的确定性通道偏移搜索，判断实际需要的是随机方向，还是任意固定的网格相位调整。该比较属于 QAViT 新实验，不反写成原论文结论。


## 接入依据与来源边界

**基线组件** 固定偏移与 decoded-W bias 折叠基础、局部输出 MSE。

**配方阶段与依赖** 补一次采样的通道 N、均值/幅度搜索及种子；更新 W 后重算补偿，保留 paper/code 噪声与目标差异。

**部署条件** 原主要 W6A6/W8A8；真四位适配可静态，INT16 N/bias 与输入加法须计成本。

**尚需核对** 论文 Uniform/激活误差与代码 Gaussian/Linear 输出误差不同；四位和目标模型尚无本地验证。
