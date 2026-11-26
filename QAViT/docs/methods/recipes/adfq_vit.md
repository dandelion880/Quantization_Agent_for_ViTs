## 2. ADFQ-ViT：异常值保护与模块目标必须一起记录

<a id="adfq_vit"></a>

**证据。** 综述 [71]；[原文 v1](https://arxiv.org/html/2407.02763v1)，§3.2–3.5、§4.1；[期刊版本](https://www.sciencedirect.com/science/article/pii/S0893608025001686)，Neural Networks 186（2025）107289；[作者仓库](https://github.com/llwx593/adfq-vit) main 已下载并核读 `quant_layer.py`、`block_recon.py`、`argsparser.py` 和 README。方法主体可建立 PTQ 原配方，但精确的代码行为应以该快照记录。

### 原配方

post-LN 激活沿 patch 处理：识别异常元素，普通元素用低位宽量化，异常元素保持高精度并参与后续计算。因此它不是把所有异常值也压入同一个 4bit 张量的算法。post-GELU 的 ShiftLog2 将输入平移到正域后做 log2 编码，解码时再还原偏移。模块重构同时调整舍入参数和激活 scale，MHA 再加 attention 相关损失；这三个部件共同构成原方法。[原文 §3.2–3.5](https://arxiv.org/html/2407.02763v1)

已确认的论文设置包括 ImageNet 1,024 张，COCO 检测/实例分割校准 1 张，优化 3,000 次，舍入参数学习率 \(3\times10^{-3}\)、激活 scale 学习率 \(4\times10^{-5}\)。AdaRound 式正则采用有界 rectified sigmoid 与退火；官方参数入口默认 `iters_w=5000`、`b_start=20`、`b_end=2`、`warmup=0.2`，README 命令把次数覆盖为 3,000。保留有效命令而非仅复制函数默认，不能与已有 AdaRound 的配置混用。[官方代码](https://github.com/llwx593/adfq-vit/tree/main/quant)、[参数入口](https://github.com/llwx593/adfq-vit/blob/main/argsparser.py)

源码提供两条重要精化。第一，[`TokenQuantizer`](https://github.com/llwx593/adfq-vit/blob/main/quant/quant_layer.py) 实际以 `abs(x)>alpha` 取异常值，每次 forward 恢复这些原始浮点值；scale/zp 则在 `inited=False` 时初始化并缓存，所以“per-patch”不等于每次动态求 scale。第二，qkv 的分支会把异常值阈值设为 5，README 示例又传 `--outlier 10`，不能将一个全局阈值假定为所有位点的最终值。每个实际模块要导出有效配置。

### primitive 与接入约束

建议 `outlier.sparse_bypass` 的输出为低精度主体、异常位置及高精度值，账本同时统计 mask/index/残差值的字节数与稀疏计算量；`rep.shift_log2` 负责平移与对数码本；`loss.adfq_module` 负责模块损失。原文把 \(AS\) 定义在 Softmax 与 V 乘积之后，而[官方 QuantAttention](https://github.com/llwx593/adfq-vit/blob/main/quant/quant_block.py)返回的是 Softmax 后、乘 V 前的 `attn`。这需要 paper/code 两份明确损失定义。

[block_recon.py](https://github.com/llwx593/adfq-vit/blob/main/quant/block_recon.py)实际调用把缓存 FP attention 送入 log_softmax，把量化 attention 送入 softmax，使用 dim=1；B×H×N×N 情况对应 head 轴，KL 为 batchmean 后除以 `lamb_r`。不能未经标注改成常见的最后一轴教师→学生 KL。代码 profile 保存这些实际行为，数学适配可以另设正常方向/归约轴，但必须单独比较。

原配方进入 `ptq_outlier_bypass`。去掉异常值旁路再做 static W4A4 是可研究适配，不能继续声称复现 ADFQ 原结果。若选固定 token 位置尺度，需要扩展现有 contracts 的静态轴语义，并覆盖多分辨率拒绝/重校准规则。post-GELU 的平移也要记录是数据初始化值还是常量，禁止在 inference 隐含重估最小值。

### 研究工作流与当前差距

QAViT 已有 log/shift/AdaRound 零件，未有这一整套异常值数据通路及模块目标。拟议 `study.outlier_protection_cost` 将异常值率、旁路成本和任务精度放在同一曲线上，再比较 clipping、静态重参数化和原始旁路。`metric.outlier_payload_bytes` 与 `metric.outlier_task_sensitivity` 是新增建议；现有 token 布局和双视图指标可直接复用。重点是确认保护的是否是语义关键值，而非仅观察最大值下降。


## 接入依据与来源边界

**基线组件** log/shift、AdaRound 单位点基础、布局与输出观测。

**配方阶段与依赖** 补异常值载荷/稀疏 FP 计算、ShiftLog2、位置尺度、模块重构；attention 目标保留 paper/code 两版。

**部署条件** 原法保留输入相关 FP 异常值，属额外计算/动态旁路；缓存位置尺度另需静态布局契约。

**尚需核对** AS 观察点、KL 方向/轴/权重和有效异常阈值存在已记录差异；去旁路版本不叫原法。
