## IPTQ-ViT

<a id="iptq-vit"></a>

**证据。** 综述 [21]；[arXiv:2511.15369](https://arxiv.org/abs/2511.15369)，§4.1–4.5、§5.1 与消融；[当前作者仓库](https://github.com/gihwan-kim/IPTQ-ViT)。现仓库标为 WACV 2026，并包含后续 IPTQ++ 扩展。已读原方法正文、主实验、当前 README 和目录说明；未执行代码。仓库当前实现不能不加区分地充当 2025 预印本的锁定版本。

### 第一层：原配方

对各非线性位置评估候选整数近似，以统一评分选择每层函数，再对组合好的模型校准。其独立 PTQ 方法包括 Data-aware Poly-GELU：根据视觉数据范围拟合近似，采用四次 erf 多项式并预量化常数；Efficient Bit-Softmax：将指数拆成整数和分数部分，整数部分用位移，分数部分用低阶近似及移位加法。没有模型训练。

选择评分同时考虑 SQNR、平方 L2 扰动与整数操作数，经 Softplus 及倒数变换后做调和聚合。原文 Eq.18 对功率比写的是 `20 log`，与常用 `10 log` 的 SQNR 口径不同；导入时应保留原文版本标识并核对作者代码，不能静默覆盖 QAViT 标准 SQNR。主分类验证为 W8A8/W4A8；COCO 有额外精度设置。权重 per-channel 对称，激活 per-layer 非对称。当前已读主文把校准张数比较指向附录 C.3/D.1，完整预算应继续绑定原附录和运行配置，不能把仓库某个 VLM 的 `--calib-samples 32` 当作原 ViT 设置。

### 第二层：primitive

新增的是**非线性实现选择器**、视觉范围下的近似系数求解与算子成本估计。量化器类型与算子实现应分别建模；同样的 W4A8 可使用不同整数非线性，产生不同精度及成本。统一评分可作为一个待验证 objective，原文的 operation count 仅是成本代理，并非设备延迟。它与现有 NMSE、AV 等指标并列，不能替代主任务裁决。

### 第三层：workflow

原文比较近似族、阶数及统一评分，并在 TVM/W8A8、batch 8 上报告设备延迟。QAViT 应先保持线性量化不变，比较统一近似与逐层选择，再观察 error/operation-count Pareto 点是否在真实设备兑现。A4、DINOv3 多尺度和 ADE 为新增适配范围。若某个 operator route 没有真实 kernel，应保留算法实验结果并标注 simulation，而非回退浮点后仍声称 integer-only。

**代码与导出边界。** 当前 README 提供 ImageNet 入口、PyTorch 非线性模拟和 CUDA kernel 路径，也明确部分 log-family 组合没有 kernel、验证与部署 kernel 模式不同，并提及 fp32 运算顺序。应锁定 commit，逐条核对实际 arithmetic，再决定 `integer_only` 标签；“目录里有 CUDA 文件”不是纯整数执行证明。完整原法进入 integer-operator profile，固定 4 bit 适配属于新实验。


## 接入依据与来源边界

**基线组件** SQNR/L2、operator 接口、成本测量基础。

**配方阶段与依赖** 补视觉范围 Poly-GELU、Bit-Softmax、按层近似选择与统一成本代理。

**部署条件** 原主 W8A8/W4A8；integer-only 与主线四位覆盖分别判定。

**尚需核对** 锁原版本与后续 IPTQ++ 差别；预算附录/实际 kernels 未全闭合，功率比 20log 需保留原指标身份。
