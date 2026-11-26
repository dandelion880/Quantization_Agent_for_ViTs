## AMP-ViT

<a id="amp-vit"></a>

**证据。** 综述 [68]；[WACV 2025 正文与补充材料入口](https://openaccess.thecvf.com/content/WACV2025/html/Tai_AMP-ViT_Optimizing_Vision_Transformer_Efficiency_with_Adaptive_Mixed-Precision_Post-Training_Quantization_WACV_2025_paper.html)，§3.1–3.3、Algorithm 1、Tables 1–4。已读正文方法及主要设置；未核实作者公开实现。作者为 Yu-Shan Tai、An-Yeu Wu。

**第一层：原配方。** 先用 SymAlign 对 LN 后激活进行逐通道平移和缩放，等价吸收至 LN 与相邻线性层参数；中心为校准激活的通道均值。随后 AutoScale 把 post-GELU 分成负值、小正值、大正值三个区域，以共享基准 scale 与整数位移表示区域 scale 比率。上下界来自平均最小值和 99.95 百分位；根据范围确定一个移位量，再用 Hessian-guided 目标搜索另一个移位量及基础 scale。固定精度版本可单独使用这两步。完整 mixed 版本从 W8A8 开始，以 `SQNR(b−1) × log(numel)` 评分逐次降低某层 W 或 A 的位宽，直到达到各自平均位宽目标；降低 GELU 位宽时重新执行 AutoScale。模型无任务训练；有等价参数改写、校准统计与离散搜索。Table 1 的固定 W4A4/W6A6 使用 32 张图；还报告 fully quantized mixed 5 bit 等设置。默认比较保留 Softmax/LN 的浮点计算，fully integer 配置应单独识别。

**第二层：primitive。** 分别登记中心化等价变换、三区域 GELU 编码器、区域 scale 比率搜索、基于 SQNR 与参数量的贪心位宽分配器。QAViT 已有 SmoothQuant、bias 类改写和 GELU 非对称量化；需要补的是精确的等价方程、三区域码字布局及分配算法，而非把名字再注册一次。4 bit 区域/符号位属于原 4 bit 码字内部，须验证总编码数量和重建表，不允许把区域标签当免费旁路。

**第三层：workflow。** 原文有 SymAlign/AutoScale 消融和不同平均位宽实验。QAViT 先在固定 4 bit 下比较中心化、AutoScale 及组合，再在同一 quantizer 下比较均匀分配与原贪心分配。记录低 bit 步骤后的真实误差和预算是否达成，验证贪心评分与实际任务降幅是否一致。已有梯度捕获可用于分类 Hessian 代理；dense 任务的反向目标需由任务适配器提供，不能默认为分类 CE。

**协议与代码状态。** 固定位宽的两个组件可进主线；完整分配器进 layer-mixed profile。论文足以拆解核心配方，作者完整代码与补充材料中的所有搜索网格尚未核实，不宣称 turnkey 复现。


## 接入依据与来源边界

**基线组件** SmoothQuant/bias、GELU 网格基础、输出误差。

**配方阶段与依赖** 补 SymAlign 精确折叠、三区域 AutoScale/尺度比搜索、SQNR×log(numel) 贪心分配。

**部署条件** 固定组件可 W4A4；完整分配属 mixed，默认 LN/Softmax FP 与 fully-integer 分开。

**尚需核对** 作者代码/完整搜索网格未核；区域与符号必须在总 bit 内，与 MPTQ 共家族而非两个独立算法。
