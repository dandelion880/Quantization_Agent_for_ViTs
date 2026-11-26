## 13. Outlier-aware Slicing / OAS-ViT（ICML 2024）

<a id="outlier-aware-slicing"></a>

**来源。** 综述[27]，[正式论文及补充材料入口](https://proceedings.mlr.press/v235/ma24f.html)、[PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/ma24f/ma24f.pdf) §3、AppendixD/F、Algorithm1。作者说明代码在Supplementary Materials；本次没有取得并运行该代码。

**paper-recipe。** 这里slicing划分的是**重构子图**，不是把离群通道切到高精度。作者发现离群值与重构粒度相关：同质ViT/DeiT可用连续多个block联合重构；Swin跨stage的downsample改变拓扑与梯度条件，细粒度更合适。方法把block拆为self-attention、output projection、MLP三个模块；检测downsample，存在时倾向每block的三个细单元，否则使用n个连续block作为重构单元。论文采样研究最多三个连续block，且更粗并非总改善任务准确率。

给一个被选子图 \(R\)，优化 \(\|R_{fp}(X_{fp})-R_q(X_q)\|_F^2\) 及舍入正则，采用QDrop激活混合。附录F：16batch×64=1024张，舍入lr1e-3、A scale lr4e-5、rounding loss权重.1、每单元20000次、drop=.5、软舍入温度指数20降2；activation初始化EMA-MSE、weight minmax。作者关于粗粒度误差下界的分析依赖拓扑及优化假设，不等于保证任意更大子图都有更高heldout准确率。

**primitive。** reconstruction_topology_classifier 输出stage/downsample与子图边界；outlier_reconstruction_profile 记录不同slice的误差集中与优化响应；slice_reconstruction 调用共用有限校准优化器。特殊架构先解析真实图，不能用模型名包含“Swin”作为唯一条件。保存边界上的输入来源、shortcut和LN是否包含，保证同一分区能够复刻。

**workflow 与接入。** QAViT可把现有传播诊断、离群统计与拓扑元数据用来形成有限候选分区，再比较固定block、A/B/C、2/3block；这种数据驱动选择是新workflow，不是作者的已验证自适应规则。必须区分减少局部loss与降低最终任务误差，同时收集峰值内存和每个slice优化耗时。最终静态W4A4可行，无推理期动态分组；作者有ImageNet ViT/DeiT/Swin低bit实验。状态：原文与附录已读；官方supp代码具体文件与所有覆盖例外仍待核。


## 接入依据与来源边界

**基线组件** 布局/传播诊断、AdaRound 与块捕获基础。

**配方阶段与依赖** 补 stage/downsample 子图解析、细单元/连续多 block 分区、QDrop 式子图重构。

**部署条件** 分的是重构子图，最终静态；不是高精度离群通道旁路。

**尚需核对** supplementary 代码具体实现/全部覆盖例外未核；不能用模型名字代替真实边界或声称越粗必越好。
