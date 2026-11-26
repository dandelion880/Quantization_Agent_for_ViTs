## MixA-Q（仅无需训练的 PTQ 分支）

<a id="mixa-q"></a>

**证据。** 综述 [116]；[ICCV 2025 正文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_MixA-Q_Revisiting_Activation_Sparsity_for_Vision_Transformers_from_a_Mixed-Precision_ICCV_2025_paper.pdf)，§3.1–3.3 中 PTQ 可独立使用的结构，及 **§4.3 PTQ Result on COCO**。已读该独立分支。其余训练分支的排除身份与依据见统一来源账本。未核实作者代码。

**第一层：原配方。** 在 Swin 的窗口 attention 中，以窗口特征 L2 范数排序，将重要窗口送高精度分支，其余送低精度分支，再 scatter 回原空间。每个 stage 的重要度只计算一次。两分支共享 MHA/FFN 权重，LN 分离；PTQ 中 bias 可分别处理。以 RepQ 的 W4A8 Mask R-CNN/Swin-T 为起点，低精度窗口 A4；使用 **1 张 COCO 训练图**校准。为兼容共享权重，scale 初始化使用完整 feature map，不能分支各自重参数化得到冲突的共享权重。压缩比例按相邻两个 Swin block 绑定，在 0–80% 以 10% 间隔搜索，NSGA-II 平衡 mAP 与 BOPs；搜索不同压缩比例时不重新校准。

**第二层：primitive。** window selector、gather/scatter、双分支精度执行、共享权重一致的重参数化、比例配置搜索分别登记。此方法降低低重要度区域精度并保留计算，不等同 token pruning。已有 per-token/per-group 容器不能代替这些路由与共享状态规则。

**第三层：workflow。** 原 PTQ 实验比较不同校准样本下的 mixed 4/8 路线与统一降低 A8→A5 的路线；效率用 backbone BOPs 衡量。QAViT 先在 window-based 模型重现这一对照，再测路由、排序与打包后的真实延迟。将方法移到 DINOv3 的全局 attention 必须重新定义分组及信息依赖，是独立适配实验，不能直接沿用 Swin 证据。固定 4 bit 主线可参考诊断，但完整双分支只能进入动态/混精度 profile。

**差距与成熟边界。** 新增重点是 runtime route 与导出后的分支行为，而非新的 scalar quantizer。原文的 1.35× 属计算量速度比，不应写成设备端到端测得加速。


## 接入依据与来源边界

**基线组件** RepQ、window 布局/均匀量化基础。

**配方阶段与依赖** 仅接 §4.3 PTQ：window L2 路由、双精度 gather/scatter、共享 W 与分支 LN/bias、比例进化搜索。

**部署条件** 原 W4A8，低重要窗 A4，属 window-dynamic/mixed；不是静态全 A4。

**尚需核对** 全局 attention 的 DINO 无原 window 条件；1.35× 是 BOP 比率，官方完整代码未核。
