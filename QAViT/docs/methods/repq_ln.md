# LN 激活尺度重参数化

来源：RepQ-ViT。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:repq_ln; pipeline.py:preprocess`。

面向LN输出跨通道差异，先得到channel affine尺度/偏移，再等价变换成共享尺度的输入。需真实直接LN→Linear边；例如位点应从catalog查询，不能凭名称猜前驱。

诊断时可对目标Linear输入调用 `profile_distribution`，令 `target=activation, views=[tensor,channel]`，比较各通道范围；仅整体min/max不能判断跨通道差异。先从实际位点目录确认LN直接边，再用独立局部方案比较变换前后误差；这不规定全模型要在哪些层采用本方法。

最终 A 必须 UINT、per_tensor/static，方法rtn或uniform_affine；W方法独立。条目填写 method/module/norm/evidence，可选 calibration={strategy:percentile,percentiles:[...]}。不接受alpha。

可选 observation=float 为FP输入适配路径，可和其它独立组件组合。observation=quantized_prefix 使用量化前缀观察；当前仅验证 joint/sequential、全计划 affine/log_sqrt2 percentile 初始化；与其它网格混用会明确拒绝，不能称任意插件皆可拼接。

伪代码核心：从每通道(scale_c,zero_c)得到共享(scale,zero)，构造 input_scale/input_shift 和对应 W/bias补偿；先检查FP等价，重拟合变换后的W，A静态冻结。

和PTF方案级互斥，同边SmoothQuant互斥。选择float观察不等于原prefix协议；必须在报告保留差异。收益需以局部及任务准确率检验。


## 独立调用与诊断

单边条目形如 `{"method":"repq_ln","module":"从catalog选择Linear","norm":"其直接LayerNorm前驱","observation":"float","calibration":{"strategy":"percentile","percentiles":[0.999,0.9999,0.99999]},"evidence":["实际测量引用"]}`。这里是接口示例，不能把占位文本作为可执行计划。W和其它边的A仍由独立插件配置。多边可通过`preprocess_groups`相同选择器表达。

对照时固定其余网格、尺度和范围，比较无变换与此变换；检查FP等价门禁，以及变换后共享尺度是否改善误差。float与quantized_prefix的观察差异也必须明确。不得把通道间差异大直接等同于该变换必然提高分类精度。

实现边界修复：float观察后显式对A使用percentile时，会在变换后的输入上重新拟合共享scale/zero，同时保留`repq_affine`执行标记；否则历史实现会在导出后重载时被严格拒绝。此分支是适配变体，不能称为原文quantized_prefix观察协议。修复不改变已选scale/zero、舍入、clipping或原prefix分支。
