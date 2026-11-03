# 尺度选择与任务相关误差

来源：PTQ4ViT / RepQ-ViT / 项目接口。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/calibration.py; coordinate.py; plugin_contracts.py:calibration`。

网格(method)与尺度策略(calibration)是两条独立选择轴。只换网格名称但保留minmax，可能仍丢失主要数值区间。

minmax：初始化；manual：显式scale_values，可选zero_values，须验证实际数量；percentile：仅affine/log_sqrt2独立observer，按张量MSE选分位数。

search：在指定真实算子输出上独立选W或A尺度，metric=mse/cosine/hessian；factors是初始scale倍率。coordinate：同算子交替选W/A或MatMul两操作数，要求joint/static、兼容uniform_full/twin网格；同组rounds/metric/loss一致。peer_precision明确另一个操作数float还是quantized。

Hessian-guided评分 mean((grad_y*(yq-y))²)，不是完整Hessian。loss=teacher_kl或classification_ce；均从calibration池真实反向得到，无优化器。teacher=相同FP模型时KL梯度可能近零、排序受舍入影响，应读取梯度/候选分数；若任务标签可用，可显式测试CE变体并注明不同于原协议，不能静默替换。

factor_grid={start,stop,count}为不含stop等距倍率，可表达较密搜索；与factors互斥。搜索范围/轮数有成本，应按证据决定。特殊softmax默认搜索split，GELU默认仅调正尺度。

局部输出MSE/梯度指标是筛查，整模型Top-1决定最终选择；不读取final做尺度选择。


## 参数核对

独立示例：`activation_calibration={"strategy":"percentile","percentiles":[0.999,0.9999,0.99999]}`只适用于声明兼容的网格；`weight_calibration={"strategy":"search","metric":"mse","factors":[0.7,0.85,1.0]}`表示真正拟合并比较三个尺度。它们不是某模型完整配方。只声明`spec.scale="minmax"`而省略calibration并没有进行上述搜索。

决定尺度前应查看分位数与极值之差、跨通道范围和目标算子误差；少量低成本比较后再选择正式策略。运行回执的strategy、trial scores、samples才证明搜索被执行，文字说明不能替代。

完整校准尺寸也属于适配验收。`grids.quantile`在元素数超过torch的2^24限制时，沿用RepQ官方NumPy线性percentile回退；不抽样、不采用分块分位数均值，不改变候选顺序/评分。小输入保持原torch路径，其它异常继续报错。DeiT-B fc2的32图（19365888元素）已验证可执行，CPU observer与同设备上游参数一致；不同设备的浮点评分/选择未证明逐位一致。该修复不表示任何完整方案已达到原文精度。
