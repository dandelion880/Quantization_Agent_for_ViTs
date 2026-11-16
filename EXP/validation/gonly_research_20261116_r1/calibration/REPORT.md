# 校准搜索空间交付与针对性数值验证

本次已将真实Transformer block调度、重估及block/整网输出目标选参接入既有数值链。未启动正式贯通实验，未修改正式static W4A4或冻结双任务迁移协议。这里的数值来自CPU上的真实PyTorch小型随机ViT、合成图像和显式夹具标签，只证明执行和选参，不表示DeiT/ImageNet或ADE收益。

## 核验与变更

已有W/A顺序、实际前向拓扑的operator调度、静态minmax/percentile/manual/search/coordinate以及局部MSE/cosine/Hessian梯度代理均已存在，没有另造同功能系统。已有target级校准参数未消费检查、GPTQ禁止伪RTN尺度搜索、AdaLog候选base必须使用search等检查保留。

实质新发现：RepQ-LN preprocess observer原来允许`{strategy:minmax,percentiles:[...]}`，但minmax并不消费percentiles。现在按策略允许精确字段，明确拒绝此类声明；没有将其静默转为另一科学选择。

原生`plan.execution`新增并执行：`calibration_unit=operator|block`（block显式给出实际路径`block_modules`）、`layer_order=forward|reverse`、`reestimate_passes`及`calibration_context=current|float`。block以真实forward中的连续位点组成，W/A阶段在每个block内完成后推进；重估重新捕获数据且临时关闭当前目标，避免双重QDQ，保留当前其它已量化目标。回执包含实际pass/block/启用W/A、样本及选中参数。新增字段同步进入宿主冻结合同和数值去重键，显式默认值不会成为伪新实验。

`ptq_agent.execution.calibration_search.search_frozen_scales`为custom数值API，接收宿主核验的已冻结native完整模型，以每个明确候选实际执行block或整网forward，选择最小目标并安装真实静态参数。支持FP原图像边界与当前Q输入的local-FP block参考，all/CLS/patch及float/Q背景；整网可选MSE/cosine/KL/CE/Top-1 error/CE logit平方梯度代理。每项都有精确定义，不把代理叫完整Hessian。候选始终相对初始冻结参数，异常恢复模型和hook。历史源码隔离类以明确数值接口兼容，测试故意确认当前`quantizers(isinstance)`看不到隔离类，但新API正常执行。

候选参数只在calibration拟合；外层search比较和正式双任务评价仍需独立执行。导出使用选中模型当前记录，附`custom_selection`报告；已有导出重载核验和保护边界保留。

配套修正：构造替换子模块时新模块默认training=True，原root.eval标志不能覆盖它们；`build_model`末尾递归eval/冻结，并在audit递归核查。这修复原H3要求完整eval状态时的真实构造缺陷。

## 验证

- 最终专项8项通过，6.053秒，含真实两block逆序W/A调度、两pass重估状态及参数变化、目标/背景改变分数、选中参数实装、所有整网metric、历史隔离类、异常恢复、W选参及导出重载逐值一致。
- 与执行合同、计划一致性、受限数值import/真实worker合测58项通过，31.991秒（`regression_final.log`）。此后仅收紧RepQ那一项无效字段及对应测试，并将选参报告写入导出；最终8项再次全过。
- 初次75项合测出现3项旧`DraftRepairRegression`选择版本夹具失败，全部保留在`regression.log`；不是数值失败，不擦除或记作通过。根代理负责全局夹具/当前工作流集成验证。
- 可复现实测入口：`python -B EXP/validation/gonly_research_20261116_r1/calibration/replay_tiny.py`。结果`tiny_numerical.json`保存源码SHA256、完整plan/audit、CPU/Python/Torch、seed=7、2block/24维/3head、3张32×32合成图像、所有候选分数、选中尺度与参数及实际样本。

六个候选倍率为0.4/0.6/0.8/1.0/1.2/1.4，改变首block QKV输入A：

| 实际内部目标/背景 | 选中倍率 |
|---|---:|
| 整网MSE / cosine / 教师KL，完整Q背景 | 1.0 |
| 整网CE（夹具标签） | 0.6 |
| 整网CE平方梯度代理 | 1.2 |
| 整网Top-1 error | 0.4（准确率平分按声明首项） |
| block MSE，完整Q背景，FP原图像参考，all/patch | 0.4 |
| 同上CLS | 0.8 |
| block MSE，float背景，FP参考 | 1.0 |
| block MSE，Q输入local-FP参考 | 0.8 |

这些差异说明目标、上下文、参考与token划分确实进入选参；也提供一个有解释力的反例：背景上游误差很大时，FP block参考与局部参考可能偏向不同尺度，不能把最大局部MSE当根因。合成夹具上排名不同不证明任一指标普遍更优。

## 适用边界与未验证

API只支持当前与历史兼容的native静态量化器接口；block目标只允许block内干预，整网目标当前要求单一Tensor输出。权重倍率重编码仅RTN/uniform网格，GPTQ/GPTAQ/AdaRound重新求解仍走其原执行器或明确custom实现。整数复合算子与block/多pass/float背景、observer_batch_size等未验证组合明确拒绝，不能静默使用旧绑定。原生GPTAQ及quantized_prefix不能同时指定float背景。

未做真实DeiT校准收益、ADE冻结迁移收益、不同随机种子稳定性、大规模候选缓存性能与完整GPU验证；没有训练或final数据选参。正式下一轮前可以由Generation按不确定性选择少量判别对照，没有新增必扫全网、固定次数或科研价值审批。
