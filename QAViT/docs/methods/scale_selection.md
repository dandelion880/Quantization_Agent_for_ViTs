# 尺度选择与任务相关误差

来源：PTQ4ViT / RepQ-ViT / 项目接口。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/calibration.py; coordinate.py; plugin_contracts.py:calibration`。

网格(method)与尺度策略(calibration)是两条独立选择轴。只换网格名称但保留minmax，可能仍丢失主要数值区间。

minmax：初始化；manual：显式scale_values，可选zero_values，须验证实际数量；percentile：仅affine/log_sqrt2独立observer，按张量MSE选分位数。

search：在指定真实算子输出上独立选W或A尺度，metric=mse/cosine/hessian；factors是初始scale倍率。coordinate：同算子交替选W/A或MatMul两操作数，要求joint/static、兼容uniform_full/twin网格；同组rounds/metric/loss一致。peer_precision明确另一个操作数float还是quantized。

Hessian-guided评分 mean((grad_y*(yq-y))²)，不是完整Hessian。loss=teacher_kl或classification_ce；均从calibration池真实反向得到，无优化器。teacher=相同FP模型时KL梯度可能近零、排序受舍入影响，应读取梯度/候选分数；若任务标签可用，可显式测试CE变体并注明不同于原协议，不能静默替换。

factor_grid={start,stop,count}为不含stop等距倍率，可表达较密搜索；与factors互斥。搜索范围/轮数有成本，应按证据决定。特殊softmax默认搜索split，GELU默认仅调正尺度。

局部输出MSE/梯度指标属于内部拟合目标；外层候选比较使用冻结search协议，最终正式判断遵守既定分类+分割双任务及排序协议；不读取final做尺度选择。


## 参数核对

独立示例：`activation_calibration={"strategy":"percentile","percentiles":[0.999,0.9999,0.99999]}`只适用于声明兼容的网格；`weight_calibration={"strategy":"search","metric":"mse","factors":[0.7,0.85,1.0]}`表示真正拟合并比较三个尺度。它们不是某模型完整配方。只声明`spec.scale="minmax"`而省略calibration并没有进行上述搜索。

决定尺度前应查看分位数与极值之差、跨通道范围和目标算子误差；少量低成本比较后再选择正式策略。运行回执的strategy、trial scores、samples才证明搜索被执行，文字说明不能替代。

完整校准尺寸也属于适配验收。`grids.quantile`在元素数超过torch的2^24限制时，沿用RepQ官方NumPy线性percentile回退；不抽样、不采用分块分位数均值，不改变候选顺序/评分。小输入保持原torch路径，其它异常继续报错。DeiT-B fc2的32图（19365888元素）已验证可执行，CPU observer与同设备上游参数一致；不同设备的浮点评分/选择未证明逐位一致。该修复不表示任何完整方案已达到原文精度。


## 完整冻结背景中的block/整网目标

自定义实验可导入 `ptq_agent.execution.calibration_search.search_frozen_scales`。它复用已经由宿主来源绑定并加载的native v3完整冻结模型，真实枚举指定静态尺度候选，不训练权重或头。调用后选中参数安装在同一模型；所有非干预位点恢复原完整量化背景。失败恢复初始状态并移除hook，不发布机制反证。

```python
from ptq_agent.execution.calibration_search import search_frozen_scales
model, receipt = ctx["load_frozen"](request["evidence_refs"][0])
selection = search_frozen_scales(
    ctx["backend"], model, ctx["data"],
    [{"id": "base", "scales": {"blocks.0.attn::attention_query": {"activation": 1.0}}},
     {"id": "clip", "scales": {"blocks.0.attn::attention_query": {"activation": 0.8}}}],
    block="blocks.0", metric="mse", reference="fp_model", context="quantized",
    token_scope="patch", count=2)
# selection可放入observations；冻结request需声明相同calibration count。
# 后续search评价应在这份选中model上另行执行，并如实交付evaluations/实际样本。
```

候选`scales`可同时包含多个位点，每个位点可选择`activation`和/或`weight`正倍率，均相对初始冻结参数，不连乘。W当前仅RTN/uniform网格重新编码原FP参考权重；GPTQ/GPTAQ/AdaRound重新求解应沿已有执行器或明确的custom实现。A支持已有静态网格，保持实际bits、粒度与方法。整数复合算子缓存绑定不在此API中隐式更新，遇到该组合会明确拒绝。返回`selected_parameters`的实际小张量值/大张量hash、全部候选得分、确切样本ID、背景/目标、选择规则及参数安装状态。

- `block="blocks.i"`：捕获真正的block输出，而不是位点输出；允许`token_scope=all/cls/patch`。改变位点须位于该block内。`reference=fp_model`对应原FP在同图像上的block输出，包含上游差异；`reference=local_fp`对应原FP block在当前冻结量化输入上的输出，隔离局部重建误差。两者回答不同问题。
- `block=None`：每个候选均实际执行整网；目标为整网输出/分类logits。此模式允许少样本教师KL、任务CE、真实校准Top-1 error与明确定义的平方梯度代理；它们是校准拟合目标，不能冒充独立准确率。
- `context=quantized`保持当前完整冻结背景；`float`在候选评分期间关闭干预集合之外的量化器，可检验FP背景选参是否迁移到Q背景。local_fp只配quantized背景。此背景变化仅用于拟合，选中模型恢复完整Q状态后另行外层评价。
- `count`只消费当前calibration视图的前count个样本；相同候选读取同一实际样本序列，不静默缩样。缓存超过`cache_mib`明确报错。可用调用方预定的小视图控制成本，不要求枚举全部组合。

| metric | 实际计算与适用范围 |
|---|---|
| mse | 每样本选定张量元素MSE，再按实际样本平均；block/整网 |
| cosine | 每样本选定张量展平后1-cosine；零范数回退MSE；block/整网。原生coordinate仍为末维cosine均值的负数，两种数值不直接混排 |
| teacher_kl | KL(原FP softmax logits ∥ 当前候选softmax logits)，温度1，样本平均；仅整网分类logits |
| task_loss | 对实际校准标签的分类cross entropy；仅整网，缺标签明确拒绝 |
| top1_error | 实际校准标签argmax错误比例，离散且少样本会平分；仅整网 |
| hessian_diag_ce | 在**初始冻结模型logits**上由真实标签CE自动微分得到grad，取mean((grad*(candidate-reference))²)，再样本平均。只是logit对角平方梯度代理，非完整Hessian，也不是原生位点梯度的同一目标 |

精确平分保留声明顺序第一项。少样本任务指标噪声、FP/Q背景和CLS/patch冲突应成为后续判别对照的依据，不能预设赢家。内层拟合、外层search候选选择、最终双任务L3各自保留数据与用途；ADE迁移只重载选中冻结包，不在ADE重新拟合。选中后仍需按已有executor导出/重载审核；将返回的selection加入`executor.report["custom_selection"]`，并从选中模型收集当前weight_record/params导出，避免拿初始拟合返回值导出旧参数。API本身不使一个诊断模型自动成为正式方案。
