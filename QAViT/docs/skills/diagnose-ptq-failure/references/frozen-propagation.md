# 从完整冻结方案定位可改善的误差传播

适用于连续方案都围绕同一量化边、局部指标和双任务收益相悖，或尚不清楚校准应在哪种上下文进行的情况。不是每轮必做的全网扫描。可从少量代表性边界入手，再用一个有判别力的干预缩小不确定性。

## 现有覆盖与新增执行入口

`profile_sensitivity` 使用简化 RTN/QDQ；`ActivationDiagnostics` 只看量化器输入误差。`profile_trial_diagnostics` / `profile_trial_ablation` 通过旧 trial 索引读取当前运行 search 测量，不能据名称推断它们已经覆盖历史完整方案、FP/Q block 边界或所有 task head。

完整方案可直接使用 `frozen_helper/profile_frozen_propagation`，调用参数见 [统一能力目录](../../../CAPABILITIES.md)；需要扩展时仍可用原 `custom_experiment / experiment_v1`。请求在顶层声明确切 `evidence_refs` 或 `solution_refs`；只提及路径、B、Top3 或 controls 中的自然语言名称不构成来源。历史方案即使退出 Top3，仍应以目录中的确切 solution id 发现并加载。`context['load_frozen'](ref)` 由宿主核验包、批准源码和协议身份，只 load、不 prepare。失败先修请求/执行，不能把缺测当反证。

可调用 `ptq_agent.execution.propagation.profile_frozen_propagation`，公开源码可由 Coding/Safeguard 的接口目录读取。示意代码（ref 来自请求参数且已在顶层引用，data 是宿主限定的数据视图）：

```python
from ptq_agent.execution.propagation import profile_frozen_propagation

def execute(request, context):
    p = context['parameters']
    model, receipt = context['load_frozen'](p['source_ref'])
    result = profile_frozen_propagation(
        context['backend'], model, context['data'],
        blocks=p['blocks'], split=p.get('split', 'calibration'),
        interventions=p.get('interventions', []),
        classification_output='output',
        source={k: receipt.get(k) for k in ('evidence_ref', 'solution_id', 'model_ref')})
    return {'observations': {'propagation': result}}
```

所选 block 路径来自本次模型目录。已绑定 block 的分组使用 profile 的 CLS、special 和 patch 范围，显式布局需与实测相符。工具保存各次位置参数、关键字参数和嵌套 RoPE，在相同的单流边界重放。多流输出或不同残差坐标需要相应适配。任务输出按其原有字段比较。

干预可选 `{'id':'b5_fp','kind':'fp_block','block':'blocks.5'}`，或 `{'id':'q0_bypass','kind':'bypass','site':确切ptq_site_id,'target':'activation'}`（target 也支持 weight）。旁路精确量化边或把当前 Q 输入交给 FP 块，其余模型数值状态冻结，不重校准；每臂结束恢复开关和 hook。模块级前处理若改变边界坐标，先确认 FP/Q 残差边界仍可比。

## 数字回答的四个问题

对每个样本、块，`x_FP` 与 `x_Q` 来自对应完整前缀；`F` 是原 FP 块，`Q` 是当前冻结方案块：

| 观测 | 实际计算 | 可回答的问题 |
|---|---|---|
| input_total | x_Q 对 x_FP 的误差 | 块前已累积多少偏差？ |
| local_added | Q(x_Q) 对 F(x_Q) | 相同实际 Q 输入下，本块新引入多少失真？ |
| upstream_mapped | F(x_Q) 对 F(x_FP) | 即使本块无量化，前缀偏差经过此块如何变化？ |
| output_total | Q(x_Q) 对 F(x_FP) | 当前完整背景下出口总偏差是多少？ |
| intervention error_to_fp / change_from_frozen | 干预臂对 FP / 当前 Q 的边界变化 | 对下游传播和任务表征的边际影响是什么？ |

误差平方不可简单相减当作贡献；`decomposition_cross_term.twice_dot` 保存 `2·<local, upstream>`。除以对应 elements 后，total MSE = local MSE + upstream MSE + cross MSE。负交叉项表示抵消，不能丢弃。各比较的 relative MSE 分母是自己的 reference 能量，不能直接相加；跨层还需参考尺度与 token 类别。

`baseline_output_error` 和每臂 `output_error` 保留命名输出，避免把 CLS 分类语义与 patch/dense 特征混成一个分数。`classification_output='output'` 仅在确认是 rank-2 logits 且有真实标签时指定：报告 KL(teacher FP || Q)、标签交叉熵、正确样本数和 Top1。少样本 Top1 离散且高方差；不指定便只报表征误差。若研究分割表征，Coding 可传入 `forward_fn(model, inputs)` 执行明确的冻结 head 并返回命名输出；本 API 不自动运行 ADE、不重新拟合 neck/head，不把 backbone patch 误差冒充 mIoU。正式双任务评价仍走原冻结迁移协议。

## 从轮廓到研究选择

- 大 output_total、小 local_added：优先核实上游输入失配或传播，避免只调当前层局部 MSE。
- 大 local_added，但冻结干预对 logits/patch 末端影响小：保留抑制/冗余解释，可能不值得继续局部选参。
- 干预改善 CLS 却恶化 patch，或 KL 改善而任务不变：保留任务冲突，考虑参考与校准目标是否偏向单一表征。不能自动认定某 metric 优于其它指标。
- FP 块替换可测量当前背景的恢复空间，但不说明可由合法 W4A4 实现。再用可行的冻结 scale/码点候选对照检验可改善性。
- 校准上下文可据结果比较 FP 输入/目标、Q 输入加条件 FP 目标，或原 FP 输出重建；也可比较 block 目标与少样本整网 KL/任务 loss。内部拟合、外层候选比较与最终双任务测试的样本/目的分开记录，不使用 final 选参。

优先选择能改变下一步决策的对照；局部负结果、任务收益不足和工程未执行分别保存。现有证据已足够时，Generation 可直接设计 H1/H2 改良或 H3 新机制，不需要额外诊断门禁。
