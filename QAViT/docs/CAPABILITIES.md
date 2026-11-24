# 可执行能力短目录

统一入口：`request_experiment({question, measurements, execution, ...})`。以下均为已有实现，直接返回原任务的 Evidence 或明确失败，不需 Coding 包装；不设实验次数门禁。科学正文仍只从 `knowledge/INDEX.md` → `docs/skills/INDEX.md` 按需读取；本页只描述调用合同，不复制科研方法。

| 用途 / service.tool | 输入 | 输出、范围与状态影响 |
|---|---|---|
| 常规观察 / `profile.<tool>`（L1） | `arguments` 指定模块/位点、统计及 count；工具名见 `context.services.profile.tools`，参数见 `EXPERIMENT_INPUTS.md` | 结构、分布、硬件或量化诊断统计与证据资产；校准样本测量和纯目录读取分别记账；不产生正式双任务成绩。候选 refit/ablation 的具体语义按工具区分。 |
| 冻结传播 / `frozen_helper.profile_frozen_propagation`（L1/L2） | 明确冻结来源；`arguments={blocks, split?, count, offset?, interventions?, token_layout?, classification_output?, view?, output?}` | 配对 FP/Q 边界误差、local/upstream/cross、干预及可选分类诊断；仅 calibration/search，默认 calibration；模型数值与量化开关保持不变，不重校准。支持坐标对齐的单图像流 block，可重放原位置参数、关键字参数及嵌套 RoPE；绑定布局可分 CLS/special/patch。分类诊断需指定实际 logits 输出。 |
| 离散尺度搜索 / `frozen_helper.search_frozen_scales`（L1/L2） | 明确冻结来源；`arguments={candidates, count, offset?, metric?, block?, reference?, context?, token_scope?, cache_mib?}` | 全部候选分数、选中项、参数与状态哈希；只在 calibration 选参，默认只评分并返回建议及生效参数的摘要。支持已有静态原生量化器接口；integer compound 需显式重绑定，仍走 custom。 |

| 双任务候选评分 / `frozen_helper.score_frozen_candidates`（L1/L2） | 同一父包、`candidates, observations_spec, objectives?` 及许可视图 | 分类 logits、指定 block、冻结分割头多层 patch 或窗口 logits；返回逐图/层/窗口误差、候选收据及各目标提名。 |
| 提交增量 / `quantize_evaluate`（L2/L3） | `parent_ref, accept_trial, evaluations` 和对应来源引用 | 从确切完整包安装被点名的试评，导出并独立冷载子包，返回父子状态及实际修改。 |

冻结来源用顶层 `evidence_refs:["EvidenceCard-…@1"]` 或 `solution_refs:["完整方案ID"]`；多个包用 `execution.source_ref` 精确指定。完整调用示例按需 `read_asset("QAViT/docs/CAPABILITIES.md")`；常规 profile 参数按需 `EXPERIMENT_INPUTS.md`。以上单设备随 CPU/CUDA backend 执行，不宣称原生低比特加速。

<!-- CALL_DETAILS -->

宿主验证资产与原实现版本，原 `controls` 引用兼容。source_ref 可用 Evidence/solution/model ID，唯一包可省略。禁止路径猜测或重校准冒充加载。

例如冻结传播：
```json
{"question":"第一块误差如何传播？","measurements":["boundaries","decomposition_cross_term"],"evidence_refs":["EvidenceCard-…@1"],"execution":{"service":"frozen_helper","tool":"profile_frozen_propagation","arguments":{"blocks":["blocks.0","blocks.1"],"split":"search","count":4}}}
```
尺度搜索使用相同请求结构，将 tool 换为 `search_frozen_scales`，arguments 例如 `{"count":4,"metric":"mse","block":"blocks.0","candidates":[{"id":"identity","scales":{"blocks.0.attn.qkv":{"activation":1.0}}},{"id":"a08","scales":{"blocks.0.attn.qkv":{"activation":0.8}}}]}`。count 也可放 `execution.calibration.count`（传播 search 则 `execution.evaluation={split:"search",count:4}`）；镜像冲突报错，null 明确使用完整该视图，不静默缩样本。校准 count 省略时仅继承已声明协议数目。

研究者仍可在 `custom_experiment/experiment_v1` 内通过 `context.load_frozen` 调用相同 helper；自定义任务头或 forward 可沿此已有路径。直接请求与函数调用复用同一数值实现。搜索 Evidence 是校准内选择，后续导出、重载和完整双任务 L3 仍需现有实验链，不能把选参分数当正式成绩。

CPU/CUDA 随授权 backend 实际设备执行；以上诊断/搜索均单设备，不声称 INT4 原生 kernel 或低比特加速。`profile.query_execution_capabilities` 查询实际设备；完整 L3 的候选间并行、同一冻结包双任务并行及纯分类分片由宿主按支持能力和空闲卡分配。单卡可串行双任务，缺卡正常排队。

按需接口：`EXPERIMENT_INPUTS.md`、`CODE_INTERFACES.md`；科学依据：`skills/diagnose-ptq-failure/references/frozen-propagation.md`、`methods/scale_selection.md`。原有单目标 Python helper 的显式安装参数为 `install_selected=True`。

原生 static uniform_affine 可为 Linear 输入、Q、转置 K、V 使用固定 token 区间。参数、适用布局和尾部长度规则见 methods/scale_selection.md。

### 在同一父包比较选择器

`arguments.candidates` 每项含 `id` 与 `scales`。空对象代表原状态。每个观察指定唯一 id、view、output、metrics。例如在绑定目录中确认位点名后提交下列 arguments。

```json
{"count":4,"candidates":[{"id":"parent","scales":{}},{"id":"p08","scales":{"blocks.8.attn::post_softmax":{"activation":0.8}}}],"observations_spec":[{"id":"cls","view":"classification","output":"classification_logits","metrics":["teacher_kl","mse"]},{"id":"transfer","view":"dense","output":"head_features","metrics":["nmse"]}],"objectives":{"class_teacher":{"cls.teacher_kl":1},"dense_features":{"transfer.nmse":1}}}
```

`block` 观察可指定下游模块、`tokens=all|cls|special|patch` 和 `reference=fp_model|local_fp`。后者计算完整父模型在该窗口传入的输入经原 FP block 的输出。候选始终走实际冻结前后缀。`head_features` 默认消费任务绑定的全部层，也可显式给出索引 `layers` 及按层号字符串键控的 `layer_weights`。分割 logits 使用同一冻结头的窗口输出。每个指标先在指定张量、窗口内归约，再按声明的层权、图内窗口均权及原图均权聚合。组合不同量纲时研究者决定归一化或权重。显式 objectives 仅用于 calibration，同分按候选 id 排序。确认阶段用独立 search 切片、省略 objectives、提交选出的不同候选；任务反馈由 evaluator 在独立视图采集。

原图经各自绑定的分类与分割预处理生成两种输入。窗口坐标、输入 SHA256、输出形状和样本身份随 Evidence 保存。两个几何视图来自同一原图；跨形状 observer 拟合及 ADE 数据域收益各需另外实测。

接受时在 `experiment.evidence_refs` 引用评分 Evidence，同时以 evidence_refs 或 solution_refs 引用父包。执行片段如下。

```json
{"service":"quantize_evaluate","training":false,"parent_ref":"父包的 model_ref","accept_trial":{"evidence_ref":"评分 Evidence 的完整引用","candidate_id":"p08"},"evaluations":[{"kind":"classification","split":"search","count":4}]}
```

下一轮可引用子包 Evidence 和其 model_ref。正式 L3 沿冻结 Brief 的全部审计和双任务评价执行。自定义格式可由获准的 `executor_v1` 提供确切状态的导出与加载。使用完整新配方时另记重建状态，并在该状态重做选择。
