# 可执行能力短目录

统一入口：`request_experiment({question, measurements, execution, ...})`。以下均为已有实现，直接返回原任务的 Evidence 或明确失败，不需 Coding 包装；不设实验次数门禁。科学正文仍只从 `knowledge/INDEX.md` → `docs/skills/INDEX.md` 按需读取；本页只描述调用合同，不复制科研方法。

| 用途 / service.tool | 输入 | 输出、范围与状态影响 |
|---|---|---|
| 常规观察 / `profile.<tool>`（L1） | `arguments` 指定模块/位点、统计及 count；工具名见 `context.services.profile.tools`，参数见 `EXPERIMENT_INPUTS.md` | 结构、分布、硬件或量化诊断统计与证据资产；校准样本测量和纯目录读取分别记账；不产生正式双任务成绩。候选 refit/ablation 的具体语义按工具区分。 |
| 冻结传播 / `frozen_helper.profile_frozen_propagation`（L1/L2） | 明确冻结来源；`arguments={blocks, split?, count, offset?, interventions?, token_layout?, classification_output?}` | 配对 FP/Q 边界误差、local/upstream/cross、干预及可选分类诊断；仅 calibration/search，默认 calibration；模型数值与量化开关保持不变，不重校准。支持坐标对齐的单图像流 block，可重放原位置参数、关键字参数及嵌套 RoPE；绑定布局可分 CLS/special/patch。分类诊断需指定实际 logits 输出。 |
| 离散尺度搜索 / `frozen_helper.search_frozen_scales`（L1/L2） | 明确冻结来源；`arguments={candidates, count, offset?, metric?, block?, reference?, context?, token_scope?, cache_mib?}` | 全部候选分数、选中项、参数与状态哈希；只在 calibration 选参，选中尺度安装到本作业加载实例，来源包不变，不默认导出新包。支持已有静态原生量化器接口；integer compound 需显式重绑定，仍走 custom。 |

冻结来源用顶层 `evidence_refs:["EvidenceCard-…@1"]` 或 `solution_refs:["完整方案ID"]`；多个包用 `execution.source_ref` 精确指定。完整调用示例按需 `read_asset("QAViT/docs/CAPABILITIES.md")`；常规 profile 参数按需 `EXPERIMENT_INPUTS.md`。以上单设备随 CPU/CUDA backend 执行，不宣称原生低比特加速。

<!-- CALL_DETAILS -->

宿主验证资产与原实现版本，原 `controls` 引用兼容。source_ref 可用 Evidence/solution/model ID，唯一包可省略。禁止路径猜测或重校准冒充加载。

例如冻结传播：
```json
{"question":"第一块误差如何传播？","measurements":["boundaries","decomposition_cross_term"],"evidence_refs":["EvidenceCard-…@1"],"execution":{"service":"frozen_helper","tool":"profile_frozen_propagation","arguments":{"blocks":["blocks.0","blocks.1"],"split":"search","count":4}}}
```
尺度搜索使用相同请求结构，将 tool 换为 `search_frozen_scales`，arguments 例如 `{"count":4,"metric":"mse","block":"blocks.0","candidates":[{"id":"identity","scales":{"blocks.0.attn.qkv":{"activation":1.0}}},{"id":"a08","scales":{"blocks.0.attn.qkv":{"activation":0.8}}}]}`。count 也可放 `execution.calibration.count`（传播 search 则 `execution.evaluation={split:"search",count:4}`）；镜像冲突报错，null 明确使用完整该视图，不静默缩样本。校准 count 省略时仅继承已声明协议数目。

研究者仍可在 `custom_experiment/experiment_v1` 内通过 `context.load_frozen` 调用相同 helper；需要自定义任务头/forward 或导出选中包时沿此已有路径。直接请求与函数调用复用同一数值实现。搜索 Evidence 是校准内选择，后续导出、重载和完整双任务 L3 仍需现有实验链，不能把选参分数当正式成绩。

CPU/CUDA 随授权 backend 实际设备执行；以上诊断/搜索均单设备，不声称 INT4 原生 kernel 或低比特加速。`profile.query_execution_capabilities` 查询实际设备；完整 L3 的候选间并行、同一冻结包双任务并行及纯分类分片由宿主按支持能力和空闲卡分配。单卡可串行双任务，缺卡正常排队。

按需接口：`EXPERIMENT_INPUTS.md`、`CODE_INTERFACES.md`；科学依据：`skills/diagnose-ptq-failure/references/frozen-propagation.md`、`methods/scale_selection.md`。旧引用和直接 Python helper 保持兼容。

原生 static uniform_affine 可为 Linear 输入、Q、转置 K、V 使用固定 token 区间。参数、适用布局和尾部长度规则见 methods/scale_selection.md。
