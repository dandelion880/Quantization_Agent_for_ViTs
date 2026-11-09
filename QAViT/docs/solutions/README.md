# 方法组件、完整方案种子与协议 Top-3

| 层次 | 权威位置 | 含义与状态 |
|---|---|---|
| 论文方法组件 | [methods/index.json](../methods/index.json) 与独立方法文档 | 定义一个量化/校准机制及其适用条件，不等于整网配方 |
| 可执行方案种子 | [seeds.json](seeds.json)、[GPTAQ](gptaq.md)、[AdaLog](adalog.md)、[RepQ](repq.md)；[baselines.py](../../tools/ptq_agent/execution/baselines.py) 的 build_baseline | 三个本地完整 static W4A4 适配起点；有来源和构造器，初始 status=pending_evaluation、top1/mIoU=null，尚非当前协议实测前三名 |
| 原论文配方参考 | [recipes.py](../../tools/ptq_agent/execution/recipes.py) 的 build_recipe | PTQ4ViT/RepQ/FQ-ViT 的另一组配置构造器，有特定结构/位宽限制；生产流程未自动调用，不替代当前严格协议种子 |
| 协议方案库 / 实测 Top-3 | [core/solutions.py](../../core/solutions.py)；研究运行 EXP/solutions/<protocol_hash>/top3.json | 同协议完整方案的实测复用库；初始可含待测种子占位，必须看每项 status/metrics，不能只凭 Top-3 名称认定已验证 |

方案库与 Recorder/Memory 分开。论文报告、局部代理改善或其它协议的结果不能填入当前实测字段；种子中的官方来源只说明算法依据。维护种子时同步构造器、配置、方法文档与相关测试，不更改历史测量。

初始化从静态种子或已有共享协议桶冻结配置和源码来源。非研究运行隔离在自己的 solutions/；各运行 Journal 保留当时快照，solution_library/<id_hash>.json 物化完整配置及可复用源码。generate/evolve/code/safeguard/record/plan_l3 获得 context.complete_solutions 短目录，通过 configuration_asset 用 read_asset 取完整内容；其它任务可先读本说明或静态种子。角色与 purpose 的实际映射以 agents/tasks.py、agents/interaction.py 为准。

每轮最终完整方案的有效真实 L3 双任务结果才有资格更新共享库，按 ADE20K mIoU、再 ImageNet Top-1 排序，平分保留旧项；已评测项先于待测种子。协议键包含模型、evaluation_protocol、quantization_contract、hard_constraints；不合并异质协议。写入加锁、原子落盘，旧方案在 archive 及原运行保留。未晋升、局部与负结果仍沿 Journal/Recorder 保存，不影响辩论裁决或限制 H3。

统一发现与维护入口见 [knowledge/INDEX.md](../../knowledge/INDEX.md)，运行内外边界见 [资源生命周期](../RESOURCE_LIFECYCLE.md)。
