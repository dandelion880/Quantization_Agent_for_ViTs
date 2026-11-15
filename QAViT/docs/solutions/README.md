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

## 独立保底 B 与冻结历史加载

B 是既有 incumbent 和同协议实测方案库中的保留结果：沿用 mIoU、再 Top-1 的原排序，失败、未完成探索和较低的新结果均不清除有效 B。B 不占 H1/H2/H3 槽位、不参加下一笔科研投入的预选，也不要求为了保底重复 L3。没有有效完整双任务成绩时，B 明确为 pending；待测种子不能充当已验证 B。`fallback_B` 给出保留模型、确切证据及历史/本轮来源；历史成绩不会转成本轮新增测量。

实测 Top-3 仍保持原存储口径，三个待测完整种子作为独立目录持续可发现，避免被实测项挤掉。H1/H2 可从任意合适的完整方案出发；方案成熟度不自动证明改良候选有效。未完成探索及反证仍存于不可变卡片和任务，不受 Top-3 容量裁剪。

跨运行冻结加载在实验请求顶层提交 `solution_refs: ["<complete_solutions 中 evaluated 的 id>"]`，自定义 `experiment_v1` 中调用 `context['load_frozen'](solution_id)`。宿主仅从本运行已提交的方案库快照派生路径，核对原 Evidence、原模型包与批准代码哈希、源版本及同协议身份；待测种子、任意路径及缺失来源均拒绝。`source_snapshot` 在本运行的 `solution_library/source/<source_version>/` 物化为隔离数值包，加载原版本 backend/executor 并执行其原身份检查，不伪装当前源码身份。自定义 executor 的 `ptq_agent` imports 定向至该隔离包，批准源码本身不改写；不支持的无别名 dotted import 明确拒绝。加载只调用 `load`，禁读数据 batches，绝不调用 `prepare` 或重新校准。返回回执区分 `measurement_origin=historical` 与本次后续新实验；读取旧成绩本身不是新增实测。原包与旧运行保持只读。

旧库项若缺少 `source_snapshot`，宿主按原运行 Journal 中该 Experiment 的确切提交时点，选择当时生效的初始源码快照或 `implementation_intervention` 快照；验证原提交链及已声明 source_version 一致后，只补充新任务绑定，不改旧库、旧卡或源文件。无法从原日志唯一确定时拒绝。源码缓存由宿主冻结请求时预先物化；数值 worker 在 Landlock 内仅核验和读取，不能自行在作业外创建来源文件。
