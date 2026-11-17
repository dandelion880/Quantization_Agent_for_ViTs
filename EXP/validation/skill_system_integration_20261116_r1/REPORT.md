# 七项科研 Skill 接入与验证报告

2026-11-16：已完成最小接入和本轮验证。沿用现有 G-only、索引/read_asset、来源快照及实验闭环，没有新增角色、固定阶段、必读资料、实验配额或宿主门禁；本轮暂缓自主 Skill 沉淀。没有启动长期正式实验，也没有真实任务收益或 Skill 科研效果提升的结论。机器可读结果见 [RESULTS.json](RESULTS.json)。

已读取项目 AGENTS、实际源码，以及包内 START_HERE、CODEX_TASK 和 SOURCE_COMPATIBILITY_NOTES；包内快照描述与当前行为逐项核对。输入包为 135,583 bytes，SHA256 为 `e5d6871fd3025788c05bb124330c0ff01b840d4019b23cf151f0ec224cf17da1`。41 个 manifest 条目全部验证，解压原件保留在 [source_package](source_package/qavit_skill_system_20261116/START_HERE.md)。直接安装映射、原件/安装件哈希及派生说明见 [integration_map.json](integration_map.json)。外部论文和 Library 原件未在本轮重新取得，相关数字仍是包作者报告的来源材料，不是本轮复现。

实际变更如下。

| 范围 | 结果 |
| --- | --- |
| [Skill 入口](../../../QAViT/docs/skills/INDEX.md) | 七项主文、16 份包内专项参考；原 `frozen-propagation.md` 字节不变并保留可发现链接。`evaluate-ptq-candidate` 替换 `assess-evidence-transfer`，不保留重复同义入口。旧原件在 Git/既有运行快照及 `before_skills/` 备份中。 |
| [共享资料](../../../QAViT/knowledge/research-skills/SOURCES.md) | 共同约定、PTQ4DINO 与五篇论文路线；适配相对路径，明确原包作者来源和验证层级。 |
| 提示词与文档 | 更新 common、Generation、Recorder、plan_l3、统一导航、接口文档、README/STRUCTURE 和[当前伪代码](PSEUDOCODE.md)。清除旧 H1 可原样保留、起点偏向 Top-3、四项 Skill 计数及合成证据混称 L0 的入口表述。H1/H2 完整方案改良、H3 开放创新、独立 B 与冻结迁移保持原职责。 |
| [离线评分包](../../../QAViT/configs/evaluation/gonly_v0_2/README.md) | Recorder/Select 标记内容修订 `r20261116`，修正 Recorder 沿用 v0_1 身份及“比赛/结案”旧术语。S1–S4 改为证据解释、投入价值、版本事实、保底与历史的不同锚点/维度；正确保留现版本也可高分，不要求制造修订或实验。保留 rules-version 入口 `gonly_v0_2`，刷新 provenance 哈希；v0_1 与既有冻结成绩不动。 |
| 离线验证材料 | 包内 CASES/REVIEW_GUIDE/TRYOUT_NOTES 原样放在 tests/skill_evals/seven_skills_20261116，排除在研究者知识读取 ACL 之外。回放、日志、来源原件和结果均位于本次 EXP。 |

研究参考与最终方法现在使用既有字段表达：`training=false` 或 `training_allowed=false` 仍是全程禁训合同；不能靠 Skill 或“仅参考”标签绕过。明确允许参考优化的 Brief 可通过现有 `custom_experiment/experiment_v1` 和 Coding→Safeguard→执行→Evidence 链声明 `resolved_choices.training=true`，在隔离副本研究受限 STE。最终方法单独声明 false，从参考优化前状态和许可 calibration 数据重新生成参数、组成员及选择，不继承 oracle 的实例答案。`training_allowed=true` 表示许可；`hard_constraints.training=true` 会要求每次执行为 true，不能误当可选许可。没有改写已有 Brief，也没有新增 STE 专用 API。

[当前兼容说明](../../../QAViT/docs/SCIENTIFIC_SKILLS_COMPATIBILITY.md) 核实了实际边界：block 调度仍可逐位点拟合；frozen scale search 每 site/target 接受正标量，W 重编码限 RTN/uniform；`hessian_diag_ce` 是初始 logits 的 CE 梯度平方；float 背景与原生 calibration_context 含义不同；CLS/patch 切分不是 DINO 专用分组；通用 Memory 仍以当前 Journal 为范围。任意逐元素组倍率、zero 直接搜索、lookahead 和独立特殊 token 分组仍需原 Coding 链明确实现，不能仅改文案声称接通。

确认并修复了一个实际状态 bug：[calibration_search.py](../../../QAViT/tools/ptq_agent/execution/calibration_search.py) 原先基线哈希包含全部量化位点，选中状态哈希却只含干预位点。因此 identity 候选倍率 1 在输出完全相同时仍报告不同状态哈希。先以新回归得到[修复前失败](state_hash_before_fix.log)，再让两边都哈希完整状态。测试同时检查 identity 输出/哈希相同，以及倍率 0.8 的状态哈希改变。只改状态摘要范围，候选、评分和数值选择未改；这是本轮唯一生产 Python 实现改动。

验证证据分层如下，不能相互替代。

| 层级 | 实际结果与证据 | 不支持的推断 |
| --- | --- | --- |
| 内容、可发现性、来源 | 7 份 Skill 格式校验通过；导航审计覆盖 7 Skill、21 Method、76 页；41 个包 manifest 条目、36 个活动 Markdown 的源码快照/哈希、96 个本地链接/锚点全部通过。见 [asset_checks.json](asset_checks.json)、[skill_validation.json](skill_validation.json)。 | 不证明研究行为或算法收益。 |
| 工程回归 | 262 个不同用例分次最终通过。主回归 260 项中首次 259 通过、1 项订阅锁竞争失败；该项随后两次单独通过。另 2 项参考权限/CPU 实验链测试通过。模块及去重用例见 [test_manifest.json](test_manifest.json)，各次结果见 [test_summary.json](test_summary.json)。覆盖 H1/H2/H3、B、修订/来源恢复、原实验链、Recorder 时点、冻结迁移合同、读取权限、快照、尺度搜索和传播。 | 不等于一次 262 项全绿，也不是完整仓库所有测试或真实 DeiT/ADE 验证。 |
| 参考权限与数值执行 | 允许条件下实际 CPU worker 对单一 scale 做一次 STE/SGD，记录 optimizer_steps=1、源模型不变；独立 forward 从同一优化前状态重建，optimizer_steps=0、无 frozen oracle 加载，返回初始 scale 而非参考优化后的值；Evidence 回到原研究者。两种全程禁止字段均拒绝参考标签覆盖。见 [参考链日志](reference_optimization_tests.log)。Coding/Safeguard 提交是测试脚本，数值 worker 是真实执行，measurement_source=synthetic_computation。 | 不是全套 PTQ4DINO；未数值验证任意分组独立性，更不是通用语义证明器。 |
| 真实 LLM 文本行为 | 4 个案例完成，共 7 次成功模型响应、6 次 read_asset；实际输出/工具回执和人工核读见 [behavior_review.json](behavior_review.json)。 | 无旧/新对照、多次采样或统计效果结论，不证明七项 Skill 都已逐项产生收益。 |
| 科研效果 | 本轮未开展真实 DeiT/DINO/ADE、ImageNet/ADE20K 正式实验，未测量算法效果。 | 不宣称精度、泛化、效率或研究质量提升。 |

失败记录完整保留。初次 39 项集成检查中的新参考测试漏写预计时长，默认 10,800 秒超过测试 Brief 的 7,200 秒预算，尚未进入 Safeguard；仅为 fixture 补 `estimated_seconds=30` 后两项参考测试通过，未放宽生产预算。主回归的唯一失败是 mock preflight 用例与同时运行的真实回放争用项目订阅锁，导致模型调用计数为 0；[隔离复现](subscription_lock_reproduction.json) 证实等待超时前 preflight/model 均未调用，[无回放争抢时复跑](preflight_after_replays.log) 通过，无需修改生产调度。修复前状态哈希红测、fixture 失败和争锁失败均保留原日志，没有抹去或改写为成功。

真实回放使用生产 Workflow、原订阅传输和正常 H3 提交结构，CLI 为 0.169.0，请求配置为 `gpt-6-astra/high`。CLI 没有回报实际模型/推理档位，审计保留 null，不能把请求配置当服务端身份验证。每例是人工构造情境，沿 tiny validation Brief 的 calibration=2/search=4 配置，只提交待检验提案；配置样本数不是实测数据。评阅要点未进入被测上下文，无额外模型评委；由本次接入执行者核读，不是独立盲评。

| 案例 | 自主读取 | 可观察行为 |
| --- | --- | --- |
| [C01](C01_replay.json) | learn-from-reference-solution、design-scale-calibration | 拒绝把训练后成员继承包装成 training-free；保留获准参考研究，提出从初态前向响应生成不等长组及独立性对照，说明需 custom 实现。 |
| [C04](C04_replay.json) | adapt-quantization-method | 从题设推导 Q/K 协变尺度下的量化不变性，撤下零效应机制；未先要求全层 profiling，后续旋转只是待验证替代。 |
| [C07](C07_replay.json) | 零 Skill | 遵守冻结禁训合同，拒绝以 Skill 建议豁免 optimizer；仍给出冻结前向研究选择，没有把未知历史 checkpoint 当已取得证据。 |
| [C09 重试](C09_retry1_replay.json) | design-scale-calibration、calibration_order、scale_selection | 区分 block 调度和联合重建，按实际多位点标量候选 API 设计二维交互比较，明确未支持的逐元素/zero/DINO 组功能。 |

C09 首次在模型启动前发生 CLI `account/read` RPC -32603 错误，[原失败](C09_replay.json) 与原总汇保留；使用新目录 `live/C09_retry1` 单独重试成功，没有覆盖先前尝试。每例最多 480 秒/6 个审计记录是本次回放的成本边界，不是新增生产门槛。四例数值任务和 Evidence 均为 0；它们只支持本次观察到的决策正确性。只触发了三种 Skill 正文；Recorder/select 的真实 LLM 评分行为及其余 Skill 尚未逐项回放。

源身份从 `13b263b30a72e2ca5d12fa04d6f50e1f26e3af8e91713f1be5fa77c6630c46a6` 变为 `81bfde84fbb11ffcedbd27c3627aa92eaff752373a6afb37ed73c8d6d2291058`，四例成功回放均使用后一身份，回放期间未修改 QAViT 核心。旧运行继续保有旧快照，本轮不回写历史测量。AGENTS、reference、v0_1 评分包及历史 EXP/runs 无跟踪文件改动；没有改调度、核心 loader 或来源校验实现。完整 delta 和快照在本次 EXP 本地保留，Git 仅开放轻量交付文件。

剩余缺口包括真实模型/下游 head 的数值验证、上述尚无内置 API 的研究建议，以及更广泛的多次真实 LLM 行为评价。现有 `gradient_audit`、源状态哈希和 frozen-load 回执能支持审查，但 `optimizer_steps=0` 不能证明不存在手写更新或隐藏 oracle 输入；独立最终算法仍需结合源码、来源与执行记录判断。本轮没有为了补齐这些缺口启动长期实验或新增自动语义门禁。

可在项目根目录复核内容与相关测试：

```bash
python -B QAViT/tests/audit_knowledge.py
python -B EXP/validation/skill_system_integration_20261116_r1/verify_assets.py
PYTHONPATH=QAViT python -B -m unittest tests.test_scientific_skills tests.test_calibration_search tests.test_reference_optimization tests.evaluation.test_gonly_rules
```

真实回放入口是 [replay.py](replay.py)，可指定 `--case C09 --attempt new_review_01`；已有目录会拒绝覆盖。它会产生新的订阅调用，不属于上述离线检查。原包、原 Skill/rubric 备份、失败日志和完整 live 快照保存在本地 EXP，没有把评阅答案加入生产知识。

原始 ZIP 已按用户要求删除。删除前再次校验路径、大小及 SHA256，删除后确认不存在；解压原件与来源记录保留。详情见 [archive_cleanup.json](archive_cleanup.json)。
