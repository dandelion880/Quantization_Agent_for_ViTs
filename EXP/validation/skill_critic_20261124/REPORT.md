# QAViT 流程 Skill、Recorder 与批量 Critic

2026-11-24。开发分三阶段。验证覆盖真实 DINOv3-B 历史决策断点、真实订阅模型调用、一项新的 CPU 数值探针及宿主回归。没有开展完整研究运行或官方终评。

## 提交与迁移

| 阶段 | 引用 | 可核对的变化 |
|---|---|---|
| gonly 起点 | `264e16b806836c6c37681b3970772851c01a66d6` | 读取实际代码、冻结试验最新报告、配置身份与恢复机制 |
| A | `b4759c8aab13c7ca5ea5da1dacf5dd9293901fa1` | 调整既有研究 Skill、工程 references、冻结网格能力说明及来源账本 |
| B | `31995c496f75d9817faa1f370de2ce8fbce9fbd2`，`ablation/gonly`，`gonly-skills-20261124` | Recorder 发布、统一发现与版本化读取；保留 gonly 对照 |
| 旧 Full | `1873ec79f8a6ccbd2d49806b8d5e058fc2b47f2f`，`full-retired-20261124` | 归档原 main 本地尖端，含当时未同步的提交 |
| 迁移 | `8c4a8cb8e6cff5c25c4ee75f637c2fd92119bcf1`，普通双父合并 | 迁移点的 `QAViT/` 与 B 逐文件相同。main 独有订阅回执另存为 CLI 171/172 验证目录下的 `full_compatibility.json`；历史 Full 卡片、schema、离线评价及回归材料仍可恢复 |
| C | `b2fad28b9cfbad1322dd72b413a1a5fddb540687`，main | 批量 Critic、身份绑定、开发反馈、费用与失败退化；后继交付提交保存本报告和可复核产物 |

## 功能接入

**人工科研资料。** 审计既有七个完整包、案例、工程资料、角色与实际执行能力后，沿原触发条件更新现有条目。K-Dense 的实验设计、统计分析落在 `experiment-patterns` 和 `ranking-validation`，覆盖原图配对、多视图相关性、替代解释、小样本与选参。Orchestra 与 Anthropic 的结构类比、问题重构落实在 `intervention-design`。ARIS 与 Shinka 的结果到结论、失败条件与原件定位进入 `evidence-and-comparison`。Superpowers 和 NVIDIA 的调试、验证、测时落在 `knowledge/coding/{reuse,debug,verify}.md`。现有 frozen scale 接口说明覆盖独立 grid 的局部 indices、倍率和合法整数 zero，分组边界由父包固定。Agent Skills 用于包格式，EvoSkill 用于从真实经历提议修订。源码使用本项目表述；每个具体文件、固定 commit、许可和适配落点集中保存在[来源账本](../../../QAViT/knowledge/research-skills/SOURCES.md)。

**Recorder。** 同一 Metareview 的可选 `skill_updates` 记录 create/revise/retire、范围、步骤、反例和精确批次来源。现有 Journal 事务一起提交科学分析、知识、游标和发布意图。宿主逐项校验并在共享锁内核对 base，写不可变包后原子写发布回执。崩溃重放沿同一操作身份恢复；非法提案和冲突保留原稿及原因，可恢复写入失败保持待处理。研究失败与工程失败均保留其实际范围，后续实验由科研角色决定。人工原件继续参加源码哈希。共享包在 `EXP/skills/versions/<sha>/<name>/`，发布记录在 `EXP/skills/releases/`。完整目录通过 `read_asset skills/INDEX.md` 搜索或分页；角色按需读 `skills/<name>/SKILL.md`、`skills/<name>@<sha>/references/origin.json`。附件内有来源运行、卡片与字段双重哈希、精确字段值。新运行固定可见版本，后续仅接收本运行的新发布；模型调用沿自己的已发快照读取。[资源生命周期](../../../QAViT/docs/RESOURCE_LIFECYCLE.md)

**Critic。** 默认独立 `gpt-6.2-astra / medium`，Generation 沿用 high。跨运行 Skill 样本的 Recorder 沿历史冻结路由 `gpt-6.2-sol / medium` 执行；新模板其他角色仍沿既有配置。宿主在生成终态后对实际提交的 H 创建一个可恢复批量任务，交付 Brief、主题、父方案、关键测量与读取入口。需要实验时，顶层 `hypothesis_ref` 固定目标及其当时的 topic、method 与 inspiration。Coding、Safeguard、执行、Evidence 与原任务续调使用原管线。请求和 Coding 绑定后的规格都检查 L3 的开发协议。Generation 接收建议、版本和测量，决定取舍、修订并分别规划最终 L3；重新综合沿用原审阅。Critic 无结果或局部路由/预算失败时保存实际状态，Generation 可以继续。全局预算、来源及执行约束仍适用。配置 `enable_critic: false` 时跳过任务和交付，旧 gonly 缺省此字段时维持原配置语义。已冻结 Full 继续通过归档代码恢复。[主线伪代码](PSEUDOCODE.md)


## 真实链路与行为

1. **从经验到下一运行的动作。** 历史 DINOv3-B 冻结 B32 的同父单块干预在 16 张图上比较了未改父包、同前缀局部 FP 替换与原 FP。第 0 块的修复使 logits KL 从约 1.058 变为 0.693，末层 patch 相对误差从约 0.194 变为 0.148，storage 误差却从约 0.165 变为 0.469。这是诊断上限，合法 A4 增量仍需另测。运行 A 的真实 Recorder 在 `MetareviewCard-032ccec68f5547fbb1179abb84b2d5ac@1` 发布 `gate-fp-block-headroom-to-static-ptq`，包 SHA 为 `4f219ad0755ef7b2392a1a199a326da757b68e65adc980159e6550c9cbcea243`。五个便携字段保存了父包、预定决策分支及误差等原始观察。使用隔离共享根的新运行 B 自主读取流程和出处，提交以“logits、四层 patch 相对改善二者取小，再最大化”提名主块，安排同父单位倍率、局部误差竞争块和冷载复核。相同 Brief、人工库及调用上限的对照，将末层 patch 改善最大且 KL 不恶化作为提名标准。两者都没有启动新候选数值作业。这个单次比较观测到选块规则变化。gonly 保存隔离验证原件。交付时从已提交 Metareview 复核五个来源字段，用同一发布事务将该真实流程迁入 main 的默认共享根，重复发布取得同一回执。新开的 `default-library-read` 运行可从普通目录恢复相同字节。该小包和发布回执随本次提交分发。[部署的流程原件](../../skills/versions/4f219ad0755ef7b2392a1a199a326da757b68e65adc980159e6550c9cbcea243/gate-fp-block-headroom-to-static-ptq/SKILL.md) · [Skill 机读回执与来源字段](AB.json)
2. **从批评到 Generation 的取舍。** 两个历史 select 断点来自同一实际研究运行的不同轮次。每侧共三次调用的比较中，Critic 两次均把为它分配的两次调用用于读取，交接时没有成稿，Generation 按原路径完成综合。第二轮增加一组五次上限的配对，双方均给出剩余调用并在最后一次提示收束，输入提供更细的干预字段入口。Critic 第三次调用提交三个版本的建议。针对 H2，它要求在部署前缀和同一滑窗上核对 blocks 2/5/8 的实际接口张量，再把末层表的替换归因于 H2。Generation 明确采纳匹配父包、核对张量与双任务分列的判据，保留三个现有 H 的版本。Generation 自审也提到父包和导出冷载。两侧均提出了具体下一行动，未对原研究 Journal 提交新 H，也未执行新开发或官方评价。[断点、版本、原始决定与逐调用成本](C.json)
3. **从请求到新 H 的宿主验证。** 单独的 CPU 验证沿生产执行链，用脚本科研决定对 H2 发起 tensor_probe。seeded tiny ViT 的一次 A4 局部 QDQ 得到输入相对 MSE **0.0309211**、算子输出相对 MSE **0.0318711**。恢复后的 Critic 收到 Evidence，Generation 使用该 Evidence 修订 H2 到 @2。Recorder 和 Generation 均收到建议及测量，新的 H2 仍标记待验证，三个槽位分别获得 L3 计划。科研角色是测试夹具，测量来自独立 CPU worker。[CPU 测量、修订与来源](cpu_feedback.json)

## 成本

input 包含 cached input；读量统计 read 工具返回的正文字符，模型调用耗时包含该调用的订阅等待和 preflight。各侧并行执行，表内时间为侧内各次调用耗时之和。各项比较共享同一侧内总配额，不同轮次和不同配额的结果分别列示。

| 断点与路径 | 调用 | input / output tokens | cached input | 读取次数 / 正文字符 | 调用秒 |
|---|---:|---:|---:|---:|---:|
| 运行 A，Recorder | 1 | 27,667 / 2,477 | 0 | 0 / 0 | 92.5 |
| 运行 B，仅人工库 | 5 | 161,111 / 2,193 | 9,472 | 15 / 35,168 | 177.5 |
| 运行 B，含已发布流程 | 5 | 133,952 / 1,861 | 9,472 | 4 / 7,367 | 197.5 |
| 第 1 轮，限三次，自审 | 2 | 53,027 / 1,648 | 0 | 3 / 16,176 | 66.2 |
| 第 1 轮，限三次，Critic + Generation | 3 | 59,504 / 1,508 | 0 | 4 / 12,933 | 92.0 |
| 第 2 轮，限三次，自审 | 3 | 92,903 / 1,582 | 0 | 5 / 29,680 | 91.3 |
| 第 2 轮，限三次，Critic + Generation | 3 | 76,071 / 1,491 | 0 | 4 / 15,952 | 82.6 |
| 第 2 轮，限五次，自审 | 5 | 155,641 / 2,181 | 0 | 13 / 40,218 | 135.5 |
| 第 2 轮，限五次，Critic + Generation | 5 | 145,628 / 2,941 | 0 | 6 / 48,459 | 145.4 |

Skill 消费侧相对人工库少 27,159 个输入 token、少 27,801 字符、多 19.9 秒。首个生产者成本单列；若只摊给这一个消费者，生产者加消费者共 161,619 输入、4,338 输出、六次调用。五次上限的 Critic 配对少 10,013 输入、多 760 输出、多读 8,241 字符、多 9.9 秒。短预算中出现的缺失读取包括隔离回放未实现的 retrieve_memory 和一个不可用字段。断点支持授权卡片及文档读取，实验请求会记录为待执行；它使用生产模型路由与提交 schema，不能替代全工作流吞吐测试。

两阶段共 **32 次有响应的订阅调用**，输入 905,504、输出 17,882，缓存命中 18,944，54 次读取，共 205,953 字符，累计调用约 1,080.6 秒。首次启动五次上限配对时，两侧各有一次相对输出路径错误的 CLI 尝试，分别约 3.9/4.6 秒，未接受任何决定，服务未返回 token 数。修正为绝对路径并使用新目录重跑。订阅回执有请求路由及目录核验，CLI 未回显实际模型和计费金额。以上不含开发者对话、自动回归与历史实验开销。实测链引用的历史 GPU worker 约 61.3 秒，本轮所有模型断点新增 GPU 数值作业为零。最终独立 CPU 交接 worker 的墙钟为 3.12 秒、CPU 为 4.87 秒、GPU 为零。开发期间的回归含重复 CPU 探针，未收集这些测试的总 CPU 时或能耗。[真实调用](C.json) · [跨运行样本](AB.json)

## 验证与复核

gonly 阶段 A 的 33 项测试、阶段 B 的 46 项、专项 9 项，以及冻结源码后重跑的 40 项回归通过，集合存在交集。七个原包和迁入默认共享根的 Recorder 包均通过 Skill 格式检查。曾在运行回归期间修改源码，恢复测试按预期阻止了跨身份续跑；上述 40 项使用固定源码重新运行。A/B 的消费者在前四次调用以读取为主，两侧同样给出第五次决策窗口；隔离验证使用显式 intervention，原始运行未改写。数值回归覆盖冻结父网格、局部增量接受/回滚、导出冷载、定制算术和同包双任务。主线还检查一次批量任务、候选/局部路由失败、恢复、版本修订、按需读取和 Recorder 来源。

核对真实历史运行 `dinov3b_static_w4a4_independent_20261121_r1` 的冻结输入，常规与历史两种配置校验均重得原 SHA `f1d79e0df05c109f954bb306faed080df837f69c0852bee854016ef7c582ddc7`。主线初轮回归暴露一条旧断言期待相同实现正文重复注入；gonly 对照单独重放得到同一失败。已改为核验实际 body_location、指向的完整实现，以及重启后的原件。此项只调整回归断言。

最终 C 提交的 25 个模块按三组执行，分别通过 56、70、40 项，共 **166 项**。结构审计通过，224 份 Python 文件完成语法检查，源码快照 404 个文件；知识审计通过，检查 81 页。实测包部署前后 `core_sha256` 都是 `12a08728e1d1d82563d7c5f51ac214f5e24984532119612d49c19bfc54b0a42b`。模块清单、各组耗时及状态见[验证回执](C.json)。

在本机，A/B 原始作业及上游文件/许可位于 gonly worktree 的同名验证目录。C 的原始作业位于 main worktree 对应目录。轻量哈希、字段、决定和用量随 Git 跟踪。需要重新核对原件时，可用 `PYTHONPATH=QAViT python -B EXP/validation/skill_critic_20261124/critic_replay.py 2 critic --calls 5 --output <新的EXP目录>`；这条命令会产生新的真实订阅调用。A/B 的复现脚本依赖 gonly 的本地冻结源与共享库。既有[冻结数值报告](../frozen_trials_20261124/REPORT.md)记录更大规模的历史成绩，本轮没有重测它们。

## 科研能力的证据边界

跨运行选块规则和一次审阅后的验证判据均出现了可追溯的变化。两个紧预算断点同时展示了新增角色的读取成本。尚无同条件独立重复所得的 ImageNet/ADE20K 新包成绩，无法据此断言科研能力、低比特精度或端到端性价比已提高。
