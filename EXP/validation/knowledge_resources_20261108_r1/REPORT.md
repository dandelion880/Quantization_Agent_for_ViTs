# 科研知识资源整理交付

基线：6c58798，任务开始时工作区干净。实现范围是静态资源导航、读取/引用兼容、文档维护与验证；未迁移历史产物，未改科研阶段、实验链、排名或预算/GPU 调度。

## 交付入口

- [完整真实目录与科研资源盘点](PROJECT_RESOURCE_INVENTORY.md)：实际文件树、原→新位置、职责/维护/调用链、运行内外边界及待讨论问题。
- [统一知识入口](../../../QAViT/knowledge/INDEX.md)：唯一资源导航与维护规则；现有正文保留 docs 下原位。
- [资源生命周期](../../../QAViT/docs/RESOURCE_LIFECYCLE.md)：常驻/按需知识、MethodLibrary 兼容、Journal/Memory/Evidence、共享方案和文档来源。
- [Skill 接入说明](../../../QAViT/docs/SCIENTIFIC_SKILLS.md)、[历史导航](../../../QAViT/docs/HISTORY.md)、[当前伪代码](PSEUDOCODE.md)。
- DELIVERY.json 为机器可读摘要；changes.json / changes.patch 为本轮增量；before/、before_manifest.json 为任务前态。原始验证材料留本目录，不写入核心资源。

## 实际修改

1. 新增 knowledge/INDEX.md；context 仅交付一个 read_asset 指针，正文不常驻。docs 与 knowledge 均走原冻结授权/哈希/分页/去重路径；原 skills/、methods/、solutions/ 入口保留，无双份正文。
2. Skill INDEX 的触发表从四个 SKILL.md 的 name/description 生成。audit_knowledge.py 仅是维护/离线检查脚本，不是运行时注册框架，不调用 LLM。四份正文及九份参考未改。
3. 方法文档以 read_asset 为常规入口。无调用方的 require_plan_documents 阅读检查已删除；数值 query_method_library/read_method_document、method_library.json 快照、插件覆盖和哈希核验、旧 require_method_documents 配置兼容保持。可选方法发现目录缺失/损坏时回退，不改变数值执行合同。
4. sources.ref 中当前授权静态文档路径兼容存入 sources.document，保留 status/supports；文档不加入 Evidence/卡片支持链、不授予新权限。显式 document 仍是作者声明，实际交付范围/版本以 read_asset 回执为准。未新增输出必填字段、引用门禁或拒绝整份科研提交；私有/未知卡片仍按原规则处理。只影响新提交，历史卡片不补写。
5. solutions README 分清组件、完整种子、原论文配方与协议实测 Top-3；种子配置、构造器、协议键、成绩排序/晋升/归档均不改。明确 Memory 仅属本运行，自动跨运行复用现限同协议方案桶。
6. README/STRUCTURE 改为当前概览，逐次历史报告集中导航；EXECUTION/METHODS/Skill 说明互相引用权威入口。新增资源回归，更新公共 Prompt 与其哈希基线。
7. AGENTS.md 仅增加下述一条结构规定，其余全文规则不变。合理技术例外明确允许，没有新审批。

## AGENTS.md 新增原文

> 新增或修改文件按职责组织：角色/任务指令放 `QAViT/agents/prompts/`；静态科研知识由 `QAViT/knowledge/INDEX.md` 统一导航，现有 Skill、Method、方案种子及来源资料保留在 `QAViT/docs/{skills,methods,solutions,references}/`，新增知识优先放 `knowledge/`；系统、合同与接口说明放 `docs/`，执行实现放 `core/`、`runtime/`、`tools/`，配置与测试放 `configs/`、`tests/`；运行内 Journal/Memory/Evidence、协议方案库及验证产物放 `EXP/`。合理技术例外可按实际依赖保留原位，并同步更新导航与相关读取入口，不为目录形式迁移历史资产。

## 验证结果

<!-- VALIDATION_START -->
| 验证 | 实际结果 | 原件 |
|---|---|---|
| 第二次完整离线回归 | 549 项中 548 项通过；1 项三 worker 重叠时间断言失败，1110.943 秒 | full_regression_r2.log |
| 首次完整回归 | 549 项中 548 项通过；1 项 worker 收尾时序断言失败，747.532 秒 | full_regression.log |
| 新资源回归 | 7/7 通过，7.857 秒；覆盖七角色、引用/权限、缺失回退、恢复与数值快照 | targeted_r2.log |
| 收尾/资源模块独立复测 | 13/13 通过，6.587 秒 | adaptive_recheck.log |
| 当前/任务前源码专项对照 | 同样的 worker 收尾与并发 9 项，各 9/9 通过，14.489 / 14.157 秒；复用相同本地依赖 | runtime_current.log、runtime_baseline.log |
| 静态审计 | 166 个 Python 文件语法/导入与动态入口通过，253 份源码快照可复现 | structure_audit.json |
| 资源一致性 | 4 Skill、21 方法，53 份当前导航/知识页面；生成表和本地链接通过 | knowledge_audit.json |
| CPU 数值集成 | 六次工具调用成功：三查询、三类 tiny synthetic 实测；重复统计缓存命中且账目不增加 | cpu_contracts/contracts.json |
| 历史 CPU/快照对照 | 工具 Schema、配置、三个统计的 summary/shapes/comparison 与旧结果逐值相同；query_structure 仅排除实测可用内存读数差异 | compatibility_comparison.json |
| 旧真实输出脚本回放 | A1 文档问题 3→0；C1 三条文档来源恢复，仅剩 1 条非卡片“理论推导”ref；机制正文不变 | citation_replay.json |
| 真实 Generation 文本烟测 | 完成；3 次真实模型调用、5 次不同 read_asset、0 submission_issues | resource_choice_skills.json |
| 真实 Judger 工程错误烟测 | 完成；1 次真实模型调用、0 资料工具调用、0 submission_issues | engineering_only_skills.json |
| 历史/规则保护 | 已核对 83 份历史文件哈希不变，40 份静态正文/索引不变；AGENTS 只增指定条款 | resource_preservation.json、historical_hashes.json |
| 旧函数依赖审计 | 当前生产代码无调用/导入，另检查 375 份历史独立/生成 Python，零匹配（副本/快照另保留） | legacy_dependency_audit.json |

真实 Generation 自主读取 adapt-quantization-method/SKILL.md、solutions/repq.md、methods/repq_ln.md、当前运行的完整方案 configuration_asset 及 designing-a-remedy.md。最终用 document 记录方法/种子，用 ref 记录本次 Brief，用 source 说明本卡推导；明确种子待评测、题设不是本次测量。无路径指令强迫其选某份 Skill。

真实烟测合计 4 次模型调用、282.688 秒、输入 79,048 tokens、输出 7,050 tokens（含 reasoning 4,622），缓存输入 3,072；0 数值任务、0 GPU 实验。请求模型为配置默认 gpt-6.1-sol，Generation high/Judger medium；CLI 未报告 actual model/effort，不能把 requested 当独立服务端证明。完整调用与用量见 live_validation.json / 各 llm_audit 原件。

运行目录按需入口无新增 LLM 调用；公共 Prompt 净增 280 字符，Skill 短目录净增 354 字符（现 1,738 字符），统一 INDEX 仅作为指针交付，正文/参考仍按需读取。这是字符差异，不是推断的 token 成本。

两次完整运行均为 548/549，失败项不同，不能写成全量全绿。首轮失败在 test_slow_teardown_past_worker_deadline_preserves_result_on_resume 的即时 live_processes 断言：检查时 worker 已不存活；该模块独立 13 项通过、第二轮此项通过。第二轮 test_dispatch_starts_three_overlapping_workers_and_recovery_accepts_once 的实际重叠为 0.103505 秒，小于固定 0.4 秒阈值；该项首轮通过。随后当前/任务前源码同样的 9 项 worker 专项各全通过。两个失败项目都曾在未改代码/测试/阈值的独立或全量运行中通过；没有丢失、重复结算或权限失败断言。它们显示短时序断言有敏感性，但本轮未证明性能变化的因果、未验证任意系统负载下的稳定性。保留原调度与阈值，风险留在报告中，不靠反复跑到绿色掩盖问题。
<!-- VALIDATION_END -->

## 失败、范围与剩余风险

新增测试初次运行时，七角色夹具未释放已占用角色槽位，另一夹具重复领取运行任务，触发既有队列保护。已修正夹具为顺序结束/单次领取，再测新增七项通过；生产调度未修改。原始 targeted.log 与修正后 targeted_r2.log 均保留。

旧真实输出回放：A1 的三条 Skill/Method ref 均恢复到有效来源；C1 的三条文档 ref 均恢复，仅“本卡理论推导”这种非卡片 ref 保留原问题记录。两份科学机制正文完全不变。当前公共 Prompt 提示用 source 自然描述推导；程序不猜未知 ref 的含义来伪造出处。回放是脚本化提交验证，不能算新增真实 LLM 结果。

没有完整 GPU 科研实验、ImageNet/ADE 正式精度、长时预算/GPU 压力或科研收益 A/B；未重新核验联网文献服务、论文链接和全部方法组合。真实烟测如有读取，仅证明该文本情境的自主调用/提交，不证明方法有效。数值必需配置/快照缺失仍按原合同失败，不能把可选导航的回退理解为绕过执行约束。

物理知识仍在 docs 中形成明确列出的兼容例外；进一步物理统一、跨运行经验导入、METHODS 与数值 Schema 的更细同步、历史本机路径可移植性和大型核心拆分留待后续讨论。源码身份会随本次知识/提示词/代码编辑改变，旧运行续跑仍须现有显式 intervention，不静默忽略哈希差异。

## 复现与回退

从项目根目录，普通离线检查：

```bash
python -B QAViT/tests/audit_knowledge.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
```

新输出目录运行 audit_structure.py、snapshot_contracts.py，命令见 QAViT/README。本目录 replay_citations.py/live_smoke.py 是本次隔离验证脚本；如复跑需复制到新的 EXP 目录，不覆盖本轮轨迹，真实脚本会消耗订阅额度。build_inventory.py 只更新其所在交付目录的盘点树/manifest，不改历史资产。

changes.patch 按 before/ 与当前文件生成，逆向可检查本次增量；回退前确认当前文件仍与 changes.json 后态哈希一致。只撤销本轮列出的文件/新入口，不使用 git reset 覆盖后续用户成果，不删除 EXP 验证证据。旧知识目录从未移动；回退无需迁移历史运行。AGENTS/README/STRUCTURE 与新增文档、Prompt 基线应随同撤回，避免留下导航/调用不一致。
