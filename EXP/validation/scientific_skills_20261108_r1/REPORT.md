# 首批科研 Skill 接入与验证

日期：2026-11-08。基于任务开始时的最新工作区（Git HEAD `8d937f0`，包含此前未提交的科研能力升级），保留原有改动。本次未执行 git reset、提交、迁移历史运行，未改数据、checkpoint 或评测协议。

已完成最小接入：Skill 是可选的本地策略资料；沿用原有 read_asset、上下文、哈希/权限、分页和续调去重。全量离线回归 542 项通过；真实角色烟测结果见下文最终汇总。

## 实际改动

| 文件/目录 | 本次增量 |
|---|---|
| QAViT/docs/skills/ | 安装原包四条 SKILL.md、九份 references 和短 INDEX；四条正文及九份参考与原包逐字一致 |
| QAViT/agents/interaction.py | 只增加短目录交付及已有 asset_locations 登记；Generation/Evolution/Judger/Recorder 可发现，目录缺失时省略 |
| QAViT/agents/prompts/common.md | 一段可选策略阅读说明：可零 Skill、按需读、复用已读内容、失败继续、无固定顺序/新增审批/H3限制 |
| QAViT/tests/test_scientific_skills.py | 六项可复用离线资源/权限/续调/回退集成回归 |
| QAViT/tests/skill_evals/ROUTING_AND_REPLAY.md | 原包评估设计独立保存；只更新开头的接入状态和报告链接，不纳入运行时目录 |
| QAViT/tests/data/prompt_readiness_baseline.json | 保留历史基线，更新 explicit_prompt_fixes/current_prompt_sha256，并记录本次授权变化原因 |
| QAViT/docs/SCIENTIFIC_SKILLS.md、QAViT/README.md、STRUCTURE.md | 配套接口、维护说明、目录及盘点入口 |
| .gitignore | 精确放行本次精简报告、伪代码、交付摘要与可复现脚本；原始运行/备份继续留本地 |

`INDEX.md` 补上 docs 相对路径的实际 read_asset 参数，并说明 profile/query 名称属于现有 profile 服务、方法文档优先直接读取、方案与 Memory 的现有入口，以及 L0 不新增实验级别。未新增工具、Schema、Skill Router、配置开关、任务类型或模型调用阶段。

以下实现与任务开始快照一致：科研主循环、H*/H4、ExperimentRequests、并行 Judger、队列与 driver、预算/GPU 资源、实验合同、static W4A4、方法/Top-3 实现、Memory/Recorder 发布机制、Coding/Safeguard、角色工具和提交合同。只有共同 Prompt 加入可选资料说明，角色分工和科研原则未改。

## 上下文与成本

目录正文为 1,384 字符（2,314 UTF-8 字节）；确定性典型 H3 请求对照从 33,989 到 36,067 字符，增加 2,078 字符，包含说明和来源位置等展示元数据。它是**字符数，不是 token 数或端到端性能测量**，见 context_cost.json。初始 Skill 正文和 references 的注入数均为零；新增路由模型调用数为零。

Skill 读取由角色自主选择，会产生原有工具续调成本；已有上下文内重复结果只保留一次。测试覆盖 INDEX 已交付后再读不重复、同一正文重复读取不重复、恢复后已读正文保留、参考文件尚未选读时不出现。显式 research_note 之前的历史读取仍服从旧压缩规则，不新增永久缓存。

## 静态、单元与集成结果

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 针对性 Skill 测试 | 6/6 通过，6.257 秒 | skills_tests.log |
| 全量离线回归 | 542/542 通过，714.538 秒 | regression.log |
| 最终结构审计 | 通过；164 个 Python 文件、248 个源码快照文件，errors=[] | structure_audit_final.json / .log |
| CPU 实际数值/接口快照 | 三项结构/插件/位点查询 + 分布、敏感性、特殊位点统计共六调用成功；三份原始统计文件，缓存复读不增加测量计数 | cpu_contracts/contracts.json |
| CLI、补丁与脚本语法 | run.py --help 正常；git diff --check 正常；交付脚本 AST 编译通过 | DELIVERY.json 中记录 |
| 原包一致性 | 四正文、九参考 SHA256 与原包一致；修改仅目录、评估状态说明及接入代码/文档 | package_manifest.json |

运行命令（均在项目根目录，输出使用新 EXP 路径）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
python -B QAViT/tests/audit_structure.py --output EXP/<新路径>/structure_audit.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B QAViT/tests/snapshot_contracts.py --output EXP/<新路径>/cpu_contracts
```

新增测试逐项验证：目录与 name/description/相对链接/递归身份；全部 Skill/参考真实 host_tool 可读；研究角色目录交付与工程/管理角色不自动交付；续调/恢复与重复读取去重；整个 Skill 树缺失仍直接提交 H3；选读不存在文件后回原任务并正常提交；冻结哈希和未授权路径边界。已有全量回归同时覆盖原无 Skill 实验循环、父子关系、并行/隔离、代码链、Recorder 与预算。

CPU 数值快照使用随机小型模型与合成数据，角色单元测试是明确的脚本夹具；二者都不冒充真实数据集上的科研成绩。全量回归结束后才启动真实订阅烟测，避免占用其共享订阅槽影响并发测试。

## 低成本真实 Agent 烟测

`live_smoke.py` 使用生产 Workflow 队列、request_body、真实默认模型/effort、订阅预检/传输、host_tool 和 submit。仅驱动指定角色任务，不启动完整 advance 科研轮次。输入情境取自原包 A1/B1/C1/D1/A4，明确标成**假设性评估题设，不是本运行 Evidence**；不允许启动数值实验或联网文献。模型可查本地资料，也可直接提交，未指名要求读某条 Skill 或参考。

A1 做同题、同角色、同配置的有/无 Skill 对照。无 Skill 条件只在烟测脚本隐藏目录和 Skill 资产权限，生产实现没有开关或分支；原目录完全缺失的路径另有单元集成测试。每案例至多六模型调用、420 秒，全批 1,800 秒，仅是烟测预算，不进入生产科研合同。

<!-- LIVE_RESULTS_START -->
七个案例均完成，真实模型调用合计 **11 次**，host_tool 读取 **9 次**，累计案例耗时 **577.681 秒**；数值实验与 GPU 任务均为零。

| 案例 | LLM 调用 | 读取 | 秒 | 人工复核的实际行为 |
|---|---:|---:|---:|---|
| A1 无 Skill | 1 | 0 | 126.262 | 无 Skill 对照；直接生成待检验 H3，保留 W/A 交互和数据边界。 |
| A1 有目录 | 2 | 3 | 166.025 | 自主同批读 A（诊断）、B（判别）及 GPTAQ 方法卡；未把激活 outlier 当既定根因。 |
| B1 有目录 | 1 | 0 | 28.338 | 零 Skill；直接识别网格/输入混杂并提出同条件对照。不能计作 B 的路由命中。 |
| C1 有目录 | 2 | 3 | 161.992 | 自主读 C（适配）和 SmoothQuant/RepQ 方法卡；提出分支内候选，保留等价性、W4 代价和反证。 |
| D1 有目录 | 1 | 0 | 27.734 | 零 Skill；区分 peer/FP 两个参考，不将 5%/1% 当传播倍率或任务收益。不能计作 D 的路由命中。 |
| A4 有目录 | 1 | 0 | 15.804 | 零 Skill；已定位 shape 错误沿原工程链继续，不当算法反证。 |
| D_policy 有目录 | 3 | 3 | 51.527 | 定向请求复用仓库准则；自主选 D 并同批读 next-step-decisions 与 evidence-scope，随后提交有条件投入判断。 |

补充 D_policy 是因前六例尚未选择参考文件而增加的**定向资料使用**烟测（`live_smoke.py --case D_policy`）。它要求复用既有准则，但没有指名 Skill、文件或读取顺序；不把它描述为完全无提示的自发触发，也不据此宣称原包全部情境已评估。

CLI 报告累计 input_tokens=192524、output_tokens=16211；reasoning_output_tokens=7493 是输出中的单列统计，不重复相加。默认请求 Generation 为 gpt-6.1-sol/high，Judger 为 gpt-6.1-sol/medium；CLI 没有独立回显 actual model/effort，审计保持 null，不把请求配置冒充实际回显。

A1 同题对照：无 Skill 为 1 次调用、0 次读取、126.262 秒、输入 16,959 / 输出 3,981 tokens；有 Skill 为 2 次调用、3 次读取、166.025 秒、输入 38,376 / 输出 5,061 tokens。两者都遵守题设/实测边界，候选不同；单次非盲重复无法判断科学质量的因果提升。目录没有新增路由调用，但有用的按需读取仍有完整上下文续调成本。

四个 Skill 正文均有真实读取记录，另有两份策略参考；没有重复读 INDEX，也没有案例内重复调用相同资产。B1、D1、A4 和无 Skill 对照直接提交，说明目录不会强制使用。原始输出与逐调用请求/回执/审计保存在 live/；基础批在 live_summary.json，补充批在 live_D_policy_summary.json，精简可复核汇总在 live_validation.json。
<!-- LIVE_RESULTS_END -->

## 交付、原包与回退

- [独立完整项目/科研资源报告](PROJECT_RESOURCE_INVENTORY.md)：逐文件目录、模块职责、资源位置/维护/角色/读取/使用/更新、调用与设计边界，以及后续讨论项。
- [接入说明](../../../QAViT/docs/SCIENTIFIC_SKILLS.md)、[当前流程伪代码](PSEUDOCODE.md)。
- 原始来源：package_manifest.json 保存 ZIP 与各文件 SHA；package/ 保存原始交接、原始 INDEX 和原始评估设计。完成接入核验后删除用户要求删除的原始压缩包，最终状态见 DELIVERY.json。
- before/ 与 before_manifest.json 保存本轮开始的有效文件；preexisting_changes.patch 单列用户此前改动；changes.patch **仅相对于该开始快照**，不是相对 Git HEAD，避免回退时误撤前次升级。
- 如需撤回，先核对当前文件仍匹配本次交付，再执行 `git apply --reverse --check EXP/validation/scientific_skills_20261108_r1/changes.patch`，人工审阅通过后反向应用。勿用 git reset。撤回不删除历史验证记录；旧运行恢复仍遵循原源码身份/intervention 规则。

## 未验证部分

本次没有长时间 GPU 研究、完整 ImageNet/ADE 双任务 L3、真实 INT4 部署测试、跨模型泛化或 Skill 科研收益的统计证明。原包所有案例、复杂多 Skill 组合、Evolution/Recorder 的全部真实情境和多随机重复盲评尚未逐一执行。少量真实角色读取/提交只能证明本次情境下通道与行为可用，不能推断所有研究问题都应使用 Skill，或已提升最终任务指标。
