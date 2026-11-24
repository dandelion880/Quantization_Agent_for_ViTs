# QAViT_gonly 工程清洗（2026-11-17）

状态：本轮工程清洗与验证已完成。最终全量回归 **767 项，765 项通过、2 项 CUDA 测试跳过，无失败/错误**；三个真实 Generation 组件回放完成，历史评价原件校验通过。机器可读结果见 [RESULTS.json](RESULTS.json)。

工作区基线为 `0217218`。修改保持现有 workflow、Journal、Memory、主题/灵感库和 Coding→Safeguard→原执行链，没有另建科研流程；没有改数据、checkpoint、冻结 Brief、历史实验、旧评分包或原分数，也没有清理运行产物。

## 问题—修改—验证

| 复核结果 | 修改与边界 | 验证入口 |
|---|---|---|
| 确认缺陷：新增 focus 使旧 Eval-L0 包依赖当前 TASKS 而无法载入 | 按各包自己的冻结 purpose/role 合同校验；未覆盖任务单列。新建 gonly_v0_3 补 focus、H3 首次/持续发展和当前 plan_l3 职责；旧包不改 | `FrozenEvaluation`、`tests/evaluation`；实际六个历史评价目录、43 个案例及其中已有 12 份接受评分原件校验 |
| 确认缺陷：可信 implementation 来源被 CUDA/资源措辞压过 | 身份校验后的结构化来源优先，包装异常保留原因链；实现回 Coding，请求回原科研申请者；同一失败实现不走数值 transport 重试 | `test_structured_error_origin...`、`test_wrapped_trusted_failure...`、`test_code_repair`、`test_dual_task_evaluation` |
| 确认缺陷：无可授权 GPU 时 ready 可无限保留 | 区分 ResourceBusy/ResourceUnavailable；普通繁忙默认 2 秒重探测，不增加尝试/预留。无任务运行且不可执行或资源等待 900 秒无进展时可靠保存 resource_blocked；resume 重探同一授权池 | `ResourceAndCostRecovery` 的真实 launch 边界、run 收尾、繁忙/兄弟任务/恢复夹具；原数值等待预算回归 |
| 确认缺陷：一条坏 topic_notes 拒收整份 Recorder 正文 | 条目内校验确切 topic 版本与来源；保留 submission_issues 原文和原因，其余正文及有效注记发布；没有放宽身份检查 | `test_invalid_topic_note...`；不存在/非确切版本/坏来源/类型错误四种情况 |
| 确认缺陷：next_step 文本用于动作匹配和新建 | 完成/暂存/恢复只按 action ID 更新；next_action_id 接续原 open 动作；new_action 显式建新动作；next_step 仅叙述 | `test_defer_pause_resume_complete...`、`test_research_topics`；真实 Generation 暂存恢复回放 |
| 确认缺陷：恢复源码时授权整个 research_run/commit 快照 | 保留完整归档；read_asset(snapshot_field, member) 只读明确绑定来源的许可成员，返回原版本、归档和成员 hash；拒绝原始容器及评价材料，旧保存权限也过滤 | `test_exact_source_member...`、历史源码加载/干预恢复回归；篡改、路径越界、旧权限与评价隔离 |
| 策略重审并修正：综合失败曾结束该轮，候选丢失 L3 接续机会 | 原有限重试结束后保存 synthesis_pending、确切候选版本与待做阶段；该轮未完成，修复后用原 resume --retry-failed 和原预算续调。宿主不编造审查、不默认 H1、不损伤 B | `SynthesisRecovery`；恢复后各候选 plan_l3 精确绑定；`test_multi_candidate_l3`、候选修订回归 |
| 确认说明矛盾：模型提示使用内部 resolved_choices.training | 模型使用 execution.training，规范化冻结记录用 execution.resolved_choices.training；旧输入兼容保留 | `test_model_training_input...`；对应 plan_l3 Prompt、实验接口和 Skill 兼容说明同步 |
| 实测风险：大历史目录全量正文、重复恢复条目和宿主扫描 | 自动主题/灵感/方法目录限制 8 条，按 ID/query 分页深读；仅完全相同的研究注记共用正文，所有证据/配置/代码定位字段保留。Journal 目录稳定时复用索引，预算增量折叠、实时计时；状态最多每秒写，目录周期扫描 5 秒，关键节点仍立即检查 | 长历史前/中/后实测、可见性/分页/原件篡改回归；现有停止、故障、预算和持久化回归 |
| 真实回放发现遗漏：worker 收尾比较整份归档，离线说明变更仍拒收输出 | v3 执行身份与完整归档身份分开，并修正 worker 成功/格式失败两条比较链；真实运行代码变化仍拒收，离线包/一般测试/退休 Prompt 保留归档但不影响执行身份 | `test_worker_accepts_offline_archive_changes...`；失败回放原件保留，受影响案例另目录重放 |

新回归集中于 [test_engineering_cleanup.py](../../../QAViT/tests/test_engineering_cleanup.py)。旧测试按已确认合同调整：综合失败保持未完成；恢复资产使用精确成员；动作复议明确 ID；可信实现来源优先于 EngineeringBlocked 同名异常，宿主阻塞与无来源的旧阻塞仍不进 Coding。另修正预检顺序夹具依赖本机 codex 路径的问题，并把宽泛异常断言加强为确切错误类别。Prompt 原始基线保留，仅记录有对应语义回归的明确修订，差异见 [prompts.diff](prompts.diff)。没有删除测试或降低数值断言来通过。

## 保留与未重复改动

Generation 仍决定方向并使用原模型路由；H1/H2 改良完整父方案且有回退；H3 可沿同一灵感修复、调参、组合、解释和简化。独立 B 不占候选槽位。三候选经 Generation 综合/必要修订后各按确切版本进入原双任务 L3，ADE20K 优先的更新规则、候选优先一卡并行与 Coding 串行保持不变。

主题、灵感、方法谱系、候选版本仍分别维护；旧候选和证据归属不回写。Coding 实现并回送问题，Safeguard 核验忠实性与协议，Recorder 记有来源的认识和接续事项，宿主仅管理身份、执行、预算与恢复。Skill 仍是可选知识，没有新增必读证明、实验次数或固定科研步骤；static W4A4、数据划分、冻结迁移、局部诊断/最终 L3、参考优化/最终 training-free 的边界沿用原合同。Judger/Evolution 仍服务历史读取与回归，没有按关键词删除。

## 成本实测

[benchmark.py](benchmark.py) 只读实际三轮历史（9,150 commits、9,180 events、2,294 cards、1.213 GB），另建 200 主题、120 灵感、800 已取消任务的明确状态夹具。各项 3–10 次，状态写 20 次，以下为中位数；原始样本见 [before_costs.json](before_costs.json)、[after_costs.json](after_costs.json)、[final_costs.json](final_costs.json)、[最终源码复测](verified_costs.json)。最终复测只读复用同一历史运行与 perf_final 状态夹具，不新建科研运行。测试与测量有 CPU 重叠及冷热缓存差异，不能当严格隔离的因果基准或生产吞吐承诺。

| 路径 | 修改前 | 最终 |
|---|---:|---:|
| 真实历史 Journal 重复刷新 | 31.50 ms | 0.022 ms |
| 队列投影 | 32.87 ms | 1.583 ms |
| 预算投影 | 38.99 ms | 10.002 ms |
| 无活动 worker 轮询 | 31.03 ms | 0.073 ms |
| 上下文构建 | 27.21 s | 17.05 s |
| 权限构建 | 19.84 ms | 10.599 ms |
| 完整目录大小扫描 | 777 ms | 756.326 ms |
| 原子状态单次写入 | 1.36 ms | 1.40 ms |
| 长目录夹具上下文 | 396,566 字符 | 19,776 字符 |
| 长目录夹具上下文构建／权限构建 | 107 ms／8,046 ms | 92 ms／312 ms |

真实历史本身没有主题目录，正文大小 82,813→83,565 字符（新增导航与接口说明），不声称对它压缩 95%。最终源码复测结果已用于表格；两个测量场景没有重叠恢复请求，不据此推断正文去重分支的速度。长目录夹具减少约 95% 字符，完整原件仍可读取。cProfile 定位的主要热点是 Memory 导航复制完整卡片；现在无筛选目录复用已校验的轻量 ACL 字段，只为结果页复制正文，筛选检索及原件 hash 校验仍保留。历史上下文仍需约 17 秒；不以取消完整校验进一步提速。

目录扫描中间测得 1.57 秒，单次扫描改善不稳定；有效改动是从约每 0.1 秒循环请求改为最多每 5 秒周期扫描，并保留任务预留/新轮检查。状态写频率从每圈改为每秒，最终收尾不节流。没有声称数值推理或模型质量提升。

## 当前维护入口

| 规则 | 原权威维护位置 |
|---|---|
| 角色、任务与可见模型接口 | `agents/tasks.py`、`agents/roles.py`、`agents/prompts/{角色,tasks/任务}.md` |
| 科学协议、请求与规范化 | `docs/STATIC_W4A4.md`、`core/static_w4a4.py`、`tools/experiments.py`；说明 `docs/EXPERIMENT_INPUTS.md` |
| 主题/灵感/方法及恢复 | `core/topics.py`、`core/inspirations.py`、`agents/interaction.py`；说明 `docs/RESOURCE_LIFECYCLE.md` |
| 执行、资源、预算、源码身份 | `runtime/{driver,task_runtime,task_worker,budget,identity}.py`、`tools/frozen.py`；说明 `docs/EXECUTION.md` |
| 可选知识与 Skill | `knowledge/INDEX.md` → `docs/skills/INDEX.md`；能力边界 `docs/SCIENTIFIC_SKILLS_COMPATIBILITY.md` |
| 离线评价与历史适用范围 | `docs/EVAL_L0.md` → `configs/evaluation/<version>`；历史报告 `docs/HISTORY.md` |

[当前伪代码](PSEUDOCODE.md) 是实现导航；本报告记录变更和证据，不建立另一套科研规则。

## 验证与剩余边界

[真实模型回放与原输出](behavior_review.json) 共三个完成案例：

| 场景 | 实际行为与边界 |
|---|---|
| 低分/未入库灵感接续 | 真实 Generation 读取原 H3 完整实现，保留同一 active 灵感并提交待验证候选。低分是显式假想条件；实际历史 H3 在来源加载阶段失败，不能声称已有负面机制测量。 |
| 已有收益的解释与简化 | 读取真实 H2 代码及部分历史结果后选择 narrow，研究码器、目标、分组与共享的必要性。明确 round 2 数值尚未核对齐、不计算跨轮增益；一次 select 读取请求失败后仍正常提交。 |
| 主题暂存恢复 | 读取实际 H3 代码片段后选择 resume；原 action-1e50a3b5cdf8038a 从 deferred→open、主题 paused→active，ID 和动作数不变；不声称数值加载已验证。 |

三个完成案例共 9 次模型调用；另保留因 worker 归档身份比较缺陷而被拒收的解释案例首试 4 次，修复后另目录重放。总计 13 次，CLI 记录输入 397,997、输出 12,408 token，缓存输入 2,944；推理输出 3,007 是输出子字段，不重复相加。请求路由均为原 Generation 的 gpt-6-astra/high；CLI 未回显实际 model/effort，实际值记未知。每案例上限 4 调用、480 秒，没有扩充科研预算绕过失败。

状态种子、动作和低分条件是程序夹具；决策及工具调用是真实模型行为；原代码、原配置和旧 L3 数值是历史证据；本轮新数值任务和 Evidence 均为 0。三个完成案例都没有读取 Skill 或启动新实验，说明不额外设置这些步骤仍可提交。未用离线 Judge 打新的科学分数。解释案例的原 Evidence 很大，回放脚本保留原件并按页读，未读部分明确保留未知；回放耗时不能代替生产吞吐或模型质量统计。

程序验证：第一批 82 项、第二批 100 项及补充边界 2 项通过（有重叠，不相加当总覆盖数）；完整 766 项集成首轮发现 2 个失败、1 个错误、2 个 CUDA 跳过：正文去重丢失证据定位字段、预检夹具依赖本机 CLI 路径、旧异常名称优先断言。三项已修正，完整失败日志 [full_regression_final.log](full_regression_final.log) 保留；受影响的 54 项回归全部通过；[最终全量复验](full_regression_verified.log)为 **767 项、765 通过、2 跳过、0 失败/错误**，耗时 939.241 秒。跳过的两项为 AdaLog CPU↔CUDA/冻结包 CUDA 传输，因本轮显式 CUDA_VISIBLE_DEVICES=''；未冒称 GPU 验证通过。最初全量基线和第一轮集成运行被主动中断以进行修复，原日志保留，不计为全量通过；清洗前的定向 Eval-L0 基线确实复现两个载入错误。结构审计 209 Python 文件、378 源码成员通过；知识导航审计 21 个方法、77 页通过；历史评价校验结果见 [historical_eval_verify.json](historical_eval_verify.json)。

剩余限制及下一次验证条件：

- 本轮没有正式长贯通、完整 ImageNet/ADE20K 或新的大模型数值测试；现有 CPU 小模型/张量回归不证明新算法科研效果。来源隔离、综合失败恢复和 GPU 长等待主要由程序夹具覆盖，900 秒时限用受控时钟验证，没有占用设备实等。
- 对原历史运行升级恢复需按原流程明确记录 implementation intervention，并使用剩余预算；预算不足按既有授权机制处理，不重置计时。资源池为空时必须有有效的明确授权才能执行，resume 不自动增加设备权限。
- 在原 H3 上开展下一次获准局部验证时，应核对确切来源包、原实现及冻结签名后检验真实加载；此前的 CPU 接口通过不能替代这一实际数值链证据。随后是否开展更多机制实验由 Generation 决定，没有新增最低实验次数。
- 完整贯通前需确认授权 GPU/物理身份可用、数据/权重/head 及样本划分仍匹配、现有磁盘与预算可承受原双任务 L3。三候选先完成真实综合；待综合记录不能作为放行结论。
- v0.3 是仓库派生的未人工校准标准；17 秒历史上下文与首次权限构建仍有优化空间。本轮已处理实测热点，保留原件校验，没有继续扩展架构或删除低分/失败资产。

复查测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
python -B QAViT/tools/evaluation --rules-version gonly_v0_3 rules
```

运行现场、逐次失败、原始模型 audit/回答、代码与快照仍在本目录及 `live/`。Git 仅精确纳入报告、汇总、脚本、成本样本和检查日志；忽略的原始大文件没有删除。
