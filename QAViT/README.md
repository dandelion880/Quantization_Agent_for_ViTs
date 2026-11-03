# QAViT V2

QAViT 让 LLM 负责量化科研推理，程序负责实验执行、身份与来源、状态和归档。唯一入口是 `run.py`，配置协议为 `qavit.research/2`。保留七角色、H1/H2/H3 初始讨论、固定 H* 与 H4 的父子挑战、真实量化工具、独立代码验证及 L0–L3 Memory。

Recorder 只需提交科研分析、下一轮建议和未决问题；事实完整归档与科研综合分别标记。逐条来源覆盖和强制审计已由程序归档取代。调用方用精简草案与短引用表达科学选择，程序生成执行镜像、固定任务身份和可追溯记录。关键数值、安全与来源检查继续执行。

## 阅读顺序与修改入口

1. `core/workflow.py`：先读 `advance()`，再读 `progress_selection()`、`progress_match()`。H1/H2/H3 → 筛查 → H1–H2、H1–H3、H2–H3 逐场结案 → 持久化 H* → 仅加强 H* 形成 H4 → 筛查 → 唯一 H4–H* 辩论 → 父子最终选择 → 可选 L3 → Recorder → 下一轮（复盘失败保留待办与原始证据，允许继续）。结案不改判；position/arbitrate 每次刷新本轮公开科研正文并附完整历史索引。
2. `agents/tasks.py`、`agents/prompts/`、`agents/roles.py`：任务目的对应角色、输出、工具和片段；公共要求 + 角色要求 + 任务目的 + 提交协议组成最终 Prompt。
3. `core/experiment_requests.py`：实验申请、科学审批、冻结、候选代码与独立参考、数值验证、Safeguard、执行、解释。
4. `tools/experiments.py`、`tools/experiment_worker.py`：冻结执行输入、校验资产、构建 Evidence；原始测量和解释分别保存。
5. `core/memory.py`、`core/journal.py`、`core/cards.py`：事实、投影、知识修订及精确/词法检索；目前没有向量检索。
   `core/execution_contract.py`、`core/evidence_scope.py` 统一冻结执行约束和核验范围；`tools/result_contract.py` 为所有 L3 校验实际结果；`tools/runtime_interface.py` 覆盖自定义服务的实际 Data/Backend 接口。
6. `runtime/driver.py`、`task_queue.py`、`task_runtime.py`、`task_worker.py`、`budget.py`：任务启动、轮询、回执、重试、进程保护、恢复与预算。

| 修改内容 | 首要位置 |
|---|---|
| 科研阶段与实验申请 | `core/workflow.py`、`core/experiment_requests.py` |
| 辩论和排名 | `core/debate.py`；提交校验在 `agents/interaction.py` |
| Prompt 和任务映射 | `agents/prompts/`、`agents/tasks.py`、`agents/roles.py` |
| 上下文、工具结果与科研正文提交 | `agents/interaction.py` |
| Memory 与 incumbent | `core/memory.py` |
| 角色模型与推理强度 | `configs/models.json`；输入校验 `runtime/configuration.py` |
| Profiling | `tools/profiling.py`；纯工具会话 `tools/ptq_agent/profiling/host.py`；完整注册 `execution/profile_tools.py` |
| 量化算法、校准与重载 | `tools/ptq_agent/execution/`，见[数值库说明](tools/ptq_agent/README.md) |
| 分类、分割及模型适配 | `tools/evaluating.py`、`segmentation.py`、`vision_models.py` |
| Coding/Safeguard 与数值隔离 | `tools/safeguards.py`、`verification_worker.py`、`runtime/guards.py` |
| 订阅调用和审计 | `runtime/model_router.py`、`subscription.py`、`codex_cli.py` |
| 路径、缓存和源码身份 | `runtime/paths.py`、`runtime/identity.py` |

`Workflow` 拥有唯一的运行状态；三个有实质逻辑的职责分区 `RoleInteraction`、`ExperimentRequests`、`WorkflowDriver` 共享该状态，不各建一份队列或 Memory。主流程可直接读出科研步骤，运行循环在 driver。

## Prompt 与模型配置

`agents/prompts/common.md` 是公共要求，七个角色文件是角色原则，`tasks/*.md` 是本次任务要求。组装时保留片段内标题、段落和列表，只清理文件两端空白；14 类任务的最终 Prompt 有固定哈希对照。修改正文会改变身份，须记录明确差异。

默认模型只定义在 `configs/models.json`：Supervisor 为 gpt-6-luna/low，Recorder 为 gpt-6-luna/high；Generation、Evolution 为 gpt-6.1-sol/high，Judger、Coding、Safeguard 为 gpt-6.1-sol/medium。显式运行输入仍可提供完整 `llm.agent_models` 覆盖，生效配置冻结在 `research_run.json`；无自动模型回退。

## 环境与命令

环境、第三方版本和只读资产路径见 [ENVIRONMENT.md](docs/ENVIRONMENT.md)，协议见 [EXECUTION.md](docs/EXECUTION.md)，方法边界见 [METHODS.md](docs/METHODS.md)。入口不安装或升级依赖。

从项目顶层运行，离线验收不启动真实 Agent：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
python -B QAViT/tests/audit_structure.py --output EXP/validation/structure_audit.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B QAViT/tests/snapshot_contracts.py --output EXP/validation/cpu_contracts
python -B QAViT/run.py --help
```

输出位置必须是新的 EXP 文件/目录，不覆盖已有材料。测试临时文件在 `EXP/cache/tmp`，结束自动清理。`validate_code_service.py`、`validate_memory_service.py` 会调用真实 LLM，`validate_numeric.py --model deit` 会加载真实资产；它们是另外授权的验证入口，本轮未执行。

角色提示词、提交视图与工具输入均围绕科研工作精简；持久化卡片仍保留完整合同。新服务合同为 qavit.experiments/2.1，历史记录不补写、不绕过源码身份/审批绑定续跑。

未来科研运行先将 `configs/research.json` 模板复制到新的 EXP 输入目录并填写 ResearchBrief、适配器、协议、约束和预算。模板故意不能直接运行：

```bash
python -B QAViT/run.py start --config EXP/<新任务>/input.json --name <新运行名>
python -B QAViT/run.py status EXP/runs/<新运行名>
python -B QAViT/run.py resume EXP/runs/<新运行名>
python -B QAViT/run.py report EXP/runs/<新运行名>
```

`answer` 回答明确澄清，`resume --retry-failed 原因` 记录显式重试；`extend-budget --seconds ... --reason ...` 只在用户授权延长时使用。恢复不重置时间或任务历史。源码变化继续要求 `--intervention` 记录并复核冻结输入，不跳过哈希。

## 运行事实与历史

每个新运行的 research_run.json 保存有效输入、源码/Prompt/配置/接口快照；cards/ 与 commits/ 是事实源。jobs/ 保存请求、审计和回执，code/ 保存运行生成代码；checkpoint/status/summary 是派生视图。自有代码递归纳入身份，第三方安装清单独立记录，模型和数据以哈希绑定。所有运行产物属于 EXP，核心没有依赖历史实验实现。

报告区分流程完成、科研复盘完成与正式实验验证成功。没有实测的推断、随机模型 CPU 运算、子集测量及完整任务评测分别标记。Recorder 延后不等于知识已被审查；已结案裁决不可因后到证据改写。

当前修改依据、历史轨迹回放、低成本验证、复杂度比较与剩余问题见[科研简化报告](../EXP/validation/research_refactor_20261103_r1/REPORT.md)，流程见[当前伪代码](../EXP/validation/research_refactor_20261103_r1/PSEUDOCODE.md)。历史报告、运行、修复材料只读保留在 EXP；旧运行不自动迁移或绕过身份检查续跑。
