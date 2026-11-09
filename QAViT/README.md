# QAViT V2

QAViT 让 LLM 负责量化科研推理，程序负责数值执行、身份与来源、状态和归档。科研运行入口 run.py，研究配置协议 qavit.research/2。保留七角色、Generation/Evolution 自主研究循环、并行 Judger、Coding 单实现→Safeguard→执行、Recorder、Supervisor 及既有预算/GPU 调度。

## 阅读与修改入口

- [科研资源统一导航](knowledge/INDEX.md)：Prompt、Contract、Skill、Method、Recipe、Memory/Evidence、Toolbox 的职责、权威位置与维护规则。
- [Eval-L0 离线评价](docs/EVAL_L0.md)：独立读取历史轨迹、统计资源、按六类角色评分；入口 `python -B QAViT/tools/evaluation --help`，不进入科研门禁。
- [实际目录与模块](../STRUCTURE.md)：当前文件分工；[完整文件盘点与调用链](../EXP/validation/knowledge_resources_20261108_r1/PROJECT_RESOURCE_INVENTORY.md) 是本轮真实工作区快照。
- [执行协议](docs/EXECUTION.md)、[实验输入速查](docs/EXPERIMENT_INPUTS.md)、[代码接口](docs/CODE_INTERFACES.md)、[static W4A4 合同](docs/STATIC_W4A4.md)：接口说明与实现共同维护。
- [资源生命周期](docs/RESOURCE_LIFECYCLE.md)：按需读取、来源、运行 Memory、跨运行方案库边界；[Skill 接入](docs/SCIENTIFIC_SKILLS.md) 与 [方法能力边界](docs/METHODS.md)。
- [历史报告导航](docs/HISTORY.md)：各次验证的适用版本和原件入口；历史成绩不代表当前协议验证。

## 主调用链

core/workflow.py 的 advance/progress_selection/progress_match 组织 H1/H2/H3→筛查→三场初始辩论分别并行推进并全部结案→持久化 H*→Evolution 仅加强 H* 得到 H4→筛查→唯一 H4–H* 挑战→最终选择→可选 L3→Recorder→下一轮。结案不改判，复盘失败保留待办与原始证据。

core/experiment_requests.py 保存研究者原始实验需求，必要时交给 Coding 单实现，再绑定执行选择、Safeguard 审查并执行；证据返回原研究者继续判断。无需新增测量时可以直接提交。Skill、文档和方案种子仅辅助选择，不改变流程或限制 H3。

agents/tasks.py、roles.py、prompts/ 定义任务/角色/工具及提示词；agents/interaction.py 组织上下文、授权读取、去重和提交绑定。core/memory.py、journal.py、cards.py 维护运行事实与知识修订；core/solutions.py 单独维护协议完整方案。runtime/driver.py、task_queue.py、task_runtime.py、budget.py、gpu_resources.py 管启动、恢复、资源与回执。tools/ 下是真实数值执行实现，tools/ptq_agent/ 是纯数值库，无旧 Agent 框架运行依赖。

## 环境与运行

先按 [ENVIRONMENT.md](docs/ENVIRONMENT.md) 准备依赖、模型服务与只读资产。入口不安装或升级依赖。角色模型默认值唯一维护于 configs/models.json；输入可完整覆盖，生效配置冻结于 research_run.json，不自动回退。运行中源码、Prompt 或知识变化仍遵守既有身份与显式 intervention 恢复规则。

将 configs/research.json 复制到新的 EXP 目录并填写 ResearchBrief、评测协议、约束、资源及预算。模板故意不能直接运行。从项目根目录执行：

```bash
python -B QAViT/run.py start --config EXP/<新任务>/input.json --name <新运行名>
python -B QAViT/run.py status EXP/runs/<新运行名>
python -B QAViT/run.py resume EXP/runs/<新运行名>
python -B QAViT/run.py report EXP/runs/<新运行名>
python -B QAViT/run.py --help
```

新输入支持 active_segments、exclude_numeric_wait 和允许 GPU 池；历史运行不静默改配置。具体预算、恢复、异常与资源行为见 EXECUTION.md；已有真实实验是否完成以各次运行原始记录为准。

## 离线验证与资料编辑

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
python -B QAViT/tests/audit_knowledge.py
python -B QAViT/tests/audit_structure.py --output EXP/validation/<新任务>/structure_audit.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B QAViT/tests/snapshot_contracts.py --output EXP/validation/<新任务>/cpu_contracts
```

测试临时文件写 EXP/cache/tmp 并自动清理；报告/快照须用新的 EXP 路径。更新 Skill frontmatter 后用 audit_knowledge.py --write-skill-index 生成触发目录，普通检查不写文件。Prompt 修改需更新 tests/data/prompt_readiness_baseline.json 的明确差异。

validate_code_service.py、validate_memory_service.py 会调用真实 LLM；validate_numeric.py 会加载真实模型资产，均须按当次任务范围与预算使用。历史验证不等于当前完整 GPU 科研性能基线。本轮整理与验证见 [交付报告](../EXP/validation/knowledge_resources_20261108_r1/REPORT.md)。
