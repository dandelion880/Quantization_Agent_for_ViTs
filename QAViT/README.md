# QAViT V2

QAViT 让 LLM 负责量化科研推理，程序负责数值执行、身份与来源、状态和归档。科研运行入口 run.py，研究配置协议 qavit.research/2。main 的 Generation（gpt-6.2-astra/high）负责主题、候选版本和最终 L3，Critic（gpt-6.2-astra/medium）对当轮提交的候选做一次可恢复的批量审阅。`enable_critic: false` 可关闭审阅。A/B 对照位于 `gonly-skills-20261124`，旧 Full 位于 `full-retired-20261124`。

## 阅读与修改入口

- [科研资源统一导航](knowledge/INDEX.md)：Prompt、Contract、Skill、Method、Recipe、Memory/Evidence、Toolbox 的职责、权威位置与维护规则。
- [Eval-L0 离线评价](docs/EVAL_L0.md)：独立读取历史轨迹、统计资源、按六类角色评分；入口 `python -B QAViT/tools/evaluation --help`，不进入科研门禁。
- [实际目录与模块](../STRUCTURE.md)：当前文件分工；[完整文件盘点与调用链](../EXP/validation/knowledge_resources_20261108_r1/PROJECT_RESOURCE_INVENTORY.md) 是本轮真实工作区快照。
- [执行协议](docs/EXECUTION.md)、[实验输入速查](docs/EXPERIMENT_INPUTS.md)、[代码接口](docs/CODE_INTERFACES.md)、[static W4A4 合同](docs/STATIC_W4A4.md)：接口说明与实现共同维护。
- [资源生命周期](docs/RESOURCE_LIFECYCLE.md)：按需读取、来源、运行 Memory、跨运行方案库边界；[Skill 接入](docs/SCIENTIFIC_SKILLS.md) 与 [方法能力边界](docs/METHODS.md)。
- [历史报告导航](docs/HISTORY.md)：各次验证的适用版本和原件入口；历史成绩不代表当前协议验证。

## 主调用链

core/workflow.py 的 advance/progress_preselection 组织 Generation 共同主题恢复/决定→H1/H2/H3 自主实验与修订→可选 Critic 批量审阅→Generation 横向综合与科学修订→H1/H2/H3 各自绑定确切版本的 L3 规划与执行→Recorder→下一轮。不发布 Judger、Evolution 或 H4。L1/L2 不代替正式评价；有效 ImageNet + ADE20K L3 由程序按原协议更新 Top-3。局部失败保留已完成候选，综合失败保存确切候选和待做 L3，本轮保持未完成，允许原任务按原预算恢复；独立 B 保留；单候选执行失败不阻断其他候选，部分成功不算全轮完成；复盘失败保留原始证据并允许继续。

core/experiment_requests.py 保存研究者原始实验需求，必要时交给 Coding 单实现，再绑定执行选择、Safeguard 审查并执行；证据返回原研究者继续判断。无需新增测量时可以直接提交。Skill、文档和方案种子仅辅助选择，不改变流程或限制 H3。

agents/tasks.py、roles.py、prompts/ 定义任务/角色/工具及提示词；agents/interaction.py 组织上下文、授权读取、去重和提交绑定。core/memory.py、journal.py、cards.py 维护运行事实与知识修订；core/solutions.py 单独维护协议完整方案；core/inspirations.py 基于 Journal 保存独立灵感及跨轮恢复引用，不复制完整方案。runtime/driver.py、task_queue.py、task_runtime.py、budget.py、gpu_resources.py 管启动、恢复、资源与回执。tools/ 下是真实数值执行实现，tools/ptq_agent/ 是纯数值库，无旧 Agent 框架运行依赖。

## 环境与运行

先按 [ENVIRONMENT.md](docs/ENVIRONMENT.md) 准备依赖、模型服务与只读资产。入口不安装或升级依赖。角色模型默认值唯一维护于 configs/models.json；Generation 默认使用 gpt-6.2-astra/high，显式配置可选已核验的 Astra/high 路由，其他角色也可显式覆盖，生效配置冻结于 research_run.json，不自动回退。运行中源码、Prompt 或知识变化仍遵守既有身份与显式 intervention 恢复规则。

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

validate_code_service.py、validate_memory_service.py 会调用真实 LLM；validate_numeric.py 会加载真实模型资产，均须按当次任务范围与预算使用。历史验证不等于当前完整 GPU 科研性能基线。初版 G-only 验证见 [交付报告](../EXP/validation/gonly_20261110_r1/REPORT.md)。

H3 默认持续发展一条活跃灵感，按确切代码、配置、结果和未完成事项接续；低分或未入 Top-3 不自动结束方向。Generation 决定继续/暂停/切换，Recorder 更新进展与引用；H1/H2 保留各自完整父方案及可回退增量，独立 B 和三候选 L3 不变。接口与边界见 [资源生命周期](docs/RESOURCE_LIFECYCLE.md#独立灵感库与跨轮接续)。当前离线检查和伪代码见 [本轮报告](../EXP/validation/inspiration_continuity_20261117_r1/REPORT.md)。

动态主题与方法谱系见 [资源生命周期](docs/RESOURCE_LIFECYCLE.md#动态研究主题与方法谱系)；每轮先恢复再由 Generation 共用选题，轮中改变追加版本，不限制原全局搜索、历史父方案或 B。该版伪代码和针对性状态/真实 Generation 行为回放见 [本轮记录](../EXP/validation/research_topics_20261117_r1/REPORT.md)。

工程清洗、恢复边界、成本实测和回放验证见 [工程报告](../EXP/validation/engineering_cleanup_20261117_r1/REPORT.md)；最新伪代码见 [当前流程](../EXP/validation/decision_20261121/PSEUDOCODE.md)。规则仍在上述原维护入口，旧报告仅描述当时版本。

模型输入与绑定示例见 [环境与资产](docs/ENVIRONMENT.md) 和 [DINOv3 独立开发集研究输入](../EXP/inputs/dinov3b_static_w4a4_independent_dev/input.json)。新运行的 model_metadata.json 保存实测块调用、token 布局、量化边与任务特征入口；各角色默认接收摘要。

下一轮 focus 和 generate 从 memory.previous_candidate_results 恢复上一轮逐候选实测及其条件。DINOv3-B 轮内使用 L3-dev，90% 目标按最终完整官方验证的各任务 FP 核对。[科研决策回放与边界](../EXP/validation/decision_20261121/REPORT.md)
