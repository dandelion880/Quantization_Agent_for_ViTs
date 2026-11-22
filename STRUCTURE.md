# 项目结构与修改入口

五项业务入口保持 QAViT/、EXP/、reference/、AGENTS.md、STRUCTURE.md；根 README 提供项目概览，.gitignore 管版本边界。以下是模块导航；**逐文件的真实完整结构**及省略范围见 [本轮盘点报告](EXP/validation/knowledge_resources_20261108_r1/PROJECT_RESOURCE_INVENTORY.md)，由报告同目录的 build_inventory.py 扫描工作区生成，不是理想目录。

| 实际位置 | 职责 |
|---|---|
| QAViT/run.py | 科研运行 CLI：start/resume/status/report/answer/extend-budget |
| QAViT/agents/prompts/ | common、当前角色/任务及历史兼容 Prompt 片段；tasks.py/roles.py 映射和组装 |
| QAViT/agents/interaction.py | 任务上下文、读取权限、分页/去重、来源与提交绑定 |
| QAViT/knowledge/INDEX.md | 统一静态科研知识导航与维护入口，正文按需读取 |
| QAViT/docs/skills/ | 七项按需 Skill、16 份包内参考及保留的冻结传播 API 参考，INDEX 触发表由 frontmatter 生成 |
| QAViT/knowledge/research-skills/ | 共同科研约定、六条完整研究路线及来源；按需 read_asset |
| QAViT/docs/methods/ | 原位保留的组件手册及 index.json；read_asset 为常规阅读入口 |
| QAViT/docs/solutions/ | 原位保留的完整方案种子及说明，不是实测排名文件 |
| QAViT/docs/references/ | PTQ 方法来源、版本、许可与历史核验线索 |
| QAViT/docs/ | 系统/执行合同、实验/代码接口、环境、资源生命周期与历史导航 |
| QAViT/core/workflow.py、experiment_requests.py | G-only 自主研究/综合修订/多候选L3/轮次推进与实验请求；保留现有职责，不拆分大型文件 |
| QAViT/core/cards.py、journal.py、memory.py、evidence.py | 合同、不可变事实、运行内知识修订、观测展示及检索 |
| QAViT/core/topics.py | Journal 上的动态问题/范围/复议信息、方法谱系与确切候选关联；Generation 决定，宿主不作科学打分 |
| QAViT/core/inspirations.py | 独立灵感的 Journal 修订、活跃状态与代码/配置/结果恢复投影；不参与 Top-3 或实验调度 |
| QAViT/core/solutions.py、static_w4a4.py | 协议专属完整方案库与必要数值边界 |
| QAViT/core/debate.py、execution_contract.py、evidence_scope.py | Full 历史裁决表兼容、冻结执行约束及证据适用范围 |
| QAViT/runtime/ | driver/队列/worker、订阅传输、预算/GPU/运行段、保护及源码身份 |
| QAViT/tools/ | 文献、实验服务、数值 worker、评测、代码审查与模型适配 |
| QAViT/tools/evaluation/ | 独立 Eval-L0 离线 CLI；只读历史运行，计量/受限 Judge/评分校验/报告，不被科研主流程导入 |
| QAViT/configs/evaluation/ | v0_1 为上传原件、gonly_v0_2 为冻结旧派生包、gonly_v0_3 为当前派生标准；均与科研自动授权隔离，版本与使用入口见 QAViT/docs/EVAL_L0.md |
| QAViT/tests/evaluation/ | Eval-L0 CPU 接口、来源、时间边界、隔离与恢复测试；不生成科研代表性样本 |
| QAViT/tools/ptq_agent/ | 纯数值实现、会话快照和工具注册；execution/baselines.py 与 recipes.py 保留实际构造器 |
| QAViT/tools/model_sources/ | 固定模型构造源码与既有许可证 |
| QAViT/configs/ | 通用研究模板、角色默认模型、数值/评测配置和只读资产清单 |
| QAViT/tests/ | 单元/离线集成回归、静态审计、CPU 快照及显式真实组件验证器；data/ 为小型基线，skill_evals/ 为评估设计 |
| EXP/runs/、inputs/、outputs/ | 历史及未来具体运行、输入、产物；Journal/Memory/Evidence 均属于各运行 |
| EXP/validation/、implementation/ | 逐次验证/实施报告、日志、源码快照及复现材料；旧目录不充当当前执行实现 |
| EXP/evaluations/ | 独立评价冻结原件、逐任务案例、Judge 回执、分项/聚合报告及本次接入验证；不回写原科研运行 |
| EXP/cache/ | 临时文件、依赖缓存和锁，不属于源码身份 |
| reference/ | 当前 coscientist 两份原始 PDF；不自动注入角色 |

`EXP/solutions/<protocol_hash>/` 是当前已有的跨运行共享方案库，按冻结协议分桶。独立验证可使用自己的 solutions/；完整方案、待测种子和独立 B 按原资格区分。不为目录美观预建空库或移动历史桶。

物理知识路径保留以兼容已有 read_asset、数值会话文档快照、源码身份和历史来源；新增知识优先归入 knowledge/，合理依赖例外见 AGENTS.md。全部资源分类、权威位置、使用角色、发现和维护规则只在 [统一导航](QAViT/knowledge/INDEX.md) 集中说明；[生命周期](QAViT/docs/RESOURCE_LIFECYCLE.md) 解释实际调用与沉淀。

Git 跟踪稳定源码、Prompt、知识、轻量配置、测试及许可证；EXP 仅按 .gitignore 的精确路径保存已检查的轻量交付，原始运行/快照/日志留本机。reference/、.runtime_pkgs/、完整 ImageNet 清单、数据、权重和凭据不入库。忽略不代表删除；环境与本地依赖见 [ENVIRONMENT](QAViT/docs/ENVIRONMENT.md)。历史导航集中在 [HISTORY](QAViT/docs/HISTORY.md)，不在这里追加每次实施日志。

当前流程见[科研决策与开发评价伪代码](EXP/validation/decision_20261121/PSEUDOCODE.md)。工程清洗验证及成本实测见[历史报告](EXP/validation/engineering_cleanup_20261117_r1/REPORT.md)。实现继续使用原 workflow/experiment_requests、Journal、Memory、主题/灵感库和调度链；没有拆分架构或删除历史实现。源码执行身份与完整归档身份分别由 runtime/identity.py 维护；研究角色通过 interaction.read_snapshot_asset 读取确切成员。历史评分包保持原件，configs/evaluation/gonly_v0_3 是当前独立派生评价版本。

之前的三轮科研、能力、Skill、灵感和主题接入报告均保留原位置；适用版本与入口见 [历史导航](QAViT/docs/HISTORY.md)。EXP 的运行/快照仍为历史原件，当前实现不从旧 EXP 导入框架代码。

新运行的 model_metadata.json 由 tools/profiling.py 复用后端目录和结构 profile 生成。tools/vision_models.py 处理源构造器与任务特征差异，模型与任务配置归 configs/，阶段目标见 [DINOv3 研究输入](EXP/inputs/dinov3b_static_w4a4/input.json)。

独立开发评价的数据适配位于 `QAViT/tools/development.py`，Coding 文件修订与原生通道分别位于 `QAViT/agents/source_workspace.py` 和 `QAViT/runtime/coding_native.py`。按需工程资料见 `QAViT/knowledge/coding/`。开发协议验证与数据清单生成入口位于 `EXP/validation/l3_dev_20261120/`。

Eval-L0 的行为回放入口位于 `QAViT/tools/evaluation/breakpoint.py`；真实案例、按组件运行的 CPU 回归和短链路验证见[断点评测集合](EXP/validation/engineering_replay_20261122/REPORT.md)。[覆盖矩阵](EXP/validation/engineering_replay_20261122/COVERAGE.md)记录各阶段可验证的职责及缺口。
