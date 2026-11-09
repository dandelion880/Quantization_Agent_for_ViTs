# 项目结构与修改入口

五项业务入口保持 QAViT/、EXP/、reference/、AGENTS.md、STRUCTURE.md；根 README 提供项目概览，.gitignore 管版本边界。以下是模块导航；**逐文件的真实完整结构**及省略范围见 [本轮盘点报告](EXP/validation/knowledge_resources_20261108_r1/PROJECT_RESOURCE_INVENTORY.md)，由报告同目录的 build_inventory.py 扫描工作区生成，不是理想目录。

| 实际位置 | 职责 |
|---|---|
| QAViT/run.py | 科研运行 CLI：start/resume/status/report/answer/extend-budget |
| QAViT/agents/prompts/ | common、七角色、任务目的片段；tasks.py/roles.py 映射和组装 |
| QAViT/agents/interaction.py | 任务上下文、读取权限、分页/去重、来源与提交绑定 |
| QAViT/knowledge/INDEX.md | 统一静态科研知识导航与维护入口，正文按需读取 |
| QAViT/docs/skills/ | 原位保留的四 Skill 与九份 references，INDEX 触发表由 frontmatter 生成 |
| QAViT/docs/methods/ | 原位保留的组件手册及 index.json；read_asset 为常规阅读入口 |
| QAViT/docs/solutions/ | 原位保留的完整方案种子及说明，不是实测排名文件 |
| QAViT/docs/references/ | PTQ 方法来源、版本、许可与历史核验线索 |
| QAViT/docs/ | 系统/执行合同、实验/代码接口、环境、资源生命周期与历史导航 |
| QAViT/core/workflow.py、experiment_requests.py | 自主研究/两阶段选择/轮次推进与实验请求；保留现有职责，不拆分大型文件 |
| QAViT/core/cards.py、journal.py、memory.py、evidence.py | 合同、不可变事实、运行内知识修订、观测展示及检索 |
| QAViT/core/solutions.py、static_w4a4.py | 协议专属完整方案库与必要数值边界 |
| QAViT/core/debate.py、execution_contract.py、evidence_scope.py | 裁决表、冻结执行约束及证据适用范围 |
| QAViT/runtime/ | driver/队列/worker、订阅传输、预算/GPU/运行段、保护及源码身份 |
| QAViT/tools/ | 文献、实验服务、数值 worker、评测、代码审查与模型适配 |
| QAViT/tools/evaluation/ | 独立 Eval-L0 离线 CLI；只读历史运行，计量/受限 Judge/评分校验/报告，不被科研主流程导入 |
| QAViT/configs/evaluation/v0_1/ | 用户上传评分包的唯一原文及哈希来源；避开 docs/knowledge 的科研自动授权范围，详见 QAViT/docs/EVAL_L0.md |
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

`EXP/solutions/<protocol_hash>/` 是已实现的研究运行共享方案库路径，只有需要时创建；本轮整理开始时尚不存在此顶层目录。验证运行的 solutions/ 位于自己的目录。不能为美观预建空库或移动历史桶。完整树仅列实际存在的目录。

物理知识路径保留以兼容已有 read_asset、数值会话文档快照、源码身份和历史来源；新增知识优先归入 knowledge/，合理依赖例外见 AGENTS.md。全部资源分类、权威位置、使用角色、发现和维护规则只在 [统一导航](QAViT/knowledge/INDEX.md) 集中说明；[生命周期](QAViT/docs/RESOURCE_LIFECYCLE.md) 解释实际调用与沉淀。

Git 跟踪稳定源码、Prompt、知识、轻量配置、测试及许可证；EXP 仅按 .gitignore 的精确路径保存已检查的轻量交付，原始运行/快照/日志留本机。reference/、.runtime_pkgs/、完整 ImageNet 清单、数据、权重和凭据不入库。忽略不代表删除；环境与本地依赖见 [ENVIRONMENT](QAViT/docs/ENVIRONMENT.md)。历史导航集中在 [HISTORY](QAViT/docs/HISTORY.md)，不在这里追加每次实施日志。
