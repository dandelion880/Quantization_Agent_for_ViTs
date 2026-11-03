# 项目结构与修改入口

业务结构保持 `QAViT/`、`EXP/`、`reference/`、`AGENTS.md`、`STRUCTURE.md` 五项。根目录 `README.md` 提供项目概览与使用入口；`.git/` 和 `.gitignore` 用于版本管理。

Git 管理 QAViT 的稳定代码、Prompt、工具、轻量配置、文档和测试，包括 `tests/data/` 的小型夹具与模型构造源码的现有许可证。新增源码默认可跟踪，不对 QAViT 设置逐文件白名单。

EXP 严格按 `.gitignore` 中的准确路径放行：`implementation/v2_structure_20261027/REPORT.md`，以及 `validation/research_refactor_20261103_r1/` 下的 `REPORT.md`、`PSEUDOCODE.md`、`DELIVERY.json`、`REPLAY_RESULT.json`、`complexity.json`。新增实验配置、指标或总结须先检查内容，再显式增加路径；原始日志、运行记录、快照和大型产物保留在本机。

`QAViT/.runtime_pkgs/`、`QAViT/configs/assets/imagenet_manifest.json`、`reference/`、权重、数据集、缓存及凭据不入库；忽略不删除或移动任何本地资产。仓库克隆不包含完整运行环境，依赖及外部资产仍按 [环境说明](QAViT/docs/ENVIRONMENT.md) 准备。配置和来源记录中的本机路径保留原义，历史报告引用的原始证据也可能仅在本机可用。提交前检查暂存内容中的凭据与不适合公开的信息。

| 位置 | 当前职责 |
|---|---|
| `QAViT/run.py` | 唯一 V2 CLI：start/resume/status/report/answer/extend-budget |
| `QAViT/core/workflow.py` | 科研阶段、初始三场依序结案→持久化 H*→唯一父子场的两阶段选择和轮次推进；结案不改判 |
| `QAViT/core/experiment_requests.py` | 实验审批、Coding、独立参考、Safeguard、执行；提交判断卡的原 Judger 主动申请后直接接收结果，其余沿独立解释链 |
| `QAViT/core/cards.py`、`journal.py` | 卡片合同、不可变事实、原子提交与重建 |
| `QAViT/core/execution_contract.py`、`evidence_scope.py` | 结构化必需执行约束归并/独立核对，Evidence 规范范围、知识边界及精确字段匹配 |
| `QAViT/core/debate.py`、`memory.py` | 直接胜负表、L0–L3 Memory、程序来源归档/科学综合状态与检索 |
| `QAViT/agents/tasks.py`、`roles.py`、`prompts/` | 任务映射、工具 Schema、公共/角色/任务 Prompt 组装 |
| `QAViT/agents/interaction.py` | 按 purpose 交付必需对象、分页目录、统一提交可见性与读取权限、工具响应去重和科研提交校验 |
| `QAViT/runtime/` | driver、队列、worker 生命周期、预算、订阅传输、路径、保护、递归身份 |
| `QAViT/tools/` | 实验协议、数值 worker、评测、Safeguard、文献和模型适配 |
| `QAViT/tools/result_contract.py`、`runtime_interface.py` | 全服务 L3 实际结果协议、自定义 Data/Backend 宿主观测及 CPU 接口夹具 |
| `QAViT/tools/data_receipt.py` | 从已有后端配置/清单及匹配校验身份派生实验数据回执；不加载模型、不代选科学参数 |
| `QAViT/tools/ptq_agent/` | 纯数值工具库；无旧 Agent/Orchestrator/Memory/LLM 框架 |
| `QAViT/tools/model_sources/`、`.runtime_pkgs/` | 保留的模型构造源码、第三方依赖及许可证 |
| `QAViT/configs/` | 通用研究模板、唯一角色模型默认值、数值后端和只读资产配置 |
| `QAViT/tests/` | 可复用离线测试、小型固定夹具与显式组件验证器；`test_minimal_fixes.py` 覆盖选择、L3 复盘/状态和 Profiling 证据读取；`test_experiment_preflight.py` 覆盖草案预检及原任务有限修复；`test_parent_evolution.py` 覆盖 H* 绑定、父子选择、恢复及下游交接；`test_serial_debate_memory.py` 覆盖按场推进、科研正文/索引、隔离和结案固定；输出写 EXP |
| `QAViT/docs/` | 当前执行协议；EXPERIMENT_INPUTS.md 为直接交付的科学调用速查；CODE_INTERFACES.md 为独立代码验证接口；环境依赖和方法说明 |
| `EXP/runs/`、`EXP/validation/` | 历史/未来运行事实、测量、代码、日志和模型产物 |
| `EXP/validation/serial_debate_memory_20261029_r1/` | 本批最小修改报告、独立提示词差异、当前流程伪代码与针对性 CPU 回归日志 |
| `EXP/validation/experiment_inputs_20261029_r1/` | 输入整理报告、数据回执核对、提示词差异、伪代码及 CPU 回归日志 |
| `EXP/validation/contract_interfaces_20261030_r1/` | 本轮合同/接口修复前复现、源码差异、当前伪代码、CPU 正反例及历史保护核对 |
| `EXP/implementation/v2_structure_20261027/` | 本次唯一整理报告、流程伪代码、原件快照、变更清单和验证日志 |
| `EXP/implementation/` 其它目录 | 历史实施事实，只读保留 |
| `EXP/cache/` | 临时文件、依赖缓存、锁；不属于源码身份 |
| `reference/` | coscientist 原论文和补充材料 |

从 [QAViT/README.md](QAViT/README.md) 的阅读顺序进入。科研主循环、实验申请、角色交互和运行设施已分开；当前实现及验证边界在 README，具体未修复问题和历史索引在[整理报告](EXP/implementation/v2_structure_20261027/REPORT.md)。

框架只依赖 QAViT 内实现和明确只读外部资产。实验生成代码可以来自当次 EXP 运行；不得从历史实施目录导入一次性框架。MVP 入口、五角色提示词、旧工作流与专用配置已删除，不保留 legacy 目录或兼容分支。数值计划 v2/v3、卡片协议等仍有实际用途的版本标识保持原义。

历史 cards、commits、日志、快照和结果不迁移、不覆盖。整理后的版本创建新运行；其自身的恢复、预算延续、哈希检查与证据追踪保留。本次结构工作没有启动真实科研任务，不代表科研问题已解决。

`EXP/validation/memfs_delivery_20261030_r1/`：Letta 官方固定版本参考、修改前快照与补丁、各 purpose 输入示例及体量对比、Recorder 增量整理/恢复和读取隔离 CPU 回归。

`EXP/validation/research_simplification_20261031_r1/`：本轮科研导向最小简化报告、调用链及当前伪代码、历史输入的只读性能复现、CPU 回归与同请求四服务集成对照。历史实验资产不覆盖。

`EXP/validation/real_research_tiny_20261031_r1/`：独立 CPU tiny ViT 真实 Judger 科研验证、两次实测 Evidence、Recorder/Generation 链路状态、完整原始轨迹、局部修复及回归。所有预算与源码干预只属于本次验证；失败材料保留。

`EXP/validation/research_refactor_20261103_r1/`：当前科研简化报告、修改前快照、源码差异、当前伪代码、历史 Recorder 隔离回放、CPU 四服务集成与回归；所有角色脚本夹具和真实 LLM 调用明确区分。
