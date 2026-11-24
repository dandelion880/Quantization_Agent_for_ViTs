# 环境、依赖与只读资产

沿用本机已验证安装。环境使用 Python 3.10.8、torch 2.1.2+cu118、torchvision 0.16.2+cu118 和 jsonschema 4.22.0。Linux Landlock 用于生成代码写隔离。

当前已验证的订阅入口使用 Codex CLI 0.172.0 和现有 ChatGPT 登录。每次调用仍检查精确版本、模型目录、推理档位和工具隔离。角色分配见 `configs/models.json`，真实结构化响应和隔离验证见 [0.172.0 兼容记录](../../EXP/validation/cli172_sync_20261124_r1/cli172_compatibility.json)。历史运行如绑定了已移除的 CLI 路径，恢复时须通过显式干预记录传输变更，参见 `EXP/validation/cli166_resume_20261031_r1`。

`.runtime_pkgs/` 是随项目交付的必要第三方运行依赖，不是临时缓存。保留 common（包括 NumPy 1.26.4、MMEngine/MMCV）、classification（timm 0.3.2）和 segmentation（timm 1.0.20、MMSeg）全部现有内容。分类与分割两套 timm 按原逻辑隔离加载。精确版本、许可证相对位置及历史来源在 [runtime_dependencies.json](runtime_dependencies.json)；历史来源目录只用于溯源，不是当前 import 路径。官方 DeiT 源码与 LICENSE 保存在 `tools/model_sources/deit/`，源文件哈希仍与资产收据核对。未推断或添加未知来源的许可证授权。

`requirements.txt` 是环境记录。文献 PDF 功能需要可选 pypdf；本轮不安装、不访问文献网站。不同平台或依赖版本需要单独复核，导入成功不表示 GPU/模型组合验证完成。

| 只读输入 | 配置位置 | 解析基准 |
|---|---|---|
| DeiT checkpoint、数据清单、模型来源 | `configs/deitb_backend.json` 的 backend | 相对配置文件目录 |
| 模型构造源码、checkpoint 哈希 | `configs/assets/deit_model_manifest.json` | 相对资产清单目录 |
| ImageNet 划分、路径和标签 | `configs/assets/imagenet_manifest.json` | 按清单中原有绝对路径读取 |
| ADE 数据、预处理、head 收据 | `configs/ade_evaluation.json` | data_root 使用原路径；head_receipt 相对 configs |
| 冻结 ADE head | `configs/assets/ade_head.json` 的 checkpoint | 原有绝对路径，加载时核对 SHA256 |
| DINOv3 资产目录、官方构造器及冻结分类头 | `configs/dinov3b_backend.json` | 已有 model_assets_manifest 的选定记录、源码及 checkpoint SHA256 |
| DINOv3 分割任务 | `configs/dinov3_ade_evaluation.json`、`configs/assets/dinov3_ade_head.json` | 从原 MMSeg checkpoint 配置获得层/预处理；逐张量核对保存的骨干并 strict 加载 neck/head |
| 历史任务关联 | `configs/assets/legacy_task_bindings.json` | 只对确切的旧资产清单摘要和 model_id 恢复原 ADE 设置 |
| 通用模型 factory | ResearchBrief 的 model.adapter.factory | `文件路径:函数`；相对路径基于项目根；冻结源码资产哈希 |

母模型、任务头、官方源码和数据沿所选资产清单的原位置只读加载。数据集原位置不搬迁、不覆盖。新 factory 应使用自包含实现或当前运行批准的实验代码；框架不使用 EXP 中的一次性框架实现。

`configs/deitb_backend.json` 是实际数值工具的后端示例，不是科研策略模板。其 loop 中的工具选项（格式、校准缓存、方法能力、诊断资源与历史工具元数据）沿用现有值；V2 的研究轮次/全局墙钟以运行输入与 BudgetLedger 为准；数值工具原有活动时间检查和调用预留保护仍按原实现保留，具体边界列在本次报告，未顺带修改预算策略。模型与推理强度仅在 `configs/models.json` 和明确冻结的运行覆盖中生效，数值后端不再包含旧 LLM 设置。

`runtime/paths.py` 根据自身位置定位项目，并将本地工具库和依赖放入 sys.path；TMPDIR、Torch/HuggingFace 等缓存指向 `EXP/cache/tmp`。worker 由绝对路径启动，可从任意工作目录导入。所有测试建议使用 `python -B`，避免把字节码缓存写入源码目录。
