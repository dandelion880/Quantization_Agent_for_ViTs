# QAViT PTQ 主线接入与验证

工作区为 `Quantization_Agent_for_ViTs` 的 `main`，接入基线为 `de7310e`。本轮产物归入此目录。 从公共请求经准备、导出、评分至确切版本恢复，各环节的实测记录见下表。输入文件清单、原字节数、SHA-256 和去向均可复核。[上传清单](PACKAGE_MANIFEST.json) · [材料映射](source_mapping.json) · [校验收据](asset_ingest.json) · [续跑记录](PROGRESS.md)

## A–E 交付

| 工作包 | 主线改动 | 本轮证据 |
|---|---|---|
| A　协议与来源 | `static_w4a4/2` 将构造过程和推理要求分别验证。计算边、真实位宽、四位码容量、输入相关选择、额外计算与覆盖例外贯穿准备、导出、重载、复用和比较。历史 v1 依据原规则恢复。原文、来源版本、组件、运行谱系和 Skill 分别有读取入口。 | [83 条来源去向](../../../QAViT/knowledge/ptq/source_coverage.json) · [严格协议实测](primary_protocol.json) |
| B　数值与配方 | 在现有 executor 接入固定轴网格、非均匀码本、配对统计、逐阶段重构、补偿、搜索、合成校准、离线分配及结构状态。源请求装配成可编辑的普通方案。优化及结构状态随模型保存。 | [52 个来源版本的入口、依赖、差异与请求](../../../QAViT/knowledge/ptq/implementation_status.json) · [组件索引](../../../QAViT/knowledge/ptq/component_status.json) · [跨进程重载](numerical/process_cold.json) |
| C　metric、任务与可比性 | preparation 保留原始输出、导数与依赖身份；冻结候选用各自完整前后缀评分。分类和 dense 绑定真实 head、输入、窗口和 mask。选择分析对独立原图重新抽样，可按 confusion 求和重算 mIoU；准备与评分成本独立记录。父包、历史引用、证据复用分别处理。 | [分类敏感度](task_metrics.json) · [dense 准备](dense_sensitivity.json) · [联合校准与同包输出](joint_views.json) · [选择质量](selection_quality.json) · [原图混淆计数](per_original_dense.json) |
| D　科研角色与 Skill | 活跃角色只常驻协议、实际能力与阅读入口。7 个既有 Skill 修订，4 个新流程进入同一检索和版本体系。主题组合八维可变量；Generation 在 Critic 反馈后决策。H1/H2 保留完整父方案与回退增量，H3 接续灵感，独立 B、各自 L3 与准确版本恢复继续由宿主管理。 | [科研 Skill 总目录](../../../QAViT/docs/skills/INDEX.md) · [实际 Generation 读取和判断](generation.json) · [Recorder 提交](recorder.json) · [精确记忆恢复](memory_restore.json) |
| E　集成验收 | 沿原有 tiny native fixture、权限边界、公共 worker、冻结比较与 Journal 执行。每个声称可调用的来源版本都有其自身的阶段请求和重载记录。 | [数值汇总](numerical/summary.json) · [公共 worker 三步](service_evidence.json) · [最终回归](verification.json) |

## 方法、组件及来源关系

账本保留 51 个 PTQ 主条目、5 个 PTQ 原配方来源缺口、14 个 QAT 排除、6 个非 PTQ 排除、5 个边界未决、1 个家族版本和 1 个算子旁证，共 83 条。ERQ TPAMI 2025/DUQ 是已收录 ICML 条目的后续版本。静态方法目录共有 72 个身份，包含原有的量化器、变换等资产；目录数不能用作独立 baseline 数。排除记录只含身份、出处、分流类别与原因。[统一来源账本](../../../QAViT/knowledge/ptq/source_coverage.json) · [实际方法目录](../../../QAViT/docs/methods/index.json)

逐方法账目中的 `source_version` 和 `source_unresolved` 给出处与材料的原始审计状态；`current_conditions`、`source_requests`、`verification` 给当前适配的条件、来源配置与行为收据。正文保留数学定义、张量布局、顺序、作者预算、例外和版本差异。组件账目按导出方案的 W、A、observer、变换、postfit、分配和结构标出使用者；论文中拟议的诊断保留其研究身份。角色可用目录返回的 `read_asset` 读取分段长文。已有运行的 `method_id` 通过 `read_method` 回到实际谱系。[逐版本账目](../../../QAViT/knowledge/ptq/implementation_status.json) · [逐组件账目](../../../QAViT/knowledge/ptq/component_status.json) · [阶段和依赖映射](../../../QAViT/knowledge/ptq/08_METHOD_COMPONENT_MAP.md)

## 数值及科研行为

- 77 组缩小的数值请求分别准备、导出，并在独立 Python 进程使用 `FrozenData` 重载。校验对比保留输入的输出哈希及实际执行状态哈希。三个已有作者分类图另有有界调用。FQ-ViT 在本轮小模型 W4 时遇到原整数 LN 零方差，成功收据为 W8A8。压缩包中的作者表格、旧工程的历史成绩和这些测量分别保留。[数值请求及汇总](numerical/summary.json) · [冷进程](numerical/process_cold.json) · [三套来源图](native_recipes.json)
- 严格 `static_w4a4/2` 实测使用有限校准 QDrop。冷包有 26 个实际覆盖边界，累计 12 个 W/A optimizer step，导出为硬 W 和固定 A。对已优化 W 做父包编辑，结果与从已部署 W 重新编码的独立计算相等，与从原始浮点 W 重算相差最高 0.007762786。相同条件下的四位静态成绩仍须正式任务评价。[主线验证](primary_protocol.json)
- 2–10 位逐位打包对比独立 Python 整数位串；另验零斜率、负乘数、偶数舍入、饱和、正贡献传播和未量化 bias。配对 ridge 与独立最小二乘最大差约 1.0×10⁻¹⁵，离散分配和穷举目标一致。实际移动图完成 unfold、局部/全局融合和桥接梯度评分；Swin 按原 shifted mask 分发和恢复，两个方向的全高分支与原块最大差 1.19×10⁻⁷，混合分支与全高分支相差 0.03137，冷包重现混合结果。[独立算术对照](edge_oracles.json) · [H-ViT 与 PracticalEdge 的原图算子对照](nonlinear_oracles.json) · [共享组件与 Trio 前提](component_checks.json) · [MobileViT](hybrid_bridge.json) · [Swin](window_graph.json)
- 11 个来源分配请求保存候选、成本、选择与冷包。CLAMP 保存同一像素池的刷新、量化种群、阶段目标与导出。DAQuant 分别验证域配对优化及冻结范围；QwT-v2 另验证固定点融合。各项训练集/校准集访问和预算取自对应请求。[来源分配](allocation_recipes.json) · [CLAMP](cyclic_images.json) · [DAQuant 静态变体](numerical/daquant_frozen/source_request.json) · [QwT-v2 的整数变体](numerical/qwt_v2_integer_fold/source_request.json)
- 同一冻结包在真实 ImageNet 分类 head 输出 `[1,1000]`，在 ADE head 的记录窗口输出 `[1,150,512,512]`。另一份 dense 反向收据记录 238865 个有效像素。联合校准的两类原图、任务来源、head 和窗口均可追溯。不同许可、任务 head、覆盖、分辨率、资源和数据划分分别限定比较范围。[同包两任务](joint_views.json) · [dense 梯度与 mask](dense_sensitivity.json) · [比较规范](../../../QAViT/knowledge/ptq/11_TASK_COMPARISON_AND_EVIDENCE.md)
- 一次完整候选选择用了 3 张校准原图和 3 张独立开发原图。候选池包含冻结父包和两个网格变体。校准 MSE 选中 0.7，独立任务 NLL 最低的是父包，选择遗憾值为 0.003777，代理指标前二包含任务最优者。两份请求保留逐图观测、同一父包身份和实测耗时。ADE 的另一份检查将一张完整原图的窗口拼回后计数，FP 和量化输出的逐类混淆矩阵都与整图归约一致。[候选选择与成本](selection_quality.json) · [校准候选](selection_calibration.json) · [独立开发候选](selection_search.json) · [ADE 逐原图核对](per_original_dense.json)
- Generation 实际读取 BiasComp 正文，区分位置偏差和跨图像误差，提出交换图像的负对照。预训练 DeiT-B 的一个 QKV 位点使用 4 张图拟合、4 张不同图检查。保留图平均局部 MSE 为零补偿 0.06925、通道偏差 0.06574、位置偏差 0.04575、位置偏差加样本斜率 0.04407，跨图像错配斜率为 0.04868。Recorder 据此归纳了适用条件和反例，Journal 恢复原 `EvidenceCard@1` 与 `MetareviewCard@1`。记忆在 L3 上下文中的可见性与正式 L3 实验分别记账。[判断与调用](generation.json) · [逐图测量](focus_measurement.json) · [拟合状态重载](focus_cold.json) · [Recorder 与恢复](memory_restore.json)

## 需要带入下一阶段的差异

| 来源 | QAViT 采用的条件及影响 |
|---|---|
| 原论文、作者代码、预算 | 比如 APH 的有符号方向差分与作者绝对值版本、FIMA 的秩与归一化、DAQuant 的坐标统计、ERQ 的 sum/mean 和增广 bias 都分版本绑定。PoMQ 出版版 E1–E3 写 L1；资料中的 L2 归约是独立选项。重算时按版本读正文及源请求。[PoMQ 出版版](https://www.mdpi.com/2076-3417/15/18/9856) · [ERQ 作者代码](https://github.com/zysxmu/ERQ/blob/main/test_quant_expand.py) |
| 编码与部署 | H-ViT Eq.18 的印刷上下界有额外码态；软件配置使用显式合法的有符号码域。Trio 原指数范围有 24 档，16 档版独立命名。QwT-v2 有软件整数累加/重定标参考。PyTorch QDQ 计时、加载耗时和真实设备端到端延迟各有不同口径。[H-ViT](https://link.springer.com/article/10.1007/s43684-025-00121-0) · [H-ViT 请求](numerical/recipe_h-vit/source_request.json) · [组件收据](component_checks.json) |
| 特定架构与数据 | MobileViT、Swin、MBConv、线性注意力、BN 统计、指定任务 head 和不同尺寸 token 都需对应真实图。已实测的微型结构与任务数据见上表。整套 COCO 检测/实例分割、点云、SR 和相应标注许可须在实际运行中另行绑定；当前 ImageNet/ADE 成绩不覆盖这些任务。 |
| 尚缺原件 | FGPTQ-ViT、UniQ-ViT、MROSPU、token-dynamic-bit-width 和 optimal-numerical-format 的已核部分保留。仍缺的分段端点、AQO/SRO 参数与时序、优化预算、动态位宽规则或格式选择及后端细节逐条见来源卡。五个边界未决条目仍保持来源记录。综述相邻条目、相似方法和数值 smoke 均不能填补这些原件。[具体来源卡](../../../QAViT/knowledge/ptq/sources) · [本轮原始来源复核](source_followup.json) · [UniQ 作者仓库](https://github.com/Dexter-Yu/UniQ-ViT) |
| 待实测设备 | FPGA、Versal AIE、Apple ARM/NEON 等需要目标硬件、编译工具链、对应 kernel 和同口径基准。软件数学、编码和图级调用状况见逐项账目，其论文准确率和原生设备字段均另列。 |

## 运行与恢复入口

从主线根目录运行，Python 使用已有环境并设置 `PYTHONPATH=QAViT`。复核已有收据的入口为 `python EXP/validation/ptq_foundation_20261125/audit_foundation.py`。仅重载现有全部数值包使用 `python EXP/validation/ptq_foundation_20261125/numerical_probe.py --cold`。重跑某个数值请求可追加 `--only recipe_qdrop`。同时核验本地权重校验和、来源绑定、公共 worker 与 Recorder 收据，可运行 `python EXP/validation/ptq_foundation_20261125/verify_delivery.py`。实际权重留在本工作区的各 `model/weights.pt`；便携请求、运行条件、哈希和收据保存在实验目录。

下一阶段从 `QAViT/configs/research.json` 复制运行配置，填写用户指定模型、实际权重与 metadata、任务/head/标签策略、校准和开发集、任务优先级、比较视图、设备和预算。保留有界源配方中的 `source_id`、`version`、`profile`、`base`、`settings` 与 `evidence`，将 fixture 的层路径、样本 ID 和预算替换为绑定资产；设置 `execution.recipe`，按需要增加 `recipe.allocation`、`calibration_generator` 或联合任务校准。公共请求可参考 `service/prepare/request.json`。有 optimizer 更新时声明 `resolved_choices.training`、校准分区、更新对象、目标和有限 step 预算，冻结评价读取已导出包。严格主线与扩展配置分别使用 `static_w4a4/2` 和 `ptq_extension/1`；后者申报实际扩展维度。[运行模板](../../../QAViT/configs/research.json) · [公共 worker 请求](service/prepare/request.json) · [研究协议](../../../QAViT/knowledge/ptq/01_STATIC_W4A4_V2.md)

新研究运行使用 `python QAViT/run.py start --config <运行配置>`；恢复同一运行使用 `python QAViT/run.py resume <运行目录>`。正式候选按各自协议和预算运行 L3，并单独建立可比的 B。本次没有启动多轮综合搜索、作者全预算复现或全量正式评测。

原资料包 `QAViT_PTQ_Foundation_20261125.zip`（258472 字节）已在验收后删除。上传清单、25 份材料的原始哈希与接入去向，以及删除时复核的原包 SHA-256 均已保留。[原包处置收据](source_disposition.json) · [材料映射](source_mapping.json)
