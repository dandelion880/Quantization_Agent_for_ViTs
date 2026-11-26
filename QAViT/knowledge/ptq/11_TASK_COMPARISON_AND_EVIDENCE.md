# 任务、对照、父方案与证据：多方法接入的执行语义

> 上传材料的待接入事项和原文预算按 2026-11-25 出处保留。实际数值入口、适配差异与证据见 [实现账目](implementation_status.json)。

状态：供本轮 Codex 实现使用的补充规格。本文补齐工作包 A/C，并约束 B/D/E 之间的结果交接；不修改原文献卡、原协议历史记录和既有覆盖数字。当前用户已授权代码接入与必要的低成本验证，完整昂贵科研运行和综合调优仍属后续工作。QAT 不进入方法、组件、候选、灵感、Skill 或实证方案库；有限校准 PTQ 优化按 01_STATIC_W4A4_V2.md 处理。

本文中的字段是需要宿主表达的事实，不是可直接传给当前 API 的新参数。Codex 应映射到现有 ResearchBrief、Experiment/Evidence、模型 manifest、Journal 和方法目录，避免另建平行数据库。数学定义继续引用 metrics_registry.md，本文集中说明这些指标在任务、比较和历史状态中如何使用。

## 1. 源码审计结论与实现差量

审计基线为包内对应的 QAViT(20261124-101904) 源码。以下差量来自源码核查，不以文稿出现新术语直接判定现有功能缺失。

| 源码锚点 | 当前事实 | 接入后需要表达的差量 |
| --- | --- | --- |
| core/solutions.py：protocol_key、rank、fallback | 按 model/evaluation_protocol/quantization_contract/hard_constraints 指纹分桶；当前排序为 mIoU 优先、Top-1 次之；B 保留已验证结果且不占探索槽 | 分开比较语境与运行协议身份；按声明的任务/资源选择策略生成视图；既有双任务排序继续可用 |
| core/solutions.py：prepared_compatibility | 当前只支持保留确切 search/final、校准集前缀扩展等狭窄迁移 | 增加按实际模型包/评价身份判断的历史证据复用，不能只删除全部一致性检查 |
| tools/frozen.py：bind_sources、bind_parent_initialization、load_refinement_parent 等 | 多条路径仍要求 training_declared/preparation_training=False | 区分准备阶段优化与最终冻结，允许已获授权的优化型 PTQ 包继承；加载仍不能重新拟合 |
| execution/frozen_trials.py | 同候选、同图像/窗口、分任务与多层特征观察、逐图归约；候选目前是 scale 编辑；local_fp 绑定固定 parent 输入 | 支持冻结的完整状态候选，明确 local_fp_parent 与 local_fp_candidate；保留旧行为兼容 |
| execution/calibration.py：capture | 站点梯度捕获当前要求二维分类 logits；教师 KL/真实标签 CE 已区分；梯度全零不静默换 MSE | 添加任务适配的 sensitivity prepare；dense 观察已有，不等于 dense 反向准备已实现 |
| tools/dual_task_evaluation.py | 分类和分割核对同一冻结包身份，可保存一侧失败时另一侧的真实部分结果 | 新配方包继续使用相同身份核对；新增准备状态不能破坏同包评价 |
| core/evidence_scope.py、core/memory.py | 实测范围由宿主事实绑定；比较经验保留逐 Evidence 条件，不以合集扩大范围 | 纳入新协议、准备过程、metric/reference、任务视图和资源事实；不让自由文本冒充宿主实测 |
| core/inspirations.py、core/topics.py、core/skills.py | 灵感、方法谱系、完整方案和 Skill 已分工；读取受 commit cutoff 约束 | 将新方法材料与这些对象关联，保持其身份、恢复及跨运行经验边界 |
| agents/interaction.py、core/memory.py 的方法目录入口 | 静态文献目录给 read_asset；read_method 读取运行中 method_id 谱系 | 明确命名空间，不能把论文 slug 当成可执行谱系 ID |

这些是本轮需要衔接的真实入口。实现应复用现有数据和状态设施；仅新增文档、枚举或 prompt 不能算完成接入。

## 2. 先分清四类对象与两个方法命名空间

### 2.1 四类对象

**方法资料。** 描述一篇原论文、一个作者版本、一套配方或 primitive；可以在尚未实现时被发现，用来提出有根据的研究问题。来源完整性和实现状态分别记录。

**完整方案。** 是能够恢复具体模型状态和构造配置的对象。可以是待测 seed、开发集已测方案或具有正式任务结果的方案；状态必须可见。方案条目指向代码、量化包、模型/任务绑定和证据，不复制一份容易过期的算法摘要来代替它们。

**运行方法谱系。** 表示 QAViT 实际产生、修订和比较过的方法身份；关联候选、父方法、父方案、主题、灵感与实施记录。它可以引用某个文献方法，但不会因为目录出现一篇论文就自动存在一个运行谱系。

**灵感与研究主题。** 记录尚未成为完整方案的机制假设、竞争解释、已做尝试和未决问题。没有可执行模型包、尚无数值、一次失败或当前排名低，都不使合理的 PTQ 灵感失去保存价值。主题说明当前研究问题，不能取代方案的构造与结果身份。

Skill 保存可复用的决策过程，Memory 保存带适用条件的经验；两者不取代上述原件。

### 2.2 文献目录 ID 与运行 method_id

静态 docs/methods/index.json 的 id 是资料/能力目录身份。角色应使用目录给出的精确 read_asset 定位文稿、来源或当前实现说明。运行工具 read_method 的 method_id 来自 Journal 中已建立的方法谱系；不能将 aphq-vit、erq-icml2024 等文献 slug 直接填入 read_method，除非宿主确实建立并返回了同名的运行身份。

新提案可引用文献来源，但没有运行谱系时按现有接口由宿主分配身份。应保存“运行 method_id 引用了哪些 method-source/recipe/primitive”的关联，而不是强行合并两个命名空间。公共工具目录只列真实可调用接口；资料目录可以显示未实现能力，并明确后续实现依赖。

## 3. 比较语境：不是一个过度拥挤的 protocol_hash

一次比较至少能还原下列事实；它们可以分布在现有对象中，无须为每次实验要求用户手工填一张新表。

| 事实组 | 需要表达的内容 | 允许变化时怎样说明 |
| --- | --- | --- |
| 预训练来源 | architecture、checkpoint/hash、适配器、必要结构常量 | 非等价结构恢复后的模型仍指向原来源，并记录变化 |
| 任务设置 | classification/segmentation/detection 等、head/neck 权重、特征层、输出语义 | 更换 head 或任务时，单独评价身份 |
| 输入/数据 | 原图 ID、split 来源、分辨率/变换/窗口/增强、label/teacher 使用 | 校准域/规模或输入策略是研究变量时单列 |
| 构造过程 | recipe/variant、更新对象、数据用途、顺序、随机性、实际优化/搜索投入 | 原配方和共同预算是不同视角 |
| 最终表示 | W/A/codebook、边覆盖、静态/动态、例外、补偿和实际 pack 状态 | 扩展 profile 并列比较，不冒充相同主配置 |
| 评价过程 | exact model package、任务视图、metric/reduction、集合范围、backend | 部分/完整结果、QDQ/原生执行分开 |
| 比较目的 | 原法复现、统一覆盖、资源效率、迁移、机制消融或选参能力 | 每个比较结论说明自己回答哪一个问题 |
| 证据来源 | paper_reported、official_recipe_reproduced、qavit_unified_protocol；本地执行范围 | 来源类别不能替代实测范围与复现状态 |

运行 protocol_hash 继续标识不可变运行条件，比较视图可以引用多个协议对象，并对每一项说明兼容关系。不要为了跨协议比较而把历史对象的哈希重写成当前值；也不要因哈希不同就丢失一切可用信息。模型包可作为父状态使用、历史成绩可用于某个比较、成本能否匹配，是三个分别判断的问题。

## 4. 任务适配应提供的真实能力

### 4.1 分类

沿用当前 classification forward、既有 head 与预处理。实际评价输出 Top-1/Top-5；NLL、teacher KL、margin 或 ECE 按问题使用。标签来源明示：真实标签、teacher pseudo-label 与无标签不互相替代。

task loss 适配器需返回可微标量或明确的 per-example loss 及归约。teacher logits 来自已绑定 teacher/head/temperature；相同 teacher/student 路径上的 KL 一阶梯度可能为零，保留该事实，不自动换 hard pseudo-label 或 MSE。

### 4.2 语义分割

复用现有 DenseTask、窗口几何、head_features 与 frozen ADE 评价。dense sensitivity/reconstruction 需另接可微 task loss 路径，正确处理空间轴、ignore mask、目标尺度、特征层和 head；不能把张量 flatten 后传给只接受二维分类 logits 的旧 capture，或在 inference_mode 下假称已准备梯度。

正式分割结果先按既有任务协议融合窗口、对齐原图，再累计 confusion matrix。分类 crop 与 dense window 可以来自同一原图，但不是同一个输入视图。分类/分割共用量化包时，backbone 参数、量化器、补偿与结构状态身份一致；各自的固定 task head 和必要模型包装可以不同并分别绑定。

### 4.3 新任务或当前无法适配的任务

用同样的最小接口说明输入视图、预测/特征、teacher/label loss、有效 mask、最终指标与 reduction。已支持分类并不意味着检测/分割已有真实执行适配；尚未接入时明确 unsupported/unmeasured 范围，同时保留可完成的任务结果，不把缺失任务填成零分，也不因缺一项删除另一项真实测量。

原论文的检测/分割结果归入作者证据；QAViT 在新任务的迁移需重新产生任务证据。模型和任务名字不写死在共享指标实现中；特殊 tokens、head 数、层宽和分辨率来自真实适配器元数据。

## 5. 数据用途与预算：继续使用既有 manifest

把“数据叫什么 split”和“这次拿它做什么”分开。沿用 calibration/search/final/dev 以及已有任务 manifest；在实际访问记录中增加用途，而不是另造四个数据系统。

- **fit/statistics。** 拟合范围、分组、补偿、权重、舍入或收集二阶矩。
- **sensitivity/teacher preparation。** 产生梯度、扰动、曲率、teacher logits 或合成数据。
- **selection。** 选候选、超参、停止点、checkpoint 或任务权重。
- **heldout diagnostic。** 检验泛化、selector、传播与机制；其结果若随后影响选择，应记录这一使用事实。
- **final evaluation。** 产生正式报告结果，不参与本次参数拟合；若已经反复用来选择，就按实际开发用途解释。

这些用途可以在允许的训练/开发划分内合理复用，并如实记录；不强制每个实验新建互不重叠的四套数据。重要的是不把使用过的集合改名成 untouched heldout，也不把默认可用池大小写成真正使用数量。

数据账本同时保存 unique original images、views/windows、tokens/pixels、label counts、teacher forward、样本呈现次数。继承父包时区分 inherited calibration IDs、incremental fit IDs、当前 job 实际读取；父包已经使用的数据仍属于累计构造事实，当前 job 没重新读取则不算新读取。重复读一个缓存不形成新的独立数据或重复实验。

公开论文采用32张、1024张或其他数量是 recipe 事实，不是 QAViT 的全局永久上限。本轮低成本验证按所验实现风险选取小输入；后续原法复现与资源比较使用其各自声明预算。

## 6. 哪些比较回答什么问题

| 比较类型 | 需要保持的关键条件 | 可以改变什么 | 结果应怎样命名 |
| --- | --- | --- | --- |
| 原配方核对 | 原模型/任务、作者精度例外、校准与优化设置尽可能一致 | 为执行所需的环境适配须记录 | 作者论文版/代码版 recipe reproduction |
| 统一覆盖比较 | 同源模型、task/head、评价、W/A和关键边覆盖 | 方法各自合法校准策略/优化机制，费用公开 | unified coverage PTQ comparison |
| 共同资源比较 | 同任务与评价；声明共同的数据/计算/存储/延迟轴 | 精度、分组、动态行为或补偿可按问题变化 | resource-matched / resource frontier |
| 机制消融 | 同父状态、数据、评价和未被研究的构造条件 | 明确的一项机制或必要交互 | ablation / interaction |
| 跨任务迁移 | 同一个已冻结 backbone 包与各任务固定FP对照 | task view/head与目标域评价 | same-package transfer |
| 各任务分别校准 | 各任务数据和对应FP对照明确 | 每个任务构造独立量化包 | task-specific PTQ |
| 多视图联合拟合 | 一个共同包；各来源/任务信号与投入公开 | 联合目标、校准比例和策略 | joint calibration / shared package |
| selector 评估 | 固定实际候选池、proxy/确认数据、reference | 被评 metric 或选择策略 | candidate selection quality |

统一位宽不要求强行相同迭代次数；资源匹配也不是把所有方法削减到最便宜方法的设置。保留原recipe表现，另在合适资源范围给共同预算对照。若只改变预算，不能把结果当纯算法消融；若改了首尾A8或保留FP矩阵，不能与全覆盖静态W4A4只按Top-1混排。

配对比较优先同一父方案和原图ID。source model 固定指同一预训练来源，不意味着 PTQ 局部重构过程中禁止更新权重。任务训练的 FP head 是任务基线的一部分；量化后再做任务训练的补救流程不因只更新head就进入本PTQ范围。

## 7. 排名、保留结果 B 与方案库

### 7.1 排名是一个已声明的视图

保留原有 mIoU→Top-1 字典序作为旧双任务协议的有效策略。新协议按本轮问题选择单任务排序、多任务 Pareto、预先声明的加权效用或资源约束下排序；不强制把所有研究变成一个综合分。

先判断条目是否可用于当前比较，再应用排序规则。缺失mIoU的条目可留在分类比较视图，不能伪造mIoU、自动按零处理，或与双任务完整方案宣称相同证据。profile不同或预算不同的条目可在对应资源/迁移视图中出现，不需要删除其方案。

多任务权重、归一化尺度和tie处理应在使用selection结果之前确定。不要每次对当前候选集合做新的min-max归一化后声称跨运行分数可比。Top-1/mIoU的微小差异可附paired不确定性；只在它会改变判断时补重复，不为所有条目强制多seed。

### 7.2 selected regret 与候选池覆盖

固定候选池 C，proxy选择 \(\hat c\)，独立确认utility为U：
\[
\mathrm{regret}=\max_{c\in C}U(c)-U(\hat c).
\]
报告C的实际候选身份、pool-best、selected、baseline和数据范围。regret回答“池内是否选对”，pool-best−baseline回答“池内有没有好方案”；两者都不能推出搜索空间的全局最优。

复杂metric需要计入prepare与score成本。只有一个候选时无法评估排序优劣；候选utility高度接近时，报告分辨率与不确定性。来自另一个模型/预算的论文结果不作为这个池里的实测oracle。

### 7.3 B、seed 与历史方案

B继续表示当前比较语境下可保留的、已有实测支持的完整结果，不占H1/H2/H3探索位置。新的失败、OOM或较差候选不使B消失。历史B显示来源运行、原协议、测量范围和new_measurement=false；不能复制到本轮后改成刚刚实测。

待测seed可以作为成熟起点被选择和构造，但没有可重载实测包时，不能伪装成已验证冻结父包或B。开发集已测方案可保留并继承其真实状态，不能在没有完整结果时进入正式全任务排行。最新方法可以先是来源卡/待实现/待测seed，而非必须等进入Top-3才允许被研究。

## 8. 父方案继承与新旧协议迁移

### 8.1 继承分三个层次

1. **引用继承。** 借鉴历史方案/论文设计并读取其配置，无权借用其分数或假设已有模型包。
2. **状态继承。** 冷加载具体冻结父包，在其副本上执行增量 PTQ；父权重、quantizer、补偿、结构、代码版本与来源绑定可恢复。
3. **证据继承。** 某些历史测量在新的比较语境中仍可作为历史证据，必须显示兼容依据与原始范围，不标为新测量。

这三层彼此独立。可引用一个老方法不等于可加载它；能够加载一个包不等于它旧的mIoU适用于新head；两个任务共用一份模型包也不等于二者采用同一数据视图。

### 8.2 常见迁移的确定处理

| 情境 | 处理 | 不能做的事 |
| --- | --- | --- |
| 旧training-free W4A4包进入允许优化的v2；包、head、评价和覆盖实际相同 | 记录兼容依据，作为历史结果/父包复用；是否补新测量取决于尚未验证的变化 | 重写旧协议、改成新实测，或因v2允许优化而拒绝旧包 |
| 新包曾做有限校准重构，最终状态冻结完整 | 允许作为合法PTQ父包；保留更新对象/数据/优化事实；加载不重新拟合 | 要求preparation_training=False、或把其伪改为False以绕过旧条件 |
| 旧成绩来自首尾A8、主比较要求这些边A4 | 保留作者/例外profile；若做全覆盖适配，产生新状态并重新测对应任务 | 把旧分数改标签成全覆盖W4A4 |
| 同包改分类/分割head、特征层或预处理 | 包可以继续作为构造起点；评价身份变化，旧任务成绩不直接迁移 | 只因backbone hash相同就复制旧指标 |
| 原校准预算不同，当前比较是共同资源效率 | 记录累计构造预算；需要等预算时产生真实新构造 | 只改manifest的calibration_samples使旧包满足新预算 |
| 搜索/评价manifest内容或使用角色改变 | 根据原图ID、变换和用途分别判断；旧结果保留原范围 | 只比较split字符串，或把开发集成绩当未触碰最终结果 |
| 动态/附加FP补偿原配方进入新研究空间 | 保留真实extended profile及成本，可以形成独立比较 | 仅修改profile名变成static，或从方法库一概删去 |
| 老包缺少某项新字段，但旧原件能确认事实 | 用指向原件的兼容记录表达已知/未知；不修改原artifact | 从当前默认值反向填造历史执行事实 |
| 老数值实现不同于当前代码 | 复用绑定的旧loader或显式迁移并核对；保存两份实现身份 | 用当前实现重新校准后称为原状态冷加载 |

旧 prepared_compatibility 的窄迁移继续适用于原情境。新迁移应围绕实际兼容关系扩展，而不是将所有保护全部关闭。兼容结论由可核对的模型、数据、执行事实支持；没有依据时保留可引用历史资料并说明需要哪项重新测量，不把未知变成全面禁止。

### 8.3 准备阶段和最终冻结分开

旧training_declared/preparation_training布尔量保留原版本语义。新的构造事实至少能区分 gradient_use=none/read_only/optimization、参数变化方式=closed_form/discrete/iterative、更新对象与范围、数据用途、最终冻结和可重载状态。旧值False不能自动推出“没有任何权重变化”，因为闭式量化/补偿也会改权重；新值有优化也不能自动推出QAT。

冻结加载、任务评价、父包读取不得启动optimizer、读取新的calibration数据或重估参数。需要增量拟合时使用独立准备阶段和子包，保存parent→child状态差异。保存模型不应遗漏补偿tensor、learned rounding硬化后的编码、分组/重排、替代结构和量化参数。

## 9. metric 与统计缓存的事实绑定

cache identity包括依赖事实，不只是metric名字、层号或数据路径。缓存的复用条件由估计目标决定；不必要的失效会浪费计算，错误的复用会改变算法。

| 缓存类型 | 绑定内容 | 变化后的行为 |
| --- | --- | --- |
| FP teacher原始logits/features及其派生概率 | FP checkpoint、head、输入/窗口、层、mask和dtype；温度化概率另绑定temperature | 只改量化候选通常可复用；改teacher/head/view按依赖重建；只改temperature可复用原始logits，重算概率及其梯度目标 |
| 当前量化前缀输入与二阶矩 | 完整前缀状态、A量化、输入变换、样本/布局、统计归约 | 上游变换/重构/补偿或A参数变化后重采；只改下游时可继续用 |
| local_fp_parent输出 | 固定parent及其前缀输入、FP block、view | parent不变可复用；候选改上游仍可作为固定parent参照，但不称同输入局部误差 |
| local_fp_candidate输出 | candidate当前前缀、FP block、view | candidate上游变更即重算；当前块内量化参数变更而输入不变可复用FP块输出 |
| site gradient/APH/FIMA | 任务loss/head、teacher/student、suffix精度、参考点、扰动、温度、秩和数据 | 固定参考proxy可跨其候选池；算法要求迭代刷新时按真实stage刷新 |
| QEP/ERQ/GPTAQ配对矩阵 | Xfp/Xq成对身份、轴、均值/和归一化、更新顺序、damping定义 | 不能只凭site和校准图数复用；父输入改变则失效 |
| 补偿回归统计 | 拟合输入、目标reference、模块、保留/归约轴、当前上游/quantizer | 改quantizer、reconstruction或上游补偿后按依赖重估 |
| candidate score/任务结果 | 冻结candidate state、任务/数据/metric/归约、backend | 同状态同评价可复用但标cached；不算独立重复 |
| 真实backend性能 | 导出图/编译artifact、硬件backend版本、shape/batch、测量定义 | 改图或测量条件需重测；数学等价不保证同延迟 |

缓存receipt应能说清本次是hit/rebuilt以及依赖匹配/变化原因；数值原件保留必要的shape、count、normalization、effective rank、dtype及数值退化状态。不要求每次保存所有activation大矩阵，可保存可再生输入绑定与充分统计。

APH/FIMA的prepare可以用反向，score继续保持冻结前向。完整重构另走optimizer执行器，不解除现有score的inference_mode后假称兼容。若缓存退化，全零梯度、奇异小矩阵或零方差须按该variant的明确数值策略处理；未指定替代目标时不静默退回MSE。

## 10. 实证结论应怎样落库

同一条Evidence区分四件事：执行请求说想做什么、宿主确认实际做了什么、程序测到什么、研究者据此推断什么。以实际执行为准记录更新和数据；不从候选宣言直接生成成功事实。

| 实际证据 | 可支持的结论 | 不支持的结论 |
| --- | --- | --- |
| 原论文/作者公开结果 | 作者在其协议上的报告与机制来源 | QAViT已接入、已复现、目标模型已有效 |
| CPU数值对照、toy模块、真实编码保存重载 | 对应数学/轴/状态/公共执行路径已验 | 作者整套成绩复现、目标任务准确率提高 |
| 少量真实数据的缩小recipe运行 | 该模型/位点/数据/预算的执行与局部观察 | 原预算完整原法复现、全验证集成绩、一般成熟结论 |
| 当前协议的完整任务测量 | 该冻结包与该任务视图下的表现 | 未测任务/模型/后端的有效性 |
| 独立重复与迁移对照 | 在已覆盖条件上的稳定性与边界 | 无条件普遍规律 |

方案库保存具体状态、成本和测量；灵感库保存负结果后的剩余问题和下一步；Memory将经验标成observed_once/conditional等真实状态；Skill提炼可复用决策过程。一次局部成功不自动升为replicated；同一缓存或重复读取不是独立复验。OOM、超时和实现错误进入工程状态，除非有有效数值证据，不作为机制被推翻。

Recorder可以引用多臂Evidence，但每臂保留其真实scope。不要取一个宽泛联合scope覆盖不同模型、不同任务或不同参考。用公开paper数字补齐本地缺失指标、将partial结果变成full_dataset、把测试替身结果写成real，均不属于有效积累。

## 11. 可直接复用的结果叙述模板

以下是正文模板，不是新的工具调用协议：

> 本比较以【预训练来源与任务/head】为对象，研究【原法核对/统一覆盖/资源/迁移/消融/selector】。父状态为【模型包与来源】，候选执行了【配方及variant】，实际更新【参数范围】，使用【数据用途、真实读取及投入】。最终部署为【覆盖、表示、静态/动态、附加模块】。本次只测得【任务/metric、样本范围和backend】；结果为【原始指标、差值与必要不确定性】。与【对照】可比的条件是【具体条件】，不同之处是【真实差异】。这些结果支持【有边界的结论】，尚不能回答【未测条件】。历史分数/缓存引用为【确切来源】，不计为本次新测量。

“方法已接入”的叙述应另附公共入口的实际执行与冷加载证据。文稿已登记、primitive已实现、完整recipe可运行、原法已复现、统一协议已验证、后端已测是分别成立的事实；不要用单个implemented布尔字段包办。

## 12. 与五工作包及验收的关系

A负责把这些身份、协议和来源关系落入既有事实模型；B负责产生真正可重载的完整PTQ状态；C负责按上述任务/数据/metric和比较条件测量；D让角色能读取并据此做研究判断；E通过12_INTEGRATION_ACCEPTANCE.md的低成本案例核实实际连接。

旧资料已具备主要数学定义和方法差异，本文件补的是历史与任务执行语义。仍需在接入过程中由真实环境补齐的不是一般性“更多计划”，而是：尚未公开的作者配方细节、具体task adapter/backend可运行性、各方法的实际API注册与完整保存格式、真实执行回执及原法/目标任务成绩。无法获取的来源字段继续保留明确状态，不能靠本文件编造。
