# QAViT 最小科研能力升级（2026-11-07）

本次在原工作流上补齐完整方案起点、static W4A4、工程交接与真实结果边界；没有重构主框架、增加 LLM 审核、改动默认预算或资源调度。历史运行及其未决裁决保持原样。真实三槽 Generation、最终77项定向回归、结构审计及接口快照检查均通过；全量回归的一项共享槽超时已隔离复验通过。

**真实失败复盘及对应修复**

依据最新源码、[原科研报告](../../inputs/deitb_training_free_static_w4a4_20261106_r1/research_report.txt)、原始 jobs/code/cards，以及同目录两份 handoff/reuse gap 记录核对：

| 事实与根因 | 本次处理与判断边界 |
|---|---|
| R1 H1 的 FX 控制流失败；24 个预期 LN 组只识别 12 个，norm1 的 shape/getattr 使用被当成额外数值消费者 | 复用已有实际张量边验证与显式仿射预处理，允许 Coding 图适配。完整 RepQ 起点不依赖全图 FX，也不要求插件组合。真实 tiny ViT 的 RepQ+GPTQ 适配校准/导出/重载通过，两条 LN 边均验证；不宣称已重跑完整 DeiT-B 比较。 |
| 自定义结果 artifacts/evaluations 的容器与行结构不一致，曾在宿主 append/values 崩溃 | 保留最新 artifacts 修复；增加有唯一 kind 的评测列表到字典的无损规范化，仍验证真实 split/sample_ids。空评测列表也无损转为空字典；歧义行返回准确的实现错误，不能猜结果归属。 |
| 新请求修复同作者旧实现时，源码和执行日志没有随请求交接 | 宿主绑定同研究角色/轮次/槽位/场次的最近实现，或显式引用公开 Evidence 的生产者；交付精确源码、错误、request/result/worker log 权限。保留立场隔离与哈希，不开放整目录。 |
| 原 custom 包能在原任务重载，experiment_v1 却不能调用其批准 loader | 增加 Evidence 绑定的 load_frozen；核对原包/源码哈希，只调用 load，阻止校准批次读取并记录回执。真实历史 H2 包已在原数值库快照下完成一次 GPU 前向，未 prepare，原模型及包未变。当前数值库直接加载旧包仍因版本不同正确拒绝；未引入跨版本绕过。 |
| engineering_blocked、空 measurements 曾被发布成 completed 数值证据 | 显式工程阻塞返回失败，交回研究者，不作为测量或触发无意义的重复 Coding 修订。缺证据可保持未决；不重判历史比赛。 |
| R2 H1 同时冻结 [-7,7] 与“不改原 GPTAQ”，形成请求内冲突 | [-8,7] 与 [-7,7] 均合法。提供显式 gptaq_signed_min 适配，在误差补偿过程内使用对应码域，不在最终权重上事后截断。保留原默认网格。允许修订冲突请求、撤下不兼容插件或回退完整方案。 |
| 对数网格的 16 个非零值加下溢零，以及 RepQ 非整数 mean(z) 饱和端点，可能实际有 17 值 | 新增显式 reserve_zero 变体，校准评分与推理共用选项；按冻结参数核验容量，不能以样本 unique 代替。完整 RepQ 适配在变换后用 affine percentile 拟合整数零点；旧公式不静默修改。 |
| per-channel affine 的恒定通道 lo==hi 会产生无效 scale；PTF/RepQ 旧方案级互斥过宽 | 仅给退化区间定义含零扩展，正常区间算式不变；冲突限于同一 LN 链，不同链实际组合测试通过。 |
| 原 CLI exit1 只留 stderr 哈希，无法确定失败原因 | 今后保留有界、脱敏、权限600的诊断文件；不把过去未知的197字节内容推断为已证明的上下文故障。 |

原 R1 H1–H2、H1–H3 的未决是工程失败与证据缺口，不能视为算法反证。原 H2 全量结果是 ImageNet Top-1 **5.374%**、ADE20K mIoU **1.2067773%**；R2 H2 的 **85.0%** 只来自另一个实例的 1000 张 search，缺完整 ImageNet/ADE，因此不自动晋升本次新协议的 Top-3。以上均是历史结果，非本次新测成绩。

**方法组件与完整方案库**

沿用 methods/ 方法卡、伪代码、真实插件和检索入口，补充官方来源、固定版本、适用条件和组合边界。独立的 docs/solutions/ 与 execution/baselines.py 提供 GPTAQ、AdaLog、RepQ 三个完整适配起点，包含量化范围、校准/执行顺序、参数、可执行构造器及版本。它们是当前本地可执行、有官方来源的成熟强基线起点，不宣称已证明是新严格双任务协议下的前三名；两项实测成绩均为 null/待测。

选择参考 [GPTAQ/GPTQv2 官方论文](https://arxiv.org/html/2504.02692v1)、[AdaLog 官方实现](https://github.com/GoatWu/AdaLog)、[RepQ-ViT 官方实现](https://github.com/zkkli/RepQ-ViT)。GPTAQ 插件沿用本地固定 d6bd2e4 版本名称，不因论文现名 GPTQv2 静默替换实现。论文分类成绩与本项目 head、patch、码本、校准及 ADE 协议不同，不填入本库实测字段。

研究运行共享 EXP/solutions/<protocol_hash>/top3.json，验证运行隔离。配置、源码快照/版本（包括显式 intervention 对应版本）、实际量化覆盖、原产物和 Evidence 均可追溯。每轮完整研究的有效真实 L3 同协议双任务结果才参与 (mIoU, Top-1) 降序更新；平分保留旧项，没有更优实测方案则保留。已完成评测项排在待测占位项前，旧版本保留在 Journal/archive 和原运行。待测不阻塞研究，也不伪造排名。

未晋升方案、局部发现、失败及负面证据继续按原 Recorder/Memory 归档与按需检索。完整方案目录只承担复用与选优。H1/H2 从其中选择基础方案，在正文自然说明保留/修改；H3 保留开放创新。不增加固定措辞、方法名或新审核字段。

**协议与主流程**

ResearchBrief 新模板使用 static_w4a4/1；Generation/Coding/Safeguard/Judger/Recorder、执行覆盖、结果、方案库和手册一致。主体 Linear/Conv W4 与输入 A4，Q/K/V/P、FC2 post-GELU A4，每组至多16值且零计入，静态参数冻结。非线性内部、LN参数、bias、token/位置编码、残差和累加允许高精度；head/neck 实际选择单独记录。粒度、对称性、校准、量化器、重参数化与补偿仍可搜索。局部诊断按本次声明范围，完整 L3 核验全边界。

保留 Generation 局部循环、Screening、三场独立并行辩论、Judger、固定 H*、Evolution/H4 父子挑战、Coding/Safeguard、Recorder；默认预算、非 Judger 单通道、Judger 并发、自适应 GPU 池和按需 Memory 均沿用。主要新增代码：core/solutions.py、core/static_w4a4.py、tools/frozen.py、execution/baselines.py；其余调用方只做配套接入。流程见 [PSEUDOCODE.md](PSEUDOCODE.md)。

**验证记录**

- 新增针对性回归：14 项通过，10.417 秒，见 upgrade_tests_r6.log。覆盖真实 tiny 数值校准/重载、合法码域/17值反例、内部精度例外、组合适配、结果接口、跨请求权限、Memory事实归档、Top-3选优和跨运行持久化。排名测试使用明确的合成指标夹具，不是科学成绩。
- 额外真实 tiny RepQ+GPTQ 适配：h1_adaptation_result.json；两条 LN→Linear 边各读取2个校准样本，导出重载 torch.equal，未测任务准确率。
- 历史 H2 原包：historical_loader_result.json，44.425秒 GPU 加载和单图前向，输出 [1,1000] 有限；使用保存的原数值库快照，未重新校准、未形成新准确率主张。
- 结构审计通过：163 Python文件、231快照文件，structure_audit.json。
- 第一轮全量回归531项曾有3个断言失败、1个夹具错误；已修复并保留 regression.log，工具 Schema 的授权增量单列保存，没有覆盖历史 golden。
- 真实三槽 Generation已通过：生产队列与订阅传输实际调用7次（H1/H2各2次、H3为3次，均包含按需读取续调），未申请数值实验，未启动L3。请求路由gpt-6.1-sol/high；CLI没有独立报告actual model/effort，因此不反推为实测身份。原始审计保留在generation_live/jobs。输入149864 tokens、输出17855 tokens（其中reasoning 9115，不重复相加）。
- 全量回归535项、652.439秒：534项通过，1项订阅并发测试因同时运行的真实Generation占用项目共享槽而超时；Generation结束后该并发组7项隔离复验全部通过（9.825秒）。原失败不隐藏，见regression_r2.log及parallel_runtime_isolated.log。
- 最终77项定向回归全部通过（43.309秒），覆盖新增15项升级用例、执行合同、计划一致性、定制执行器、导出重载、数值导入/隔离、Prompt及工具Schema；见final_targeted_r2.log。最后补齐旧v2 RTN的静态容量核验、classifier命名的可选FP例外，以及空评测列表的无损规范化。最终结构审计及actual CPU接口快照通过，见structure_audit_delivery.json、delivery_contracts/contracts.json。

**真实生成的三个假设**

三份原始HypothesisCard全文见 [HYPOTHESES.json](HYPOTHESES.json)，验证摘要见 generation_validation.json。以下为人工复核后的摘要，没有代写或改写原提交：

| 槽位 | 实际输出与基础方案使用 |
|---|---|
| H1 | “保留完整GPTAQ静态W4A4种子：原始FP输入对齐补偿”。实际读取gptaq_w4a4完整配置，绑定源码版本，原样保留完整种子；设计同码本G_delta置零对照检验原FP输入对齐的贡献，不强加组合。 |
| H2 | “RepQ静态W4A4的非概率激活分通道裁剪：保留LN重参数化的瓶颈适配”。实际读取repq_w4a4配置与方法卡，依据历史B-A诊断优先研究非概率A，保留原方案、LN共享A及概率量化；设计分张量/分通道×minmax/裁剪四组对照，明确历史诊断不证明当前种子同样受限。 |
| H3 | “行质量缺陷感知的静态Value锚点：在四位PV内消除概率量化的共模误差”。以AdaLog为可执行底座，提出PV=P(V−c)+c的静态坐标变换、按概率行和缺陷回归锚点，并设计FP概率消融区分Value中心化解释；恢复常量发生在投影输入A4之前。 |

三者均符合各自研究定位，说明量化边界、失败/反驳条件及尚未实测的范围。H1/H2自然说明基础方案，无新字符串门禁；H3提出新机制但没有宣称世界首创。此次输入使用历史真实事实摘要，角色均如实标记没有独立复验历史原件，不把摘要充作本次Evidence。

**剩余限制**

本次不运行昂贵的完整双轮 L3。三个严格协议适配起点的全量 ImageNet/ADE 成绩仍待测，不能声称质量提升或已有实证 Top-3。通用自定义算法的码本计算仍由原有 Safeguard 审查其实现，宿主独立核验实际边界、冻结回执、重载、数据范围和状态不变；这不是任意源码的形式化证明。旧包精确复现需要匹配原数值库版本。本次所有数值运行均清楚标明 tiny、单图或历史全量证据范围。

最终源码指纹：`820442b80960daf5b9fbc3ab2a5ab8f59f8fdf28138d2fe3281ec51153a1f34e`；真实Generation使用`e8be585dc7f9e69e8be661901c3233a0fb698672ab1a5091fca38297b0d1408e`，完整源码保存于其research_run.json。两版本间仅补上述v2 RTN/classifier兼容边界、空评测列表规范化及回归案例，三假设提示词、Generation逻辑、Top-3策略和DeiT配方未改变。详见[DELIVERY.json](DELIVERY.json)。
最终完整源码另存于本目录final_source_snapshot.json；所有历史失败、第一次回归失败和并发资源冲突日志均保留。
