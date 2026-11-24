# 三轮 QAViT G-only 正式科研实验

三轮均完成全量双任务 L3、Recorder 和托管清理。分割指标逐轮小幅改善，分类未超过初始历史 B；H3 没有有效数值验证，不能声称已验证创新机制。

## 修复、验证与冻结

分支 `ablation/gonly`；修复提交 `70a14ac`，冻结 core SHA256 `6f5db3ec908f9e44ae2616b1e2f8ad6a8d2468b3f7ceffbcbe5c1f4a87ad2dec`。源码、输入、初始 Memory、方案库和 B 的启动前原件见 [experiment_identity.json](experiment_identity.json) 与 [input.json](input.json)。运行中核心源码未改，无外部 intervention、暂停恢复或预算延期；正式结束后另有B比较时点的工程修复（见后文），未用于本轮；原冻结快照保持不变，原manifest/summary不回写。

修复覆盖候选修订提交前后幂等恢复、严格臂身份投影、子进程异常类型/来源及 Coding 回送、恢复实际请求/deadline、逐轮 L3 与总体覆盖状态；接口、提示词、测试、文档同步。保持原 Coding→Safeguard、Memory/Recorder、方案库/B、预算/资源机制；无新增科学门禁、最低次数、强制修订、新 Skill、小函数式 H3、Judger/Evolution。

固定源码集成95项全过，专项故障注入及真实 CPU 修复链通过。并行开发期间的源码身份拒绝及所有中间失败保留，未弱化检查。新 GPU 工程验证 search64+ADE1 的单/双卡质量指标、样本、整数计数/混淆矩阵逐值一致，测量重叠12.353678秒；过期旧请求与新deadline、真实身份拒绝及释放通过。仅为工程小样本，未计正式 L3。

另用独立新 tiny 状态完成6次真实 Generation调用、3次真实PyTorch合成计算：反证返回→H2@1修订@2→重新预选→确切@2参数绑定并执行L2，9项最终核查通过。保留资源字段拒绝及验证器误将 synthetic_computation 要求为 real 的初次失败；只读修正核验，没有重呼模型或补造测量。详见 [修复与验证报告](../../validation/gonly_hardening_20261116_r1/REPORT.md)。

官方CLI同步并固定0.169.0；原隔离和精确版本拒绝保留，三种既定科研模型及sol/ultra监控真实传输探针通过。科研请求仍为Generation `gpt-6-astra/high`，Coding/Safeguard `gpt-6.1-sol/medium`，Recorder `gpt-6-luna/high`。监控会话可见metadata为`gpt-6.1-sol/ultra`；服务端actual均未回显，保持unknown。

## 协议与正式结果

官方非蒸馏DeiT-B，training-free、FP32 reference QDQ、static_w4a4/1；固定32张分类校准，ImageNet final 50,000张＋ADE20K val 2,000张。固定search池1,000；ADE直接迁移冻结状态，原neck/head不训练、不重新校准。原模型、数据、任务头及预处理身份保留。每轮都从宿主核验的独立重载实例实际完成全量计分；每次输出parity仅核验1张search，未扩大为全图parity。全部Evidence原件hash和代码/Safeguard/选择/候选版本绑定通过；3个L3是本次新执行，缓存复用0。

| 方案 | Top-1 | Top-5 | ADE mIoU | mAcc | aAcc | Δ初始B Top-1 / mIoU（百分点） |
|---|---:|---:|---:|---:|---:|---|
| 配对FP（各轮相同） | 81.796 | 95.594 | 39.829582 | 50.606542 | 78.753181 | — |
| 初始历史B AdaLog（未在本次重测） | 72.482 | 90.294 | 28.055648 | 37.788110 | 71.391383 | — |
| R1 H2 | 72.350 | 90.212 | 28.459682 | 38.045538 | 71.526040 | -0.132 / +0.404034 |
| R2 H2 | 72.394 | 90.210 | 28.479136 | 38.107880 | 71.592908 | -0.088 / +0.423489 |
| R3 H2 | 72.190 | 90.236 | 28.555078 | 38.227104 | 71.501445 | -0.292 / +0.499430 |

R3相对R2：Top-1 −0.204点、mIoU +0.075941点。最终按原mIoU优先、Top-1次序保留R3为B/incumbent，Top3为R3/R2/R1；初始历史B移出Top3，但其启动快照、代码、包和历史原件完整保留。每轮预选时B独立可用，不占H槽；没有以保底成绩冒充新候选成绩。

相对FP，R3仍低9.606点Top-1、11.274505点mIoU。差值是同协议单次观察，不是独立种子重复或置信结论；不支持Full/G-only架构因果比较、首创性、稳健泛化或原生INT4加速。QDQ延迟不可当作原生低比特吞吐。

## 逐轮科研与反馈利用

| 轮次 | H1 / H2 / H3及实际决策 |
|---|---|
| R1 | H1从完整AdaLog提出LN→Linear通道平衡；H2用两项真实L2收窄为首层Q逐head、K共享尺度；H3空间误差反馈保留理论提案。Generation保留K/深层CLS和部分head反例，选H2完成全量L3。 |
| R2 | H1依据前轮联合W/A与CLS反例改双token约束；H2在同一冻结包/真实量化背景中用条件PV目标选码，L2显示CLS反向与共享Q对照更好Top-1，仍以未知迁移收益为由自主选H2。H3固定码本二元联合机制因来源绑定失败无有效测量。 |
| R3 | H1完整量化背景L2保留FC1局部收益、撤回任务增益保证；H2从固定九配置库用32cal教师KL选首层共享minmax Q，256search虽KL改善但Top-1未变、PV/logit误差恶化。Generation选H2，全量结果显示分类回退、分割小幅改善。H3三次加载拒绝后仍无有效机制证据。 |

R2、R3确实读取并引用前轮Meta-review/Evidence，候选正文说明反馈如何改变干预、约束与预测；R2/R3 H2的feedback_used字段为空，但sources有确切Meta-review引用与影响说明，不能据空字段断言未利用。三个Recorder均完成，R1/R2复盘实际包含配对FP与量化损失，旧ADE FP交付缺口未重现。本次正式运行没有调用revise_candidate发布新版本；该工具的实际闭环由独立准备验证覆盖，不将生成前的方案收窄伪称正式版本修订。

现有Skill正文实际读取1次、Memory retrieve 9次、成功read_card 84次；目录交付不算Skill调用。真实Memory利用有出处，Skill与科研收益没有独立因果证据。没有新增或自主沉淀Skill。

终态人工核验发现R2/R3 Recorder把候选与已经晋升后的当前B比较，并据相同model_ref弱化新增差异；两份原包身份实际不同，独立CPU比较也确认首层Q状态变化。这种比较时点混淆进入两条L2知识，降低后续科研综合可靠性。原Meta-review/Memory完整保留，详见scientific_memory_diagnosis.json；不得把相同当前B身份当作未发生新执行。此后只补工程上下文：显式交付预选时B与复盘当前B，不替Recorder作科学判断、不重写历史或追加正式轮次。结束后的工程补修20项回归通过；只读历史核验14,204文件未变，未重新调用Recorder，不能声称历史Memory已纠正或新提示词实际模型效果已验证。修复后core为92dbd9fc776a441b6e2b782f41874b6223cca04e16a0215742c4d120aa0c6c57，正式结果仍绑定原6f5db3快照。修复验证见postrun_engineering_fix及[后续工程报告](../../validation/gonly_postrun_B_context_20261115_r1/REPORT.md)。

## 失败与干预

17次数值执行：9份有效Evidence（6个L2、3个全量L3），8次失败。R1漏交分类样本/指标封套、R2包装eval状态、R2 L3路径类型错误均沿原Coding→Safeguard内部修复后成功。H3在R2两次、R3三次因未授权冻结来源失败：原引用仅写controls.evidence_refs，顶层evidence_refs/solution_refs缺失，编译没有丢合法来源。不能由改源码补授权，重复修复消耗是实际局限；失败既不是机制反证也不是有效创新验证。另有执行前calibration.split补全及工具读取拒绝，原始反馈与源码均保留。

外部科研干预0；监控者只做只读核验和归档，未代写候选、改评测或放松校验。13个终态failed任务包含历史数值/工具失败，终态活动任务0；不把历史失败列表当作尚在运行。完整原因、内部9次实现修订、17份Safeguard审查及准确执行版本见RESULTS.json及原始Journal。

## 成本与预算闭合

Agent活跃 **12189.096秒 / 203.152分钟**，上限14,400秒；余2210.904秒。排除纯数值等待14111.012秒 / 235.184分钟；单一闭合segment墙钟26300.107秒 / 438.335分钟。重叠Coding/推理/审核正确计活跃，无预算重置、延期或额外轮次；因三轮完成停止，不填满剩余额度。

| 科研角色 | 调用次数 | Input token | Output token | Cached input | Reasoning output |
|---|---:|---:|---:|---:|---:|
| coding | 70 | 3,875,805 | 147,576 | 141,440 | 13,946 |
| generation | 71 | 3,693,159 | 70,609 | 201,600 | 20,524 |
| safeguard | 37 | 2,224,481 | 11,621 | 95,872 | 3,584 |
| recorder | 4 | 280,709 | 10,487 | 55,808 | 6,067 |
| 合计 | 182 | 10,074,154 | 240,293 | 494,720 | 44,121 |

182次模型调用全部完成、用量齐全；Router累计7491.136秒，模型区间并集7069.354秒，均不冒充纯推理时延。GPU分配墙钟 **15802.223 GPU秒 / 4.390 GPU小时**；worker累计同值，峰值并发2；失败worker 301.900秒。此为分配区间，不是GPU kernel计算、能耗或货币账单。完整L3均走custom未声明并行能力的单GPU兼容路径，不能宣称双卡正式加速。

准备单列：10次真实模型调用，input208,969、output6,465，含独立Generation6次与传输探针4次；tiny数值worker9.227秒。真实GPU工程检查单/双卡各自elapsed见RESULTS.json，整个准备GPU租约区间未独立计量，保持unknown，不补造精确总量；这些费用均不混入正式科研预算。

外部Codex准备、修复、监控和报告会话截至2026-11-15T22:52:59.676795+00:00：input **198,112,569**（cached **196,959,360**）、output **238,583**（reasoning **75,702**）。这是root和4个子代理的重复上下文累计，不是唯一材料量；不含该时点之后的最终交付。程序采样每30/180秒且不调用模型，但外部会话Token仍很大，不能称整个监控过程低Token。

cached input是input子字段，reasoning output是output子字段，不重复相加。以上三账相互区分；订阅实际货币成本和服务端actual模型未知。外部账可由external_cost.py重算。

## 终态、证据入口与交付

终态completed；workflow_completed、formal_validation_success、scientific_review_complete、state_saved、monitor_cleanup_completed均true，expected/validated/completed_rounds均[1,2,3]，supervisor退出0，finalization_errors=[]，预算segment闭合。自有launcher、worker、CLI退出，5张GPU均0MiB，已用UUID租约可独占重获。详见monitor_terminal_review.json；未用report命令覆盖托管清理字段。

| 轮次 | 正式Evidence | 执行 |
|---|---|---|
| R1 | `EvidenceCard-34132bb00ead4c6f8f6b8ecbc4a26107@1` | `task-ee70cbb59cd25c6937cef641-a0001` |
| R2 | `EvidenceCard-46fbe7b34ad44eea8f72d04756d69074@1` | `task-cb87f2f8ee12da235667770c-a0001` |
| R3 | `EvidenceCard-ad61016bc2954c3695078601425890ae@1` | `task-bd1f367496c783dd998c7cff-a0001` |

完整机器结果：[RESULTS.json](RESULTS.json)；托管原件：[summary.json](../../runs/deitb_static_w4a4_gonly_20261115_r2/summary.json)，同run内Journal/cards/jobs/code/Memory保留。round_audit.py、make_monitor_draft.py、compare_frozen_payloads.py、external_cost.py、finalize_delivery.py为可重算派生入口。原始包、日志、请求/响应和大索引留本机忽略目录；Git只跟踪源码、轻量输入/报告和核验入口，不覆盖历史、数据、checkpoint或main。

实验支持自主诊断、反证利用、单点量化选择与双任务反馈贯通；分割约半个百分点的单次改善伴随分类下降。未完成项是H3机制有效实证、独立重复/置信范围、源绑定错误的科研侧可靠使用、历史Memory比较时点混淆、FP差距主因及Full-vs-G-only因果比较。没有把历史成绩、缓存或小样本替代正式成果。
