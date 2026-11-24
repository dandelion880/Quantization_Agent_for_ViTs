# G-only 科研能力修订与针对性验证

本轮完成程序、工具契约、真实校准搜索及冻结误差传播诊断的配套修改；没有启动下一轮正式贯通实验。正式方案仍为training-free static W4A4，ImageNet与ADE双任务、既定数据划分及冻结迁移协议不变。H1/H2改良完整方案、H3创新、B独立保底和Generation科研决策责任保留。

代码已具备下一次贯通的相应能力，但**当前计算环境尚不满足立即启动条件**：GPU不可用，原DeiT历史GPU包在CPU上严格重载失败。Tiny真实组件、原H3函数与工程恢复已经验证；不能将其表述为原DeiT H3已取得机制结论或新校准策略已有正式收益。

## 先核验，再修改

阅读项目AGENTS、最新源码、三轮正式报告及最新postrun/hardening报告后确认：

| 问题 | 已有实现及本轮处理 |
|---|---|
| Recorder比较时点 | 037211e已有B_transition补丁；保留并回归，没有重复重写。追加纠正两条实际错误L2，并使历史检索带替代记录入口 |
| 错误知识传播 | 原旧Meta-review/L2错误未更正；新增显式维护append_correction，仍走scope/Evidence/supersedes核验，禁止同幂等键换正文 |
| 完整方案退出Top3 | 方案包原件存在，available/bind_solutions仅查Top3；现从已提交library/archive/considered事件发现和加载，含从未入榜的有效完整结果；新运行冻结同协议共享archive |
| 声明配置未消费 | 保留已有有效W/A、局部目标及参数拒绝逻辑；修复RepQ minmax却声明percentiles被忽略的问题，明确拒绝而不猜科学选择 |
| 反馈与接口 | 精简方案目录与重复历史观测，保留关键反例、条件、覆盖与精确入口；同步schema、提示词、方法、Skill及公共数值源码权限 |
| H3恢复 | structured controls来源归一、绑定原执行源码、确切身份别名/唯一包选择；请求错误回Generation，实现错误回Coding，宿主/资源错误保留相应回执，子进程和异常链保留来源细节 |

完整历史方案目录只读本运行已提交快照，不在角色读取时扫描可变共享库；保留既有时间边界和来源真实性核验。精简反馈不自动裁判科学价值，长历史仍按需读取。

## 已追加的历史更正

原正式运行 `deitb_static_w4a4_gonly_20261115_r2` 仅追加维护commit **9150**，替代 `knowledge-9d818a1cedfec85cfed8` 与 `knowledge-c96c9cb2ec4fd44c2c95`。新卡 `MetareviewCard-843410c1b498418da755a616f2c90747@1` 明确author=program、维护操作者、原因及model_called=false，不冒充Recorder调用或正式轮次。

R2/R3候选与预选时B的包身份不同；晋升后当前B与候选相同不能否定新执行和改动。记录相邻完整方案的真实差异，同时保留“没有同期AdaLog-only消融、独立重复或稳健机制因果证明”的限制。

11,442份原cards/commits在追加前后hash比较通过；原head与上一份只读核验一致，整条链重新审计通过，research_run/summary/status哈希与之前记录一致。Journal正常更新派生checkpoint到9150，原正式summary/status仍描述9149完成状态。首次验证器误把派生checkpoint也当作应保持不变的原件，导致最后断言失败；记录该失败后修正核验规则，只读复核，没有重发更正或恢复科研运行。

L2当前检索返回两条version=2；原复盘检索返回knowledge_revisions。测试还验证旧读取时点仍见旧版、原文本不被覆盖、重复提交幂等和改正文被拒。详见 [更正摘要](memory_correction_summary.json)、[精确修订内容](memory_correction_plan.json)、[可复核入口](correct_memory.py)。大完整回执保留本机，不纳入Git。

## 可执行科研能力及数值证据

- 原生校准新增真实block调度、前向/逆向层序、多pass重估、current/float背景；全部进入冻结执行合同、去重键和真实拟合回执。自定义`search_frozen_scales`在完整冻结模型上做block输出重建及少样本整网目标，保存样本、全部候选分数、选中尺度/参数并实际安装；历史隔离类兼容、异常恢复和导出重载均测试。
- metric实际含义分别为MSE、1-cosine、FP教师KL、标签CE、Top1 error以及明确定义的logit平方梯度代理；后者不是完整Hessian。内部拟合只读calibration；外层search和最终双任务分开。三张合成图像上，整网MSE/KL选1.0、CE选0.6、梯度代理选1.2；block all/patch选0.4、CLS选0.8、float背景选1.0、local-FP参考选0.8。这证明选项实际改变选参，不证明哪种更优。[校准报告与复现](../gonly_research_20261116_r1/calibration/REPORT.md)
- `profile_frozen_propagation`支持所选真实block的配对FP/Q输入输出、FP block在Q输入上重放，区分local_added、upstream_mapped、total及有符号交叉项；支持CLS/patch与命名任务表征、冻结FP block替换/W或A旁路，记录下游影响和状态恢复。两张合成图像上局部新增MSE约1e-5而上游映射约3.7e-3；FP block替换改善logits，首层Q旁路无改善。只支持当前背景的诊断解释，不能把FP替换收益当合法W4A4可达收益。[诊断报告与复现](../gonly_research_20261116_r1/diagnostics/REPORT.md)
- 原H3请求明确来源的兼容规范化和恢复链通过；原机制函数在新tiny完整AdaLog冻结包上完成2cal/1search、两个block、五臂测量。实际拟合lambda、P/V与PV捕获一致、lambda_one控制通过；search各臂同分，属于无机制收益支持的有效组件结果。[H3报告与复现](../research_recovery_20261116_r1/contracts/REPORT.md)

已有`diagnose-ptq-failure` Skill已增强，新增冻结传播参考，目录/frontmatter/knowledge导航同步。已验证Generation上下文发现目录、read_asset读取正文及参考；没有声称真实LLM使用Skill后科研质量已提高。本轮未扩展自动Skill蒸馏框架。

## 验证与剩余范围

固定最终源码集成 **238项全部通过（170.716秒）**；知识审计4个Skill、21个Method、55页通过，Skill格式校验通过，34个变更Python文件语法解析及git diff --check通过。最终结果、源码身份和日志摘要见 [RESULTS.json](RESULTS.json)。并行开发阶段触发的源码身份拒绝、旧L3 fixture过期绑定、H3原包CPU加载失败、tiny新插入模块training标志问题和缺标签诊断拒绝全部保留；修复实际问题后重新核验，没有弱化身份/数据/冻结保护。

新增构造器递归eval/freeze及audit检查，修复root.eval后新插入子模块仍training=True的问题。旧preflight/引用夹具改为经过真实G-only候选生成/预选建立绑定；迟到证据夹具只追加Evidence而不误建新候选；Skill链接检查排除代码示例，不再把load_frozen调用当链接。没有削弱生产版本合同。正式结果与原冻结源码绑定保持原样。最终复核另发现来源绑定误从卡片封套取sequence的实现错误，已改取Journal索引，并用真实提交、源码intervention和恢复测试替代掩盖问题的mock；其修复中的一次过宽替换引起旧historical_source的NameError，也由完整回归捕获并精确修复。移除无算法必要性的重估次数上限，仍受原资源/超时控制；选参回执补全逐位点W/A启用状态和配置hash。

尚未验证或明确不支持：

- 当前NVML初始化失败、CUDA不可见。原DeiT包12个fc2融合effective_bias的CPU重建hash与GPU原件不同，其它audit一致；历史包仅存对应hash，未伪造张量或放宽校验。需恢复GPU后重放原包/H3，并做真实DeiT校准与诊断小样本验证。
- 新校准/诊断的正式DeiT、ADE收益、GPU性能、独立种子重复及真实Generation/Recorder模型行为改善未验证。诊断可接显式冻结task forward，但未验证真实ADE head路径。
- 整网选参当前为单Tensor输出；W倍率重编码支持RTN/uniform，其他求解器沿原执行链或custom。整数复合算子、observer_batch_size与新block/多pass/float组合明确拒绝；不是已交付可任意组合的能力。
- 新Skill只交付可发现/可读/有数值API支持的初版，没有强制全网扫描、次数、换位点或科学价值门禁。

本轮Git交付仅稳定源码、提示词、方法/Skill、测试与精确白名单中的轻量报告/复现脚本/数值摘要；模型包、原始运行、缓存及大日志留本机。下一轮正式启动仍由用户另行发起，本轮未自动启动。
