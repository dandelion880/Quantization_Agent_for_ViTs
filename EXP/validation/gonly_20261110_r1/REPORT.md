# G-only 第一轮简化验证

## 基线与范围

main / origin/main 基线为 `8800006b687c56205ad624bae2ad7156c326d873`，本地 annotated tag 为 `full-baseline-20261110-8800006`。实现位于 `ablation/gonly` 的独立 Git worktree；不合并 main，不覆盖历史实验。源码共用 Git 对象库；`.runtime_pkgs`、ImageNet manifest、reference 及忽略的历史报告链接回原有资产，本轮产物只写本 worktree 的 EXP。

## 实现

- 工作流为 H1/H2/H3 自主研究 → Generation 证据预选 → Generation L3 规划 → 原实验链 → Recorder → 下一轮。取消筛查、三场辩论、H*→H4、父子比较和旧排名调度，不生成 H4。历史卡片/评分包的兼容读取保留。
- Generation 可在生成和综合候选时读取知识/Memory、申请真实 L1/L2、解释正负反馈并修订；预选冻结候选版本和证据引用。保留三槽位原分工、可选判别实验 Skill，不增加最低实验次数、必读门槛或科研审核。
- Generation 的 generate/select/plan_l3 固定 `gpt-6-astra/high`；默认配置、运行覆盖和直接 transport 路由均检查。目录/档位/账户预检沿用原实现，不自动降级。服务端未回显模型的字段保留 null，回显冲突不能接受。
- 局部失败保留已完成候选；预选失败可回退到已提交 H1，并明确只是可用性兜底。已执行失败交给 Recorder，未开始就被预算取消的任务不额外启动复盘。恢复不重开已被采用的旧选择；Full 历史运行只读，不能在此分支静默转为 G-only。
- Coding 单实现、Safeguard、实验服务、资源调度、静态 W4A4 和数值证据校验保留。仅有效的 ImageNet + ADE20K 完整双任务 L3 可按原规则更新 Top-3；提前持久化有效结果，不依赖 Recorder 成功。无需所有 H 都做 L3。
- Eval-L0 保留原评分包字节与哈希，适配 Generation/plan_l3；新 select 轨迹和费用完整记录，但没有对应 v0.1 量表，明确排除科学质量评分，不伪用旧辩论排名标准。

关键实现：`core/workflow.py`、`agents/{tasks,interaction,roles}.py`、`agents/prompts/`、`runtime/{configuration,model_router,driver}.py`、`core/memory.py`；既有实验与方案库实现未改写。当前算法见 [PSEUDOCODE.md](PSEUDOCODE.md)。

## CPU 与接口验证

| 检查 | 结果 |
|---|---|
| 全量 `unittest discover -s QAViT/tests -p 'test_*.py' -v` | **625 项通过，515.387 秒**；`full_cpu_final.log` |
| 结构/语法/源码快照 | 通过；182 Python 文件，287 个快照文件；`structure_audit_final.json` |
| 知识目录 | 通过；4 Skills、21 Methods、54 页；`knowledge_audit_final.log` |
| 当前合同快照 | 通过；`cpu_contracts_final/contracts.json` |
| Eval-L0 原评分包校验 | 通过；`evaluation_rules.json` |
| 实际 Generation 目录/账户预检 | 通过；`generation_preflight.json` |

G-only 回归覆盖三条 H、全部候选选择、L1/L2 申请/回传、证据后的选择修订、候选版本冻结、L3 规划/执行/延期、Recorder、两轮交接、选择失败 H1 兜底、全部生成失败、预算关闭、持久回执/中断恢复及禁止模型降级。数值合同和双任务晋升规则由既有完整回归覆盖；CPU 夹具不是正式科研结果。

## 小规模真实验证

### 自主研究与 Anytime（live_gonly_r2）

真实订阅 + seeded tiny ViT / 合成 RGB32，禁用 L3，预算900秒。共16份调用审计，15次真正开始模型调用；保存 H1/H2、4条实际 torch L2 Evidence、Generation 的 H1 预选及第一轮 Recorder。第一轮 H3 在账户只读预检遭遇传输错误（未调用模型），其缺失被显式交给预选；没有阻塞既有成果或第二轮。

第二轮接收前轮 Memory 后自主增加 count=1 校准对照，得到决策相关的排序反转：

| 校准张数 | 静态 A4 粒度 | 局部算子输出 NMSE | 同4张 search 的整网输出 NMSE |
|---|---|---:|---:|
| 2 | per_tensor | 0.03187111017 | 0.00148388759 |
| 2 | per_channel | 0.00525990191 | 0.00022766676 |
| 1 | per_tensor | 0.00853767002 | 0.00032487156 |
| 1 | per_channel | 0.00204406054 | 0.00035894608 |

只量化 FC2 输入，FP 权重与其余位点不变；每一对使用同条件，所有 search 回放样本及顺序相同。第二对的局部输出排序与整网输出排序相反；不同校准数的局部统计视图不同，不能当作相同分母的跨组性能。它不证明任务精度收益。H1/H2 明确保留完整成熟种子并收窄主张，预选说明 H2 在首轮没有额外收益证据、缓存不算独立重复，也没有重做相同实验。

运行在既有预留策略下于796.25秒收口，状态 `budget_stopped`、`state_saved=true`，仅首轮完成；第二轮候选和第二轮复盘未完成，未达到两轮三槽位全部提交的目标。真实反例已持久化。完整跨轮提交、恢复与 L3 路径另由 CPU 回归覆盖，不能把该烟测写成完整两轮科研成功。

### 真实 DeiT 单实现反馈链（real_deit_code_r2）

**通过**：Generation → Coding → Safeguard → GPU执行 → Generation引用返回Evidence并提交H1。预训练 DeiT-B，GPU0，`blocks.0.attn.proj` 输入；新 `quantizer_v1` 单实现以 `max(quantile(abs(X),0.995)/7,1e-12)` 拟合静态 per_tensor A4，ties-to-even，裁剪到[-8,7]。

实际 L1 仅使用4张 calibration 图像、605184个激活元素；input_relative_mse=0.13588649435，operator_output_relative_mse=0.05084246407。**没有消费 search/final**；请求中的search count=4并不是已执行搜索测评。无训练、权重保持FP，不代表完整W4A4。Generation明确指出两种归一化误差的分母不同，不能把数值之比当作性能改进；完整GPTAQ种子的组合仍待测。证据、代码及审查引用见 `service_validation.json`。GPU已释放。

### 独立 H3 补测

`live_h3_r2` **通过**：一次真实 `gpt-6-astra/high` 调用（111.62秒）提交 H3“固定A4码本下的跨token守恒舍入”。提案改变跨token舍入的联合分配，保留冻结尺度及15码字，说明误差均值界、裁剪误差局限、竞争解释和正负结果如何改变主张。它把上述反例准确标为Brief转述的历史数据，未伪造本次Evidence；本次没有新增数值测量。算法尚未实现或验证，不能据此声称新颖性、性能收益或完整跨轮科研成功。

此前 `live_h3_r1` 在180秒单调用限时及随后剩余预算内未能提交，原件保留。补测总预算仍为300秒，单调用可使用剩余预算，取消传输重试，并要求简洁的可检验提案；没有降低模型/effort、延长旧运行预算或改写科研宿主。

### 轨迹与评价接口

所有已结束运行（含失败原件）的 Journal 哈希链全部校验通过。小 ViT 实际轨迹完成 Eval-L0 prepare/verify：5个可评分任务、630个冻结来源文件；select明确标记无对应量表，其原文与资源仍保留。没有额外调用独立质量Judge，也未产生新科学质量分数。

## 已诊断的失败

早期探索回归暴露了旧 Full 工作流预期、worktree 缺少被忽略的历史链接及开发期间源码身份变化等问题，记录保留在 `EXP/gonly_initial_cpu.log`。随后全量回归的两处失败分别为真实订阅调用占用共享锁干扰预检夹具，以及未开始任务被取消后触发多余 Recorder；已隔离验证顺序并修复归档条件，上表最终全量已通过。

首次真实烟测 `live_gonly_r1` 的验证脚本把 `execution_guidance` 写为字符串，现有接口按字典读取而中断；当时没有数值实验或假设成果。原件保留，修正测试输入后使用新目录 `live_gonly_r2`，没有改写冻结运行配置。

`real_deit_code_r1` 的测试输入错误覆盖了后端样本池大小，触发“显式清单数量不一致”检查；无有效Evidence，Generation如实提交未验证候选，验证器判为 incomplete。修正方式仅是在新输入中保留原样本池定义，由实验请求选择4张校准图像；新目录 `real_deit_code_r2` 通过。没有修改清单、数据划分或宿主检查。

## 复现与边界

从本 worktree 根目录运行 CPU 命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest discover -s QAViT/tests -p 'test_*.py' -v
```

受限真实烟测入口为 `live_smoke.py --name <新的运行名>`（900秒预算、tiny ViT合成输入、禁用L3）；`--h3-only` 为300秒的独立H3补测。DeiT代码链使用原入口：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B QAViT/tests/validate_code_service.py --output EXP/validation/gonly_20261110_r1/real_deit_code_recheck1 --config EXP/validation/gonly_20261110_r1/real_deit_code_input_r2.json --question-file EXP/validation/gonly_20261110_r1/real_deit_code_question.txt --wall-seconds 600
```

新运行必须使用新输出目录，暂停恢复不重置预算。CPU回归与真实订阅测试应顺序运行，避免共享订阅锁干扰。原始输入、Journal、代码快照、模型调用、worker 回执和测量结果留在本地 EXP；轻量摘要、哈希和路径在 [DELIVERY.json](DELIVERY.json)。

未运行完整 ImageNet/ADE20K 双任务评测，未验证 G-only 相对 Full 的最终精度、科学质量或成本收益，也未为 select 发明新的质量评分量表。本轮不变更数据、checkpoint 或正式评价协议。
