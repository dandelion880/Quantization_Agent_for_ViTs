# 本次 G-only 贯通实验算法与预算

两轮，每轮 H1/H2/H3 → Generation 预选 → Generation L3 规划 → Coding（需要新增绑定时）→ Safeguard → 原协议数值执行 → Recorder → 下一轮。Agent 活跃预算 10800 秒，active_segments，exclude_numeric_wait=true；L1/L2/L3 独立默认超时 3600/10800/86400 秒。

以下沿用当前 G-only v1 伪代码；实际完成位置以 Journal、summary.json 与最终报告为准。

# QAViT G-only 伪代码

```text
Input: ResearchBrief, 固定评价协议, 预算, 方案库 Top-3
Generation := gpt-6-astra / high（覆盖、模型目录及回显冲突均不允许降级）
Restore Journal / Queue / Memory / Budget；原始事实不可改写

for round in remaining rounds:
    for H in [H1 成熟方法兜底, H2 瓶颈改进, H3 开放创新]:
        Generation 读取 Brief、绑定反馈及按需静态知识/Memory
        while Generation 决定继续研究且预算允许:
            自主定义能区分竞争机制、影响下一步决定的 L1/L2 实验
            原需求持久化；新代码走 Coding 单实现 → 绑定 → Safeguard
            现有服务沿原冻结接口执行；两条路径均返回实际执行结果
            返回 Evidence 或失败 → Generation 解释正负结果并修订当前方案
        提交该 H 的候选、实际支持与限制；已完成事实即时保留

    Generation 综合已完成 H1/H2/H3 及证据，按需继续取证
    固定预选卡：候选版本、selected_slot、理由、证据、未解问题
    局部失败保留已有候选；预选失败可用已提交 H1 兜底，明确未比较
    不生成 H4，不调度筛查、辩论、父子比较或旧排名

    if L3 enabled and candidate selected:
        Generation 规划原 ImageNet + ADE20K 双任务协议
        经原 Coding/Safeguard/执行链运行选定候选，不强制所有 H 都做 L3
        失败/澄清交回 Generation 的计划修订任务，或如实延期
        program 验证真实 L3、完整数据、固定模型、静态 W4A4 和同协议身份
        program 按原 (mIoU, Top-1) 比较更新 Top-3，不依赖 Recorder 成功

    Recorder 综合新增事实、未决问题与失败，更新 Memory
    Recorder 失败：保存待复盘材料，后续 Generation 接收原始事实和缺口
    保存 round_finished；进入下一轮

budget stop / interruption:
    停止 worker → 接收持久回执 → 保存有效 L3 方案与所有阶段成果
    保存未完成状态与复盘待办；不造实验、假说、排名或成功状态
resume:
    核验冻结身份，复用既有版本/执行结果，不重置预算或重开已采用的旧决定
```

L1/L2 用于机制研究和候选筛选。正式性能结论与 Top-3 更新只接受有效双任务 L3。Skill 是可选策略资料，不新增阶段、最低实验次数或科学审核。
