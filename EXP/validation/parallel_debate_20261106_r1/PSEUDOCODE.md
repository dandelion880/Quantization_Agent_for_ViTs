# QAViT 当前流程（2026-11-06）

```text
Supervisor（程序）持有唯一 Journal、不可变卡片、任务队列和原墙钟预算
加载 ResearchBrief、角色模型路由、任务提示词及已有提交/尝试/回执
恢复只继续未完成任务；每次模型调用重新形成独立提交快照

for research_round in 原预算轮次:
    Generation 依原自主实验/Memory循环提交 H1、H2、H3
    Judger Screening 依原规则筛查三项
    repeat until 三个初始配对都正式 match_closed:
        for pair in [H1-H2, H1-H3, H2-H3]:
            独立按本场持久状态推进，不等待其它配对
            每exchange依原规则：A立场 + B立场 → 中立仲裁
            继承本场历史裁决、明确采用证据、工具反馈和双方候选/Screening
            同exchange A/B分支仍由原任务祖先及提交快照隔离
            不自动绑定其它场position/verdict正文；Memory仍按需检索/读取
            随时可请求实验、Memory、方法手册等原工具
            仲裁按原规则提前结案/continue，最多5回合
        dispatch + poll（单Supervisor，worker不直接写科研Journal）:
            LLM总数 <= runtime.llm_concurrency（默认4，可设1..4）
            Judger <= 3；同pair <= 1；优先较久未获调用的配对
            其它角色合计 <= 1（原串行约束）
            异步worker合计 <= llm_concurrency+1（默认5，含文献）
            订阅：项目Judger三槽，其它角色原单槽；同账号/模型/隔离/费用预检
            实验申请正常入队，不因Coding正忙拒绝：
                原申请者 → 单份Coding → 绑定执行规格 → Safeguard
                → 数值执行（CPU/GPU共用项目单槽）→ Evidence → 原申请者续调
            不同请求可在Coding/Safeguard/数值阶段流水推进
            每次尝试独立execution_id/目录/哈希/耗时/usage/error
            全局时间沿原墙钟或active_segments结算，不累加worker耗时
    三场match_closed全部存在后才select_parent：
        明确直接胜者/唯一或无合格方案 → 程序引用已有裁决
        循环、平局、未决 → 原Judger排名与不确定性处理
    H*固定 → Evolution沿原自主研究流程生成H4 → Screening
    只安排H4-H* → 原final rank → 原L3规划/执行/延期处理
    Recorder原增量科研综合 → Memory → 下一轮Generation
```

本轮不改变研究预算估计、实验标准、算法、角色及停止边界；显式保存并发1的历史运行继续遵守原配置。历史科研结果不覆盖。
