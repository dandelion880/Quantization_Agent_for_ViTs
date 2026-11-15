# 当前 G-only 研究伪代码

```text
读取不可变Brief、协议、预算、源码身份；恢复沿用Journal和既有计时
加载同协议方案库；独立保存B=有效incumbent/实测完整方案
保持待测种子可发现；历史成绩明确标为历史
for round within original scope/budget:
    H1: 基于明确完整方案，提出已有方法启发的改良
    H2: 基于明确完整方案，依据真实瓶颈诊断提出改良
    H3: 探索新机制
    Generation可按需request_experiment，无最低次数或逐候选实验门禁
    原请求 → Coding(必要时) → Safeguard → 数值 → Evidence/失败 → 原任务
    投影关键证据：FP/Q/delta+unit、分臂、范围、协议/模型、失败与反证
    长样本身份保留hash/count/精确读取入口，原证据不变
    while Generation尚未提交预选:
        若改变科学干预：revise_candidate(确切父版本,完整候选,理由)
        原子发布新版本；恢复同工具任务幂等；旧证据仅作依据
        淘汰过期预选任务，以全部当前候选版本重开预选
        按下一笔科研投入价值选H1/H2/H3或null；B不参与成熟度竞赛
    plan_l3绑定selection_ref与hypothesis_ref的确切版本
    工程失败沿原Coding/计划任务修复，不能静默改科学定义
    同冻结模型包：有两张可租GPU且加载能力支持时
        主卡ImageNet ∥ 次卡ADE；每任务内部仍单卡
        同包hash/模型/协议检查；部分失败保存；统一deadline/资源清理
    否则显式串行并记录原因；原预算口径不变，不累加子任务时间
    仅有效原双任务正式测量有资格更新incumbent/方案库/B
    Recorder综合证据、用稳定question_id关闭问题，保留未完成探索
```

历史冻结方案只经已核验来源绑定加载；不重校准、不把历史成绩算作本轮新测量。无Judger/Evolution，不生成新Skill或自主沉淀Skill。本验证不启动下一轮完整贯通。
