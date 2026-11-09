# QAViT 当前科研流程与独立 Eval-L0

科研主流程保持原样：

```text
ResearchBrief + 原科研预算/运行段/源码身份
  → Generation H1/H2/H3（可请求工具与实验，读取真实回执，再继续同一角色任务）
  → 各候选 screen
  → H1–H2、H1–H3、H2–H3 各自 A/B position 汇合后 arbitrate
  → 已结案直接关系明确时程序 rank；需暂定选择时 Judger rank
  → 持久化唯一 H*
  → Evolution 仅从 H* 形成 H4 → screen → H4–H* 对辩与最终选择
  → 按原协议可选 plan_l3 / 合理 defer
  → Recorder 复盘与原 Memory/方案库机制 → 后续轮次

任一合法实验请求：
  原始科学要求 → 需要时 Coding 单实现 → 原宿主绑定 → Safeguard
  → 原预算/GPU/执行合同下数值执行 → Evidence 或失败回执
  → 原请求者续调；既有缓存与独立新执行仍分开

本轮未插入新角色、审核、必填字段、评分门禁或额外科研调用。
```

完全独立的离线旁路：

```text
用户显式选择已有 run 和新的 EXP/evaluations/<eval_id>
  → 读取评分包唯一原文；核验来源哈希和当前 purpose→role 覆盖
  → 直接只读 commits/cards/jobs；不构造生产 Journal/Workflow
  → 校验提交链、卡片、实际请求与回执；保存所读原字节的 SHA256 对象
  → 归并 task_id + purpose 下全部 execution_id / step / attempt
      排除程序 rank、Supervisor 程序身份、没有 LLM 执行的科研任务
      保留失败、缺日志、预检与孤立执行的独立资源计量
  → 依据每次实际发送请求冻结角色可见上下文
      不沿菜单/引用补原文；分页/截断/未读保留
      自己的输出/源码与宿主接受的提交分别保存
      后到实验及其他角色结果只进入 posthoc 报告
  → 计算 Token/资源/时间口径/重试/工具与已有执行状态
  → 写出无科研分数的准备报告

用户显式选择已准备 case（或显式 --all）：
  → 冻结身份 + 适用唯一 Rubric + 原 Judge Prompt + 实际资料
  → 身份盲化/脱敏；不传 posthoc；过大输入明确未评分
  → 独立 worker 复用原已核验纯订阅传输
      固定请求 gpt-6.1-sol / medium；无 GPU/工具 host/科研队列或预算
      独立尝试上限/超时/锁/原始回执；额度或隔离故障停止
  → 原 Schema + 身份 + criterion 集合/顺序 + 状态/分值 + 来源/摘录校验
      失败只影响离线任务；保存失败，不伪造分数
      中断后可接收同一次已完成响应，不重复调用
  → 程序计算 Q 与 scored/applicable/total、Unknown、N/A、置信度和覆盖
  → role×purpose 分组；Generation 按 slot、rank 按 stage
  → 独立列原科研成本与 Judge 开销，保存递增版本报告

未来人工专家抽样、科研代表任务、因果回放/数值验证均由用户后续安排。
```
