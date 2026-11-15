# 当前 G-only 贯通伪代码

```text
核对不可变输入、源码、协议、预算；首次运行冻结初始 Memory、方案库及独立 B
恢复沿用 Journal、实际请求与原累计计时，不改写历史失败
for round in 1..configured_max_rounds within cumulative active budget:
    H1 基于完整方案，从已有方法启发提出改进
    H2 基于完整方案，从真实瓶颈提出改进
    H3 探索新机制；所有角色自主决定是否请求有价值实验
    request -> Coding(必要时) -> Safeguard -> 数值 -> Evidence/真实错误 -> 原请求者
    交付 FP/Q/delta、单位、样本范围、分臂身份、反证及精确来源
    Generation 预选中可 revise_candidate，复用同一事务恢复孤立卡字节
    修订发布下一不可变版本；旧证据仅是依据，重新预选绑定确切版本
    Generation 规划 L3 -> 原实现链 -> 同包全量 ImageNet/ADE（资源允许则双卡）
    子任务先核验身份再传递原异常 code/origin/日志；实现错误回送 Coding 重审
    记录逐轮规划、请求、执行、有效 Evidence，保留 B/依法发布或晋升完整方案
    Recorder 综合本轮事实及下一轮问题，下一轮使用 Memory/方案库反馈
仅当全部目标轮具有有效 L3，报告总体数值验证成功；流程/复盘状态分别报告
排除纯数值执行与等待；重叠科研/Coding/审核按活跃区间并集计时
预算耗尽接收已启动数值任务，停止新科研；必要清理并保存未完成阶段
```

本任务验证目录名中的 20261116 为初建时日期标签，实际起止时间以 UTC 日志为准（开始于 2026-11-15）。验证状态与正式科研分开，无新增 Skill 或科学审批门禁。
