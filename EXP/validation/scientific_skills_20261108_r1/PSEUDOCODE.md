# QAViT 当前流程伪代码（2026-11-08）

依据当前 `core/workflow.py`、`core/experiment_requests.py`、`agents/interaction.py` 和 `runtime/driver.py`；Skill 只改变资源发现，不增加控制流阶段。下面是实现摘要，不是替代源码的额外规则。

```text
输入：用户 ResearchBrief、明确模型/数据与评测合同、运行预算和资源配置
初始化：冻结有效输入、源码/Prompt 身份；建立 Journal、任务队列、预算和 Memory
        初始化本协议完整方案库（三个成熟待测起点或已存在的同协议 Top-3）

while 当前授权范围/预算尚允许推进：
    接收完成的 worker 回执，核验来源与执行身份，发布事实
    依队列依赖、角色并发、GPU 租约及预算调度；不由 Supervisor 代替调度

    对当前轮 H1、H2、H3 建立独立 Generation 任务
    对每个研究任务，按需重复：
        组装 common + 角色 + purpose 提示词与原有工具/提交合同
        交付 ResearchBrief、任务绑定事实/反馈、预算、现有方法/方案目录
        若为 Generation/Evolution/Judger/Recorder 且 Skill INDEX 可读：
            交付短 INDEX，登记原有正文交付位置；不加载 Skill/参考正文
        角色自主决定：
            read_card / retrieve_memory → 返回可见历史正文或精确展开入口
            read_asset → 按原哈希/权限/分页读取所选知识或来源
                         相同结果复用/去重；读取失败回原任务，非科研否决
            literature_search/fetch → 返回真实来源或失败，不能假装已读
            request_experiment → 按下方实验链执行，结果回同一研究任务
            submit → 程序绑定固定身份并发布科研正文
            clarify → 仅保留真正缺失且影响科学定义的问题
        可以零 Skill、零实验直接提出待验证假设；没有 Skill 路由模型

    对已提交 H 做原有筛查
    三场初始比赛各自独立推进（Judger 最多三并发）：
        H1–H2，H1–H3，H2–H3
        每场：按当前争点提交双方立场 → 仲裁 → 继续交锋或结案
        本场私有/同回合对方未汇合立场按原可见性隔离
        结案固定，包括 tie / undetermined；后到证据不改判
    等待三场均结案
    从已有直接结果/资格持久化选择 H*；必要时由原 Judger 做选择分析

    若存在合格 H*：
        Evolution 仅基于固定父卡 H* 生成 H4
        同样可自主读取/实验/续调；不得另选或增加父卡
        筛查 H4；仅推进 H4–H* 父子比赛
    根据原两阶段选择规则确定最终候选

    若运行允许 L3 且有最终候选：
        Judger 提交正式实验草案（或明确 defer_reason）
        执行原实验链，保留有效 Evidence；解释不成为证据发布前置
    程序归档增量来源，按需要运行 Recorder：
        科学分析、下一轮指引/问题、可选知识更新
        绑定原始 Evidence 条件和比较臂；不新增全量逐条审计
        科学复盘失败保留 pending，与事实已归档分开；依原逻辑可继续
    用同协议有效完整双任务结果考虑更新独立 Top-3，保留旧来源
    发布 round_finished，将绑定反馈带入下一轮

正常结束/预算停止/失败：停止并确认 worker，保存真实完成范围、未完成事项与回执
恢复：使用 Journal/冻结输入/身份与剩余预算，不重置历史或绕过源码变更规则
```

```text
实验链(request, 原研究任务)：
    保存研究者原始科学目的与所需测量；宿主补确定性身份、资源和条件
    若已有内置服务且无需源码：
        冻结计划 → 原有执行核验/缓存 → 数值 worker
    否则：
        Coding 交付一份候选实现及必要 execution_choices
        程序绑定执行 → Safeguard 对照原始需求审查 → 数值 worker
        修订/失败/澄清沿原有恢复路径处理，不另设 Skill 审核
    验证真实回执、范围、导出重载和必要结果；生成 Evidence 或失败回执
    结果返回原任务，由原研究者解释并决定继续实验/修订/提交
    L3 满足原合同的双任务结果可更新运行内 incumbent；与场内裁决独立
```
