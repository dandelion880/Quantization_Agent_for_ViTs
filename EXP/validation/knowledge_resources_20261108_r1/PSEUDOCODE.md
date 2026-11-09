# 当前科研流程伪代码

本次资源整理不改变调度。以下依据 core/workflow.py、core/experiment_requests.py、agents/interaction.py、runtime/driver.py；省略工程细节而保留研究关系。

```text
初始化或恢复 Workflow：
    冻结/核验研究输入、源码身份、预算及可用资源
    从本运行 Journal 重建队列、运行状态和 L0–L3 Memory
    初始化/恢复协议方案库快照（待测种子不带虚构分数）

按预算及现有 Supervisor/异常处理推进轮次：
    H1/H2/H3 各完成 Generation 自主研究任务
    筛查后，三场初始 Judger 辩论独立并行推进、分别结案
    全部结束后保存 H*；已结案判断不再改写
    Evolution 只加强 H*，经同一自主研究循环形成 H4
    筛查 H4 后只开展 H4 与 H* 的父子挑战并作最终选择
    如科学判断与预算需要，沿现有 plan_l3/实验链完成 L3
    有效完整真实双任务结果按原规则更新 incumbent/同协议 Top-3
    Recorder 综合本批科研材料；失败/待补来源继续 pending
    若预算及条件允许，带反馈进入下一轮

每个研究任务可反复：
    按角色/purpose 组装 Brief、任务、预算、相关短卡及工具接口
    给 knowledge/INDEX 的小指针（不读取正文）
    对相应角色交付 Skill/Method/完整方案短目录
    研究者自主选择：直接提交、读取资料/历史、或请求实验
        read_asset → 原授权快照与哈希 → 文本分页 → 去重 → 续调
        retrieve_memory/read_card → 当前运行可见来源 → 续调
        request_experiment → 冻结原科学需求
            已有服务沿原链执行
            需新代码时：Coding 单实现 → 宿主绑定 → Safeguard → 执行
            归档程序 Evidence/失败与真实条件 → 返回原研究者
    Skill 可以跳过、组合；读取失败不新增门禁或强制测量

提交：
    保留有效科研正文；可选附加条目按现有逐条规则处理
    sources.ref 为 Journal 卡片引用
    兼容将其中当前授权静态文档路径存为 sources.document
    文档不进入 Evidence 支持链，声明不替代实际读取/测量
    Journal 原子发布，程序维护来源；历史已发布卡片不改写
```

本轮没有新增科学阶段、LLM 路由、记忆层、输出必填字段或审批。具体合同、角色权限、计时/GPU 与恢复语义以源码为准；本伪代码属于本次 EXP 交付快照。
