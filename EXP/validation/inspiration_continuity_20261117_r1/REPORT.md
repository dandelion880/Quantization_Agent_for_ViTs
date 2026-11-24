# 独立灵感库与 H3 连续演化

本次在现有 gonly 源码上补齐灵感保存、跨轮接续和原代码交接；没有启动真实科研运行或改变评测协议。

## 修改

- `QAViT/core/inspirations.py`：独立于完整方案库，以原 Journal 保存简短灵感的稳定 ID、修订、active/paused、进展、问题、下一步、引用和已吸收部分。仅保存不排实验；切换只暂停旧方向。Generation 决策，Recorder 仅更新已有进展和引用。
- `workflow.py` / `interaction.py` / `experiment_requests.py`：H3 默认绑定单一活跃灵感，普通研究、最终候选、修订和 L3 关联确切身份；综合取证可显式提供 inspiration_id。由既有请求、代码 manifest、源码快照和 Evidence 投影恢复视图，含失败/未完成产物，不重复存储完整方案。Coding/Safeguard 沿已有 implementation_history 取得确切源码，不从摘要重新生成。原读序号、哈希核验、幂等恢复与预算仍适用。
- H1/H2 增加兼容的 solution_derivation：完整父方案 ID、实质改动、回退方式及宿主绑定的父记录哈希。下一轮保留各槽位派生入口，代码结果记录已交付父代码 ID；吸收可用灵感部分时引用确切 H1/H2 候选。旧卡片无需回填字段，库外完整父方案沿原正文来源定义。
- 同步 Generation、Recorder、Coding、任务提示词、相关科研约定、生命周期、导航和 Prompt 哈希基线。删除“每轮必须新机制”的误导；修正 Skill 中可能与三个最终候选全部 L3 冲突的表述。

## 低成本检查

命令从项目根目录运行，均设置 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`：

```bash
python -B -m unittest QAViT.tests.test_inspirations QAViT.tests.test_multi_candidate_l3 QAViT.tests.test_candidate_revision QAViT.tests.test_code_asset_ownership QAViT.tests.test_fallback_solutions QAViT.tests.test_structure.StructureContracts.test_final_prompts_match_baseline_or_explicit_fixes QAViT.tests.test_structure.StructureContracts.test_model_schemas_expose_science_without_persistent_identity_fields -v
python -B -m unittest QAViT.tests.test_inspirations -v
python -B QAViT/tests/audit_knowledge.py
```

- 主批 39 项通过；补充修复版本精确绑定后，灵感专项 7 项通过。两批合计 40 个不同用例，重复运行不算新增覆盖。日志为本目录 `tests.log`、`continuity_final.log`，轻量摘要见 `RESULTS.json`。
- 脚本化低分 H3 未进入 Top-3，重启与下一轮仍保持同一灵感 ID、原代码 SHA256、原始配置、结果和 Recorder 待办；下一轮 Coding 直接收到原源码。不同修复版本分别绑定各自执行与结果，未完成父请求下的子执行 Evidence 仍可恢复。
- 保存无代码灵感不创建实验、不占数值预算；同 ID 更新、单活跃、暂停/恢复/切换、工具发布中断重放、过期更新拒绝、旧快照隔离、源码篡改拒绝均覆盖。
- Recorder 可记录已吸收部分，不能暂停/切换；H1/H2 各自父方案可追溯；独立 B 与 ADE20K 排序保留；三个最终候选全部进入 L3，完成顺序、局部失败、精确候选绑定和候选修订恢复保持有效。
- 静态知识链接/Skill 索引、修改文件语法和 `git diff --check` 检查结果记录在 `RESULTS.json`。

这些是宿主状态、合同和交接检查。涉及低分、完整指标形状及排名的输入均为明确脚本夹具；不代表真实 ADE20K/ImageNet 成绩或真实 LLM 决策质量。

## 保持与限制

普适量化目标、冻结 Brief、量化约束、数据/checkpoint、评测协议、Top-3 更新、B 保留和实验预算未变；灵感无额外候选槽位，成熟且按原规则达标的结果仍进入正式方案库。历史实验、原始结果与已有源码资产未回写。

灵感库目前在本运行内跨轮和重启复用，没有跨运行自动合并或自动迁移代码资产；旧运行不会凭空生成已审核灵感，升级恢复仍须原有显式 intervention。父方案实质改进、是否继续及下一步科学价值由 Generation 判断，宿主只校验身份与保存事实。没有开展新一轮真实 LLM/GPU 科研、搜索空间重设计、组件消融或大规模验证，因此不宣称科研性能提升。

最新流程：[PSEUDOCODE.md](PSEUDOCODE.md)。
