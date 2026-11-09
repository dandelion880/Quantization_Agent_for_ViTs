# Eval-L0：独立历史科研评价

此入口只读取已有运行，评分输出不进入科研 Journal、Memory、Top-3、任务队列、预算、实验合同或运行门禁。生产 `run.py` 不导入评价模块；评分不启动实验或 GPU。用户提供的 v0.1 评分包是唯一规则来源，人工抽样、代表性科研任务、Eval-L1 配对回放与额外数值验证不在本版范围。

## 文件与来源

| 位置 | 职责 |
|---|---|
| [configs/evaluation/v0_1/](../configs/evaluation/v0_1/README.md) | 原包 15 个文件，逐字节保留；六类 Rubric、两个 Prompt、Schema、原则、接入与验证要求、来源 |
| [provenance.json](../configs/evaluation/v0_1/provenance.json) | 原 ZIP 文件名/SHA256、各成员路径/字节数/SHA256；`rules` 命令验证完整性 |
| [tools/evaluation/](../tools/evaluation/__main__.py) | 只读冻结、计量、评分校验、独立 Judge 和报告 CLI |
| [tests/evaluation/](../tests/evaluation/test_eval_l0.py) | CPU 接口和故障注入测试；夹具没有科研金标准，不是科研样本生成器 |
| `EXP/evaluations/<eval_id>/` | 本次冻结原件、案例、Judge 原始回执和报告；每次准备使用新目录 |

评分规则选择 `configs/evaluation/` 是有意的技术例外：科研 `RoleInteraction.knowledge_roots()` 自动授权 `docs/` 和 `knowledge/` 文本。把 Rubric 放入它们会扩大被评价者可读的科研材料。此位置无需修改生产权限代码，也不把评分规则加入静态科研知识 INDEX 或角色 Prompt。接入文档只提供运维入口，不复制规则正文。

## 使用

从项目顶层运行，Python 环境需要既有 `jsonschema`（项目已声明 4.22.0）。入口不安装依赖、不加载数值后端、不要求 GPU。

```bash
python -B QAViT/tools/evaluation rules
python -B QAViT/tools/evaluation prepare EXP/runs/<已有运行名> --output EXP/evaluations/<新eval_id>
```

`prepare` 无 LLM 调用，生成全部实际可观察角色任务并立即给出资源报告。`manifest.json.cases` 列出 case_id、原 task_id 和 task_key。按任务明确指定真实评分对象：

```bash
python -B QAViT/tools/evaluation judge EXP/evaluations/<eval_id> --case <case_id或task_id> --max-attempts 2 --timeout 300
python -B QAViT/tools/evaluation report EXP/evaluations/<eval_id>
python -B QAViT/tools/evaluation verify EXP/evaluations/<eval_id>
```

可重复传 `--case`；明确需要评全部已准备任务时用 `--all`。不存在自动抽样或生成科研任务的命令。默认评分器固定为 `gpt-6.1-sol / medium`，不读取 `configs/models.json` 的科研角色路由。`--cli-path` 可选择已核验版本的本机 Codex CLI；不自动升级、不降级模型、不切换 API 后端。

`--max-attempts` 为该评价案例累计上限（1–3，默认 2），不是每次重启重新计数；鉴权、额度、计费边界、隔离、模型身份等故障会结束本次评分会话。默认每次模型尝试含预检限时 300 秒；`--max-input-characters` 默认 600000。超长材料保留完整冻结对象并报告 `input_limit`，不静默裁剪。可在确认模型上下文允许时显式提高；CLI 上下文错误仍作为评价失败记录。

准备目录禁止复用。评分重跑复用已经通过校验的结果；崩溃后可接收已完成但尚未校验的同一次调用，失败尝试原件保留。若 `prepare` 中断，保留其目录用于审计，换新 eval_id 重新冻结；不把半份准备结果标为可评分。报告按 `reports/report-0001.{json,md}` 递增保存，`latest_report.json` 仅为派生导航。

## 真实材料与可见性

读取器直接校验提交序列、提交链哈希、TaskCard/事件一致性、卡片内容哈希、请求/回执绑定和调用身份。避免构造会创建目录或初始化生产缓存的 Journal/Workflow。历史文件只读；冻结原字节按 SHA256 保存到 `sources/`，来源清单保留原运行相对路径。读取过程中源文件若被改写会失败；读取后新追加的提交不属于本次冻结时界。

主单位为一个 `task_id + purpose`，其全部 step/attempt、失败/修复、实际模型输出和最终宿主绑定提交均保留。`parent_task`、程序 L3 请求的 `origin_task`、依赖和工具请求把研究者、Coding、Safeguard、执行与回传关联起来；`previous_request_task` 保留前次修订来源，每个任务可直接定位完整 TaskCard。子任务的质量/费用不重复加到父任务科研分数。没有真实调用的程序 rank、Supervisor 程序调度、仅预检任务不计为 LLM 科研样本，其资源尝试仍保留。六类角色、11 个 purpose 按 `agents/tasks.py` 校验；不存在的 interpret 不补样本或零分。

Judge 只看到保存的实际请求中的 system_prompt/context/messages/工具定义/响应合同、角色原响应及本任务自己的提交。`input_versions`、卡片目录、资产权限、source_handles 不等于已读取正文；不会沿这些引用补入完整卡片。工具分页、`required_reads/required_assets` 和未读范围原样保留。逐调用 read_view 序号用于约束时间，H4 的父方案使用原请求版本；不从最终状态替换父方案，不将同 exchange 的未汇合另一方立场或后到 L3 填入原输入。

完整实验关系、执行回执与既有 Evidence 另列在 `run.json` 和报告的 posthoc 通道，**不传给质量 Judge**。真实、synthetic/test_double、失败、中断、缓存及局部/完整范围以原始字段为准；不依据运行成功自动认定算法正确，不把缓存或同次对照算成独立重复。完整数值数组和代码保留原件；报告链接可回查，未验证的来源保留缺口。

缺失的调用/原文/卡片保留 Unknown；哈希损坏、身份冲突等完整性错误不会放行成可确信的来源。有完整观测的部分可以按 Rubric 得分，合理 `tie/undetermined/continue/defer_reason`、无新增知识或合法局部诊断没有程序扣分。

## 计量与评分

程序逐执行统计启动、完成、失败、未知状态、Token 及子字段、任务状态区间、重试/格式修复、工具种类/状态、已有实验状态。请求与实际回显模型/effort 分开；CLI 未回显的实际身份为 null。CLI 内部重连次数不可观察，不能把它当作已知零重试。`reasoning_output_tokens` 不再次加到 output，缓存 Token 也单列。`model_router.elapsed_seconds` 含锁等待及预检，纯推理时延和单独锁等待不可推知；并发调用累计时间不冒充墙钟。任务队列与 waiting 区间、运行段和程序记录的纯数值等待分别报告；未闭合区间不猜结束时间。

缺用量报告已知合计和缺失数；全缺为 null。默认订阅货币成本 Unknown。可选 `prepare --pricing <JSON>` 保存**带来源与日期的估计费率**：

```json
{
  "source": "用户提供的明确费率来源",
  "as_of": "YYYY-MM-DD",
  "currency": "USD",
  "basis": "token_rate_estimate_not_subscription_charge",
  "models": {
    "模型名": {
      "input_per_million": 1,
      "cached_input_per_million": 0.1,
      "output_per_million": 2,
      "input_includes_cached": true
    }
  }
}
```

上述数字仅说明格式，**不是真实价格**。计价必需用量缺失时总额 null、已知小计分列；实际模型未知时明确标记按请求模型估计。非零缓存写入需要额外提供 `cache_write_per_million` 与 `cache_write_additional`。费率估计不是订阅账单。

Judge 结果先按原始 Draft 2020-12 Schema 校验，再核对 case/role/purpose/rubric、criterion 的完整有序集合、状态/分值、source_id、已展示范围内的 JSON pointer 与摘录。`scored` 要有可复查来源；程序不能证明理由的科学真伪，仍需后续专家校准。无效输出只影响本次离线评价。

`Q = 100 × sum(score)/(2 × scored)`；无 scored 时为 null。N/A 不进入适用分母，Unknown 进入适用分母；同时报告 scored/applicable/total、Unknown、N/A、逐项置信度与有效证据覆盖。按 role × purpose 分组，Generation 再分 H1/H2/H3，rank 再分 parent/final；未评任务数独立呈现。报告不设置通过阈值、不硬合质量与成本，也不声称假说或 Prompt 的因果收益。

Judge 的调用、失败、Token 和耗时另列 `judge_overhead`。评分独立使用输出目录的锁与尝试上限，不占用科研队列/预算/订阅锁；若复用同一登录，供应商账户额度仍是共享的，不能声称创建了独立订阅账户。

## Judge 隔离与复核

独立 worker 仅加载已核验的 `runtime/codex_cli.py`、`runtime/subscription.py` 和纯 Schema 数据类；在独立进程内绕开 `runtime/__init__.py`，不载入 paths、router、driver、GPU 或 Budget。复用其只读账户/额度预检、严格模型档位核对、无工具 host 检查和响应解码，不复制维护另一套 CLI 隔离策略。CLI 版本不受支持时失败，不放松规则。GPU 环境为空；外部应用、网络搜索、shell、MCP、skills、子 agent 等功能禁用，工具 host 必须实际报告关闭。

评分材料中的指令、代码及网页均为数据；没有科研工具定义或执行动作。只有静态 Judge System 进入控制指令，case 模板在插入数据前完成占位符定位。模型名/推理档位尽量盲化，凭据模式、账户信息和环境路径脱敏。原字节保存在本地受限权限的来源库，供授权环境审计，不直接复制到可读报告。原始文件和 Judge 请求、answer、解码结果、audit、校验状态在独立目录中可复核。此措施限制操作能力，不保证 LLM 在语义上绝不受提示注入影响。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B -m unittest discover -s QAViT/tests/evaluation -p 'test_*.py' -v
```

标准 QAViT CPU 回归仍可用原 `unittest discover -s QAViT/tests` 命令。评价实现和资源与其他自有文件一样纳入既有递归源码身份；恢复历史科研运行仍遵守原有显式干预要求，本模块不排除文件、不改哈希语义、不自动续跑。
