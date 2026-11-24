# 单实现实验代码接口

需要新代码的实验：原研究者提交目的、所需结果和科学约束 → 程序保存不可改写的原始需求并提供已知条件 → Coding 交付单份源码及必要 execution_choices → 程序绑定完整执行记录 → Safeguard 对照原始需求、当前源码和执行条件审查 → 程序执行并发布真实 Evidence → 返回原研究者继续分析。

Coding 无须等待完整执行规格冻结。原始请求与最终执行 ExperimentCard 分别持久化，后者绑定 request_ref/request_sha256；源码 manifest 绑定原始请求。程序生成批准记录，绑定原始需求、实际执行条件、当前源码和核心实现；任一实质输入变化后不得复用旧批准。历史 Reference 记录仅供读取，不参与新流程。

不要求研究者指定内部变量名、固定输出树、独立参考、数学/tiny 模型夹具或双实现容差。科学预测可省略。Safeguard 提交 approved 和 rationale；程序核验接口与来源，科学审查重点是统计位点、量化位宽、对照条件、数据使用及指标含义，合理的结果组织或说明格式差异不构成拒绝。

Coding 必须返回完整 source；interface 可由已知执行条件绑定，explanation 和 execution_choices 按需提供。当前接口为：
- quantizer_v1：fit(x, parameters)、apply(x, state, parameters)。fit 使用声明的校准数据，apply 不修改输入或冻结状态。
- executor_v1：build_plan(backend, cfg, experiment_payload, parameters)，原生 v3 可选 refine_frozen(backend, model, data, parameters)，在本次准备的完整状态上校准选择并明确安装 W/A，返回 JSON 选择记录。宿主审计并导出实际选中状态，核对重载模型全部参数、buffer 和同图输出。独立加载和迁移只使用 load。其他表示可提供 make_executor(backend, cfg, data)，自行实现配套 prepare/audit/export/load。
- experiment_v1：execute(experiment_payload, context)。context 提供 candidate_model、reference_model、data、backend、configuration、parameters、directory；reference_model 是受保护的浮点基准模型。返回 observations 及需要的 artifacts；artifacts 可省略，空 `{}` 兼容归一为 `[]`，非空使用 `[{"path":"文件路径","sha256":"64位小写十六进制SHA256","bytes":非负整数字节数}]`。sample_selection 由宿主按真实读取生成，实现可省略，显式提供时须与实际一致。

实施错误携带当前源码与具体反馈返回 Coding，就地修订同一实验；每版重新绑定执行条件及审批。不得重跑未修改的失败代码。NameError、UnboundLocalError 等实际发生在提交实现中的异常同样进入修订；预算、资产与宿主接口问题单独保留工程原因；原始科学定义有实质歧义时才回到原研究者。修订共享现有有界额度，错误和旧版本保持可追溯。

串行与并行共用异常来源判定。双任务子进程返回结构化 `error.code/message/origin`；父进程先核对冻结模型、方案、包与实际请求 hash，再转送原始类型和来源，附 `error.subprocess.worker/receipt/worker_log`。批准实现调用栈或其返回值合同错误仍回 Coding→Safeguard；超时、资源或宿主失败保留工程分类，无回执退出标为 `ChildProcessError/host`。身份不符的回执不能用其自报来源覆盖宿主拒绝。已成功兄弟任务保留为 partial result，失败回执另列；整体失败不发布完整 Evidence。恢复时 `run_experiment` 将当前实际请求的 `deadline_unix` 经 `evaluate_services` 显式传入双任务/分类分片父进程及子请求，不从旧 `request.json` 猜测时限；单一数值任务时限及预算不会按子进程重复计时。

结构检查允许 torch、numpy、math、statistics、itertools、functools、collections、copy、dataclasses、typing、random、scipy、ptq_agent；文件、进程和网络能力由宿主提供。结构检查只约束可执行接口和工程边界，不构成数学正确性证明。真实执行继续检查有限值、数据身份、基准不变性和实际结果；浮点基准、基线对照与真实误差测量继续保留。

正式 ServerBackend 的 reset() 检查浮点源模型保持 eval、参数不需梯度及权重/buffer 状态不变，不修改权重、不拟合量化参数。已有执行器和校准捕获可调用它，不能以空实现替代。


已有算法不需要生成上述包装函数：在同一 execution 中直接提交数值 `plan`，不得同时指定新代码/interface。宿主先统一补齐唯一确定的表示字段（格式对应 bits、per_tensor 空 axis/group_size、禁用目标的空 method），再在草案阶段检查完整 plan Schema/未解决项，执行阶段加载实际 backend 后，在任何校准前完整检查冻结位点、位宽、参数、粒度/轴、顺序和 FP 例外，再通过 RTNExecutor 版本适配入口调用 v2 或 v3 执行器。未知算法/observer 不能由内置执行器执行，须走新增实现流程；该入口不接受任意源码，不绕过新代码审查。新实现未提供 plan 时才自动补 requires_code=true/interface=executor_v1。

`context.public_interface_assets` 提供当前代码版本下的公共 `model.py`（v2/v3 适配入口）、`pipeline.py`、`contracts.py`、`plugin_contracts.py`、`calibration.py`、`plugin_math.py`，code/safeguard 均可用 read_asset 分页读取，权限按每次请求保存哈希。其他源码与候选资产不因此开放。默认 `RTNExecutor(backend, config, data)` 适配 v2/v3；v3 数值实现仍为 QuantizationExecutor。公共交接为 `prepare(plan, shapes) -> (model, weights, activations, audit)`、`audit(model, plan)`、`export(model, plan, weights, activations, directory, audit) -> manifest`、`load(directory) -> (loaded_model, manifest)`。使用内置执行器时，prepare 拟合并保存冻结量化状态，export 导出，宿主独立 loader/audit 核验并使用重载实例评测。提供 make_executor 的自定义算法可用自己的计划、方法名、observer 和参数表示，仍须实现既有 prepare/export/load 接口；宿主实际调用批准的加载实现，检查导出资产和重载前后真实输出。自报 load 成功不作为核验。独立 evaluate_artifact 支持 plan v2/package v1 和 plan v3/package v2 的标准模型包，自定义包须通过对应批准实现加载，不能伪装为原生包。

evaluations 支持 `{"kind":"classification","split":"search","count":16}`；该示例不规定全局评测任务。L3 按 ResearchBrief.evaluation_protocol 的任务、范围和完成条件执行；结构化 evaluations 中的 kind/split/count 由预检与证据检查共同核对，未声明项不猜测、不自动增加分类或分割。协议要求 full_dataset=true 时，内置评测的实测结果须确为完整数据。缩减评测仅支持其声明范围；L3 层级或流程完成不代表完整数据评测。ade_frozen 已在绑定的 DeiT-B 和 DINOv3-B/16 骨干与各自冻结分割头上实测。其他骨干、特征接口及头须按其绑定单独验证。其它模型可使用用户声明且绑定源码的 factory backend；新下游任务需经审查的实验适配。L3 不设置 L2 精度阈值，保留原预算与完整实验检查。

自定义冻结包的根 manifest 由批准的 loader 解释。宿主从原 Evidence 绑定模型身份、源码和完整资产树，含嵌套父包。

所有 L3 服务共用实际结果门禁。execution.evaluations 的每项必须给出 kind、split（ade_frozen 固定为 ade_validation）及显式 count；可列 metrics 和 full_dataset。结果逐项检查指标、唯一样本 ID、顺序/数量和完整范围；重复 kind 或不支持的 offset/字段明确拒绝。没有标准评测的自定义 L3，须在 Brief.evaluation_protocol.required_results 列出 `{path,schema}`（路径相对于 observations），不能仅填写“completed”。有 batch_size、calibration_samples、search_samples、data_protocol、manifest_sha256 等结构化协议字段时，也核验实际身份/范围。结果缺失不能发布为完成 Evidence、命中缓存或被报告为验证成功。

协议可明确 `reload_before_evaluation: true`。内置 quantize_evaluate 始终由宿主独立 load/audit，并使用该重载实例进行评测；evaluate_artifact 同样评测实际加载实例。custom_experiment（experiment_v1 只声明 execute，不声明独立 loader）的标准 L3 评测、完整模型或重载要求必须返回本作业内标准 v2/v3 模型包的 model_dir，宿主重新加载、审计并执行评测；插件自报的重载标志/标准成绩不能作为通过依据。任意新增评测接口仍须注册和核验，不能伪造已支持。

真实数据接口为 `data.ids(split)` 和 `data.batches(split)`，split 是字符串（如 `"calibration"`），不是 execution.calibration 配置字典；宿主在解码前绑定已声明的 calibration/evaluation 视图。ids 和 batches 直接给出已切片的顺序，勿再次应用 offset。L1/L2 custom_experiment 的 parameters.diagnostic_view 接受 split（calibration 或 search）、count、可选 offset 和 ordering_seed；其 identity.diagnostic_view 提供原始 ordered_pool_ids。宿主按真实迭代回执核对样本。batches 每项为 `(image_tensor, sample_id)`，调用 `backend.forward_model(reference_model, image_tensor)`，不能将整个二元组传入模型。Safeguard 核对真实数据分支；宿主记录实际读取和前向范围。

执行条件按本次实际操作绑定：L1/L2 局部或浮点对照可观察整网；L3 才继承最终候选要求。仅做权重统计可以不读取图像、不声明 calibration/sample_selection；实际读取由宿主记录。training-free 允许对独立张量的只读梯度诊断，禁止优化器更新并保护源模型。方法解释与可选逐位点 reason/evidence 不代替核验，也不作为机械必填字段。

正常准备阶段可获取的真实 activation shape 在该阶段捕获；预检只延后尚无 shape 的维度核验，不填虚构 shape。捕获后、prepare 和模型重载仍执行完整数值与能力检查。显式 bits/format、轴/粒度等冲突仍报告具体字段，多个科学选择不由程序猜测；自定义 make_executor 不套内置计划格式。

数值代码允许 contextlib、operator、bisect、heapq、decimal、fractions 等明确登记的标准库。结构检查分别报告未支持模块、被限制的文件/进程/网络模块和具体禁用调用；不无条件开放所有标准库。宿主仍绑定源码审批、执行身份、写入范围及数据来源，结构检查不等同于对任意 Python 代码的完整安全证明。
# 公开数值源码

Coding 与 Safeguard 的 `public_interface_assets` 索引包含已发布的数值执行组件及共享量化数学依赖（包括 grids、codecs、sites、operators、metrics 等）。按需 `read_asset` 时仍核对本次调用冻结的路径与哈希；该入口不授权其他分支的实现/资产，也不赋予文件、进程或网络执行能力。

实验草案可保留研究者明确采用的 `evidence_refs`。角色提交时先按调用快照核对可见性与引用类型；程序创建请求、绑定代码及冻结执行卡时，将这些证据同时放入封套 `refs`。该传播不改变科研内容，不放宽 Journal 的类型、封套或角色隔离检查。

新 static W4A4 的共同规则见 [STATIC_W4A4.md](STATIC_W4A4.md)。完整方案起点通过 context.complete_solutions 的 configuration_asset 获取，方法插件仍独立。Coding 可以适配图和数学表示，注明差异；无需逐字官方实现。

`experiment_v1` 可通过 `context['load_frozen'](evidence_ref)` 获得 `(model, receipt)`：请求顶层 `evidence_refs` 必须引用可见、真实导出并经宿主重载的 Evidence；已知兼容位置 `controls.evidence_refs` 合并到规范字段，原始草案保留。宿主绑定原批准加载源码及整个包的哈希，只调用 load，不调用 prepare，加载过程禁止读取校准/评测 batch，返回 actual frozen_loads 回执；不接受任意路径加载。载入后的局部消融仍由本次 Coding/Safeguard 审查，原包只读。`load_frozen(ref)` 也接受同一已绑定包的确切 `model_ref`；仅当绑定唯一完整包时可省略 ref；只匹配已绑定来源，多个引用指向同一包可合并，多个不同包匹配则拒绝选择。缺来源/选择歧义以 `RequestRevisionRequired(origin=request)` 返回 Generation，新请求重新经过原执行链；Coding 不修冻结授权。包哈希/资产读取失败属 host，资源失败属 resource，实现错误仍交 Coding。所有加载失败都不产生机制反证。

自定义评测推荐 `observations.evaluations={"classification":{"metrics":{"top1":实际值},"split":"search","sample_ids":实际ID列表,"sample_count":实际数量,"full_dataset":false}}`。也接受每行带唯一 kind 的列表，宿主只做无损映射；不猜指标归属，不补造样本身份。纯局部多臂对比可将臂结果放 observations.arms，避免把多个模型混成一个标准分类结果。阻塞返回 `status="engineering_blocked"`（可在 observations 内）和 reason；宿主回传工程失败，不发布 completed 数值证据，也不自动重复修订同一停止桩。

历史已评测完整方案也可用 `ExperimentCard.solution_refs` 引用运行冻结目录中的 solution id，并将该 id 传给 `context['load_frozen']`。它使用与历史包匹配的隔离数值源码和原批准 executor；原包、原 Evidence 和源快照必须全部可核验。当前任务的模型、数据和资产身份仍须匹配，只允许既有设备迁移。失败不会降级为重新校准；原历史精度与本次新测量分开记录，详见 [方案库](solutions/README.md)。

### 证据交付投影 v2（G-only）

`core.evidence.evidence_facts` 是 Generation、Recorder、Memory 共用的 `key_facts`：保持 execution/experiment、协议 hash、冻结模型身份、实测范围及结果状态；标准双任务指标同时显示 FP/Q、Q−FP 和 percent/percentage_points。缺失值保持 `null` 和 `missing_FP_or_Q`，不由摘要猜测。新数值回执直接保留 ADE baseline 指标；旧卡仅可从其已绑定、位于原 run 下、hash 与样本/量化指标均匹配的 `segmentation.json` 补读，标记 historical_artifact_projection，不发布新测量或修改旧卡。

大型样本 ID 列表只在交付视图折叠为有序 hash/count/read_card，原始卡片、资产与精确字段读取不变。超过 4000 字符的 scope.quantization 和 provenance.execution_facts.quantization 逐份投影为各自的 hash/count/精确字段入口，避免重复展开 97 位点的完整冻结配置；短表和 key_facts 中的模型/位宽/协议/测量不变。按 arm/arm_id/arm_name、variant/variant_id、method/condition/treatment 分开数值统计和精确叶节点标签；结构化身份以完整内容 hash 区分。显式 arms/arm_results/variants/treatments 容器保留原字典键或列表索引，不把无标签臂或含数字的臂名折叠成位点均值。key_facts 支持字典及列表 arms，并保留身份、失败状态及精确读取入口；单次多臂不能当独立重复。明确记录的限制、反证和整网臂结果优先直达；对其余大表保留覆盖说明和精确查询入口。

持久启动意图仅在任务/输入版本一致、请求正文完全一致且原 deadline 仍有效并不超出当前上限时复用。正文或资源变化、旧 deadline 失效时写独立 request-<hash>.json，保留原件；工作进程、环境和回执均绑定最终领取的确切请求。数值 worker 从其实际 request 传递 deadline_unix 到串行/并行评测及子任务，不从作业目录中同名的旧 request.json 猜测截止时间；预算累计和已领取尝试不重置。

开放问题由 `question_records` 提供稳定 `question_id`。Recorder 用 `resolved_question_refs` / `abandoned_question_refs` 关闭指定问题，旧版原文数组仍兼容；摘要中的“已解决”不作模糊匹配，也不自动作科学判断。新字段必须指向历史已交付问题，重复关闭幂等，历史复盘不变。

Recorder 的 `B_transition.before_l3` 从所引用综合任务（兼容历史预选）的首次持久快照重建 B；兼容缺少该任务的历史数据时使用决定卡发布时点，无决定则使用已记录 round_started，无可核实锚点明确 unavailable。`after_l3` 固定在本批 recording_batch.through_sequence，逐份携带 snapshot_sequence、来源和方案库事件序号；恢复、延期及预算收尾不把后轮 B 混成本轮晋升后 B。`solution_considerations` 只列该批已记录的本轮更新结果，不推断晋升。原 `fallback_B` 保持当前读取快照的兼容含义。以上均为只读交付，不重写原 Memory、不改变排序/晋升、也不把历史成绩转为同期控制。

完整冻结方案传播诊断可直接请求 `execution.service="frozen_helper", tool="profile_frozen_propagation"`，也兼容在现有 `experiment_v1` 中调用 `ptq_agent.execution.propagation.profile_frozen_propagation`：经 `context["load_frozen"](ref)` 读取明确引用的模型，再选择 block 边界和可选冻结干预。返回配对 FP/Q 输入与输出、同 Q 输入的 FP block 重放、局部新增/上游映射/总误差和交叉项、CLS/patch 与命名任务输出；记录样本和状态哈希，禁止 final 选点、不重校准。API、来源声明示例及覆盖限制见 [冻结传播参考](skills/diagnose-ptq-failure/references/frozen-propagation.md)。直接适配与 custom 路径复用同一数值 helper，均经原来源绑定与 Evidence 返回链；不新增实验配额或替代双任务评价。参数和状态影响见 [能力目录](CAPABILITIES.md)。


校准搜索可直接请求 `frozen_helper` 的 `search_frozen_scales`；其复用的数值入口为 `ptq_agent.execution.calibration_search.search_frozen_scales`（已在Coding/Safeguard公共源码读取范围）。它接收`ctx['load_frozen']`已核验加载的模型，兼容历史源码隔离的同构量化器接口；以明确候选倍率在calibration数据上实际选参并返回完整分数/样本/选中参数。block输出参考、CLS/patch和整网指标的准确语义、导出时收集选中状态的示例见 [scale_selection](methods/scale_selection.md)。它不代替外层search或正式双任务L3。

获准研究参考优化与最终 training-free 的已有字段、产物边界和真实工具限制见 [科研 Skill 兼容说明](SCIENTIFIC_SKILLS_COMPATIBILITY.md)。STE 需现有 custom 实现；全程禁训合同不会被研究用途或 Skill 覆盖。

在完整分割评测中，宿主可绑定已验证方案的 FP 混淆矩阵。数据字节及顺序、模型和头、推理实现、运行环境和协议均一致时可复用 FP；源文件及其校验和写入测量，第一张图的特征核验与量化模型的全量运行照常执行。

### 文件交付与开发评价

Coding 可用 edit_source 从授权的 Python 父版本精确替换一段，后续编辑引用上次返回的 base，最终提交 revision。宿主读取该任务的完整文件并按原规则计算不可变实现身份。保留直接提交 source。read_asset 支持 find 和 symbol 定位后分页。开发视图仅由 evaluator 在冻结包重载后构造，拟合使用选定的 calibration 视图。

新任务可显式配置 llm.coding_native_tools。原生工作区跨同任务续调保留，工作区的 implementation.py 用于最终交付。公共和父源码按哈希复制到只读 sources。原生 shell 只用于 CPU 工程调试，数值研究走现有 GPU worker。启动时实际探测权限配置，失败时 llm_audit.coding_channel 记录原因并沿用宿主编辑通道。本轮环境只能验证回退，原生执行和用量需在支持 namespace 的环境继续验证。

已有完整包复测时，在实验需求的 solution_refs 指定库内模型身份，execution 使用 evaluate_artifact 和当前阶段的 evaluations；同时绑定多个包时用 execution.source_ref 明选一份。宿主解析原始执行器、源版本和模型文件。不传新的 calibration，也无需 Coding 重写加载逻辑。新算法只需更改原生数值计划时实现 build_plan；如需在校准集上改动已构建的量化参数，可在同一模块实现 refine_frozen(backend, model, data, parameters)，返回实际选择的诊断字典。完整自定义执行器继续支持同一份源码的导出、独立重载和审计。

对资产或已绑定快照成员使用 `find` 或 `symbol` 时，`matched=false` 表示该次检索没有交付正文。可使用回包的 `read_asset` 参数打开同一授权正文首页，再按需翻页。

`parent_ref` 使宿主从已验证的完整包或增量子包加载。原生 v3 保留父计划和全部网格。缺少 parent_ref 的 `refine_frozen` 从本次 prepare 的重建状态开始。独立自定义执行器可实现 `prepare_from_parent(plan, model, manifest)` 并沿原 `audit/export/load` 接口返回一致的完整子包。若配合已评分的 `accept_trial` 和 `requires_code=true`，传入的 model 已含所选修改，宿主逐一核验准备后的状态、冷载状态和实际输出。自定义 exporter 须封存原父校准身份与新增身份的有序并集，并声明 preparation_training。可用独立原生成员的倍率、按现有网格扁平索引的 `{"indices":[0],"factor":0.9,"zero":0}`（按需指定 factor/zero）；整数 zero 受当前 codec 验证。自定义码本、PTF 和复合整数算子遵从各自的数学与重绑定规则。多视图数值调用及评分 Evidence 的正式接受示例见 [能力目录](CAPABILITIES.md)。
