# 数值实现与独立验证接口

新代码的 validation 必须有明确数学规格、cases、atol、rtol。量化器案例可包含 seed、shape、distribution(normal/uniform)、scale，或显式 values/calibration_values，以及案例 parameters。invariants 可含 code_count；只有 per_tensor 能使用全张量唯一值数量检查码本，不能因此误杀 per_channel 或更改原始公式。

每个数值案例须有 seed；executor_v1 案例另须 calibration_count、evaluation_count，experiment_v1 案例另须 context。Safeguard 或数值比较拒绝后，原申请最多发布两次 Coding 修订任务，携带原源码引用和拒绝反馈；冻结规格不变，每次修订的独立 Reference 仅从同一冻结规格与接口文档重新实现，不接收候选源码、旧审查或候选错误反馈。Coding 只负责候选源码；Reference 的接口实现问题由该独立重建处理，不要求 Coding 改写参考。执行中的 TypeError/AttributeError 也可携带实际错误、worker.log 和原源码返回 Coding，与审查拒绝共用两次修订额度。每版候选重新数值验证、审查并使用独立执行任务，旧拒绝和失败保留；耗尽后返回 rejected/failed，不重复执行旧源码。预算、资产或科学定义错误不走此实现修订分支。

若工具申请的 Coding 修订明确 clarify，说明冻结规格无法由实现修复，宿主取消该修订并将原问题、冻结卡和审查引用作为 rejected 结果交回仍等待的原科研任务。原任务自行决定是否提交修订草案或保留证据缺口；不会自动补案例、改规格或裁决。没有仍等待的原申请任务时保留正常用户澄清入口。

Coding/reference 返回的 `interface` 必须是精确接口名字符串，例如 `"quantizer_v1"`，不能把函数签名附在该字段后。完整签名和解释放 `explanation`。当前结构门禁允许的导入根为 torch、numpy、math、statistics、itertools、functools、collections、copy、dataclasses、typing、random、scipy、ptq_agent；其它依赖须先明确声明并核验。文件、进程和网络能力由宿主提供，不在数值插件内自行创建。此结构检查不代替数学审查。

quantize_evaluate 需要新增实现时声明 `requires_code=true, interface=executor_v1`。实现 `build_plan(backend,cfg,experiment_payload,parameters)`，可选 `make_executor(backend,cfg,data)`。计划遵守EXECUTION.md 中的数值合同与冻结选择。独立参考和候选来自同一 ExperimentCard，在不同 Python 进程执行；参考设计阶段看不到候选源码或候选数值结果。

已有算法不需要生成上述包装函数：在同一 execution 中直接提交数值 `plan`，不得同时指定新代码/interface。宿主在草案阶段检查 plan Schema/未解决项，执行阶段加载实际 backend 后，在任何校准前完整检查冻结位点、位宽、参数、粒度/轴、顺序和 FP 例外，再调用原 QuantizationExecutor。未知算法/observer 仍不能通过现有数值 Schema 与能力检查，须走新增实现流程；该入口不接受任意源码，不绕过新代码审查。新实现未提供 plan 时才自动补 requires_code=true/interface=executor_v1。

`context.public_interface_assets` 提供当前代码版本下的公共 `pipeline.py`、`contracts.py`、`plugin_contracts.py`、`calibration.py`、`plugin_math.py`，code/reference/safeguard 均可用 read_asset 分页读取，权限按每次请求保存哈希。其他源码与候选资产不因此开放。`QuantizationExecutor(backend, config, data, *, diagnostic=False)` 的公共交接为 `prepare(plan, shapes) -> (model, weights, activations, audit)`、`audit(model, plan)`、`export(model, plan, weights, activations, directory, audit) -> manifest`、`load(directory) -> (loaded_model, manifest)`。prepare 拟合并保存冻结量化状态，export 导出，宿主独立 loader/audit 核验并使用重载实例评测；不能通过自报 load 通过门禁。现有方法与 observer 按公共合同声明；未知 observer 不因提供了字符串名称就得到注册。

evaluations 支持 `{"kind":"classification","split":"search","count":16}`；该示例不规定全局评测任务。L3 按 ResearchBrief.evaluation_protocol 的任务、范围和完成条件执行；结构化 evaluations 中的 kind/split/count 由预检与证据检查共同核对，未声明项不猜测、不自动增加分类或分割。协议要求 full_dataset=true 时，内置评测的实测结果须确为完整数据。缩减评测仅支持其声明范围；L3 层级或流程完成不代表完整数据评测。已有 ade_frozen 是明确兼容 DeiT 的冻结 head 适配器，不能宣称自动适用所有 ViT。其它模型可使用用户声明且绑定源码的 factory backend；新下游任务需经审查的实验适配。L3 不设置 L2 精度阈值，保留原预算与完整实验检查。

custom_experiment 声明 `requires_code=true, interface=experiment_v1`，实现 `execute(experiment_payload,context)`。context 提供 candidate_model、reference_model、data、backend、configuration、parameters、directory。返回 observations、原始 artifacts(path/sha256/bytes) 和实际 sample_selection。冻结验证案例的 context 应提供明确的小规模可复现输入。插件的文件写入限制在本次作业，依赖与资产须预先声明。

`experiment_v1` 另须 `validation.interface_cases` 非空数组。每项显式含 seed、shape（batch=1 的张量形状）、sites（模块路径数组）、counts（calibration/search/final 的正整数计数）。这是 CPU 接口夹具，不是科研采样默认值：使用生产 VerifiedManifestData、真实二元组 batch 与 forward_model(model,image_tensor) 接口。草案预检在 Coding 前核对每个案例能容纳冻结 calibration/evaluation 的 split、offset、count（显式 null 仍为该视图剩余范围）；如8张要求与2张池冲突，返回作者修订，不自动改样本数。候选与独立参考分别执行数学及接口案例，两者均通过后才可批准；接口案例不能替代真实模型验证。

夹具的 `backend.spec.family` 为 `interface_fixture`。BCHW 输入且 sites 含 `patch_embed.proj` 时，当前夹具是随机小型 ViT：patch_size=16、embed_dim=24、num_heads=3、mlp_ratio=2、num_classes=1000，深度为 sites 中最大 block 索引加1；这些值不等于真实 DeiT-B 的结构。其实际模块可通过 `backend.model` 和 `backend.modules()` 核对；夹具统计不能作为真实模型的头数、覆盖或效果证据，真实模型约束仍以冻结资产和科研规格为准。

所有 L3 服务共用实际结果门禁。execution.evaluations 的每项必须给出 kind、split（ade_frozen 固定为 ade_validation）及显式 count；可列 metrics 和 full_dataset。结果逐项检查指标、唯一样本 ID、顺序/数量和完整范围；重复 kind 或不支持的 offset/字段明确拒绝。没有标准评测的自定义 L3，须在 Brief.evaluation_protocol.required_results 列出 `{path,schema}`（路径相对于 observations），不能仅填写“completed”。有 batch_size、calibration_samples、search_samples、data_protocol、manifest_sha256 等结构化协议字段时，也核验实际身份/范围。结果缺失不能发布为完成 Evidence、命中缓存或被报告为验证成功。

协议可明确 `reload_before_evaluation: true`。内置 quantize_evaluate 始终由宿主独立 load/audit，并使用该重载实例进行评测；evaluate_artifact 同样评测实际加载实例。custom_experiment 的标准 L3 评测、完整模型或重载要求必须返回本作业内的 model_dir，宿主重新加载、审计并执行评测；插件自报的重载标志/标准成绩不能作为通过依据。任意新增评测接口仍须注册和核验，不能伪造已支持。

真实数据接口为 `data.ids(split)` 和 `data.batches(split)`，split 是字符串（如 `"calibration"`），不是 execution.calibration 配置字典；后者的 count/offset 须由实现按冻结定义核对和选取。batches 每项为 `(image_tensor, sample_id)`，调用 `backend.forward_model(reference_model, image_tensor)`，不能将整个二元组传入模型。数值案例的 context.activations 路径不会验证这些真实数据接口，Safeguard 仍须核对真实分支。

