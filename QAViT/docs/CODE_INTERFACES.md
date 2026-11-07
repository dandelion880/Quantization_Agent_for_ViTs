# 单实现实验代码接口

需要新代码的实验：原研究者提交目的、所需结果和科学约束 → 程序保存不可改写的原始需求并提供已知条件 → Coding 交付单份源码及必要 execution_choices → 程序绑定完整执行记录 → Safeguard 对照原始需求、当前源码和执行条件审查 → 程序执行并发布真实 Evidence → 返回原研究者继续分析。

Coding 无须等待完整执行规格冻结。原始请求与最终执行 ExperimentCard 分别持久化，后者绑定 request_ref/request_sha256；源码 manifest 绑定原始请求。程序生成批准记录，绑定原始需求、实际执行条件、当前源码和核心实现；任一实质输入变化后不得复用旧批准。历史 Reference 记录仅供读取，不参与新流程。

不要求研究者指定内部变量名、固定输出树、独立参考、数学/tiny 模型夹具或双实现容差。科学预测可省略。Safeguard 提交 approved 和 rationale；程序核验接口与来源，科学审查重点是统计位点、量化位宽、对照条件、数据使用及指标含义，合理的结果组织或说明格式差异不构成拒绝。

Coding 必须返回完整 source；interface 可由已知执行条件绑定，explanation 和 execution_choices 按需提供。当前接口为：
- quantizer_v1：fit(x, parameters)、apply(x, state, parameters)。fit 使用声明的校准数据，apply 不修改输入或冻结状态。
- executor_v1：build_plan(backend, cfg, experiment_payload, parameters)，需要时提供 make_executor(backend, cfg, data)。
- experiment_v1：execute(experiment_payload, context)。context 提供 candidate_model、reference_model、data、backend、configuration、parameters、directory；reference_model 是受保护的浮点基准模型。返回 observations 及需要的 artifacts；artifacts 可省略，空 `{}` 兼容归一为 `[]`，非空使用 `[{"path":"文件路径","sha256":"64位小写十六进制SHA256","bytes":非负整数字节数}]`。sample_selection 由宿主按真实读取生成，实现可省略，显式提供时须与实际一致。

实施错误携带当前源码与具体反馈返回 Coding，就地修订同一实验；每版重新绑定执行条件及审批。不得重跑未修改的失败代码。NameError、UnboundLocalError 等实际发生在提交实现中的异常同样进入修订；预算、资产与宿主接口问题单独保留工程原因；原始科学定义有实质歧义时才回到原研究者。修订共享现有有界额度，错误和旧版本保持可追溯。

结构检查允许 torch、numpy、math、statistics、itertools、functools、collections、copy、dataclasses、typing、random、scipy、ptq_agent；文件、进程和网络能力由宿主提供。结构检查只约束可执行接口和工程边界，不构成数学正确性证明。真实执行继续检查有限值、数据身份、基准不变性和实际结果；浮点基准、基线对照与真实误差测量继续保留。

正式 ServerBackend 的 reset() 检查浮点源模型保持 eval、参数不需梯度及权重/buffer 状态不变，不修改权重、不拟合量化参数。已有执行器和校准捕获可调用它，不能以空实现替代。


已有算法不需要生成上述包装函数：在同一 execution 中直接提交数值 `plan`，不得同时指定新代码/interface。宿主先统一补齐唯一确定的表示字段（格式对应 bits、per_tensor 空 axis/group_size、禁用目标的空 method），再在草案阶段检查完整 plan Schema/未解决项，执行阶段加载实际 backend 后，在任何校准前完整检查冻结位点、位宽、参数、粒度/轴、顺序和 FP 例外，再通过 RTNExecutor 版本适配入口调用 v2 或 v3 执行器。未知算法/observer 不能由内置执行器执行，须走新增实现流程；该入口不接受任意源码，不绕过新代码审查。新实现未提供 plan 时才自动补 requires_code=true/interface=executor_v1。

`context.public_interface_assets` 提供当前代码版本下的公共 `model.py`（v2/v3 适配入口）、`pipeline.py`、`contracts.py`、`plugin_contracts.py`、`calibration.py`、`plugin_math.py`，code/safeguard 均可用 read_asset 分页读取，权限按每次请求保存哈希。其他源码与候选资产不因此开放。默认 `RTNExecutor(backend, config, data)` 适配 v2/v3；v3 数值实现仍为 QuantizationExecutor。公共交接为 `prepare(plan, shapes) -> (model, weights, activations, audit)`、`audit(model, plan)`、`export(model, plan, weights, activations, directory, audit) -> manifest`、`load(directory) -> (loaded_model, manifest)`。使用内置执行器时，prepare 拟合并保存冻结量化状态，export 导出，宿主独立 loader/audit 核验并使用重载实例评测。提供 make_executor 的自定义算法可用自己的计划、方法名、observer 和参数表示，仍须实现既有 prepare/export/load 接口；宿主实际调用批准的加载实现，检查导出资产和重载前后真实输出。自报 load 成功不作为核验。独立 evaluate_artifact 支持 plan v2/package v1 和 plan v3/package v2 的标准模型包，自定义包须通过对应批准实现加载，不能伪装为原生包。

evaluations 支持 `{"kind":"classification","split":"search","count":16}`；该示例不规定全局评测任务。L3 按 ResearchBrief.evaluation_protocol 的任务、范围和完成条件执行；结构化 evaluations 中的 kind/split/count 由预检与证据检查共同核对，未声明项不猜测、不自动增加分类或分割。协议要求 full_dataset=true 时，内置评测的实测结果须确为完整数据。缩减评测仅支持其声明范围；L3 层级或流程完成不代表完整数据评测。已有 ade_frozen 是明确兼容 DeiT 的冻结 head 适配器，不能宣称自动适用所有 ViT。其它模型可使用用户声明且绑定源码的 factory backend；新下游任务需经审查的实验适配。L3 不设置 L2 精度阈值，保留原预算与完整实验检查。

所有 L3 服务共用实际结果门禁。execution.evaluations 的每项必须给出 kind、split（ade_frozen 固定为 ade_validation）及显式 count；可列 metrics 和 full_dataset。结果逐项检查指标、唯一样本 ID、顺序/数量和完整范围；重复 kind 或不支持的 offset/字段明确拒绝。没有标准评测的自定义 L3，须在 Brief.evaluation_protocol.required_results 列出 `{path,schema}`（路径相对于 observations），不能仅填写“completed”。有 batch_size、calibration_samples、search_samples、data_protocol、manifest_sha256 等结构化协议字段时，也核验实际身份/范围。结果缺失不能发布为完成 Evidence、命中缓存或被报告为验证成功。

协议可明确 `reload_before_evaluation: true`。内置 quantize_evaluate 始终由宿主独立 load/audit，并使用该重载实例进行评测；evaluate_artifact 同样评测实际加载实例。custom_experiment（experiment_v1 只声明 execute，不声明独立 loader）的标准 L3 评测、完整模型或重载要求必须返回本作业内标准 v2/v3 模型包的 model_dir，宿主重新加载、审计并执行评测；插件自报的重载标志/标准成绩不能作为通过依据。任意新增评测接口仍须注册和核验，不能伪造已支持。

真实数据接口为 `data.ids(split)` 和 `data.batches(split)`，split 是字符串（如 `"calibration"`），不是 execution.calibration 配置字典；后者的 count/offset 须由实现按冻结定义核对和选取。batches 每项为 `(image_tensor, sample_id)`，调用 `backend.forward_model(reference_model, image_tensor)`，不能将整个二元组传入模型。Safeguard 核对真实数据分支；宿主记录实际读取和前向范围。

执行条件按本次实际操作绑定：L1/L2 局部或浮点对照可观察整网；L3 才继承最终候选要求。仅做权重统计可以不读取图像、不声明 calibration/sample_selection；实际读取由宿主记录。training-free 允许对独立张量的只读梯度诊断，禁止优化器更新并保护源模型。方法解释与可选逐位点 reason/evidence 不代替核验，也不作为机械必填字段。

正常准备阶段可获取的真实 activation shape 在该阶段捕获；预检只延后尚无 shape 的维度核验，不填虚构 shape。捕获后、prepare 和模型重载仍执行完整数值与能力检查。显式 bits/format、轴/粒度等冲突仍报告具体字段，多个科学选择不由程序猜测；自定义 make_executor 不套内置计划格式。

数值代码允许 contextlib、operator、bisect、heapq、decimal、fractions 等明确登记的标准库。结构检查分别报告未支持模块、被限制的文件/进程/网络模块和具体禁用调用；不无条件开放所有标准库。宿主仍绑定源码审批、执行身份、写入范围及数据来源，结构检查不等同于对任意 Python 代码的完整安全证明。
# 公开数值源码

Coding 与 Safeguard 的 `public_interface_assets` 索引包含已发布的数值执行组件及共享量化数学依赖（包括 grids、codecs、sites、operators、metrics 等）。按需 `read_asset` 时仍核对本次调用冻结的路径与哈希；该入口不授权其他分支的实现/资产，也不赋予文件、进程或网络执行能力。

实验草案可保留研究者明确采用的 `evidence_refs`。角色提交时先按调用快照核对可见性与引用类型；程序创建请求、绑定代码及冻结执行卡时，将这些证据同时放入封套 `refs`。该传播不改变科研内容，不放宽 Journal 的类型、封套或角色隔离检查。
