# QAViT 多 ViT 接入验收

**已有条件启动当前协议下的 DINOv3-B/16 正式量化研究。** 已用可信权重、真实图像和原冻结头跑通分类、分割、局部科研与候选重建。本轮止于小样本验收。DINOv3-B 的全量 FP/量化基线、正式 B 和算法增益均待实测。正式研究目标、训练许可、样本数与双任务协议保存在独立[运行输入](../../inputs/dinov3b_static_w4a4/input.json)。[可核对结果](RESULTS.json)

## 从输入到执行

- 新运行从用户选定的 backend 配置和资产清单取得构造器、源码、权重和任务资产，核对模型 ID、源码与权重摘要、张量及标签映射。读取实际模块、patch 尺寸、block 路径、头数和 prefix，并与资产声明交叉检查。有校准图时复用现有结构 profiler 测量一次 FP 调用。新 Brief 绑定资产指纹，运行中的 `model_metadata.json` 保存完整目录与调用记录。本次两模型交给 focus 的模型摘要各约 3 KB，约 90 KB 的详细模型资料按需读取。仅有权重的接入也实测返回结构与量化目录，此时图像池和实测前向保持未知。[绑定与摘要](../../../QAViT/tools/profiling.py)
- 加载与任务特征的差异集中在薄适配层。DeiT 保留原生分类前向和旧冻结头的动态位置插值。DINOv3 分类用原实现的归一化 CLS 和已有线性头，分割调用原生 `get_intermediate_layers`。训练快照内的骨干张量逐项比对，只取对应 neck 与 decode head。分割使用本头的层号、预处理和滑窗。旧 DeiT 资产依其确切清单摘要和模型 ID 找回原任务配置。[模型与任务适配](../../../QAViT/tools/vision_models.py)
- 修正了默认单 CLS 后全为 patch、block 只接一个张量、固定分类维度、固定输入预处理、全局校准数量上限及共享分割设置等假设。诊断和选参使用相同 token 区间，block 重放保留嵌套位置参数、关键字参数及 RoPE。局部探测通过原算子处理 QKV 掩码，在真实 Q/K/V/P 操作数处观察和干预。冻结导出、回读、跨任务评测与候选状态共用身份检查。[token 与 block 语义](../../../QAViT/tools/ptq_agent/profiling/adapters.py) · [冻结诊断](../../../QAViT/tools/ptq_agent/execution/propagation.py)
- 新协议隔离资产、任务、硬约束和方案库。历史包使用归档实现恢复，再核对模型、数据和 FP 原始状态。全程禁训仍由 Brief 决定，可单独声明终评 `final_training=false`。本轮研究配置选择全程禁训。研究阶段的参考优化须有相应运行授权，最终要求独立 training-free 时会检查冻结来源。[实验输入](../../../QAViT/docs/EXPERIMENT_INPUTS.md) · [当前伪代码](PSEUDOCODE.md)

## 真实双模型结果

在 V100 上，两者使用相同的工程验收方法。分类输入为 224×224，2 张 calibration 拟合静态 uniform_minmax W4A4，2 张互斥 search 图比较 FP 与从磁盘重载的量化模型。分割沿用同一冻结骨干与分类校准尺度，跑 1 张 ADE20K 验证图。两套冻结头按各自的配置消费第 2、5、8、11 层特征，编号从 0 起算。表内数值均为百分数，样本 ID、排序种子、权重哈希、实测调用和包身份见结果文件。[样本与身份记录](RESULTS.json)

| 验收项 | DeiT-B | DINOv3-B/16 |
|---|---:|---:|
| 图像序列 | 1 CLS + 196 patch | 1 CLS + 4 storage + 196 patch |
| 必需且实测冻结的 W/A 边界 | 146/146 | 146/146 |
| 首张 search 图上导出前后分类输出 | 逐值相等 | 逐值相等 |
| 首个真实分割裁剪，4 层最大相对 MSE | 8.88×10⁻¹² | 0 |
| 分类 Top-1，FP → 量化，2 张 | 100 → 0 | 50 → 0 |
| 分割 mIoU，FP → 量化，1 张 | 52.2829 → 0.1496 | 65.7401 → 0.2801 |

146 个边界包括 patch 投影和 48 个 block 线性层各自的权重与输入，以及 12 层注意力的 Q、K、V、P。两模型都验证了首尾 block 原参重放、分组误差、局部尺度搜索、QKV 与转置 K 的单点探测。DINO 的分类 logits 与冻结头直接作用于原生 CLS 的结果逐值相等，storage 和 patch 切片与原生特征逐值相等。在分割真实裁剪上，以原分类特征通路的独立 block hooks 核对原生分割特征提取。两套任务头与骨干的评测前后摘要均一致。DINO 分割头实际训练 checkpoint 内的模型与验证预处理定义，和绑定的任务配置逐项一致。这里运行的是 FP32 参考 QDQ。[双模型验收程序](verify_numeric.py)

## 角色、恢复与复现

同一 Generation focus 模板在两份实际模型输入上的摘要和协议各不相同，模板哈希相同。两份正式双任务计划均通过原 L3 规划合同的静态核验。没有提交 L3 数值任务。原共同主题及复议、H1/H2 父方案与回退、H3 灵感、B、候选修订与三候选 L3 仍由原工作流负责。[配置与角色入口核验](verify_context.py)

DINO 组件运行中，Generation 从真实位点目录选出 `blocks.0.mlp.fc2`。Coding 将 W 按输出通道、A 按输入通道设为非对称 UINT4、uniform_affine、minmax，其余主体与分类头保留 FP。经 Safeguard 审核，在 1 张 calibration 和 2 张 search 上得到整网 logits 相对 MSE 0.0256134，FP 和局部候选的 Top-1 均为 50%。Generation 将证据写入 H3，Recorder 建议下一步固定 W 与样本、对照两种激活粒度，暂缓粒度收益判断。局部包没有晋升 B。获批源码 SHA256 为 `3712e2eeeab11fee3642610eb914677da772718a871293d349782fab8223221b`。用它从源权重重新准备后，规范计划、模型状态摘要、实际校准 ID 以及两张图的 logits 均与原执行一致。[获批源码](approved_candidate.py) · [重建程序](verify_candidate.py)

实际拒绝了 DeiT 加载 DINO 包、同一 DINO 在不同协议下加载包，以及直接用本次局部包进入正式研究协议。后一对除协议外的身份字段逐项相等。显式允许备选执行适配器时也核对实载骨干、分类头和数据协议。历史 DeiT 包用发表时归档的实现、原批准源码和权重重载，在真实 search 图完成前向。两张 search 图上，当前 DeiT FP 分类 logits 与归档原加载器逐值相等。两份新验收运行的 B 均为待验证。历史冻结 Brief、成绩和包保留原件。[身份与历史重载核验](verify_identity.py)

CPU 回归收集 769 项，767 项通过，2 项 CUDA 专用检查跳过。在 V100 上重跑对应模块，5 项全部通过，包含这两项。覆盖科研选择、预算、恢复、Safeguard、证据和历史评测。组件验收的调试运行独立留档。第一次没有建立活跃计时段，下一次在 Coding 提交后用完了 15 次调用预算。成功的运行使用原队列重新建立，曾因验收脚本缺少 Recorder 上下文停下，随后在原运行、原预算上通过标准 Recorder 入口完成，已完成的数值作业未重复执行。角色快照与交付版本的 SHA 都在结果文件。历史批准随宿主版本冻结，原实验按发表时的快照复验。单独使用交付版和获批源码也重建出了相同的模型状态与 logits。[回归与运行记录](RESULTS.json)

## 复现与适用范围

从项目根目录执行。源码、权重、两个任务头及数据沿用本机已绑定位置。验收脚本为新结果创建独立目录，保留已有同名运行。第一条命令只依赖可版本化的获批源码、记录和配置及本机真实模型数据，不需要原角色运行目录。

```bash
V=EXP/validation/multi_vit_20261118_r1
CUDA_VISIBLE_DEVICES=0 python -B "$V/verify_candidate.py"
CUDA_VISIBLE_DEVICES=0 python -B "$V/verify_numeric.py" QAViT/configs/dinov3b_backend.json "$V/dino_repeat"
CUDA_VISIBLE_DEVICES=1 python -B "$V/verify_numeric.py" QAViT/configs/deitb_backend.json "$V/deit_repeat"
python -B "$V/verify_context.py" --deit-dir "$V/deit_repeat" --dino-dir "$V/dino_repeat" --tag _repeat
PATH=/root/.local/bin:$PATH python -B "$V/live_roles.py" roles_repeat
CUDA_VISIBLE_DEVICES=0 python -B "$V/verify_candidate.py" roles_repeat
CUDA_VISIBLE_DEVICES=0 python -B "$V/verify_identity.py" roles_repeat --dino-dir "$V/dino_repeat"
```

本轮实证覆盖上述两个骨干、既有分类与 ADE 冻结头、当前的单图像流和静态 W4A4 参考实现。DINO 分类头是既有 ImageNet 子集训练的线性头。其他骨干、其他任务头、训练式参考在 DINO 上的数值行为、完整 50,000 张分类与 2,000 张分割评测、原生低精度内核及显著精度提升均未在本轮验证。
