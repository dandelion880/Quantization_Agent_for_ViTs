## 11. QDrop（ICLR 2022）

<a id="qdrop"></a>

**来源。** 综述[54]，[论文](https://arxiv.org/pdf/2203.05740) §3.3、Algorithm1、§4，[官方仓库](https://github.com/wimh966/QDrop)。

**paper-recipe。** QDrop解决小校准集上低位宽块重构的过拟合，并让舍入优化见到激活量化误差。冻结浮点teacher，缓存 FP块输入、量化前缀输入及FP块输出。每个元素以概率p取消激活量化，既作用于块输入，也作用于块内量化激活：用统一定义写成 \(\tilde a=m\odot a_q+(1-m)\odot a_{fp}\)，\(m\sim\mathrm{Bernoulli}(1-p_{\rm drop})\)。块输出重构loss加AdaRound舍入正则，优化权重舍入变量及相应A尺度；推理时全部目标激活恢复量化、舍入硬化、随机掩码消失。

原文p_drop=.5，每块/层20000次，ImageNet1024张、COCO256张、NLP1024个样本，W为channelwise。原实验保留首尾层8bit；还特别区别“只首层输入/末层输入8bit”和BRECQ比较配置“首层输出也8bit”，表中†属于后一种。不能以同样W4A4标题忽略覆盖差异。当前官方 main_imagenet.py 的 prob/input_prob 默认1.0，README 的 QDrop 命令才显式设为 .5；block_recon.py 使用 batch32、舍入 Adam 默认lr .001、A lr4e-5 cosine。复现必须保存命令行覆盖而非只读默认参数。不同实现有的把drop_prob定义成“保留量化概率”，.5时数值相同但参数语义相反，必须以mask公式登记。

**primitive。** mixed_fp_quant_activation、adaround_block_reconstruct、rounding_hardening。掩码种子、元素/通道/token粒度、是否只混合输入/同时块内部都要声明。单独只在输入混合是有效变体，但不是完整原算法。QAViT现有AdaRound可复用舍入器，缺随机激活路径、阶段缓存与块级可训练执行器。

**workflow 与接入。** 相同base量化器比较p=0、.5、1，分别观察训练重构、heldout重构和任务表现；不用“calibration loss最低”直接选择p。QDrop可作FIMA/APHQ等共有优化底座，注册时继承代码版本与mask配置，避免把同一base改动重复计算为多篇方法贡献。它是作者明确的有限数据PTQ；最终static W4A4可行，原首尾8bit作为独立作者profile。作者原实验主要CNN/NLP，ViT成熟证据来自后续方法公开迁移；状态：原算法及实验配置、当前主入口/块重构/README命令已读，ViT实现按后续作者recipe分别保存。


## 接入依据与来源边界

**基线组件** AdaRound 舍入/正则/硬化、FP/Q 输入捕获。

**配方阶段与依赖** 补块输入和块内激活的随机 FP/Q 混合、A scale 优化、块调度；终态关闭随机 drop。

**部署条件** 优化中混合合法，推理全量化后可静态；作者首尾 8bit 配置另存。

**尚需核对** mask 概率含义、README 覆盖默认值和首尾例外须锁定；原证据主为 CNN/NLP，ViT 采用后续明确 recipe。
