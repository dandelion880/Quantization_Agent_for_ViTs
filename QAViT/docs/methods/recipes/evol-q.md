## 19. Evol-Q / Jumping Through Local Minima（ICCV 2023）

<a id="evol-q"></a>

**身份与来源。** 综述 [114]。[正式页面](https://openaccess.thecvf.com/content/ICCV2023/html/Frumkin_Jumping_Through_Local_Minima_Quantization_in_the_Loss_Landscape_of_ICCV_2023_paper.html)，[作者修订稿](https://arxiv.org/pdf/2308.10814)，[官方代码](https://github.com/enyac-group/evol-q) joint_evol_opt.py、utils.py、main.py。继承 CPT-V 的对比 loss 和进化 scale 搜索，扩展实验与损失面分析。

**paper-recipe。** 基础方法是 FQ-ViT；保存浮点 teacher predictions，按 attention block 逐个搜索尺度，种群15、tournament10、每块3循环、10遍、1000无标签图。论文先生成扰动种群；每步以对比目标选父代、扰动、替换最差候选，重复后取最优。ViT 的主要报告是 W8A8/W4A8/W3A8；文中的 W4A4 附加表含 CNN，不能当成 ViT A4 成熟证据。原文重复校准集实验也显示收益有波动，适合在 QAViT 中研究排序稳定性。

**当前官方代码的实际行为。** 初始种群复制15份同一初始 scale，随后每循环仅产生一个新候选；反向遍历 attention blocks，只改其子 QLinear 的 quantizer.scale，未遍历所有 activation quantizers。mutate() 使用绝对值保证非负，半径是4/3bit \(10^{-3}\)、8bit \(10^{-4}\)，与论文文字次序相反。默认 batch100、\(\tau=3\)；对比 loss 为归一化 logit 相似度的交叉熵乘 \(2\tau\)。这些版本差异必须保留，不能将代码实现描述成全 block W/A scale 都已优化。

另一个影响公平性的行为：每块优化后调用 val_loader，按 validation Top-1 保存最佳 checkpoint，最后重新加载最佳。作者代码复现应如实记录这个额外选模标签预算；统一 QAViT 协议应在独立 selection split 选 checkpoint，或预先固定最终轮，评测集只做最后测量。后一改变需另标 adapted recipe。

**primitive 与 workflow。** 共用 CPT-V 的 contrastive scorer、scale mutation、population state；metadata 记录 scale coverage、遍历顺序、采样有无放回、初始种群和实际 forward 数。研究时先锁住同一初始化，再分别比较论文行为、代码行为、统一预算变体；DINO 分类 logits 或 dense features 的对比目标必须独立命名，不能叫作者原配方。最终固定 W4A4 在机制上可接入，但 A4 表现需验证。状态：正文/修订稿与关键源码已核，代码与论文差异已登记；未执行复现。


## 接入依据与来源边界

**基线组件** FQ 组件、冻结评分、teacher 缓存，与 CPT-V 共享搜索骨架。

**配方阶段与依赖** 补有版本的进化 scale 配方、真实搜索覆盖/顺序/种群；选择集选模与最终评测分离。

**部署条件** 最终静态；原 ViT 主要 A8，A4 适配需独立验证。

**尚需核对** 当前代码只改子 QLinear scale；扰动半径、种群初始化、验证集选模与论文不同，不能统一写全 W/A 搜索。
