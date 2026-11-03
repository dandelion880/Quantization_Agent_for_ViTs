# AdaRound：固定网格上的舍入优化

来源：ICML 2020 Up or Down? Adaptive Rounding for Post-Training Quantization（PMLR119/nagel20a）。官方Qualcomm AIMET 1.35.1固定commit `7b60b5253b0a983132c2930ee91c76df748511d9`，v1/adaround的wrapper/loss/optimizer；源文件、BSD-3-Clause声明与hash见shared/reference_sources/aimet_adaround_7b60b525。实现 `execution/adaround.py`。

合法：Linear W，INT/UINT4/6/8等既有整数格式，per_tensor或per_channel(axis=0)。weight_method=adaround，必须显式options.rounding_grid=uniform_full/uniform_affine/uniform_minmax和rounding_iterations=1..20000。先依weight_calibration=minmax/manual/search冻结网格；signed/unsigned必须与网格相符。无默认迭代数，短预算不等于论文收敛。

只alpha可训练：h=clamp(sigmoid(alpha)*1.2-.1,0,1)，alpha由W/scale的小数部分初始化；W'=scale*(clamp(floor(W/scale)+h+zero)-zero)。目标是当前实际输入流到原始FP模型该模块输出的重建（feature平方和、token-row均值），加beta余弦退火舍入正则。Adam仅接收alpha，默认lr=.001、reg=.01、warmup=.2、beta20→2；可显式rounding_lr/reg/warmup/batch_size(默认32图)/seed。每次完整使用选中图的全部token，不随机裁剪token。迭代数和首末loss记入报告；异常不得声称优化完成。

推理前用alpha>=0离散冻结，只导出packed整数码和固定scale/zero，不导出软权重或优化器。原始W、网格、FP teacher、受保护head都不求导或更新。网格/输入仿射改变后旧舍入失效，必须重拟合；与同一W的RTN/GPTQ/GPTAQ互斥，不默认串接优化。可先W后A，或先A后W；报告实际输入来自FP还是量化路径。与AdaLog bias组合时使用实时W补偿参与目标。

项目适配：从AIMET的Conv/分类框架抽离Linear token-row目标，不自动加邻接激活函数、不进行整块/整网训练，也不承诺相同预算下论文精度。成本：逐位点成对CPU缓存、alpha/梯度/Adam状态以及iterations×batch_images局部前后向；显存不足即失败，不能缩减配置预算。
