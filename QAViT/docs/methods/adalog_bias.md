# AdaLog：独立输入平移与 bias 重参数化

来源：AdaLog，官方 commit `75a57d2515521bb1711159ea229d1a37665f3fed`，`quant_layers/linear.py:reparam_bias`。实现 `execution/adalog.py:bias_transform`、`pipeline.py:PipelineLinear.effective_bias`。

仅已验证的 GELU→Linear 边；preprocess.method=adalog_bias，module为目标Linear，norm=null，不使用alpha或observer。令c=0.16997124254703522，输入x'=x+c；最终bias'=bias-c*decoded_W的输入列求和。模型原bias、head与原checkpoint不变，变换在副本上显式执行。

必须先建立输入变换，再拟合该边的W/A。可搭配合法独立 A/W 插件，并不强制选择AdaLog；一个输入边只允许一个预处理，不与同边SQ/RepQ/twin-GELU叠加。若改变变换，必须重新拟合其W/A；局部比较须将联动目标全部列入 mutable_targets，否则拒绝。

关键依赖：最终补偿取当前实际反量化权重，包含AdaRound硬舍入或GPTAQ更新结果。effective_bias每次从实际使用的W计算，不能缓存初始FP补偿；只读W旁路时使用被旁路后的参考W，确保代数一致。保存scale/shift，重载从已验证编码重建补偿。

成本为逐边统计检查及每次forward的W向量乘；reference QDQ，没有原生加速声明。不修改相邻GELU或LN参数；真实GELU负界、FP等价和保护检查均须通过。
