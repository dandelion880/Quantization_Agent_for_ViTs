# repq 完整 static W4A4 种子

joint/sequential：W为per-channel affine percentile，普通A为affine percentile；真实LN→Linear边使用float观察的显式仿射重参数化，变换后affine percentile重新拟合整数零点（避免原mean(z)端点17值），P为预留零log√2。patch输入A4，head/neck FP。区别于官方patch A8与prefix观察，须重新测量。

完整可执行构造：`ptq_agent.execution.baselines.build_baseline(backend, "repq", quantize_head=False)`，返回普通v3计划，可由标准RTNExecutor准备、导出、重载和评测。完整参数在构造源码中，模型/数据/seed/校准数量/评测协议由当前 solution.configuration 固定；不另设默认预算。`configuration_asset` 提供当前完整配置及来源版本。

ImageNet Top-1：待测（null）；ADE20K mIoU：待测（null）。不能继承论文数字、其它模型成绩或局部search成绩。来源及固定commit见 seeds.json / references/ptq_methods.json；官方材料优先，允许合理自主适配。

伪代码：展开真实计算边界 → 按该方案分配独立W/A/预处理组件 → 仅在calibration拟合 → 冻结 → export/load验证 → 同协议ImageNet与ADE完整评测。若某插件不兼容，保留原方案，明确修订选择后回退成熟完整配置继续研究。
