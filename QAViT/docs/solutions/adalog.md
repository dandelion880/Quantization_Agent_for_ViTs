# adalog 完整 static W4A4 种子

joint/sequential：W为per-channel affine percentile；普通A为affine percentile，P及FC2输入使用预留零的AdaLog，FC2使用独立bias补偿。base候选19/37/55/74、scale倍率.5/.75/1/1.25，以局部输出MSE搜索。这是既有插件组合适配，不是论文全套FPCS。

完整可执行构造：`ptq_agent.execution.baselines.build_baseline(backend, "adalog", quantize_head=False)`，返回普通v3计划，可由标准RTNExecutor准备、导出、重载和评测。完整参数在构造源码中，模型/数据/seed/校准数量/评测协议由当前 solution.configuration 固定；不另设默认预算。`configuration_asset` 提供当前完整配置及来源版本。

ImageNet Top-1：待测（null）；ADE20K mIoU：待测（null）。不能继承论文数字、其它模型成绩或局部search成绩。来源及固定commit见 seeds.json / references/ptq_methods.json；官方材料优先，允许合理自主适配。

伪代码：展开真实计算边界 → 按该方案分配独立W/A/预处理组件 → 仅在calibration拟合 → 冻结 → export/load验证 → 同协议ImageNet与ADE完整评测。若某插件不兼容，保留原方案，明确修订选择后回退成熟完整配置继续研究。
