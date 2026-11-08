# 分布异常、空间位置与误差传播

**适用**：已在特定量化边或中间层观察到异常分布、误差或下游表现冲突。核心问题是“它是否真的在驱动所关心的任务误差？”

## 从统计找到有判别力的测量

1. **选择恰当坐标**：raw/RMS histogram、q99/q99.9/absmax、token RMS、channel RMS、能量占比、量化后截断/舍入误差。区分绝对幅值与相对影响；异常数据的能量占比会让整体 NMSE 被少数位置主导。
2. **按结构划分**：ViT 的 CLS、patch、register/storage 等 token 可能承担不同语义；先确认目标模型确实存在这些 token。选择 layer、输入/输出量化边、LN/GELU/Softmax、Q/K/V、attention 输出及 LayerScale/残差前后的位置。
3. **定义相互竞争的机制**：异常值只是拉大 scale，还是它确实造成大部分编码误差？某通道大幅值是“统计热点”，还是量化器位宽/码点在该通道特别不合适？局部失真是否沿残差传播到 logits？
4. **选择最短干预**：必要时固定其余量化器，比较对应量化边的量化/旁路，或改变一个 scale/网格策略；观察最邻近算子和下游 1～2 个有解释力的位置。仅“撤回后误差下降”不能自动证明它对最终任务性能有正向贡献。

## 工具关联

- `profile_distribution`：tensor/token/channel 汇总；对比 dtype、calibration 与分析样本身份。
- `profile_plugin_sites`：根据可用插件范围观察 Q/K/V、post-Softmax、post-GELU、post-LN 等特殊位点；先查询实际站点。
- `profile_trial_diagnostics` / `profile_trial_ablation`：有真实冻结候选时作定向比较，不对不存在的 trial 盲调。
- 若现有 profile 没有 `token × channel` 联合矩阵，不应凭两个边缘统计推断每个 token-channel 交叉位置；确有需要再申请小范围原始采集。

## 传播判断

- 同时报告“误差在哪里引入”和“在哪里变成输出损失”。残差/LN/LayerScale 可能压制、重加权或传播局部误差。
- 某处的绝对 MSE 增大不必然使分类/分割任务下降；若下游信号较弱，优先考虑更敏感的机制或停止该方向。
- 比较同一量化前缀下的 peer 可定位干预边际效果；与原 FP 参考比较回答整体恢复程度。两者不可互换。

## 示例与反例

**示例**：发现 DINOv3 某 block 的 storage token q99.9 极高且贡献大部分能量；下一步应该检查该 token 是否实质驱动 A4 误差，再看干预后的下游变化，而不是立刻为 storage token 设计独立量化规则。

**反例**：只看到跨模型 raw RMS 差异就断言“一切低比特失效都由特殊 token 引起”。这是相关性过度解释。
