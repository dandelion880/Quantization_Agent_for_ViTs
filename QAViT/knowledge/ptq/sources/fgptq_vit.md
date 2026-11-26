## 5. FGPTQ-ViT：保留确定的机制归属，补齐全文后签发配方

<a id="fgptq_vit"></a>

**证据。** 综述 [98]；[Springer 正式条目](https://link.springer.com/chapter/10.1007/978-981-99-8546-3_7)、[出版社 Crossmark](https://crossmark.crossref.org/dialog/?doi=10.1007%2F978-981-99-8546-3_7)。作者为 Caihua Liu、Hongyang Shi、Xinyu He；PRCV 2023，首次在线 2023-12-26，纸本 2024，页 79–90。综述的作者字段需纠正，不能把合法的纸本年份简单判错。此次取得摘要与书目信息，没有取得全文，也未核实官方代码。

### 已确认的原配方范围

正式摘要确认它是面向 post-Softmax/post-GELU 非均衡分布的分段 PTQ，使用自适应分段点搜索，分段点及搜索采用二的幂结构；使用 32 张校准图，给出 ImageNet 8bit 验证。除此之外，摘要不能确定每段码字数量、正负分配、阈值搜索目标、端点含义、权重与首尾层例外，更不能证明全 W4A4 的结果。[出版社条目](https://link.springer.com/chapter/10.1007/978-981-99-8546-3_7)

所以这里不是一个可直接执行的“完整复现 recipe”。研究目录应将它登记为 `ptq_recipe_pending_fulltext`，保留获取正式全文的具体任务；获取后必须核读分段 quantizer 定义、piecewise-point 搜索算法和实验设置表，才能把字段升为 verified。不得以已有 PTQ4ViT twin 实现加一个方法名，就将 FGPTQ 计为完成。

### 可组合 primitive 的接入槽位

建议的 `search.power2_piecewise_boundary` 接受固定码字预算、合法阈值集合、校准样本与误差目标，输出分段边界、各段 scale 与完整代码表。这个接口来自 QAViT 对同类方法的抽象，不是已核实的 FGPTQ 官方 API；目前不填原论文未知的默认超参。它应与 twin quantizer 共享编码验证器，但保留各自的分段数学与码字分配。

### 研究工作流与当前差距

拟议研究问题是：在相同真实码字总数下，硬件可实现的 power-of-two 分段能否比自由阈值获得更好的成本/误差折中。应先以已经有全文和代码证据的 PTQ4ViT/TSPTQ 实现搭建这一比较，再把 FGPTQ 的原配方作为待补充项。这样的目录能完整解释综述条目的去向，同时不会让后续 Codex 被迫猜论文。static W4A4 不登记为 FGPTQ 已验证配置。

待全文到位后的验收不是“补一条参考文献”：必须能由记录重建每个整数码的解码值、分段边界左右闭开规则以及搜索时真正变化的变量。若这些与现有 twin 完全等价，只增加 recipe 和证据，不重复实现；若有新的分辨率分配，再增加最小必要 primitive。

