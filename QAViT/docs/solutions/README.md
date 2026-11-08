# 完整方案 Top-3

与 methods/ 组件库、Recorder/Memory 知识库分开。种子为本地已接入、有来源与成熟论文依据的 GPTAQ、AdaLog、RepQ 三种完整适配方案；没有当前严格协议的双任务成绩，因此不是已实测证明的前三名，也不把论文成绩填入实测字段。AdaLog官方README的DeiT-B W4A4校准Top-1为78.03%，但覆盖/码本/校准不同，不能当作本适配成绩。官方链接见 seeds.json。

每个运行初始化冻结三条配置、源码快照引用与版本。研究运行共享 EXP/solutions/<protocol_hash>/top3.json；验证运行隔离在自己的目录。仅每轮最终完整方案的有效真实L3双任务结果可晋升，先比ADE20K mIoU，再比ImageNet Top-1；相等保留旧项。已评测项优先于尚无成绩的占位种子，待测不伪装数值排名。协议包含模型/数据、量化约束与评测设置，不合并异质协议。并发写入加锁、原子落盘、Journal保留当时快照；旧方案在archive及原运行中保留。

Generation、Coding与Recorder收到三个短目录，configuration_asset可精确读取完整配置/执行参数/源码版本/产物；新生成的实现还随方案复制可读源文件。不增加Generation字段审查或LLM门槛。未晋升结果继续沿原Memory事实与Recorder综合机制沉淀。
