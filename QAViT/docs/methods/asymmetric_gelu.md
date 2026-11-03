# 观测式 GELU 正负尺度

来源：项目适配。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:fit_activation/activation_qdq`。

本方法是适配变体，不能标成论文原双区间。与twin_gelu的区别：负尺度来自校准池实测负幅度，正尺度独立；options.negative_factor显式缩放负尺度。

位点、INT4/6/8、per_tensor/static、互斥规则同twin_gelu。可用于检查实际GELU实现或输入只覆盖部分负区间的情况。未观测负值时clamp避免零尺度，但不能推断未来输入无负值。

正尺度可search/manual；当前coordinate只接原twin_gelu，不支持本变体。与twin为替代，不叠加。评价需包括未参与拟合的数据。

