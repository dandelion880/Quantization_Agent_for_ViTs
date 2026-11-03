# RTN / 通用有限网格

来源：基础量化。固定论文/源码 revision 见 [来源索引](../references/ptq_methods.json)（本项目内的只读来源记录，历史上游路径不参与运行导入）。

实现：`tools/ptq_agent/execution/plugin_math.py:encode_rtn; codecs.py:fit/qdq`。

用途：普通分布的低成本对照；也提供 NF4/FP4/FP8 等已登记格式。不能认为 minmax 就能保留低位激活精度。

伪代码（signed INT）：s=absmax(x)/(2^(b-1)-1); q=clip(round(x/s),-qmax,qmax); y=q*s。每个独立尺度组使用同一定义。

spec 使用实际 format/bits；整数 scale=minmax，INT symmetric=true，UINT symmetric=false。per_tensor 的 axis/group_size=null；W per_channel axis=0。A per_channel 使用位点目录的activation_channel_axis，并按实测rank换算为非负axis：Linear/Q/V/LN通常为末轴，QK第二操作数K已转置为[B,H,C,N]，feature为axis=2；NCHW卷积输入为axis=1。不能把K的token位置轴误标为通道。static A不能使用per_token/per_group或NV/MX的逐位置块尺度。

weight_calibration 与 activation_calibration 独立，可 minmax/manual/search；search 在真实局部输出上比较 mse/cosine/hessian。GPTQ 是不同权重算法。

诊断：检查正/负区间误差、被置零比例、通道差异；负支全部消失时应比较其它合法网格或尺度策略，不要只改名称。W/A 的阶段顺序也影响 observer。

成本：minmax 较低；scale search 需更多局部前向。注册格式不意味着 V100 有相应 kernel。

