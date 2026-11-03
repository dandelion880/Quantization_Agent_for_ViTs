"""RTN reference 格式编码；真实 bit packing 与浮点计算明确分开。

FP8/FP4 最近偶数舍入，饱和到最大有限值；NF4 最近 codebook 值，等距取低索引。
MXFP4 使用 OCP v1 推荐的 floor(log2(amax))-2 共享指数；NVFP4 使用
16 元素 E2M1 + E4M3 块尺度 + FP32 全局尺度，无 RHT/随机舍入/训练配方。
"""

from copy import deepcopy
import numpy as np
import torch

from ..schemas import require
from .contracts import BLOCKS, FORMATS, RAW_FLOATS

DTYPES = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}
# 固定 NF4 数学码本，来源与舍入约定见 docs/FORMATS.md。
NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def positive_table(fmt, device):
    if fmt in ("FP4_E2M1", *BLOCKS):
        values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    else:
        e, m, bias = (4, 3, 7) if fmt == "FP8_E4M3FN" else (5, 2, 15)
        end = 127 if e == 4 else 124  # E4M3FN 的 0x7f 是 NaN；E5M2 的 exponent=31 不编码。
        values = []
        for code in range(end):
            exponent, mantissa = code >> m, code & ((1 << m) - 1)
            values.append(
                (mantissa / 2**m) * 2 ** (1 - bias)
                if exponent == 0
                else (1 + mantissa / 2**m) * 2 ** (exponent - bias)
            )
    return torch.tensor(values, dtype=torch.float64, device=device)


def nearest(x, table, *, ties_even=True):
    hi = torch.searchsorted(table, x.contiguous()).clamp(0, len(table) - 1)
    lo = (hi - 1).clamp_min(0)
    dl, dh = (x - table[lo]).abs(), (table[hi] - x).abs()
    upper = dh < dl
    if ties_even:
        upper |= (dh == dl) & ((hi % 2) == 0)
    return torch.where(upper, hi, lo)


def view(x, spec):
    if spec["granularity"] == "per_group":
        size = spec["group_size"]
        return x.reshape(*x.shape[:-1], x.shape[-1] // size, size)
    return x


def ranges(x, spec):
    mode = spec["granularity"]
    if mode == "per_tensor":
        return x.amin(), x.amax()
    if mode == "per_channel":
        dims = tuple(i for i in range(x.ndim) if i != spec["axis"])
        return x.amin(dims, keepdim=True), x.amax(dims, keepdim=True)
    return x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)


def fit(x, spec):
    x = view(x.detach().to(torch.float64), spec)
    require(x.numel() > 0 and bool(torch.isfinite(x).all()), "非有限量化输入", code="numerical_error")
    fmt = spec["format"]
    if fmt in RAW_FLOATS:
        return {}
    lo, hi = ranges(x, spec)
    amax = torch.maximum(lo.abs(), hi.abs())
    if fmt in BLOCKS:
        if fmt == "MXFP4":
            exp = torch.where(amax > 0, torch.floor(torch.log2(amax)) - 2, torch.zeros_like(amax)).clamp(
                -127, 127
            )
            return {"scale_codes": (exp + 127).to(torch.uint8)}
        global_scale = (amax.max() / (6 * 448)).to(torch.float32)
        global_scale = torch.where(global_scale == 0, torch.ones_like(global_scale), global_scale)
        require(bool(torch.isfinite(global_scale)), "NVFP4 全局尺度非有限", code="numerical_error")
        codes = nearest(amax / 6 / global_scale, positive_table("FP8_E4M3FN", x.device))
        return {"scale_codes": codes.to(torch.uint8), "global_scale": global_scale}
    if fmt.startswith(("INT", "UINT")):
        bits = FORMATS[fmt]
        if spec["symmetric"]:
            qmax = 2 ** (bits - 1) - 1
            scale, zero = amax / qmax, torch.zeros_like(amax)
        else:
            qmin = 0 if fmt.startswith("UINT") else -(2 ** (bits - 1))
            qmax = qmin + 2**bits - 1
            lo, hi = torch.minimum(lo, torch.zeros_like(lo)), torch.maximum(hi, torch.zeros_like(hi))
            scale = (hi - lo) / (qmax - qmin)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero = (qmin - torch.round(lo / scale)).clamp(qmin, qmax)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return {"scale": scale, "zero": zero}
    maximum = 1 if fmt == "NF4" else float(positive_table(fmt, x.device)[-1])
    scale = amax / maximum
    return {"scale": torch.where(scale == 0, torch.ones_like(scale), scale)}


def scale_for(params, spec, device):
    if spec["format"] == "MXFP4":
        return torch.pow(2.0, params["scale_codes"].to(device=device, dtype=torch.float64) - 127)
    if spec["format"] == "NVFP4":
        return positive_table("FP8_E4M3FN", device)[params["scale_codes"].to(device).long()] * params[
            "global_scale"
        ].to(device)
    return params["scale"].to(device)


def encode(x, spec, params=None):
    """params 非空表示静态校准；不会用推理输入重新拟合静态尺度。"""
    require(bool(torch.isfinite(x).all()), "量化输入非有限", code="numerical_error")
    params = fit(x, spec) if params is None else params
    fmt, bits = spec["format"], FORMATS[spec["format"]]
    if fmt in RAW_FLOATS:
        codes = x.detach().to(DTYPES[fmt])
        require(bool(torch.isfinite(codes).all()), f"{fmt} 转换溢出", code="numerical_error")
        return codes, params
    data = view(x.to(torch.float64), spec)
    scale = scale_for(params, spec, data.device)
    require(bool(torch.isfinite(scale).all()), "尺度非有限", code="numerical_error")
    z = torch.where(
        scale > 0, data / torch.where(scale > 0, scale, torch.ones_like(scale)), torch.zeros_like(data)
    )
    if fmt.startswith(("INT", "UINT")):
        qmin = 0 if fmt.startswith("UINT") else -(2 ** (bits - 1))
        qmax = qmin + 2**bits - 1
        if spec["symmetric"]:
            qmin = -qmax
        q = (torch.round(z) + params["zero"].to(data.device)).clamp(qmin, qmax).to(torch.int32)
        codes = (q & (2**bits - 1)).to(torch.uint8)  # INT 为二补码；不伪装成 UINT。
    elif fmt == "NF4":
        codes = nearest(z, torch.tensor(NF4, device=data.device, dtype=torch.float64), ties_even=False).to(
            torch.uint8
        )
    else:
        codes = (
            nearest(z.abs(), positive_table(fmt, data.device)) | (torch.signbit(z).long() << (bits - 1))
        ).to(torch.uint8)
    return codes.reshape(x.shape), params


def decode(codes, spec, params):
    fmt, bits = spec["format"], FORMATS[spec["format"]]
    if fmt in RAW_FLOATS:
        return codes.to(torch.float32)
    q = view(codes.long(), spec)
    if fmt.startswith(("INT", "UINT")):
        if fmt.startswith("INT"):
            q = torch.where(q >= 2 ** (bits - 1), q - 2**bits, q)
        values = q - params["zero"].to(q.device)
    elif fmt == "NF4":
        values = torch.tensor(NF4, device=q.device, dtype=torch.float64)[q]
    else:
        values = positive_table(fmt, q.device)[q & (2 ** (bits - 1) - 1)]
        values = torch.where((q & 2 ** (bits - 1)) != 0, -values, values)
    result = (values * scale_for(params, spec, q.device)).reshape(codes.shape).to(torch.float32)
    require(bool(torch.isfinite(result).all()), "反量化结果非有限", code="numerical_error")
    return result


def qdq(x, spec, params=None):
    codes, params = encode(x, spec, params)
    return decode(codes, spec, params).to(x.dtype)


def pack(codes, bits):
    """低位优先紧密存储，INT6 不用 INT8 的大小冒称 6-bit 存储。"""
    data = codes.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
    binary = ((data[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return torch.from_numpy(np.packbits(binary, bitorder="little").copy())


def unpack(packed, bits, shape, device):
    count = int(np.prod(shape))
    binary = np.unpackbits(packed.cpu().numpy(), bitorder="little")[: count * bits].reshape(count, bits)
    data = (binary * (1 << np.arange(bits))).sum(1).astype(np.uint8).reshape(shape)
    return torch.from_numpy(data).to(device)


def encode_weight(weight, spec):
    codes, params = encode(weight, spec)
    return {
        "spec": deepcopy(spec),
        "shape": list(weight.shape),
        "data": codes.cpu() if spec["format"] in RAW_FLOATS else pack(codes, spec["bits"]),
        "params": {k: v.cpu() for k, v in params.items()},
    }


def decode_weight(record, device):
    spec = record["spec"]
    codes = (
        record["data"].to(device)
        if spec["format"] in RAW_FLOATS
        else unpack(record["data"], spec["bits"], record["shape"], device)
    )
    return decode(codes, spec, record["params"])


def validate_params(params, spec, shape):
    """拒绝错误形状/尺度的持久化参数，避免广播静默改变量化定义。"""
    fmt = spec["format"]
    if fmt in RAW_FLOATS:
        require(params == {}, "直接浮点转换不应带尺度", code="invalid_artifact")
        return
    keys = (
        {"scale_codes", "global_scale"}
        if fmt == "NVFP4"
        else (
            {"scale_codes"}
            if fmt == "MXFP4"
            else {"scale", "zero"} if fmt.startswith(("INT", "UINT")) else {"scale"}
        )
    )
    require(set(params) == keys, "量化参数字段不符", code="invalid_artifact")
    mode = spec["granularity"]
    if mode == "per_tensor":
        expected = []
    elif mode == "per_channel":
        expected = [shape[i] if i == spec["axis"] else 1 for i in range(len(shape))]
    elif mode == "per_group":
        expected = [*shape[:-1], shape[-1] // spec["group_size"], 1]
    else:
        expected = [*shape[:-1], 1]
    for key, value in params.items():
        required_shape = [] if key == "global_scale" else expected
        dtype = (
            torch.uint8 if key == "scale_codes" else torch.float32 if key == "global_scale" else torch.float64
        )
        require(
            torch.is_tensor(value)
            and list(value.shape) == required_shape
            and value.dtype == dtype
            and bool(torch.isfinite(value).all()),
            "尺度 dtype/形状/有限值不符",
            code="invalid_artifact",
        )
        if key == "scale_codes":
            require(
                bool((value <= (126 if fmt == "NVFP4" else 254)).all()),
                "尺度包含非有限编码",
                code="invalid_artifact",
            )
        elif key == "zero":
            bits = FORMATS[fmt]
            low = 0 if fmt.startswith("UINT") else -(2 ** (bits - 1))
            high = low + 2**bits - 1
            require(
                bool(((value == value.round()) & (value >= low) & (value <= high)).all()),
                "zero point 越界或非整数",
                code="invalid_artifact",
            )
            require(not spec["symmetric"] or bool((value == 0).all()), "对称 zero point 必须为零")
        else:
            require(bool((value > 0).all()), "尺度必须为正", code="invalid_artifact")


def validate_record(record, spec, shape):
    require(
        set(record) == {"spec", "shape", "data", "params"}
        and record["spec"] == spec
        and record["shape"] == shape,
        "权重编码的规格/形状不同",
        code="invalid_artifact",
    )
    data = record["data"]
    if spec["format"] in RAW_FLOATS:
        require(
            data.dtype == DTYPES[spec["format"]]
            and list(data.shape) == shape
            and bool(torch.isfinite(data).all()),
            "浮点权重编码无效",
            code="invalid_artifact",
        )
    else:
        require(
            data.dtype == torch.uint8 and list(data.shape) == [(int(np.prod(shape)) * spec["bits"] + 7) // 8],
            "紧密权重编码的 dtype/长度不同",
            code="invalid_artifact",
        )
    validate_params(record["params"], spec, shape)
