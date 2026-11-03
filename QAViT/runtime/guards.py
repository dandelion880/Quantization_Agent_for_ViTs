"""执行保护：Landlock 写边界和训练权限审计；只在 worker 内安装。"""
from pathlib import Path
import os
from runtime.paths import ROOT, LIBRARY

GRADIENT_AUDIT = {"optimizer_steps": 0, "measurement_gradient_calls": 0, "other_gradient_calls": 0}

def install_gradient_guard(training_free):
    """覆盖所有任务(含profiling)：禁优化器更新；仅允许已核验的只读梯度统计。"""
    import inspect
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    from runtime.paths import LIBRARY
    trusted = (LIBRARY / 'execution/calibration.py').resolve()
    def step(optimizer, args, kwargs):
        if training_free:
            raise ValueError("training-free禁止任何优化器梯度更新，包括profiling")
        GRADIENT_AUDIT["optimizer_steps"] += 1
    handle = register_optimizer_step_pre_hook(step)
    original_grad, original_backward = torch.autograd.grad, torch.autograd.backward
    def grad(*args, **kwargs):
        caller = Path(inspect.currentframe().f_back.f_code.co_filename).resolve()
        measured = caller == trusted
        GRADIENT_AUDIT["measurement_gradient_calls" if measured else "other_gradient_calls"] += 1
        if training_free and not measured:
            raise ValueError("training-free不执行未验证的自定义梯度路径；仅开放旧capture只读梯度统计")
        return original_grad(*args, **kwargs)
    def backward(*args, **kwargs):
        GRADIENT_AUDIT["other_gradient_calls"] += 1
        if training_free:
            raise ValueError("training-free禁止backward更新路径；只读梯度统计使用已验证capture")
        return original_backward(*args, **kwargs)
    torch.autograd.grad, torch.autograd.backward = grad, backward
    return handle


def protect_files(directory):
    """Landlock内核写隔离；只允许当前job与EXP临时目录写入。"""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock不可用，未执行生成代码")
    write_access = sum(1 << i for i in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    if abi >= 2:
        write_access |= 1 << 13
    if abi >= 3:
        write_access |= 1 << 14
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]
    ruleset = Ruleset(write_access)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path in (Path(directory), ROOT / "EXP/cache/tmp"):
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(write_access, pfd)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)
    return {"mechanism": "Linux Landlock", "abi": abi, "write_roots": [str(directory), str(ROOT / "EXP/cache/tmp")]}

