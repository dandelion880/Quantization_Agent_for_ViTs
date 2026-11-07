"""Project-wide GPU leases, inherited by workers and released on process exit.

Placement is operational, not a scientific choice. Busy cards leave tasks ready;
multi-GPU evaluation uses whatever is free, including the original single GPU.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from core.cards import ContractError, fingerprint
from runtime.paths import ROOT


class ResourceBusy(Exception):
    pass


def inventory():
    """NVML indices, CUDA UUIDs and device minors differ in restricted containers."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=5, check=True)
    uuids = {int(row.split(',')[0]): row.split(',')[1].strip() for row in result.stdout.splitlines()}
    minors = {}
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
        if 'GPU UUID' in fields and 'Device Minor' in fields:
            minors[fields['GPU UUID'].strip()] = int(fields['Device Minor'].strip())
    return [{'index': gpu, 'uuid': uuid, 'device_node': '/dev/nvidia' + str(minors[uuid])}
            for gpu, uuid in uuids.items() if uuid in minors]


def visible_uuids():
    """Ask CUDA in a disposable process: its ordinal order need not be NVML's.

    Device enumeration creates no model/context; the Supervisor never initializes
    CUDA. Container cgroups and CUDA_VISIBLE_DEVICES are both respected.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1') or os.environ.get('NVIDIA_VISIBLE_DEVICES') in ('', 'none', 'void'):
        return []
    if not any(Path('/dev').glob('nvidia[0-9]*')):
        return []
    script = '''
import ctypes, json, uuid
cuda = ctypes.CDLL('libcuda.so.1')
status = cuda.cuInit(0)
if status == 100:
    print('[]'); raise SystemExit
if status: raise RuntimeError('cuInit: ' + str(status))
count = ctypes.c_int()
if cuda.cuDeviceGetCount(ctypes.byref(count)): raise RuntimeError('cuDeviceGetCount')
values = []
for ordinal in range(count.value):
    device, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    if cuda.cuDeviceGet(ctypes.byref(device), ordinal): raise RuntimeError('cuDeviceGet')
    if cuda.cuDeviceGetUuid(ctypes.byref(raw), device): raise RuntimeError('cuDeviceGetUuid')
    values.append('GPU-' + str(uuid.UUID(bytes=bytes(raw))))
print(json.dumps(values))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=15, check=True)
    return json.loads(result.stdout)


def available_inventory(settings):
    """Explicit hardware descriptions replace unavailable NVML/proc metadata only.

    They do not enlarge CUDA visibility. UUID visibility can also be supplied by
    CUDA_VISIBLE_DEVICES on systems where driver enumeration is unavailable.
    """
    try:
        visible = visible_uuids()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        tokens = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        container = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'all').split(',')
        if (settings.get('gpu_inventory') and tokens and all(t.startswith('GPU-') for t in tokens)
                and (container == ['all'] or all(t.startswith('GPU-') for t in container))):
            visible = tokens
        else:
            raise ContractError('无法可靠识别 CUDA 可见设备；可集中设置 runtime.gpu_inventory '
                                '及 allowed_gpus，并以 CUDA_VISIBLE_DEVICES=GPU-UUID,... 指定可见性') from exc
    if not visible:
        return []
    try:
        rows = settings.get('gpu_inventory')
        rows = inventory() if rows is None else rows
        if (not isinstance(rows, list) or any(not isinstance(r, dict) or type(r.get('index')) is not int
                or r['index'] < 0 or not isinstance(r.get('uuid'), str) or not r['uuid'].startswith('GPU-')
                or not isinstance(r.get('device_node'), str) or not r['device_node'].startswith('/dev/nvidia')
                or not r['device_node'][len('/dev/nvidia'):].isdigit() for r in rows)
                or any(len({r[k] for r in rows}) != len(rows) for k in ('index', 'uuid', 'device_node'))):
            raise ContractError('runtime.gpu_inventory 须为不重复的 index/uuid/device_node 对照表')
        selected = [dict(row) for row in rows if row['uuid'] in visible]
        for name in ('NVIDIA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS'):
            value = os.environ.get(name)
            if value is None or value == 'all':
                continue
            selectors = value.split(',')
            allowed = set()
            for item in selectors:
                if item.startswith('GPU-'):
                    allowed.add(item)
                elif item.isdigit():
                    allowed.add(int(item))
                elif '-' in item and all(v.isdigit() for v in item.split('-')) and len(item.split('-')) == 2:
                    start, end = map(int, item.split('-'))
                    allowed.update(range(start, end + 1))
                else:
                    raise ContractError('无法解释分配设备范围 ' + name + ': ' + value)
            if name == 'NVIDIA_VISIBLE_DEVICES' and any(
                    type(s) is int and s not in {r['index'] for r in rows} for s in allowed):
                # An outer ordinal can be renumbered inside a container. Neither
                # ignore the allocation nor guess which remaining UUID it meant.
                raise ContractError('NVIDIA_VISIBLE_DEVICES 数字无法对应当前设备；请使用 UUID 范围或 '
                                    'runtime.gpu_inventory 显式硬件映射')
            selected = [r for r in selected if r['uuid'] in allowed or r['index'] in allowed]
        if visible and not rows:
            raise ContractError('可见 GPU 缺少 UUID/设备节点映射；请集中提供 runtime.gpu_inventory')
        return selected
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError('GPU 身份/设备节点探测不可用；请在 runtime.gpu_inventory 集中提供硬件对照表') from exc


def resolve_runtime_resources(settings):
    """Resolve once for a new run; persist authorization and UUIDs for recovery."""
    allowed = settings.get('allowed_gpus', 'auto')
    if allowed != 'auto' and (not isinstance(allowed, list) or any(
            not (type(g) is int and g >= 0 or isinstance(g, str) and g.startswith('GPU-')) for g in allowed)):
        raise ContractError('runtime.allowed_gpus 须为 auto 或 NVML 编号/GPU UUID 数组；[] 表示禁用 GPU')
    rows = available_inventory(settings) if allowed != [] else []
    selected = rows if allowed == 'auto' else [row for row in rows if row['index'] in allowed or row['uuid'] in allowed]
    # An unavailable explicitly authorized device is not replaced with a host GPU.
    settings['allowed_gpus'] = [r['index'] for r in selected]
    if selected:
        settings['gpu_bindings'] = selected
    else:
        settings.pop('gpu_bindings', None)
    if allowed != []:
        settings['gpu_selection'] = {'requested': allowed, 'resolved': settings['allowed_gpus'],
                                     'identity': 'NVML index labels bound to CUDA UUIDs for this run'}
    if settings.get('experiment_concurrency', 'auto') == 'auto':
        settings['experiment_concurrency'] = max(1, len(selected))


def gpu_bindings(gpus, settings=None):
    if not gpus:
        return []
    if settings is None:  # legacy/direct callers, NVML indices
        rows = inventory()
        by_index = {row['index']: row for row in rows}
    else:
        rows = available_inventory(settings)
        saved = settings.get('gpu_bindings')
        if saved:
            by_uuid = {row['uuid']: row for row in rows}
            by_index = {row['index']: {**by_uuid[row['uuid']], 'index': row['index']}
                        for row in saved if row['uuid'] in by_uuid}
        else:
            by_index = {row['index']: row for row in rows}
    return [by_index[g] for g in gpus if g in by_index]


def placement(experiment, settings, brief=None):
    from tools.experiment_worker import backend_device
    payload = experiment['payload']
    spec, resources = payload['execution'], payload['resources']
    device = backend_device(spec)
    cuda = str(device).startswith('cuda')
    pool = ([resources['gpu']] if resources.get('gpu') is not None
            else list(dict.fromkeys(settings.get('allowed_gpus', [])))) if cuda else []
    requested = resources.get('gpu_count', 1)
    from tools.result_contract import classification_shard_fallback
    fallback = classification_shard_fallback(payload, brief or {})
    supported = cuda and fallback is None
    # An explicit gpu pins the primary; other authorized GPUs may supply shards.
    if supported and requested > 1 and pool:
        pool += [g for g in settings.get('allowed_gpus', []) if g not in pool]
    useful = max((r.get('count') or requested for r in spec.get('evaluations', [])), default=requested)
    return {'gpu_pool': pool, 'gpu_count': min(requested, useful) if supported else 1,
            'pinned_gpu': resources.get('gpu') if cuda else None,
            'parallelism': {'requested_gpus': requested, 'classification_shards_supported': bool(supported),
                            'fallback': None if supported else (fallback or 'CPU single-device execution')}}


_busy_cache = (0, set())


def externally_busy():
    """Avoid already occupied CUDA cards as well as QAViT leases; no GPU mutation."""
    global _busy_cache
    if time.monotonic() - _busy_cache[0] < 1:
        return _busy_cache[1]
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise ContractError('无法读取 GPU 资源状态: ' + result.stderr.strip())
    busy, indices = set(), {}
    for line in result.stdout.splitlines():
        index, uuid, memory, utilization = (v.strip() for v in line.split(','))
        index, memory, utilization = int(index), int(memory), int(utilization)
        indices[uuid] = index
        if memory > 32 or utilization > 0:
            busy.update((index, uuid))
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    if apps.returncode:
        raise ContractError('无法读取 GPU 进程占用: ' + apps.stderr.strip())
    for uuid in apps.stdout.splitlines():
        if uuid.strip() in indices:
            busy.update((uuid.strip(), indices[uuid.strip()]))
    _busy_cache = (time.monotonic(), busy)
    return busy


def acquire(pool, count=1, pinned=None, *, bindings=None):
    if not pool:
        return [], []
    busy = externally_busy()
    fds, selected = [], []
    identities = {r['index']: r['uuid'] for r in bindings} if bindings is not None else {}
    try:
        for gpu in pool:
            identity = identities.get(gpu, gpu)
            if identity not in busy:
                path = ROOT / 'EXP/cache/v2_resources' / (fingerprint('gpu:' + str(identity)) + '.lock')
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                else:
                    fds.append(fd)
                    selected.append(gpu)
            if gpu == pinned and gpu not in selected:
                break
            if len(selected) >= count:
                break
        if not selected or (pinned is not None and pinned not in selected):
            raise ResourceBusy('GPU busy; task remains queued')
        return selected, fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
