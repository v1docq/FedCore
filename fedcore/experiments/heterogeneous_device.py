"""One declared physical GPU for guard, recovery and TensorRT; no discovery."""
import os
import re

_binding = None
_verified = None


def _current_context_uuid():
    """CUDA Driver API: identify the actual current context, not NVML ordering.

    cuCtxGetDevice and cuDeviceGetUuid_v2 are queried only during an explicitly
    requested hardware run. A zero-element tensor establishes Torch's context;
    no training or workload inference precedes the identity check.
    """
    import ctypes
    import uuid as uuid_module
    import torch
    torch.empty(0,device='cuda:0')
    driver = ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else ctypes.CDLL('libcuda.so.1')
    device = ctypes.c_int()
    get_device = driver.cuCtxGetDevice
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]; get_device.restype = ctypes.c_int
    if get_device(ctypes.byref(device)) != 0:
        raise RuntimeError('Cannot identify the current CUDA context device')
    raw = (ctypes.c_ubyte*16)()
    get_uuid = driver.cuDeviceGetUuid_v2
    get_uuid.argtypes = [ctypes.c_void_p,ctypes.c_int]; get_uuid.restype = ctypes.c_int
    if get_uuid(ctypes.byref(raw),device) != 0:
        raise RuntimeError('Cannot verify the current CUDA device UUID')
    return 'GPU-' + str(uuid_module.UUID(bytes=bytes(raw)))


def validate_gpu_uuid(value):
    if (type(value) is not str or re.fullmatch(
            r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',value) is None):
        raise ValueError('An explicit stable GPU UUID is required')
    return value


def bind_cuda_uuid(uuid):
    """Call in the owned worker before baseline/recovery or any CUDA init.

    CUDA visibility must contain exactly the same UUID monitored by the guard.
    Never override an initialized or differently provisioned CUDA environment.
    A marker belongs to this process, so a fork cannot inherit a false proof.
    """
    import torch
    global _binding
    uuid = validate_gpu_uuid(uuid)
    expected = (os.getpid(),uuid)
    if _binding == expected:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise RuntimeError('CUDA visibility changed after binding')
        return
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU binding must precede the first CUDA initialization')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,uuid):
        raise RuntimeError('Declared GPU UUID conflicts with the inherited CUDA visibility')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    _binding = expected


def require_cuda_binding(uuid):
    """Validate an existing binding and the sole visible logical device."""
    import torch
    global _verified
    bind_cuda_uuid(uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The declared physical GPU must be the sole visible CUDA device')
    torch.cuda.set_device(0)
    actual = _current_context_uuid()
    if actual.lower() != uuid.lower():
        raise RuntimeError('Actual CUDA context UUID does not match the monitored physical GPU')
    _verified = dict(pid=os.getpid(),declared_uuid=uuid,actual_uuid=actual,
                     evidence='CUDA Driver cuCtxGetDevice + cuDeviceGetUuid_v2')
    return dict(_verified)
