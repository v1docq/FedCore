import sys

import torch
import fastai.torch_core as faitc
if hasattr(faitc, '_has_mps'):
    from fastai.torch_core import _has_mps
else:
    _has_mps = lambda *args, **kwargs: False
from fastcore.basics import defaults


def default_device(device_type: str = None):
    """Return or set default device. Modified from fastai.

    Args:
        device_type: 'cuda' or 'cpu' or None (default: 'cuda'). If None, use CUDA if available, else CPU.

    Returns:
        torch.device: The default device: CUDA if available, else CPU.

    """
    if device_type == "CUDA":
        device_type = defaults.use_cuda
    elif device_type == "cpu":
        defaults.use_cuda = False
        return torch.device("cpu")

    if device_type is None:
        if torch.cuda.is_available() or _has_mps() and sys.platform != "darwin":
            device_type = True
        else:
            return torch.device("cpu")
    if device_type:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        if _has_mps():
            return torch.device("mps")
    return torch.device("cpu")

def extract_device(nn: torch.nn.Module):
    return next(iter(nn.parameters())).device
