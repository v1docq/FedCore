import sys

import torch
from fastai.torch_core import _has_mps
from fastcore.basics import defaults


def default_device(device_type: str = "CUDA"):
    """Return or set default device. Modified from fastai.

    Args:
        device_type: 'CUDA' or 'CPU' or None (default: 'CUDA'). If None, use CUDA if available, else CPU.

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
