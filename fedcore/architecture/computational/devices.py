"""Utility helpers for selecting and querying computation devices.

This module provides small wrappers around PyTorch device handling
to:

* check availability of a specific device type (``cuda``, ``mps``, ``cpu``);
* choose a default device based on explicit user preference or
  best-available hardware;
* extract the device from an existing neural network module.
"""

import torch
import logging


def check_device(dev_type: str) -> bool:
    """Check whether a given device type is available.

    Parameters
    ----------
    dev_type : str
        Device type to check. Supported values:
        * ``"cuda"`` – CUDA-capable GPU;
        * ``"mps"`` – Apple Metal Performance Shaders backend;
        * ``"cpu"`` – CPU device (always available).

    Returns
    -------
    bool
        ``True`` if the requested device type is available and can be used,
        ``False`` otherwise.

    Raises
    ------
    ValueError
        If an unknown device type is requested.
    """
    if dev_type == 'cuda':
        return torch.cuda.is_available()
    elif dev_type == 'mps':
        return torch.backends.mps.is_available()
    elif dev_type == 'cpu':
        return True
    raise ValueError('Unknown device')


def default_device(device_type: str = None) -> torch.device:
    """Select a default computation device.

    If ``device_type`` is explicitly provided and available, it is used.
    Otherwise the function picks the first available device in the order:
    ``"cuda"``, ``"mps"``, ``"cpu"``.

    Parameters
    ----------
    device_type : str, optional
        Preferred device type (``"cuda"``, ``"mps"``, or ``"cpu"``).
        If ``None``, the best available device is chosen automatically.

    Returns
    -------
    torch.device
        Selected PyTorch device object.

    Notes
    -----
    The function also logs the selected device using the root logger.
    """
    if device_type is not None and check_device(device_type):
        logging.info(f'Trying to use device <{device_type}>')
        if device_type == 'cuda':
            selected = torch.device(torch.cuda.current_device())
        elif device_type == 'mps':
            selected = torch.device('mps')
        elif device_type == 'cpu':
            selected = torch.device('cpu')
    else:
        available_device = next(
            (dev for dev in ('cuda', 'mps', 'cpu') if check_device(dev))
        )
        if available_device == 'cuda':
            selected = torch.device(torch.cuda.current_device())
        elif available_device == 'mps' or 'cpu':
            selected = torch.device(device=available_device)
    logging.info(f'Device <{selected}> is selected')
    return selected


def extract_device(nn: torch.nn.Module) -> torch.device:
    """Infer device from a neural network module.

    The device is inferred from the first parameter of the module.
    If the module has no parameters, :func:`default_device` is used as a
    fallback.

    Parameters
    ----------
    nn : torch.nn.Module
        PyTorch module from which to extract device information.

    Returns
    -------
    torch.device
        Device on which the module parameters are allocated, or the default
        device if the module has no parameters.
    """
    try:
        return next(iter(nn.parameters())).device
    except StopIteration:
        return default_device()
