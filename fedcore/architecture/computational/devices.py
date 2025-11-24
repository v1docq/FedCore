import torch
import logging


def check_device(dev_type: str):
    if dev_type == 'cuda':
        return torch.cuda.is_available()
    elif dev_type == 'mps':
        return torch.backends.mps.is_available()
    elif dev_type == 'cpu':
        return True
    raise ValueError('Unknown device')


def default_device(device_type: str = None):

    if device_type is not None and check_device(device_type):
        logging.info(f'Trying to use device <{device_type}>')
        if device_type == 'cuda':
            selected = torch.device(torch.cuda.current_device())
        elif device_type == 'mps':
            selected = torch.device('mps')
        elif device_type == 'cpu':
            selected = torch.device('cpu')
    else:
        available_device = next((dev for dev in ('cuda', 
                                                 'mps', 
                                                 'cpu'
                                                 ) if check_device(dev)))
        if available_device == 'cuda':
            selected = torch.device(torch.cuda.current_device())
        elif available_device == 'mps' or 'cpu':
            selected = torch.device(device=available_device)
    logging.info(f'Device <{selected}> is selected')
    return selected


def extract_device(nn: torch.nn.Module):
    try:
        return next(iter(nn.parameters())).device
    except StopIteration:
        return default_device()
