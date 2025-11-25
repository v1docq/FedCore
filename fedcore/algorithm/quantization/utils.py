"""
Utilities for quantization operations.

Contains quantization-specific utilities, cleaned from reassembly logic.
"""

from copy import deepcopy
from functools import partial
import inspect
from typing import Literal

import torch
from torch.ao.quantization.quantize import (
    quantize, quantize_dynamic, quantize_qat
)
from torch.ao.quantization.utils import get_qconfig_dtypes
from torch.nn.quantized import FloatFunctional
import torch.nn as nn
import torchvision.models.resnet as resnet

from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.abstraction.delegator import IDelegator


def uninplace(model: nn.Module):
    """Convert in-place operations to out-of-place for quantization compatibility."""
    for name, module in model.named_modules():
        if hasattr(module, 'inplace') and module.inplace:
            module.inplace = False


def get_flattened_qconfig_dict(qconfig_mapping):
    """Convert QConfigMapping to flat dictionary."""
    # Implementation depends on QConfigMapping structure
    # This is a placeholder - actual implementation would extract configs
    return {}


class ResidualAddWrapper(nn.Module):
    """Wrapper for ResNet residual blocks to support quantization."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.skip_add = FloatFunctional()
        self.qconfig = getattr(module, 'qconfig', None)

    def forward(self, x):
        identity = x
        out = self.module.conv1(x)
        out = self.module.bn1(out)
        out = self.module.relu(out)
        out = self.module.conv2(out)
        out = self.module.bn2(out)

        if self.module.downsample is not None:
            identity = self.module.downsample(x)

        out = self.skip_add.add_relu(out, identity)
        return out


class QDQWrapping(nn.Module, IDelegator):
    """Quantization-Dequantization wrapper for modules."""
    
    __non_redirect = {'mode', 'base', 'quant', 'dequant', '_order', '_is_rnn', '__h'}
    
    def __init__(self, base, mode='pre', qconfig=None):
        super().__init__()
        self.base = base
        self.mode = mode
        
        if mode == 'pre':
            self._order = ['quant', 'base']
        elif mode == 'both':
            self._order = ['quant', 'base', 'dequant']
        elif mode == 'last':
            self._order = ['base', 'dequant']
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        from torch.ao.quantization.stubs import QuantStub, DeQuantStub
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        self._is_rnn = isinstance(self.base, nn.RNNBase)
        self.__h = None

    def __repr__(self):
        d = {
            'pre': f'{self.quant}\n{self.base}',
            'both': f'{self.quant}\n{self.base}\n{self.dequant}',
            'last': f'{self.base}\nFinal {self.dequant}'
        }
        return d[self.mode]

    def forward(self, x, *args, **kwargs):
        for layer in self._order:
            module = getattr(self, layer)
            if layer == 'base':
                out = module(x, *args, **kwargs)
                if self._is_rnn:
                    x, self.__h = out
                else:
                    x = out 
            else:
                x = module(x)
        return x

    def __call__(self, *input, **kwargs):
        result = super().__call__(*input, **kwargs)
        return (result, self.__h) if self._is_rnn else result

    def __getattr__(self, name):
        if name not in self.__non_redirect:
            return getattr(super().__getattr__('base'), name)
        else:
            return super().__getattr__(name)


class QDQWrapper(Accessor):
    """Utility class for quantization operations."""
    
    __conventional_modules = {cls[1] for cls in inspect.getmembers(torch.nn.modules, inspect.isclass)}

    @classmethod
    def __is_conventional_module(cls, module: nn.Module):
        return type(module) in cls.__conventional_modules

    @classmethod
    def __qconfig_requires_qdq(cls, module):
        qconfig = getattr(module, 'qconfig', None)
        try:
            act_type = get_qconfig_dtypes(qconfig)[0]
        except (TypeError, IndexError, AttributeError):
            act_type = None
        return act_type is torch.float32

    @staticmethod
    def is_leaf_quantizable(module: nn.Module, 
                            example_inputs: tuple, 
                            mode: Literal['qat', 'static', 'dynamic']):
        """Check if module can be quantized."""
        if example_inputs[0] is None:
            return False
        module = deepcopy(module)
        module.train(mode == 'qat')
        module.to(default_device())

        def run_fn(model: nn.Module, *example_inputs: tuple):
            model(*example_inputs)

        p_f = {
            'qat': partial(quantize_qat, run_fn=run_fn, run_args=example_inputs),
            'static': partial(quantize, run_fn=run_fn, run_args=example_inputs),
            'dynamic': quantize_dynamic,
        }[mode]

        if (example_inputs and isinstance(example_inputs[0], torch.Tensor) and 
                example_inputs[0].dtype not in {torch.int16, torch.int32, torch.int64, torch.int8}):
            m = QDQWrapping(module, 'pre')
        else:
            m = module
        m.qconfig = getattr(module, 'qconfig')

        try:
            qm = p_f(m)
            qm(*example_inputs)
            return True
        except (RuntimeError, ValueError, TypeError) as e:
            # Log the specific error for debugging
            print(f"[QDQWrapper] Quantization failed for module {type(module).__name__}: {e}")
            module.qconfig = None
            return False

    @classmethod
    def add_quant_entry_exit(cls, m: nn.Module, *example_input, allow: set = None, mode='static'):
        """Add quantization entry/exit points to model."""
        device = default_device()
        
        example_input = tuple(
            inp.to(device) if hasattr(inp, 'to') else inp for inp in example_input
        )

        with torch.no_grad():
            for name, module in m.named_modules():
                if allow and type(module) not in allow:
                    continue
                
                if cls.__is_conventional_module(module) and cls.is_leaf_quantizable(module, example_input, mode):
                    if cls.__qconfig_requires_qdq(module):
                        wrapped = QDQWrapping(module, 'both')
                        cls.set_module(m, name, wrapped)


# Export only quantization-related functionality
__all__ = [
    'uninplace',
    'get_flattened_qconfig_dict', 
    'ResidualAddWrapper',
    'QDQWrapper',
    'QDQWrapping'
]
