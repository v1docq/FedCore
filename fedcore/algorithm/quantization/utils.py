from copy import deepcopy
from functools import partial
import inspect
from typing import Callable, Dict, Literal, Optional, Union

import torch
from torch.ao.quantization.quantize import (
    has_no_children_ignoring_parametrizations,
    quantize, 
    quantize_dynamic, 
    quantize_qat
)
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.stubs import QuantStub, DeQuantStub
from torch.ao.quantization.utils import get_qconfig_dtypes
from torch.nn.quantized import FloatFunctional
import torch.nn as nn
import torchvision.models.resnet as resnet

from fedcore.models.network_impl.decomposed_layers import (
    IDecomposed, 
    DecomposedLinear,
    DecomposedEmbedding, 
    DecomposedConv1d,
    DecomposedConv2d
)
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.abstraction.delegator import IDelegator
from fedcore.architecture.comptutaional.devices import extract_device, default_device


__all__ = [
    'ParentalReassembler',
    'QDQWrapper',
    'QDQWrapping',
    'uninplace',
    'reset_qconfig',
    'RecreatedDecomposed',
    'get_flattened_qconfig_dict'
]

QConfigMapping = QConfigAny

def get_flattened_qconfig_dict(qconfig_mapping: QConfigMapping) -> Dict[Union[Callable, str], QConfigAny]:
    """ flatten the global, object_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it's not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    flattened: Dict[Union[Callable, str], QConfigAny] = {"": qconfig_mapping.global_qconfig}
    for obj, qconfig in qconfig_mapping.object_type_qconfigs.items():
        flattened[obj] = qconfig
    for obj, qconfig in qconfig_mapping.module_name_qconfigs.items():
        flattened[obj] = qconfig
    return flattened


def _recreate_embedding(module):
        assert isinstance(module, torch.nn.Embedding)
        new = torch.nn.Embedding(
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.sparse,
            module.weight,
            getattr(module.weight, 'device', default_device()),
            getattr(module.weight, 'dtype', torch.float32)
        )
        new.load_state_dict(module.state_dict())
        return new

def _recreate_linear(module):
    assert isinstance(module, torch.nn.Linear)
    raise NotImplementedError


class RecreatedDecomposed(nn.Sequential):
    __non_redirected = {'forward', '__init__', 'routing', '0', '1'}
    def __init__(self, *args, routing=None):
        super().__init__(*args)
        self.routing = routing or {}

    def __getattr__(self, name):
        if not name in self.__non_redirected:
            name, module = self.routing.get(name, (name, '0'))
            return getattr(super().__getattr__(module), name)
        else:
            return super().__getattr__(name)
    

def _recreate_decomposed_linear(
        L: DecomposedLinear
        ):
    U, Vh = L.U.detach(), L.Vh.detach()
    h = U.size(-1)
    new = RecreatedDecomposed(
        nn.Linear(L.in_features, h, bias=False),
        nn.Linear(h, L.out_features, bias=True),
        routing={'out_features': ('out_features', '1'), 
                 'weight': ('weight', '1'),
                 'bias': ('bias', '1')}
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
    if getattr(L.bias, 'data', None) is not None:
        new[-1].bias.data = L.bias.data
    return new

def _recreate_decomposed_embedding(E: DecomposedEmbedding):
    U, Vh = E.U.detach(), E.Vh.detach()
    h = U.size(-1)
    new = RecreatedDecomposed(
        nn.Embedding(E.num_embeddings, h),
        nn.Linear(h, E.embedding_dim, False),
        routing={'embedding_dim': ('out_features', '1'),
                 'weight': ('weight', '1'),
                 'bias': ('bias', '1'),}
    )
    new[0].weight.data = U
    new[-1].weight.data = Vh.T
    new._is_recreated = True
    return new

def _recreate_decomposed_conv2d(C: DecomposedConv2d):
    U, Vh = C.U.detach(), C.Vh.detach()
    assert U.ndim == 4, 'Non composed layers are not supported'
    out_1, in_1, k_11, k_12 = Vh.size()
    out_2, in_2, k_21, k_22 = U.size()
    new = RecreatedDecomposed(
        nn.Conv2d(in_1, out_1, (k_11, k_12), groups=C.groups, **C.decomposing['Vh'], bias=False),
        nn.Conv2d(in_2, out_2, (k_21, k_22), **C.decomposing['U'], bias=True),
        routing={
            'in_channels': ('in_channels', '0'),
            'out_channels': ('out_channels', '1'),
            'groups': ('groups', '0'),
            'weight': ('weight', '1'),
            'bias': ('bias', '1'),
        }
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
    new[-1].bias = C.bias
    new._is_recreated = True
    return new

def _recreate_decomposed_conv1d(C: DecomposedConv1d):
    U, Vh = C.U.detach(), C.Vh.detach()
    assert U.ndim == 3, 'Non composed layers are not supported'
    out, r, k_2 = U.size()
    r, in_, k_1 = Vh.size()
    C1 = nn.Conv1d(in_, r, k_1, C.stride, C.padding, C.dilation, C.groups, bias=False)
    C2 = nn.Conv1d(r, out, k_2, bias=True)
    C1.weight.data = Vh
    C2.weight.data = U 
    C2.bias = C.bias
    new = RecreatedDecomposed(
        C1, 
        C2,
        routing={
            'out_channels': ('out_channels', '1'),
            'weight': ('weight', '1'),
            'bias': ('bias', '1'),
        }
    )
    new._is_recreated = True
    return new


class ResidualAddWrapper(nn.Module):
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

class ParentalReassembler(Accessor):    
    supported_layers = {
        torch.nn.Embedding: _recreate_embedding,
        # torch.nn.Linear: _recreate_linear
    }

    supported_decomposed_layers = {
        DecomposedLinear: _recreate_decomposed_linear,
        DecomposedEmbedding: _recreate_decomposed_embedding,
        DecomposedConv2d: _recreate_decomposed_conv2d,
        DecomposedConv1d: _recreate_decomposed_conv1d,
    }
            
    @classmethod
    def _fetch_module(cls, module: nn.Module):
        device = default_device()
        is_decomposed = isinstance(module, IDecomposed)
        supported = cls.supported_decomposed_layers if is_decomposed else cls.supported_layers
        for supported in supported:
            if isinstance(module, supported) and (is_decomposed or not type(module) is supported):
                return supported, is_decomposed
        return None, is_decomposed
     
    @classmethod
    def _handle(cls, module, type):
        supported = cls.supported_decomposed_layers if issubclass(type, IDecomposed) else cls.supported_layers
        return supported[type](module)

    @classmethod
    def convert(cls, module):
        associated, is_decomp = cls._fetch_module(module)
        if associated is None:
            return None
        new_module = cls._handle(module, associated)
        return new_module
    
    @classmethod
    def reassemble(cls, model: nn.Module, additional_mapping: dict = None):
        """additional mapping for cases such as 'nn.ReLU6 -> nn.ReLU' in format"""
        device = extract_device(model)
        try:
            device_type = device.type
        except:
            device_type = device
        if additional_mapping:
            for name, module in model.named_modules():
                t = type(module)
                if not t in additional_mapping:
                    continue
                cls.set_module(model, name, additional_mapping[t]())
        for name, module in model.named_modules():
            new_module = cls.convert(module)
            if new_module:
                cls.set_module(model, name, new_module.to(device))
            elif isinstance(module, (resnet.BasicBlock, resnet.Bottleneck)) and device_type != "cuda":
                wrapped_module = ResidualAddWrapper(module)
                cls.set_module(model, name, wrapped_module.to(device))
                print(f"[ParentalReassembler] Residual block '{name}' wrapped with ResidualAddWrapper.")

        assert all(device == p.device for p in model.parameters()), "[ParentalReassembler] Device mismatch!"
        return model


def uninplace(model):
    """Sets all `inplace` values to False"""
    if hasattr(model, 'inplace'):
        model.inplace = False
    for child in model.children():
        uninplace(child)


def are_qconfigs_equal(qconfig1, qconfig2) -> bool:
    return get_qconfig_dtypes(qconfig1) == get_qconfig_dtypes(qconfig2)


def reset_qconfig(model: nn.Module, mapping=Dict[nn.Embedding, Optional[QConfigAny]]):
    for m in model.modules():
        t = type(m)
        if t in mapping:
            m.qconfig = mapping[t]
    return model


class QDQWrapper(Accessor):
    __conventional_modules = {cls[1] for cls in inspect.getmembers(torch.nn.modules, inspect.isclass)}

    @classmethod
    def __is_conventional_module(cls, module: nn.Module):
        return type(module) in cls.__conventional_modules
    
    @classmethod
    def __qconfig_requires_qdq(cls, module):
        qconfig = getattr(module, 'qconfig', None)
        try:
            act_type = get_qconfig_dtypes(qconfig)[0]
        except:
            act_type = None
        return act_type is torch.float32
    
    @staticmethod
    def is_leaf_quantizable(module: nn.Module, 
                            example_inputs: tuple, 
                            mode: Literal['qat', 'static', 'dynamic']):
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
        except Exception:
            module.qconfig = None
            return False 

    @classmethod
    def _replace_dicts(cls, d):
        for name, branch in d._layers.items():
            last_q = None
            for module in branch.modules():
                if isinstance(module, QDQWrapping):
                    if module.mode == 'pre':
                        last_q = module
                    else:
                        last_q = None
            if last_q is None: continue
            del d[name]
            d[name] = QDQWrapping(branch, 'last', last_q.qconfig)   
            
    @classmethod
    def add_quant_entry_exit(cls, m: nn.Module, *example_input, allow: set=None, mode='static'):
        allow = allow or set()
        m.eval()
        device = next(m.parameters()).device
        example_input = tuple(
            inp.to(device) if hasattr(inp, 'to') else inp for inp in example_input
        )

        with torch.no_grad():
            modules_order = cls.get_layers_order(m, *example_input)
            names_order = cls.get_names_order(m, *example_input)
            name_input = cls.get_name_input_mapping(m, *example_input)

            def _is_parametrizable(name: str):
                module = cls.get_module(m, name)
                is_leaf = cls.is_leaf_module(module)
                return (
                    (type(module) in allow
                    or
                    (has_no_children_ignoring_parametrizations(module) and not is_leaf
                    or is_leaf and cls.is_leaf_quantizable(module, name_input[name], mode)
                    ))
                    and bool(getattr(module, 'qconfig', None))
                )

            is_parametrizable = [
                _is_parametrizable(name) for name in names_order
            ]

            if len(is_parametrizable) > 1:
                for i in range(1, len(is_parametrizable)):
                    if (not is_parametrizable[i - 1] and is_parametrizable[i] 
                        # or is_change(i) #TODO
                        ):
                        module = cls.get_module(m, names_order[i])
                        new_module = QDQWrapping(module, 'pre')
                        cls.set_module(m, names_order[i], new_module)
                    elif (is_parametrizable[i - 1] and not is_parametrizable[i]
                        # or is_change(i)
                        ):
                        module = cls.get_module(m, names_order[i])
                        new_module = QDQWrapping(module, 'post', qconfig=modules_order[i - 1].qconfig)
                        cls.set_module(m, names_order[i], new_module)
                    else:
                        pass
            [cls._replace_dicts(module) for module in modules_order if isinstance(module, torch.nn.ModuleDict)]
            if is_parametrizable[0]:
                cls.set_module(m, names_order[0], QDQWrapping(cls.get_module(m, names_order[0]), 'pre'))
            if is_parametrizable[-1]:
                cls.set_module(m, names_order[-1], QDQWrapping(cls.get_module(m, names_order[-1]), 'last'))
        return m


class QDQWrapping(nn.Module, IDelegator):
    __non_redirect = {'base', 'quant', 'dequant', '_order', 'qconfig', 'forward', '__call__'}

    def __init__(self, base, mode='pre', qconfig=None):
        super().__init__()
        self.mode = mode
        self.base = base
        self._order = {'pre': ['quant', 'base'],
                       'post': ['dequant', 'base'],
                       'both': ['quant', 'base', 'dequant'],
                       'last': ['base', 'dequant']}[mode]
        self.quant = QuantStub(qconfig or getattr(self.base, 'qconfig', None)) if 'quant' in self._order else None
        self.dequant = DeQuantStub(qconfig or getattr(self.base, 'qconfig', None)) if 'dequant' in self._order else None
        if mode == 'post':
            assert qconfig, 'For post mode you need to specify the previous layer\'s qconfig'
        self.qconfig = qconfig or getattr(self.base, 'qconfig', None)
        self._is_rnn = isinstance(self.base, nn.RNNBase)
        self.__h = None

    def __repr__(self):
        d = {
            'pre': f'{self.quant}\n{self.base}',
            'post': f'{self.dequant}\n{self.base}',
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
