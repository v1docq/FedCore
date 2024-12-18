from typing import Dict, Optional

import torch
from torch.ao.quantization.quantize import has_no_children_ignoring_parametrizations
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.stubs import QuantStub, DeQuantStub
from torch.ao.quantization.utils import _get_path_of_module, get_qconfig_dtypes
import torch.nn as nn

from fedcore.models.network_impl.layers import IDecomposed, DecomposedLinear, DecomposedEmbedding
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.abstraction.delegator import IDelegator


__all__ = [
    'ParentalReassembler',
    'QDQWrapper',
    'QDQWrapping',
    'uninplace',
    'reset_qconfig'
]

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
            getattr(module.weight, 'device', torch.device('cpu')),
            getattr(module.weight, 'dtype', torch.float32)
        )
        new.load_state_dict(module.state_dict())
        return new

def _recreate_linear(module):
    assert isinstance(module, torch.nn.Linear)
    raise NotImplementedError


class RecreatedDecomposed(nn.Sequential):
    pass
    

def _recreate_decomposed_linear(
        L: DecomposedLinear
        ):
    U, S, Vh = L.U.detach(), L.S.detach(), L.Vh.detach()
    U = U * S
    h = S.size(0)
    new = RecreatedDecomposed(
        nn.Linear(L.in_features, h, bias=False),
        nn.Linear(h, L.out_features, bias=True)
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
    if getattr(L.bias, 'data', None) is not None:
        new[-1].bias.data = L.bias.data
    return new

def _recreate_decomposed_embedding(E: DecomposedEmbedding):
    U, S, Vh = E.U, E.S, E.Vh
    h = S.size(0)
    new = RecreatedDecomposed(
        nn.Embedding(E.num_embeddings, h),
        nn.Linear(h, E.embedding_dim, False)
    )
    new[0].weight.data = U
    new[-1].weight.data = (torch.diag(S) @ Vh).T
    new._is_recreated = True
    return new


class ParentalReassembler(Accessor):    
    supported_layers = {torch.nn.Embedding: _recreate_embedding,
                        torch.nn.Linear: _recreate_linear}
    
    supported_decomposed_layers = {
        DecomposedLinear: _recreate_decomposed_linear,
        DecomposedEmbedding: _recreate_decomposed_embedding,
    }
            
    @classmethod
    def _fetch_module(cls, module: nn.Module):
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
        if is_decomp:
            new_module = cls._decomposed_handle(module, associated)
        else:
            new_module = cls._handle(module, associated)
        return new_module
    
    @classmethod
    def reassemble(cls, model: nn.Module, additional_mapping: dict=None):
        """additional mapping for cases such as 'nn.ReLU6 -> nn.ReLU in format """
        if additional_mapping:
            for name, module in model.named_modules():
                t = type(module)
                if not t in additional_mapping:
                    continue
                cls.set_module(model, name, additional_mapping[t]())
        for name, module in model.named_modules():
            new_module = cls.convert(module)
            if new_module is None:
                continue
            cls.set_module(model, name, new_module)
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
    def get_layers_order( model: nn.Module, example_input):
        order = []
        hooks = []
        def add_hook(m):
            def forward_hook(module, input, output):
                order.append(module)
            registered_hook = m.register_forward_hook(forward_hook)
            hooks.append(registered_hook)
        model.apply(add_hook)
        model(example_input)
        [hook.remove() for hook in hooks]
        return order
    
    def __fetch_names(root: nn.Module, order: list):
        names_order = []
        for submodule in order:
            names_order.append(_get_path_of_module(root, submodule))
        return names_order
    
    @classmethod
    def __qconfig_requires_qdq(cls, module):
        from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit
        qconfig = getattr(module, 'qconfig', None)
        need_entry = False
        try:
            act_type = get_qconfig_dtypes(qconfig)[0]
        except:
            act_type = None
        return (need_entry or act_type is torch.float32)
            
    @classmethod
    def add_quant_entry_exit(cls, m: nn.Module, example_input):
        m.eval()
        modules_order = cls.get_layers_order(m, example_input)
        names_order = cls.__fetch_names(m, modules_order)
        def parametrizable_order(modules_order):
            is_parametrizable = [
                (has_no_children_ignoring_parametrizations(module) 
                    and bool(getattr(module, 'qconfig', None))
                    and cls.__qconfig_requires_qdq(module)
                )
                for module in modules_order
            ]
            return is_parametrizable
        is_parametrizable = parametrizable_order(modules_order)
        def is_change(i):
            return (
                getattr(modules_order[i - 1], 'qconfig', None) !=
                getattr(modules_order[i], 'qconfig', None)
            )

        if len(is_parametrizable) > 1:
            for i in range(1, len(is_parametrizable)):
                if (not is_parametrizable[i - 1] and is_parametrizable[i] 
                    # or is_change(i)
                    ):
                    cls.set_module(m, names_order[i], QDQWrapping(modules_order[i], 'pre'))
                if (is_parametrizable[i - 1] and not is_parametrizable[i]
                    # or is_change(i)
                    ):
                    cls.set_module(m, names_order[i - 1], QDQWrapping(modules_order[i - 1], 'post'))
        if is_parametrizable[0]:
            cls.set_module(m, names_order[0], QDQWrapping(modules_order[0], 'pre'))
        if is_parametrizable[-1]:
            cls.set_module(m, names_order[-1], QDQWrapping(modules_order[-1], 'post'))
        return m

class QDQWrapping(nn.Module, IDelegator):
    __non_redirect =  {'base', 'quant', 'dequant', '_order', 'qconfig', 'forward', '__call__'}
        
    def __init__(self, base, mode='pre'):
        super().__init__()
        self.base = base
        self.quant = QuantStub(getattr(base, 'qconfig', None)) if mode == 'pre' else None
        self.dequant = DeQuantStub(getattr(base, 'qconfig', None)) if mode == 'post' else None
        self._order = ['quant', 'base'] if mode == 'pre' else ['base', 'dequant']
        self.qconfig = getattr(self.base, 'qconfig', None)
    
    def forward(self, x, *args, **kwargs):
        for layer in self._order:
            x = getattr(self, layer)(x, *args, **kwargs)
        return x
        
    def __getattr__(self, name):
        if name not in self.__non_redirect:
            return getattr(super().__getattr__('base'), name)
        else:
            return super().__getattr__(name)
