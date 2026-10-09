"""Atomic module decomposition: all replacements are prepared before mutation."""
import torch
from fedcore.models.network_impl.decomposed_layers import IDecomposed, DecomposableLayers


def _map_decomposed_cls(inst):
    return None if isinstance(inst,IDecomposed) else DecomposableLayers.get(type(inst))


def decompose_module(model,decomposing_mode=True,decomposer='svd',compose_mode=None,decomposer_params=None):
    replacements, aliases = [], {}
    def prepare(parent):
        for name,module in parent._modules.items():
            if module is None:
                continue
            cls = _map_decomposed_cls(module)
            if cls is not None:
                if id(module) not in aliases:
                    aliases[id(module)] = cls(module,decomposing_mode=decomposing_mode,decomposer=decomposer,
                                              compose_mode=compose_mode,decomposer_params=decomposer_params)
                replacements.append((parent,name,aliases[id(module)]))
            else:
                prepare(module)
    root_cls = _map_decomposed_cls(model)
    if root_cls is not None:
        return root_cls(model,decomposing_mode=decomposing_mode,decomposer=decomposer,
                        compose_mode=compose_mode,decomposer_params=decomposer_params)
    prepare(model)
    for parent,name,replacement in replacements:
        setattr(parent,name,replacement)
    return model


def _load_svd_params(model,state_dict,prefix=''):
    for name,module in model.named_modules():
        if isinstance(module,IDecomposed):
            key = prefix + (name+'.' if name else '')
            module.set_U_S_Vh(state_dict[key+'U'],state_dict[key+'S'],state_dict[key+'Vh'])


def load_svd_state_dict(model,decomposing_mode,state_dict_path,compose_mode=None,decomposer_params=None):
    state_dict = torch.load(state_dict_path,map_location='cpu',weights_only=True)
    model = decompose_module(model,decomposing_mode,compose_mode=compose_mode,decomposer_params=decomposer_params)
    _load_svd_params(model,state_dict)
    model.load_state_dict(state_dict)
    return model


__all__ = ['decompose_module','load_svd_state_dict']

# Historical public entry point.
decompose_module_in_place = decompose_module
