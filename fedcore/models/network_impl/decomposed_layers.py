"""Decomposed layers with canonical matrix factors and explicit assembly state.

Convolution groups are independent matrices. ``S`` is a trainable coefficient,
not a singular spectrum after training; use ``canonicalize`` before truncation.
"""
from typing import Optional
from copy import deepcopy
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from fedcore.algorithm.low_rank.decomposer import DECOMPOSERS, Decomposer

__all__ = ['IDecomposed', 'DecomposedLinear', 'DecomposedEmbedding',
           'DecomposedConv1d', 'DecomposedConv2d']


def _diag_tensor_check(t):
    return torch.diag_embed(t) if t.ndim == 1 else t


class IDecomposed:
    """Mixin storing U[..., rows, rank], S[..., rank], Vh[..., rank, cols]."""
    _weight_name = ['weight']
    _compose_mode_matrices = {'one_layer': ['W'], 'two_layers': ['U', 'Vh'],
                              'three_layers': ['U', 'S', 'Vh']}

    def __init__(self, decomposing_mode=True, method='svd', compose_mode=None,
                 decomposer_params=None):
        if compose_mode not in (None, 'one_layer', 'two_layers', 'three_layers'):
            raise ValueError(f'Unknown compose mode: {compose_mode}')
        if not isinstance(method,Decomposer) and method not in DECOMPOSERS:
            raise ValueError(f'Unknown decomposer: {method}')
        if isinstance(method,Decomposer) and decomposer_params:
            raise ValueError('Configure a decomposer instance before passing it; do not combine with decomposer_params')
        self.decomposing_mode = decomposing_mode
        self.method = method
        self.decomposer_params = dict(decomposer_params or {})
        self.compose_mode = compose_mode
        self.inference_mode = False
        self._representation = 'one_layer'
        self._initial_params_num = sum(p.numel() for p in self.parameters())
        self._weight_requires_grad = self.weight.requires_grad
        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
        if decomposing_mode is not None and decomposing_mode is not False:
            self.decompose()
        self._register_state_dict_hook(self._save_representation_metadata)

    def representation_metadata(self):
        """Version the actual stored form, independently of the requested compose mode."""
        solver = self.method if isinstance(self.method, str) else next(
            (name for name, cls in DECOMPOSERS.items() if type(self.method) is cls), None)
        if solver is None:
            raise ValueError('The decomposer is not supported by the checkpoint allowlist')
        parameters = dict(self.decomposer_params)
        if not isinstance(self.method, str):
            import inspect
            for name in inspect.signature(type(self.method).__init__).parameters:
                if name == 'self' or not hasattr(self.method, name):
                    continue
                value = getattr(self.method, name)
                if value is None or type(value) in (str, bool, int, float):
                    parameters[name] = value
        return {'version': 1, 'layer_type': type(self).__name__,
                'representation': self._representation, 'decomposing_mode': self.decomposing_mode,
                'compose_mode': self.compose_mode, 'inference_mode': self.inference_mode,
                'solver': solver, 'decomposer_params': parameters,
                'bias': getattr(self, 'bias', None) is not None, 'groups': getattr(self, 'groups', 1),
                'shapes': {name: list(parameter.shape) for name, parameter in self._parameters.items()
                           if name in ('weight', 'U', 'S', 'Vh', 'bias') and parameter is not None}}

    @staticmethod
    def _save_representation_metadata(module, state, prefix, metadata):
        metadata['fedcore_svd'] = module.representation_metadata()

    def decompose(self, W=None):
        W = self._weight_to_matrix(self.weight) if W is None else W
        decomposer = deepcopy(self.method) if isinstance(self.method,Decomposer) else DECOMPOSERS[self.method](**self.decomposer_params)
        matrices = W.unbind(0) if W.ndim == 3 else (W,)
        results = []
        for matrix in matrices:
            u, s, vh = decomposer.decompose(matrix)
            # CUR may return a dense middle matrix. Convert its actual operator.
            if s.ndim != 1:
                u, s, vh = torch.linalg.svd(u @ s @ vh, full_matrices=False)
            results.append((u, s, vh))
        factors = tuple(torch.stack([r[i] for r in results]) for i in range(3)) if W.ndim == 3 else results[0]
        self.set_U_S_Vh(*factors)
        self.register_parameter('weight', None)
        self._representation = 'three_layers'

    def set_U_S_Vh(self, u, s, vh, **kwargs):
        if getattr(self, '_fedcore_tied_factors', False):
            raise ValueError('Replace tied factors jointly at model level')
        s = s.reshape(u.shape[:-2] + (u.shape[-1],))
        if u.shape[-1] != vh.shape[-2] or s.shape[-1] != u.shape[-1]:
            raise ValueError('Incompatible U/S/Vh shapes')
        for name, value in (('U', u), ('S', s), ('Vh', vh)):
            self.register_parameter(name, Parameter(value.detach().clone(), requires_grad=self._weight_requires_grad))
        self.register_parameter('weight', None)
        self._representation = 'three_layers'
        self.inference_mode = False

    def get_U_S_Vh(self):
        return self.U, self.S, self.Vh

    def _get_threshold(self):
        return None

    def _weight_to_matrix(self, weight):
        return weight

    def _matrix_to_weight(self, matrix):
        return matrix

    def factor_matrix(self):
        if self._representation == 'one_layer':
            return self._weight_to_matrix(self.weight)
        left = self.U if self.S is None else self.U * self.S.unsqueeze(-2)
        return left @ self.Vh

    def _get_weights(self):
        return self.factor_matrix()

    def _get_composed_weight(self):
        return self._matrix_to_weight(self.factor_matrix())

    def canonicalize(self):
        """Re-SVD the actual operator; invariant to factor scale and sign."""
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(self.factor_matrix(), full_matrices=False)
        self.set_U_S_Vh(u, s, vh)
        return u, s, vh

    def _evaluate_compose_mode(self):
        bias_cost = self.bias.numel() if getattr(self, 'bias', None) is not None else 0
        factor_cost = self.U.numel() + self.Vh.numel() + bias_cost
        return 'two_layers' if factor_cost < self._initial_params_num else 'one_layer'

    def compose_weight_for_inference(self):
        mode = self.compose_mode or (self._evaluate_compose_mode() if self.U is not None else 'one_layer')
        if getattr(self, '_fedcore_tied_factors', False) and mode != self._representation:
            raise ValueError('Compose tied factors jointly at model level before changing their representation')
        if mode == 'one_layer':
            self._one_layer_compose()
        elif mode == 'two_layers':
            if self._representation == 'one_layer':
                self.decompose()
            self._two_layers_compose()
        elif self._representation != 'three_layers':
            self.canonicalize()
        self.compose_mode = mode
        self.inference_mode = True

    def _one_layer_compose(self):
        if getattr(self, '_fedcore_tied_factors', False):
            if self._representation != 'one_layer':
                raise ValueError('Compose tied factors jointly at model level')
            return
        weight = self._get_composed_weight().detach().clone()
        self.register_parameter('weight', Parameter(weight, requires_grad=self._weight_requires_grad))
        for name in ('U', 'S', 'Vh'):
            self.register_parameter(name, None)
        self._representation = 'one_layer'

    def _two_layers_compose(self):
        if getattr(self, '_fedcore_tied_factors', False) and self._representation != 'two_layers':
            raise ValueError('Compose tied factors jointly at model level')
        if self.S is not None:
            self.register_parameter('Vh', Parameter((self.S.unsqueeze(-1) * self.Vh).detach().clone(), requires_grad=self._weight_requires_grad))
            self.register_parameter('S', None)
        self._representation = 'two_layers'

    def _three_layers_compose(self):
        if self._representation != 'three_layers':
            self.canonicalize()

    def _anti_three_layers_compose(self):
        if self._representation != 'three_layers':
            self.canonicalize()

    def _eliminate_extra_params(self, names):
        for name in names:
            self.register_parameter(name, None)

    def compose(self):
        self._one_layer_compose()
        self.compose_mode = 'one_layer'

    def _forward1(self, x):
        return self.forward(x)

    _forward2 = _forward1
    _forward3 = _forward1


def _factory(base, device, dtype):
    return {'device': device or base.weight.device, 'dtype': dtype or base.weight.dtype}


def _copy_parameters(target, base):
    with torch.no_grad():
        target.weight.copy_(base.weight)
        target.weight.requires_grad_(base.weight.requires_grad)
        if getattr(base, 'bias', None) is not None:
            target.bias.copy_(base.bias)
            target.bias.requires_grad_(base.bias.requires_grad)
    target.train(base.training)


class DecomposedLinear(nn.Linear, IDecomposed):
    def __init__(self, base_module, decomposing_mode=True, decomposer='svd', compose_mode=None,
                 decomposer_params=None, device=None, dtype=None):
        nn.Linear.__init__(self, base_module.in_features, base_module.out_features,
                           base_module.bias is not None, **_factory(base_module, device, dtype))
        _copy_parameters(self, base_module)
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)

    def forward(self, x):
        if self.U is None:
            return F.linear(x, self.weight, self.bias)
        left = self.U if self.S is None else self.U * self.S.unsqueeze(-2)
        return F.linear(F.linear(x, self.Vh), left, self.bias)


class DecomposedEmbedding(nn.Embedding, IDecomposed):
    def __init__(self, base_module, decomposing_mode=True, decomposer='svd', compose_mode=None,
                 decomposer_params=None, device=None, dtype=None):
        # Sparse factor gradients are not supported by dense matrix products.
        if base_module.sparse and decomposing_mode not in (False, None) and compose_mode != 'one_layer':
            raise ValueError('Sparse Embedding requires compose_mode=one_layer')
        nn.Embedding.__init__(self, base_module.num_embeddings, base_module.embedding_dim,
                              padding_idx=base_module.padding_idx, max_norm=base_module.max_norm,
                              norm_type=base_module.norm_type, scale_grad_by_freq=base_module.scale_grad_by_freq,
                              sparse=base_module.sparse, **_factory(base_module, device, dtype))
        _copy_parameters(self, base_module)
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)
        if self.sparse and self.U is not None:
            self.compose()

    def forward(self, x):
        if self.U is None:
            return F.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.sparse)
        # max_norm acts on complete rows, never on the low-rank coordinates.
        if self.max_norm is not None:
            with torch.no_grad():
                indices = x.unique()
                norms = self._get_composed_weight()[indices].norm(p=self.norm_type, dim=-1)
                scale = (self.max_norm / (norms + 1e-7)).clamp(max=1)
                self.U[indices] *= scale.unsqueeze(-1)
        weight = self._get_composed_weight()
        return F.embedding(x, weight, self.padding_idx, None, self.norm_type,
                           self.scale_grad_by_freq, False)


class _DecomposedConv:
    def _weight_to_matrix(self, weight):
        g, out, inp = self.groups, self.out_channels // self.groups, self.in_channels // self.groups
        grouped = weight.reshape(g, out, inp, *self.kernel_size)
        if self._spatial:
            grouped = grouped.permute(0, 1, 3, 2, 4).reshape(g, out*self.kernel_size[0], inp*self.kernel_size[1])
        else:
            grouped = grouped.reshape(g, out, -1)
        return grouped[0] if g == 1 else grouped

    def _matrix_to_weight(self, matrix):
        g, out, inp = self.groups, self.out_channels // self.groups, self.in_channels // self.groups
        if self._spatial:
            matrix = matrix.reshape(g, out, self.kernel_size[0], inp, self.kernel_size[1]).permute(0,1,3,2,4)
        return matrix.reshape(self.out_channels, inp, *self.kernel_size)

    def factor_weights(self):
        """Return grouped convolution kernels and their spatial arguments."""
        g, out, inp = self.groups, self.out_channels // self.groups, self.in_channels // self.groups
        r = self.U.shape[-1]
        left = self.U if self.S is None else self.U*self.S.unsqueeze(-2)
        if self._spatial:
            first = self.Vh.reshape(g*r, inp, 1, self.kernel_size[1])
            last = left.reshape(g, out, self.kernel_size[0], r).permute(0,1,3,2).reshape(self.out_channels,r,self.kernel_size[0],1)
            args1 = {'stride': (1,self.stride[1]), 'padding': (0,self.padding[1]), 'dilation': (1,self.dilation[1])}
            args2 = {'stride': (self.stride[0],1), 'padding': (self.padding[0],0), 'dilation': (self.dilation[0],1)}
        else:
            first = self.Vh.reshape(g*r, inp, *self.kernel_size)
            last = left.reshape(self.out_channels, r, *([1]*len(self.kernel_size)))
            args1 = {'stride':self.stride, 'padding':self.padding, 'dilation':self.dilation}
            args2 = {'stride':1, 'padding':0, 'dilation':1}
        return first, last, args1, args2

    def forward(self, x):
        if self.U is None or isinstance(self.padding, str):
            return self._conv_forward(x, self._get_composed_weight(), self.bias)
        first, last, args1, args2 = self.factor_weights()
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            args1['padding'] = 0
            args2['padding'] = 0
        operation = F.conv2d if len(self.kernel_size) == 2 else F.conv1d
        return operation(operation(x, first, groups=self.groups, **args1), last,
                         self.bias, groups=self.groups, **args2)


class DecomposedConv2d(_DecomposedConv, nn.Conv2d, IDecomposed):
    def __init__(self, base_module, decomposing_mode='channel', decomposer='svd', compose_mode=None,
                 decomposer_params=None, device=None, dtype=None):
        if decomposing_mode is True:
            decomposing_mode = 'channel'
        if decomposing_mode not in ('channel','spatial',False,None):
            raise ValueError('Conv2d decomposition must be channel or spatial')
        self._spatial = decomposing_mode == 'spatial'
        nn.Conv2d.__init__(self, base_module.in_channels, base_module.out_channels, base_module.kernel_size,
                          base_module.stride, base_module.padding, base_module.dilation, base_module.groups,
                          base_module.bias is not None, base_module.padding_mode, **_factory(base_module,device,dtype))
        _copy_parameters(self, base_module)
        self.decomposing = {'type':decomposing_mode}
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)


class DecomposedConv1d(_DecomposedConv, nn.Conv1d, IDecomposed):
    def __init__(self, base_module, decomposing_mode=True, decomposer='svd', compose_mode=None,
                 decomposer_params=None, device=None, dtype=None):
        if decomposing_mode not in (True,'channel',False,None):
            raise ValueError('Conv1d supports channel decomposition')
        self._spatial = False
        nn.Conv1d.__init__(self, base_module.in_channels, base_module.out_channels, base_module.kernel_size,
                          base_module.stride, base_module.padding, base_module.dilation, base_module.groups,
                          base_module.bias is not None, base_module.padding_mode, **_factory(base_module,device,dtype))
        _copy_parameters(self,base_module)
        IDecomposed.__init__(self,decomposing_mode,decomposer,compose_mode,decomposer_params)


DecomposableLayers = {nn.Linear:DecomposedLinear, nn.Embedding:DecomposedEmbedding,
                     nn.Conv1d:DecomposedConv1d, nn.Conv2d:DecomposedConv2d}
