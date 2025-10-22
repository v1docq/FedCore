"""
Utilities for model reassembly and quantization entry/exit wrapping.

This module provides two main capabilities used by FedCore's quantization flow:

1) **Parental reassembly** — converting certain layers into quantization-friendly
   equivalents and reconstructing decomposed layers (e.g., low-rank factorizations)
   into small sequential blocks that standard quantization tooling can understand.
   See :class:`ParentalReassembler` and helpers like
   :func:`_recreate_decomposed_linear`, :func:`_recreate_decomposed_conv2d`, etc.

2) **Q/DQ wrapping** — inserting :class:`torch.ao.quantization.stubs.QuantStub`
   and :class:`torch.ao.quantization.stubs.DeQuantStub` around leaf modules or
   model boundaries to ensure well-formed quantization regions and to improve
   success of PTQ/QAT preparation. See :class:`QDQWrapper` and :class:`QDQWrapping`.

There are also small helpers for qconfig handling and for making model modules
"uninplace" (i.e., disabling in-place ops prior to quantization passes).

Notes
-----
* The reassembly is intentionally conservative and focuses on commonly used
  layers (Embedding, decomposed Linear/Conv/Embedding). Unsupported cases
  pass through unchanged.
* The Q/DQ wrapping is guided by the model's execution order on example inputs
  and by the per-module ``qconfig`` presence.
"""

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
    """Flatten ``QConfigMapping`` into a dict accepted by ``propagate_qconfig_``.

    The function merges global/object-type/module-name qconfigs into a single
    dictionary where keys are either callables (ops / module classes) or string
    module-name patterns supported by PyTorch.

    Parameters
    ----------
    qconfig_mapping : QConfigMapping
        Mapping with fields ``global_qconfig``, ``object_type_qconfigs``,
        and ``module_name_qconfigs``.

    Returns
    -------
    Dict[Union[Callable, str], QConfigAny]
        A flattened mapping:
        ``{"": global_qconfig, <callable>: qconfig, "module_name": qconfig}``.

    Notes
    -----
    The ``module_name_regex`` branch is intentionally ignored here to keep
    parity with ``propagate_qconfig_`` support.
    """
    flattened: Dict[Union[Callable, str], QConfigAny] = {"": qconfig_mapping.global_qconfig}
    for obj, qconfig in qconfig_mapping.object_type_qconfigs.items():
        flattened[obj] = qconfig
    for obj, qconfig in qconfig_mapping.module_name_qconfigs.items():
        flattened[obj] = qconfig
    return flattened


def _recreate_embedding(module: nn.Embedding) -> nn.Embedding:
    """Recreate an ``nn.Embedding`` object with the same state.

    Used to "normalize" modules produced by upstream transformations so that
    PyTorch quantization can operate on standard layers.
    """
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


def _recreate_linear(module: nn.Linear):
    """Recreate a standard Linear layer (placeholder for future needs)."""
    assert isinstance(module, torch.nn.Linear)
    raise NotImplementedError


class RecreatedDecomposed(nn.Sequential):
    """A thin sequential wrapper around reassembled decomposed layers.

    The class supports **attribute routing** so that external code can access
    attributes (e.g., ``weight``, ``bias``, ``out_features``) on the composite
    block as if it were a single layer. The routing map is provided via the
    ``routing`` argument and redirects requested attributes to the underlying
    submodule that owns them.

    Parameters
    ----------
    *args : nn.Module
        Submodules constituting the reconstructed block (e.g., two Linear layers).
    routing : dict, optional
        Mapping ``{attr_name: (redirected_attr_name, submodule_name)}``.

    Notes
    -----
    Only attributes not in ``__non_redirected`` will be routed. Everything else
    falls back to the default ``nn.Sequential`` behavior.
    """
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
    

def _recreate_decomposed_linear(L: DecomposedLinear) -> RecreatedDecomposed:
    """Reassemble a :class:`DecomposedLinear` into two standard Linear layers."""
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


def _recreate_decomposed_embedding(E: DecomposedEmbedding) -> RecreatedDecomposed:
    """Reassemble a :class:`DecomposedEmbedding` into (Embedding -> Linear)."""
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


def _recreate_decomposed_conv2d(C: DecomposedConv2d) -> RecreatedDecomposed:
    """Reassemble a :class:`DecomposedConv2d` into two Conv2d layers."""
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


def _recreate_decomposed_conv1d(C: DecomposedConv1d) -> RecreatedDecomposed:
    """Reassemble a :class:`DecomposedConv1d` into two Conv1d layers."""
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
    """Wrap a ResNet BasicBlock/Bottleneck to use quantized add+relu.

    Replaces the residual addition with :class:`FloatFunctional.add_relu`
    so that static/dynamic quantization flows have a compatible fused op.

    Notes
    -----
    The wrapper preserves the original block structure and parameters and
    copies its ``qconfig`` if present.
    """
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
    """
    Reassemble and normalize modules prior to quantization.

    Responsibilities
    ----------------
    - Convert supported standard layers (e.g., :class:`nn.Embedding`) and
      decomposed layers (e.g., :class:`DecomposedLinear`) into equivalent
      "recreated" modules that are easier to quantize.
    - For CPU targets, wrap ResNet residual blocks with :class:`ResidualAddWrapper`
      to expose :class:`FloatFunctional.add_relu`.

    Class Attributes
    ----------------
    supported_layers : Dict[type, Callable]
        Mapping from supported standard layer types to recreate functions.
    supported_decomposed_layers : Dict[type, Callable]
        Mapping from decomposed layer types to recreate functions.
    """
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
        """Check whether the module is supported and whether it is decomposed.

        Returns
        -------
        Tuple[type|None, bool]
            Matched type key from the corresponding mapping (or ``None``),
            and a boolean flag indicating a decomposed layer.
        """
        device = default_device()
        is_decomposed = isinstance(module, IDecomposed)
        supported = cls.supported_decomposed_layers if is_decomposed else cls.supported_layers
        for supported in supported:
            if isinstance(module, supported) and (is_decomposed or not type(module) is supported):
                return supported, is_decomposed
        return None, is_decomposed
     
    @classmethod
    def _handle(cls, module, type):
        """Dispatch to a recreate function based on the matched type."""
        supported = cls.supported_decomposed_layers if issubclass(type, IDecomposed) else cls.supported_layers
        return supported[type](module)

    @classmethod
    def convert(cls, module):
        """Convert a single module if supported; otherwise return ``None``."""
        associated, is_decomp = cls._fetch_module(module)
        if associated is None:
            return None
        new_module = cls._handle(module, associated)
        return new_module
    
    @classmethod
    def reassemble(cls, model: nn.Module, additional_mapping: dict = None):
        """Rewrite the model in-place by reassembling supported modules.

        Parameters
        ----------
        model : nn.Module
            The model to process.
        additional_mapping : dict, optional
            Extra replacements of the form ``{OldClass: NewClass}``, e.g.,
            ``{nn.ReLU6: nn.ReLU}``.

        Returns
        -------
        nn.Module
            The same model instance with certain modules replaced/wrapped.

        Raises
        ------
        AssertionError
            If parameter devices become inconsistent during rewriting.
        """
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


def uninplace(model: nn.Module):
    """Set ``inplace=False`` recursively for modules that expose this attribute."""
    """Sets all `inplace` values to False"""
    if hasattr(model, 'inplace'):
        model.inplace = False
    for child in model.children():
        uninplace(child)


def are_qconfigs_equal(qconfig1, qconfig2) -> bool:
    """Return True if two qconfigs have the same dtype triple (act/wt/obs)."""
    return get_qconfig_dtypes(qconfig1) == get_qconfig_dtypes(qconfig2)


def reset_qconfig(model: nn.Module, mapping=Dict[nn.Embedding, Optional[QConfigAny]]):
    """Reset per-module ``qconfig`` according to a provided mapping.

    Parameters
    ----------
    model : nn.Module
        Model whose modules will be visited.
    mapping : Dict[type, Optional[QConfigAny]]
        Map from module type to the qconfig to assign (or ``None`` to clear).

    Returns
    -------
    nn.Module
        The same model instance for chaining.
    """
    for m in model.modules():
        t = type(m)
        if t in mapping:
            m.qconfig = mapping[t]
    return model


class QDQWrapper(Accessor):
    """
    Insert Quant/DeQuant stubs around quantizable regions of a model.

    The wrapper inspects execution order (using example inputs) and decides
    where to place 'entry' (QuantStub) and 'exit' (DeQuantStub) points to
    delimit quantizable subgraphs. This improves the robustness of PTQ/QAT
    preparation and conversion, especially when some modules are not eligible
    for quantization.

    Key methods
    -----------
    - :meth:`add_quant_entry_exit`: main entry point to augment a model.
    - :meth:`is_leaf_quantizable`: quick feasibility check for leaf modules.

    Notes
    -----
    This class inherits from :class:`Accessor` to get convenience methods like
    ``get_layers_order``, ``get_names_order``, ``get_name_input_mapping``,
    ``get_module``, and ``set_module``.
    """
    __conventional_modules = {cls[1] for cls in inspect.getmembers(torch.nn.modules, inspect.isclass)}

    @classmethod
    def __is_conventional_module(cls, module: nn.Module):
        """Return True if the module is a standard torch.nn class (helper)."""
        return type(module) in cls.__conventional_modules
    
    @classmethod
    def __qconfig_requires_qdq(cls, module):
        """Heuristic: check if the qconfig implies floating-point activations."""
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
        """Check if a leaf module can be quantized under a given mode.

        Runs a small sandboxed quantization attempt (PTQ/QAT) on a deepcopy of
        the module and feeds ``example_inputs`` to ensure shapes/dtypes match.

        Parameters
        ----------
        module : nn.Module
            Leaf module to probe.
        example_inputs : tuple
            Example inputs captured for this node during model tracing.
        mode : {'qat', 'static', 'dynamic'}
            Quantization workflow.

        Returns
        -------
        bool
            True if the module can be quantized and executed with the example inputs.
        """
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
    def _replace_dicts(cls, d: nn.ModuleDict):
        """Replace leading/trailing QDQ wrappers inside ModuleDict branches.

        This compacts multiple adjacent wrappers to a single 'last' wrapper
        when a branch ends with a quantizable leaf.
        """
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
        """Insert Quant/DeQuant stubs around quantizable regions of a model.

        The algorithm:
          1) Evaluate the model once to capture execution order and input maps.
          2) For each module in order, decide if it is 'parametrizable' for the
             chosen mode based on allow-list, qconfig presence, and leaf checks.
          3) Insert 'pre' (QuantStub) when transitioning from non-quantizable to
             quantizable; insert 'post' (DeQuantStub) on the opposite transition.
          4) Ensure first/last quantizable modules have boundary wrappers.
          5) Compact wrappers inside :class:`nn.ModuleDict` branches.

        Parameters
        ----------
        m : nn.Module
            Model to be augmented (modified in-place).
        *example_input :
            Example input(s) used to trace execution order and dtype paths.
        allow : set[type], optional
            Explicit set of module types allowed to be quantized.
        mode : {'static', 'dynamic', 'qat'}, default='static'
            Target quantization workflow.

        Returns
        -------
        nn.Module
            The same model instance for chaining.
        """
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
    """A delegating module that wraps a base op with Quant/DeQuant stubs.

    Modes
    -----
    - ``'pre'``  : QuantStub → base
    - ``'post'`` : DeQuantStub → base  (requires the previous qconfig)
    - ``'both'`` : QuantStub → base → DeQuantStub
    - ``'last'`` : base → final DeQuantStub

    The wrapper keeps the base module's interface and forwards attribute access
    to it (except a small set of internal attributes).

    Parameters
    ----------
    base : nn.Module
        The wrapped module.
    mode : {'pre','post','both','last'}, default='pre'
        Wrapping layout.
    qconfig : Any, optional
        Explicit qconfig to attach (required for 'post' mode).

    Notes
    -----
    RNN modules are handled specially: the wrapper's ``__call__`` returns
    ``(output, hidden_state)`` to match the original signature.
    """
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
        """Execute wrapped modules in the configured order."""
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
        """Return ``(output, hidden)`` for RNNs; otherwise just output."""
        result = super().__call__(*input, **kwargs)
        return (result, self.__h) if self._is_rnn else result
        
    def __getattr__(self, name):
        """Delegate attribute access to the base module unless internal."""
        if name not in self.__non_redirect:
            return getattr(super().__getattr__('base'), name)
        else:
            return super().__getattr__(name)
