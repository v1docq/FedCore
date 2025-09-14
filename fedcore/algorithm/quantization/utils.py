from copy import deepcopy
from functools import partial
import inspect
from typing import Callable, Dict, Literal, Optional, Union
import argparse

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
    DecomposedConv2d,
    DecomposedConv1d
)
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.abstraction.delegator import IDelegator
from fedcore.architecture.comptutaional.devices import extract_device, default_device

# TransMLA integration imports
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

# TransMLA configuration
TRANSMLA_PATH = Path(__file__).parent.parent.parent.parent / "external" / "transmla_core"
TRANSMLA_AVAILABLE = False
TRANSMLA_ERROR = None

# TransMLA function placeholders
partial_rope = None
low_rank_qkv = None
modify_config = None
get_dataset = None
prepare_dataloader = None
prepare_test_dataloader = None
evaluate_ppl = None


class TransMLAImporter:
    """Simple TransMLA import handler"""
    
    @staticmethod
    def initialize() -> Tuple[bool, Optional[str]]:
        """Initialize TransMLA imports"""
        transmla_path = TRANSMLA_PATH
        
        # Check if path exists
        path_exists = transmla_path.exists()
        assert path_exists, "TransMLA core module not found in external/transmla_core"
        
        # Setup paths
        transmla_str = str(transmla_path)
        
        transmla_str not in sys.path and sys.path.insert(0, transmla_str)
        
        # Import modules
        import utils as transmla_utils
        import partial_rope as transmla_partial_rope
        import lora_qkv as transmla_lora_qkv
        import modify_config as transmla_modify_config

        # Make functions globally available
        global partial_rope, low_rank_qkv, modify_config
        global get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl
        
        partial_rope = transmla_partial_rope.partial_rope
        low_rank_qkv = transmla_lora_qkv.low_rank_qkv
        modify_config = transmla_modify_config.modify_config
        get_dataset = transmla_utils.get_dataset
        prepare_dataloader = transmla_utils.prepare_dataloader
        prepare_test_dataloader = transmla_utils.prepare_test_dataloader
        evaluate_ppl = transmla_utils.evaluate_ppl

        return True, None


# Initialize TransMLA
TRANSMLA_AVAILABLE, TRANSMLA_ERROR = TransMLAImporter.initialize()


__all__ = [
    'Reassembler',
    'ParentalReassembler',
    'AttentionReassembler',
    'TransMLA',
    'DeferredConversion',
    'ReassemblerFactory',
    'TransMLAConfig',
    'QDQWrapper',
    'QDQWrapping',
    'uninplace',
    'reset_qconfig',
    'RecreatedDecomposed',
    'get_flattened_qconfig_dict',
    'TRANSMLA_AVAILABLE',
    'TRANSMLA_ERROR',
    'get_transmla_status',
    # Backward compatibility
    'LLMReassembler'
]

QConfigMapping = QConfigAny


def get_transmla_status():
    """
    Returns detailed information about TransMLA status
    
    Returns:
        dict: Dictionary with TransMLA status information
    """
    status = {
        'available': TRANSMLA_AVAILABLE,
        'error': TRANSMLA_ERROR,
        'path_exists': TRANSMLA_PATH.exists(),
        'recommendations': []
    }

    if TRANSMLA_AVAILABLE:
        return status
    
    # Generate recommendations based on error type
    error_str = str(TRANSMLA_ERROR) if TRANSMLA_ERROR else ""
    
    recommendation_map = {
        "transformers": ["Update transformers: pip install transformers>=4.52.4"],
        "core module not found": [
                "Create transmla_core module: mkdir external/transmla_core",
                "Copy required files to external/transmla_core/"
        ]
    }
    
    for error_key, recommendations in recommendation_map.items():
        if error_key in error_str:
            status['recommendations'].extend(recommendations)
            break
    else:
        # Default recommendation if no specific error matched
            status['recommendations'].append(
                "Install TransMLA dependencies: pip install -e external/transmla_core/"
            )

    return status


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
                 'bias': ('bias', '1'), }
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

class Reassembler(Accessor):
    """Base class for reassembling neural network modules"""

    supported_layers = {}
    supported_decomposed_layers = {}

    @classmethod
    def _fetch_module(cls, module: nn.Module):
        """Determines if a module is supported for conversion"""
        is_decomposed = isinstance(module, IDecomposed)
        supported = cls.supported_decomposed_layers if is_decomposed else cls.supported_layers
        for supported_type in supported:
            if isinstance(module, supported_type) and (is_decomposed or not type(module) is supported_type):
                return supported_type, is_decomposed
        return None, is_decomposed

    @classmethod
    def _handle(cls, module, module_type):
        """Processes module conversion according to its type"""
        supported = cls.supported_decomposed_layers if issubclass(module_type, IDecomposed) else cls.supported_layers
        return supported[module_type](module)

    @classmethod
    def convert(cls, module):
        """Converts a single module"""
        associated, is_decomp = cls._fetch_module(module)
        if associated is None:
            return None
        new_module = cls._handle(module, associated)
        return new_module

    @classmethod
    def _apply_additional_mapping(cls, model: nn.Module, additional_mapping: dict):
        """Applies additional mappings for module replacement"""
        if not additional_mapping:
            return

        for name, module in model.named_modules():
            module_type = type(module)
            if module_type in additional_mapping:
                cls.set_module(model, name, additional_mapping[module_type]())

    @classmethod
    def _traverse_modules(cls, model: nn.Module, pre_hook=None, post_hook=None):
        """
        Unified method for traversing model modules with optional hooks
        
        Args:
            model: Model to traverse
            pre_hook: Function called before processing each module (name, module) -> bool
                     Returns True to continue processing, False to skip
            post_hook: Function called after processing each module (name, module, result) -> None
        """
        device = extract_device(model)
        
        for name, module in model.named_modules():
            # Pre-processing hook
            if pre_hook and not pre_hook(name, module):
                continue
                
            # Main conversion logic - use base Reassembler convert method
            new_module = Reassembler.convert(module)
            if new_module:
                cls.set_module(model, name, new_module.to(device))
                
            # Post-processing hook
            if post_hook:
                post_hook(name, module, new_module)

    @classmethod
    def _convert_modules(cls, model: nn.Module):
        """Converts all supported modules in the model"""
        cls._traverse_modules(model)

    @classmethod
    def _validate_device_consistency(cls, model: nn.Module):
        """Validates device consistency of model parameters"""
        devices = {p.device for p in model.parameters()}
        if len(devices) > 1:
            raise RuntimeError(f"[{cls.__name__}] Device mismatch! Found devices: {devices}")

    @classmethod
    def reassemble(cls, model: nn.Module, additional_mapping: dict = None, **kwargs):
        """Main method for model reassembly"""
        cls._apply_additional_mapping(model, additional_mapping)
        cls._convert_modules(model)
        cls._validate_device_consistency(model)
        return model


class ParentalReassembler(Reassembler):    
    supported_layers = {
        torch.nn.Embedding: _recreate_embedding,
        # torch.nn.Linear: _recreate_linear
    }

    supported_decomposed_layers = {
        DecomposedLinear: _recreate_decomposed_linear,
        DecomposedEmbedding: _recreate_decomposed_embedding,
        DecomposedConv2d: _recreate_decomposed_conv2d,
    }

    @classmethod
    def _resnet_post_hook(cls, name: str, original_module: nn.Module, converted_module: nn.Module):
        """Post-processing hook for ResNet blocks"""
        if converted_module is None and isinstance(original_module, (resnet.BasicBlock, resnet.Bottleneck)):
            device = extract_device(original_module)
            wrapped_module = ResidualAddWrapper(original_module)
            # Получаем модель из контекста (будет передана через замыкание)
            if hasattr(cls, '_current_model'):
                cls.set_module(cls._current_model, name, wrapped_module.to(device))
                print(f"[ParentalReassembler] Residual block '{name}' wrapped with ResidualAddWrapper.")

    @classmethod
    def _convert_modules(cls, model: nn.Module):
        """Converts modules with additional ResNet block processing using hooks"""
        # Сохраняем ссылку на модель для использования в хуках
        cls._current_model = model
        try:
            cls._traverse_modules(model, post_hook=cls._resnet_post_hook)
        finally:
            # Очищаем ссылку после использования
            if hasattr(cls, '_current_model'):
                delattr(cls, '_current_model')


class TransMLAConfig:
    """Configuration for TransMLA conversion"""

    def __init__(self, 
                 freqfold: Union[str, int] = "auto",
                 collapse: Union[str, int] = "auto", 
                 qk_mqa_dim: int = 64,
                 q_lora_rank: Optional[int] = None,
                 kv_lora_rank: int = 512,
                 balance_kv_ratio: float = 1.0,
                 use_qkv_norm: bool = False,
                 cal_dataset: str = "wikitext2",
                 cal_nsamples: int = 128,
                 cal_batch_size: int = 8,
                 cal_max_seqlen: int = 256,
                 ppl_eval_batch_size: int = 2,
                 deepseek_style: bool = True,
                 dtype: str = "bf16",
                 device: str = "auto",
                 seed: int = 42):
        self.freqfold = freqfold
        self.collapse = collapse
        self.qk_mqa_dim = qk_mqa_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.balance_kv_ratio = balance_kv_ratio
        self.use_qkv_norm = use_qkv_norm
        self.cal_dataset = cal_dataset
        self.cal_nsamples = cal_nsamples
        self.cal_batch_size = cal_batch_size
        self.cal_max_seqlen = cal_max_seqlen
        self.ppl_eval_batch_size = ppl_eval_batch_size
        self.deepseek_style = deepseek_style
        self.dtype = dtype
        self.device = device
        self.seed = seed

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _convert_model_to_mla(model: nn.Module, tokenizer, config: TransMLAConfig, save_path: Optional[str] = None):
    """
    Complete MLA conversion of model using TransMLA
    Simple implementation without try/except blocks
    """
    print("[TransMLA] Starting MLA conversion...")
    
    # Prepare calibration data
    dataset = get_dataset(config.cal_dataset)
    train_loader = prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=config.cal_max_seqlen,
        batch_size=config.cal_batch_size,
        nsamples=config.cal_nsamples,
        seed=config.seed,
    )
    test_loader = prepare_test_dataloader(
        dataset=dataset["test"], 
        tokenizer=tokenizer, 
        batch_size=max(1, config.ppl_eval_batch_size)
    )
    
    # Stage 1: Partial RoPE
    config_dict = config.to_dict()
    
    # Handle automatic collapse calculation
    collapse_value = config.collapse
    if collapse_value == "auto":
        head_dim = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
        model.config.head_dim = head_dim
        collapse_value = head_dim // config.qk_mqa_dim
        print(f"[TransMLA] Auto collapse: {collapse_value}")
    
    config_dict["collapse"] = int(collapse_value)
    
    model = partial_rope(model, tokenizer, train_loader, test_loader, **config_dict)
    
    # Process partial_rope result
    freqfold_value = config.freqfold
    if isinstance(model, tuple):
        if freqfold_value == "auto":
            freqfold_value = model[1]
            print(f"[TransMLA] Auto freqfold: {freqfold_value}")
        model = model[0]
    
    config_dict["freqfold"] = freqfold_value
    
    # Stage 2: Low-rank QKV
    model = low_rank_qkv(model, tokenizer, train_loader, test_loader, **config_dict)
    
    # Save model if path specified
    if save_path:
        print(f"[TransMLA] Saving model to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        import argparse
        config_path = os.path.join(save_path, "config.json")
        modify_config(model, config_path, argparse.Namespace(**config_dict))
    
    print("[TransMLA] MLA conversion completed successfully!")
    return model


class AttentionReassembler(Reassembler):
    """
    Simple attention reassembler following Zen of Python principles
    """

    supported_layers = {}
    supported_decomposed_layers = {}

    @classmethod
    def convert(cls, model: nn.Module, mode: str = 'standard', **kwargs):
        """Simple conversion method without complex conditions"""
        conversion_map = {
            'standard': cls._convert_standard,
            'trans-mla': cls._convert_trans_mla
        }
        
        converter = conversion_map.get(mode)
        assert converter, f"Unknown mode: {mode}"
        
        return converter(model, **kwargs)

    @classmethod
    def _convert_standard(cls, model: nn.Module, additional_mapping: dict = None, **kwargs):
        """Standard conversion"""
        additional_mapping and cls._apply_additional_mapping(model, additional_mapping)
        cls._convert_modules(model)
        cls._validate_device_consistency(model)
        return model

    @classmethod
    def _convert_trans_mla(cls, model: nn.Module, tokenizer=None, config: Optional[TransMLAConfig] = None,
                          save_path: Optional[str] = None, additional_mapping: dict = None, **kwargs):
        """TransMLA conversion - simple implementation"""
        assert tokenizer, "TransMLA conversion requires tokenizer"
        assert TRANSMLA_AVAILABLE, f"TransMLA not available: {TRANSMLA_ERROR}"
        
        # Apply mappings
        additional_mapping and cls._apply_additional_mapping(model, additional_mapping)
        
        # Prepare config
        config = config or TransMLAConfig()
        
        # Update config with kwargs
        for key, value in kwargs.items():
            hasattr(config, key) and setattr(config, key, value)
        
        # Convert
        model = _convert_model_to_mla(model, tokenizer, config, save_path)
        cls._validate_device_consistency(model)
        return model


class DeferredConversion:
    """Simple deferred conversion container"""
    
    def __init__(self, conversion_type: str, model: nn.Module, **kwargs):
        self.conversion_type = conversion_type
        self.model = model
        self.kwargs = kwargs
        self.executed = False
        self.result = None
    
    def execute(self):
        """Execute the deferred conversion"""
        assert not self.executed, "Conversion already executed"
        
        conversion_map = {
            'trans-mla': TransMLA._execute_conversion
        }
        
        converter = conversion_map.get(self.conversion_type)
        assert converter, f"Unknown conversion type: {self.conversion_type}"
        
        self.result = converter(self.model, **self.kwargs)
        self.executed = True
        return self.result


class TransMLA(AttentionReassembler):
    """Specialized TransMLA reassembler"""

    @classmethod
    def convert(cls, model: nn.Module, tokenizer, config: Optional[TransMLAConfig] = None,
                save_path: Optional[str] = None, deferred: bool = False, **kwargs):
        """TransMLA conversion with optional deferred execution"""
        conversion_args = {
            'model': model,
            'tokenizer': tokenizer,
            'config': config,
            'save_path': save_path,
            **kwargs
        }
        
        return DeferredConversion('trans-mla', **conversion_args) if deferred else cls._execute_conversion(**conversion_args)

    @classmethod
    def _execute_conversion(cls, model: nn.Module, tokenizer, config: Optional[TransMLAConfig] = None,
                           save_path: Optional[str] = None, **kwargs):
        """Execute TransMLA conversion immediately"""
        return cls._convert_trans_mla(
            model=model,
            tokenizer=tokenizer,
            config=config,
            save_path=save_path,
            **kwargs
        )


# Backward compatibility alias - will be deprecated
LLMReassembler = AttentionReassembler


class ReassemblerFactory:
    """Simple factory for reassemblers"""
    
    _reassemblers = {
        'attention': AttentionReassembler,
        'trans-mla': TransMLA,
        'standard': ParentalReassembler,
        'parental': ParentalReassembler,
    }
    
    @classmethod
    def get_reassembler(cls, reassembler_type: str = 'standard'):
        """Get reassembler class"""
        reassembler = cls._reassemblers.get(reassembler_type)
        assert reassembler, f"Unknown reassembler type: {reassembler_type}"
        return reassembler
    
    @classmethod
    def convert_model(cls, model: nn.Module, reassembler_type: str = 'standard', **kwargs):
        """Convert model with specified reassembler"""
        reassembler_class = cls.get_reassembler(reassembler_type)
        
        conversion_methods = {
            'attention': lambda: reassembler_class.convert(model, **kwargs),
            'trans-mla': lambda: reassembler_class.convert(model, **kwargs),
            'standard': lambda: reassembler_class.reassemble(model, **kwargs),
            'parental': lambda: reassembler_class.reassemble(model, **kwargs)
        }
        
        converter = conversion_methods.get(reassembler_type)
        assert converter, f"No converter for type: {reassembler_type}"
        
        return converter()


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
        except (TypeError, IndexError, AttributeError):
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
        except (RuntimeError, ValueError, TypeError) as e:
            # Log the specific error for debugging
            print(f"[QDQWrapper] Quantization failed for module {type(module).__name__}: {e}")
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
