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
    DecomposedConv2d
)
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.abstraction.delegator import IDelegator
from fedcore.architecture.comptutaional.devices import extract_device, default_device

# TransMLA integration imports
import sys
import os
from pathlib import Path

# Add TransMLA to Python path
TRANSMLA_PATH = Path(__file__).parent.parent.parent.parent / "external" / "TransMLA"
TRANSMLA_AVAILABLE = False
TRANSMLA_ERROR = None

if TRANSMLA_PATH.exists():
    original_path = sys.path.copy()
    transmla_path = str(TRANSMLA_PATH)

    if transmla_path not in sys.path:
        sys.path.insert(0, transmla_path)

    try:
        import transformers
        from packaging import version

        required_version = "4.52.0"
        current_version = transformers.__version__

        if version.parse(current_version) < version.parse(required_version):
            TRANSMLA_ERROR = f"TransMLA requires transformers>={required_version}, current version: {current_version}"
        else:
            # Add path to transmla directory for direct import
            transmla_dir = str(TRANSMLA_PATH / 'transmla')
            if transmla_dir not in sys.path:
                sys.path.insert(0, transmla_dir)

            # Direct import of modules (works according to debugging)
            import utils as transmla_utils
            import partial_rope as transmla_partial_rope
            import lora_qkv as transmla_lora_qkv
            import modify_config as transmla_modify_config

            # Extract needed functions
            partial_rope = transmla_partial_rope.partial_rope
            low_rank_qkv = transmla_lora_qkv.low_rank_qkv
            modify_config = transmla_modify_config.modify_config
            get_dataset = transmla_utils.get_dataset
            prepare_dataloader = transmla_utils.prepare_dataloader
            prepare_test_dataloader = transmla_utils.prepare_test_dataloader
            evaluate_ppl = transmla_utils.evaluate_ppl

            TRANSMLA_AVAILABLE = True

    except ImportError as e:
        TRANSMLA_ERROR = f"TransMLA import failed: {e}"
    except Exception as e:
        TRANSMLA_ERROR = f"TransMLA initialization error: {e}"
    finally:
        if not TRANSMLA_AVAILABLE:
            sys.path = original_path
else:
    TRANSMLA_ERROR = "TransMLA submodule not found in external/TransMLA"


__all__ = [
    'Reassembler',
    'ParentalReassembler',
    'LLMReassembler',
    'TransMLAConfig',
    'QDQWrapper',
    'QDQWrapping',
    'uninplace',
    'reset_qconfig',
    'RecreatedDecomposed',
    'get_flattened_qconfig_dict',
    'TRANSMLA_AVAILABLE',
    'TRANSMLA_ERROR',
    'get_transmla_status'
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
        'path_exists': TRANSMLA_PATH.exists() if 'TRANSMLA_PATH' in globals() else False,
        'recommendations': []
    }

    if not TRANSMLA_AVAILABLE:
        if "transformers" in str(TRANSMLA_ERROR):
            status['recommendations'].append(
                "Update transformers: pip install transformers>=4.52.4"
            )
        if "submodule not found" in str(TRANSMLA_ERROR):
            status['recommendations'].extend([
                "Initialize submodule: git submodule update --init --recursive",
                "Or clone TransMLA manually to external/TransMLA"
            ])
        if not status['recommendations']:
            status['recommendations'].append(
                "Install TransMLA dependencies: pip install -r external/TransMLA/requirements.txt"
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
    U, S, Vh = L.U.detach(), L.S.detach(), L.Vh.detach()
    U = U * S
    h = S.size(0)
    new = RecreatedDecomposed(
        nn.Linear(L.in_features, h, bias=False),
        nn.Linear(h, L.out_features, bias=True),
        routing={'out_features': ('out_features', '1'), 
                 'bias': ('bias', '1')}
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
    if getattr(L.bias, 'data', None) is not None:
        new[-1].bias.data = L.bias.data
    return new

def _recreate_decomposed_embedding(E: DecomposedEmbedding):
    U, S, Vh = E.U.detach(), E.S.detach(), E.Vh.detach()
    h = S.size(0)
    new = RecreatedDecomposed(
        nn.Embedding(E.num_embeddings, h),
        nn.Linear(h, E.embedding_dim, False),
        routing={'embedding_dim': ('out_features', '1')}
    )
    new[0].weight.data = U
    new[-1].weight.data = (torch.diag(S) @ Vh).T
    new._is_recreated = True
    return new

def _recreate_decomposed_conv2d(C: DecomposedConv2d):
    U, S, Vh = C.U.detach(), C.S.detach(), C.Vh.detach()
    assert U.ndim == 4, 'Non composed layers are not supported'
    out_1, in_1, k_11, k_12 = Vh.size()
    out_2, in_2, k_21, k_22 = U.size()
    new = RecreatedDecomposed(
        nn.Conv2d(in_1, out_1, (k_11, k_12), groups=C.groups, **C.decomposing['Vh']),
        nn.Conv2d(in_2, out_2, (k_21, k_22), **C.decomposing['U']),
        routing={
            'in_channels': ('in_channels', '0'),
            'out_channels': ('out_channels', '1'),
            'groups': ('groups', '0')
        }
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
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
    def _convert_modules(cls, model: nn.Module):
        """Converts all supported modules in the model"""
        device = extract_device(model)

        for name, module in model.named_modules():
            new_module = cls.convert(module)
            if new_module:
                cls.set_module(model, name, new_module.to(device))

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
    def _convert_modules(cls, model: nn.Module):
        """Converts modules with additional ResNet block processing"""
        device = extract_device(model)

        for name, module in model.named_modules():
            new_module = cls.convert(module)
            if new_module:
                cls.set_module(model, name, new_module.to(device))
            elif isinstance(module, (resnet.BasicBlock, resnet.Bottleneck)):
                wrapped_module = ResidualAddWrapper(module)
                cls.set_module(model, name, wrapped_module.to(device))
                print(f"[ParentalReassembler] Residual block '{name}' wrapped with ResidualAddWrapper.")


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
    
    Args:
        model: Source model for conversion
        tokenizer: Model tokenizer
        config: TransMLA configuration
        save_path: Path for saving (optional)
    
    Returns:
        Model converted to MLA
    """
    if not TRANSMLA_AVAILABLE:
        print(f"[Warning] TransMLA not available: {TRANSMLA_ERROR}")
        status = get_transmla_status()
        print("[Info] Recommendations for fixing:")
        for rec in status['recommendations']:
            print(f"  • {rec}")
        print("[Warning] Returning original model without conversion.")
        return model

    print("[TransMLA] Starting MLA conversion...")

    try:
        # Prepare calibration data
        print("[TransMLA] Preparing calibration data...")
        dataset = get_dataset(config.cal_dataset)
        train_loader = prepare_dataloader(
            dataset=dataset["train"],
            tokenizer=tokenizer,
            max_seqlen=config.cal_max_seqlen,
            batch_size=config.cal_batch_size,
            nsamples=config.cal_nsamples,
            seed=config.seed,
        )

        # test_loader always needed for auto freqfold detection, even if eval disabled
        test_loader = prepare_test_dataloader(
            dataset=dataset["test"], 
            tokenizer=tokenizer, 
            batch_size=max(1, config.ppl_eval_batch_size)  # Minimum 1 for auto detection
        )

        # Stage 1: Partial RoPE
        print("[TransMLA] Stage 1: Applying Partial RoPE...")
        config_dict = config.to_dict()

        # Automatic collapse calculation if needed
        if config.collapse == "auto":
            head_dim = (model.config.head_dim if hasattr(model.config, "head_dim") and 
                       model.config.head_dim is not None else 
                       model.config.hidden_size // model.config.num_attention_heads)
            model.config.head_dim = head_dim
            config_dict["collapse"] = head_dim // config.qk_mqa_dim
            print(f"[TransMLA] Auto collapse: {config_dict['collapse']} (head_dim={head_dim} / qk_mqa_dim={config.qk_mqa_dim})")
        else:
            config_dict["collapse"] = int(config.collapse)

        model = partial_rope(model, tokenizer, train_loader, test_loader, **config_dict)

        # Process partial_rope result
        if isinstance(model, tuple):
            if config.freqfold == "auto":
                config_dict["freqfold"] = model[1]
                print(f"[TransMLA] Auto freqfold: {config_dict['freqfold']}")
            model = model[0]

        # Stage 2: Low-rank QKV
        print("[TransMLA] Stage 2: Applying Low-rank QKV...")
        model = low_rank_qkv(model, tokenizer, train_loader, test_loader, **config_dict)

        # Save model if path specified
        if save_path:
            print(f"[TransMLA] Saving model to {save_path}...")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Modify configuration
            config_path = os.path.join(save_path, "config.json")
            modify_config(model, config_path, argparse.Namespace(**config_dict))

        print("[TransMLA] MLA conversion completed successfully!")
        return model

    except Exception as e:
        print(f"[TransMLA Error] Error during conversion: {e}")
        print("[TransMLA] Returning original model...")
        # Mark model as unconverted
        if hasattr(model, '_mla_conversion_failed'):
            model._mla_conversion_failed = True
        else:
            setattr(model, '_mla_conversion_failed', True)
        return model


def _convert_attention_to_mla(module: nn.Module):
    """
    Compatibility stub. Real conversion is performed in _convert_model_to_mla
    """
    print(f"[MLA] Attention layer detected: {type(module).__name__}")
    return module


def _convert_linear_to_mla_compatible(module: nn.Linear):
    """
    Compatibility stub. Real conversion is performed in _convert_model_to_mla
    """
    print(f"[MLA] Linear layer detected: {module}")
    return module


class LLMReassembler(Reassembler):
    """Reassembler for large language models with support for various reassembly types"""

    # Basic supported layers (inherited from Reassembler)
    supported_layers = {}
    supported_decomposed_layers = {}

    # MLA-specific mappings
    mla_attention_layers = {
        # Here will be various types of attention layers
        # nn.MultiheadAttention: _convert_attention_to_mla,
        # Add as needed
    }

    mla_linear_layers = {
        nn.Linear: _convert_linear_to_mla_compatible,
    }

    @classmethod
    def _get_reassembly_mapping(cls, reassembly_type: str):
        """Returns mapping for specific reassembly type"""
        mappings = {
            'mla': {
                **cls.mla_attention_layers,
                **cls.mla_linear_layers,
            },
            # Can add other reassembly types:
            # 'gqa': {...},
            # 'standard': {...},
        }

        if reassembly_type not in mappings:
            raise ValueError(f"Unsupported reassembly type: {reassembly_type}. "
                           f"Available types: {list(mappings.keys())}")

        return mappings[reassembly_type]

    @classmethod
    def _convert_modules_with_type(cls, model: nn.Module, reassembly_type: str):
        """Converts modules according to specified reassembly type"""
        device = extract_device(model)
        type_mapping = cls._get_reassembly_mapping(reassembly_type)

        for name, module in model.named_modules():
            # Сначала пытаемся стандартную конвертацию
            new_module = cls.convert(module)
            if new_module:
                cls.set_module(model, name, new_module.to(device))
                continue

            # Then try type-specific conversion
            module_type = type(module)
            if module_type in type_mapping:
                converted_module = type_mapping[module_type](module)
                if converted_module != module:  # If module was changed
                    cls.set_module(model, name, converted_module.to(device))

    @classmethod
    def reassemble(cls, model: nn.Module, reassembly_type: str = 'standard', 
                   additional_mapping: dict = None, tokenizer=None, 
                   mla_config: Optional[TransMLAConfig] = None,
                   save_path: Optional[str] = None, return_success_info: bool = False, **kwargs):
        """
        Main method for model reassembly with support for various types.
        
        Args:
            model: Model for reassembly
            reassembly_type: Reassembly type ('mla', 'standard', etc.)
            additional_mapping: Additional module mappings
            tokenizer: Tokenizer (required for MLA conversion)
            mla_config: Configuration for MLA conversion
            save_path: Path for saving MLA model
            return_success_info: If True, returns (model, success_info)
            **kwargs: Additional parameters
        
        Returns:
            Reassembled model or (model, success_info) if return_success_info=True
        """
        print(f"[LLMReassembler] Starting model reassembly with type: {reassembly_type}")

        # Initialize success information
        success_info = {"type": reassembly_type, "successful": True, "error": None}

        # Apply additional mappings
        cls._apply_additional_mapping(model, additional_mapping)

        # Special handling for MLA
        if reassembly_type == 'mla':
            if not TRANSMLA_AVAILABLE:
                print(f"[Warning] TransMLA not available: {TRANSMLA_ERROR}")
                status = get_transmla_status()
                print("[Info] To solve the problem:")
                for rec in status['recommendations']:
                    print(f"  • {rec}")
                print("[Warning] Using stubs instead of full MLA conversion.")
                cls._convert_modules_with_type(model, reassembly_type)
            elif tokenizer is None:
                print("[Warning] MLA conversion requires tokenizer. Using stubs.")
                cls._convert_modules_with_type(model, reassembly_type)
            else:
                # Full MLA conversion with TransMLA
                if mla_config is None:
                    mla_config = TransMLAConfig()
                    print("[Info] Using default MLA configuration")

                # Update configuration from kwargs
                for key, value in kwargs.items():
                    if hasattr(mla_config, key):
                        setattr(mla_config, key, value)

                model = _convert_model_to_mla(model, tokenizer, mla_config, save_path)

                # Check if MLA conversion was successful
                if hasattr(model, '_mla_conversion_failed') and model._mla_conversion_failed:
                    print("[LLMReassembler] MLA conversion failed - model remains unchanged")
                    success_info["successful"] = False
                    success_info["error"] = "MLA conversion failed"
                    # Remove flag to avoid cluttering the model
                    delattr(model, '_mla_conversion_failed')
                else:
                    print("[LLMReassembler] MLA conversion completed successfully")

        elif reassembly_type == 'standard':
            cls._convert_modules(model)
        else:
            cls._convert_modules_with_type(model, reassembly_type)

        # Check device consistency
        cls._validate_device_consistency(model)

        print(f"[LLMReassembler] Model reassembly completed")

        if return_success_info:
            return model, success_info
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
