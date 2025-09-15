"""
Core reassembly classes and utilities.

Contains the base reassembly infrastructure moved from quantization utils.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.comptutaional.devices import extract_device
from fedcore.models.network_impl.decomposed_layers import IDecomposed


class RecreatedDecomposed(nn.Sequential):
    """Sequential container for recreated decomposed modules."""
    
    def __init__(self, *modules, routing: Dict = None):
        super().__init__(*modules)
        self.routing = routing or {}
        self._is_recreated = False


class Reassembler(Accessor):
    """Base class for reassembling neural network modules."""

    supported_layers = {}
    supported_decomposed_layers = {}

    @classmethod
    def _fetch_module(cls, module: nn.Module):
        """Determines if a module is supported for conversion."""
        is_decomposed = isinstance(module, IDecomposed)
        supported = cls.supported_decomposed_layers if is_decomposed else cls.supported_layers
        for supported_type in supported:
            if isinstance(module, supported_type) and (is_decomposed or not type(module) is supported_type):
                return supported_type, is_decomposed
        return None, is_decomposed

    @classmethod
    def _handle(cls, module, module_type):
        """Processes module conversion according to its type."""
        supported = cls.supported_decomposed_layers if issubclass(module_type, IDecomposed) else cls.supported_layers
        return supported[module_type](module)

    @classmethod
    def convert(cls, module):
        """Converts a single module."""
        associated, is_decomp = cls._fetch_module(module)
        if associated is None:
            return None
        new_module = cls._handle(module, associated)
        return new_module

    @classmethod
    def _apply_additional_mapping(cls, model: nn.Module, additional_mapping: dict):
        """Applies additional mappings for module replacement."""
        if not additional_mapping:
            return

        for name, module in model.named_modules():
            module_type = type(module)
            if module_type in additional_mapping:
                cls.set_module(model, name, additional_mapping[module_type]())

    @classmethod
    def _traverse_modules(cls, model: nn.Module, pre_hook=None, post_hook=None):
        """
        Unified method for traversing model modules with optional hooks.
        
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
        """Converts all supported modules in the model."""
        cls._traverse_modules(model)

    @classmethod
    def _validate_device_consistency(cls, model: nn.Module):
        """Validates device consistency of model parameters."""
        devices = {p.device for p in model.parameters()}
        if len(devices) > 1:
            raise RuntimeError(f"[{cls.__name__}] Device mismatch! Found devices: {devices}")

    @classmethod
    def reassemble(cls, model: nn.Module, additional_mapping: dict = None, **kwargs):
        """Main method for model reassembly."""
        cls._apply_additional_mapping(model, additional_mapping)
        cls._convert_modules(model)
        cls._validate_device_consistency(model)
        return model


class ParentalReassembler(Reassembler):
    """Reassembler for standard neural network modules."""
    
    def __init__(self):
        # Import here to avoid circular imports
        from .decomposed_recreation import (
            _recreate_embedding, _recreate_decomposed_linear,
            _recreate_decomposed_embedding, _recreate_decomposed_conv2d
        )
        
        self.supported_layers = {
            torch.nn.Embedding: _recreate_embedding,
        }

        self.supported_decomposed_layers = {
            # These will be imported from decomposed_recreation module
        }


class AttentionReassembler(Reassembler):
    """Simple attention reassembler following Zen of Python principles."""

    supported_layers = {}
    supported_decomposed_layers = {}

    @classmethod
    def convert(cls, model: nn.Module, mode: str = 'standard', **kwargs):
        """Simple conversion method without complex conditions."""
        conversion_map = {
            'standard': cls._convert_standard,
            'trans-mla': cls._convert_trans_mla
        }
        
        converter = conversion_map.get(mode)
        assert converter, f"Unknown mode: {mode}"
        
        return converter(model, **kwargs)

    @classmethod
    def _convert_standard(cls, model: nn.Module, additional_mapping: dict = None, **kwargs):
        """Standard conversion."""
        if additional_mapping:
            cls._apply_additional_mapping(model, additional_mapping)
        cls._convert_modules(model)
        cls._validate_device_consistency(model)
        return model

    @classmethod
    def _convert_trans_mla(cls, model: nn.Module, tokenizer=None, config=None,
                          save_path: Optional[str] = None, additional_mapping: dict = None, **kwargs):
        """TransMLA conversion - delegates to TransMLA module."""
        from .transmla_reassembler import convert_model_to_mla
        
        assert tokenizer, "TransMLA conversion requires tokenizer"
        
        # Apply mappings
        if additional_mapping:
            cls._apply_additional_mapping(model, additional_mapping)
        
        # Convert using TransMLA
        model = convert_model_to_mla(model, tokenizer, config, save_path)
        cls._validate_device_consistency(model)
        return model


# Simple reassembler registry with lazy loading for TransMLA
def _get_transmla_class():
    """Lazy import of TransMLA to avoid circular imports."""
    from .transmla_reassembler import TransMLA
    return TransMLA

REASSEMBLERS = {
    'attention': AttentionReassembler,
    'standard': ParentalReassembler,
    'parental': ParentalReassembler,  # Alias for backward compatibility
    'trans-mla': _get_transmla_class,  # Lazy loading
}


def get_reassembler(reassembler_type: str = 'standard'):
    """Get reassembler class by type."""
    if reassembler_type not in REASSEMBLERS:
        available = list(REASSEMBLERS.keys())
        raise ValueError(f"Unknown reassembler type '{reassembler_type}'. Available: {available}")
    
    reassembler = REASSEMBLERS[reassembler_type]
    
    # Handle lazy loading for trans-mla
    if callable(reassembler) and reassembler_type == 'trans-mla':
        return reassembler()  # Call the lazy loader function
    
    return reassembler


def convert_model(model: nn.Module, reassembler_type: str = 'standard', **kwargs):
    """Convert model using specified reassembler."""
    reassembler_cls = get_reassembler(reassembler_type)
    
    # Use convert() for modern reassemblers, reassemble() for legacy ones
    if reassembler_type in ['attention', 'trans-mla']:
        return reassembler_cls.convert(model, **kwargs)
    else:
        return reassembler_cls.reassemble(model, **kwargs)
