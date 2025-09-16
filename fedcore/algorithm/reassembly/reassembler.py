from copy import deepcopy
from typing import Optional, Literal, Dict, Any
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
import torch
from torch import nn

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.algorithm.reassembly.hooks import ReassemblyHooks
from .core_reassemblers import AttentionReassembler
from fedcore.architecture.comptutaional.devices import default_device, extract_device


class BaseReassembler(BaseCompressionModel):
    """Base class for model reassembly after compression.
    
    This pipeline node performs reassembly of compressed models for optimization
    targeting various architectures, including TransMLA.
    
    Supported reassembly modes:
        - 'parental': Parental reassembly of decomposed layers
        - 'trans-mla': Conversion to TransMLA architecture with attention optimization
        - 'custom': Custom reassembly with additional mappings
    
    Args:
        reassemble_mode: Reassembly mode ('parental', 'trans-mla', 'custom')
        reassemble_config: Configuration object (e.g., TransMLAConfig)
        tokenizer: Tokenizer (required for trans-mla mode)
        additional_mapping: Additional module mappings
    """
    
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params or {})
        
        # Main reassembly parameters
        self.reassemble_mode = params.get("reassemble_mode", "parental")
        self.reassemble_config = params.get("reassemble_config", None)
        self.tokenizer = params.get("tokenizer", None)
        self.additional_mapping = params.get("additional_mapping", None)
        
        # Device for computations
        self.device = default_device()
        
        # Parameter validation
        self._validate_params()
    
    def __repr__(self):
        return f"BaseReassembler({self.reassemble_mode})"
    
    def _validate_params(self) -> None:
        """Validate reassembly parameters."""
        valid_modes = {'parental', 'trans-mla', 'custom'}
        if self.reassemble_mode not in valid_modes:
            raise ValueError(
                f"Unsupported reassembly mode '{self.reassemble_mode}'. "
                f"Available modes: {valid_modes}"
            )
        
        if self.reassemble_mode == 'trans-mla':
            self._validate_transmla_requirements()
    
    def _validate_transmla_requirements(self) -> None:
        """Validate requirements for TransMLA mode."""
        if not self.tokenizer:
            raise ValueError(
                "Tokenizer is required for 'trans-mla' mode. "
                "Please provide 'tokenizer' parameter."
            )
    
    def fit(self, input_data: InputData) -> nn.Module:
        """Perform model reassembly.
        
        Args:
            input_data: Input data with model for reassembly
            
        Returns:
            Reassembled model
        """
        print(f"[BaseReassembler] Starting reassembly in mode: {self.reassemble_mode}")
        
        # Initialize model with hooks
        self.model_before = super()._init_model(input_data, [ReassemblyHooks])
        self.model_after = deepcopy(self.model_before)
        
        # Perform reassembly
        self.model_after = self._perform_reassembly(self.model_after)
        
        # Move to device
        self.model_after.to(self.device)
        
        # Estimate parameters (if example input is available)
        try:
            example_batch = self._get_example_input(input_data)
            self.estimate_params(example_batch, self.model_before, self.model_after)
        except Exception as e:
            print(f"[BaseReassembler] Warning: failed to estimate parameters: {e}")
        
        # Mark structure change
        self.model_after._structure_changed__ = True
        
        print(f"[BaseReassembler] Reassembly completed successfully")
        return self.model_after
    
    def _perform_reassembly(self, model: nn.Module) -> nn.Module:
        """Perform reassembly based on the mode.
        
        Args:
            model: Model for reassembly
            
        Returns:
            Reassembled model
        """
        if self.reassemble_mode == 'parental':
            return self._reassemble_parental(model)
        elif self.reassemble_mode == 'trans-mla':
            return self._reassemble_transmla(model)
        elif self.reassemble_mode == 'custom':
            return self._reassemble_custom(model)
        else:
            raise ValueError(f"Unsupported reassembly mode: {self.reassemble_mode}")
    
    def _reassemble_parental(self, model: nn.Module) -> nn.Module:
        """Parental model reassembly.
        
        Args:
            model: Model for reassembly
            
        Returns:
            Reassembled model
        """
        
        reassemble_kwargs = {'additional_mapping': self.additional_mapping}
        
        reassembled_model = AttentionReassembler.convert(
            model, 
            mode='parental',
            **reassemble_kwargs
        )
        
        return reassembled_model
    
    def _reassemble_transmla(self, model: nn.Module) -> nn.Module:
        """Reassemble model to TransMLA architecture.
        
        Args:
            model: Model for reassembly
            
        Returns:
            Reassembled model in TransMLA format
        """
        
        reassemble_kwargs = {
            'tokenizer': self.tokenizer,
            'config': self.reassemble_config,
            'additional_mapping': self.additional_mapping
        }
        
        reassembled_model = AttentionReassembler.convert(
            model, 
            mode='trans-mla',
            **reassemble_kwargs
        )
        
        return reassembled_model
    
    def _reassemble_custom(self, model: nn.Module) -> nn.Module:
        """Custom model reassembly.
        
        Args:
            model: Model for reassembly
            
        Returns:
            Reassembled model
        """
        print("[BaseReassembler] Performing custom reassembly")
        
        # Apply additional mappings if available
        if self.additional_mapping:
            self._apply_additional_mapping(model, self.additional_mapping)
        
        return model
    
    def _apply_additional_mapping(self, model: nn.Module, additional_mapping: Dict[type, type]):
        """Apply additional module mappings.
        
        Args:
            model: Model to process
            additional_mapping: Dictionary of module type mappings
        """
        for name, module in model.named_modules():
            module_type = type(module)
            if module_type in additional_mapping:
                new_module_class = additional_mapping[module_type]
                new_module = new_module_class()
                self._set_module(model, name, new_module)
    
    def _set_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Set new module in model by name.
        
        Args:
            model: Model
            name: Module name
            new_module: New module
        """
        names = name.split('.')
        parent = model
        for n in names[:-1]:
            parent = getattr(parent, n)
        setattr(parent, names[-1], new_module)
    
    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Return model after training."""
        return self.model_after if output_mode == 'fedcore' else self.model_before
    
    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Prediction using reassembled model."""
        return self.model_after if output_mode == 'fedcore' else self.model_before
    
    @classmethod
    def create_for_transmla(cls, 
                           base_params: Optional[OperationParameters] = None,
                           config = None,
                           tokenizer = None,
                           additional_mapping: Optional[Dict] = None) -> 'BaseReassembler':
        """Factory method for creating TransMLA reassembler.
        
        Args:
            base_params: Base parameters
            config: TransMLA configuration
            tokenizer: Tokenizer
            additional_mapping: Additional mappings
            
        Returns:
            Configured BaseReassembler instance
        """
        params_dict = {}
        
        # Add base parameters
        if base_params:
            params_dict.update(
                base_params.to_dict() if hasattr(base_params, 'to_dict') 
                else base_params
            )
        
        # Add TransMLA parameters
        params_dict.update({
            'reassemble_mode': 'trans-mla',
            'reassemble_config': config,
            'tokenizer': tokenizer,
            'additional_mapping': additional_mapping
        })
        
        final_params = OperationParameters()
        for key, value in params_dict.items():
            setattr(final_params, key, value)
        return cls(final_params)
