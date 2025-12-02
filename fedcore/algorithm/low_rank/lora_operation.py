"""
LoRA (Low-Rank Adaptation) implementation for FedCore.

This module provides BaseLoRA class for applying Low-Rank Adaptation
to neural network models, supporting both custom models and HuggingFace models.

Supported Layer Types:
    - nn.Linear (fully connected layers)
    - nn.Conv2d (convolutional layers for CNNs like EfficientNet, ResNet)
    - nn.Embedding (embedding layers for transformers)

Model Support:
    - HuggingFace Transformers (with optional PEFT library integration)
    - Custom PyTorch models (CNNs, MLPs, etc.)
    
Key Features:
    - Parameter-efficient fine-tuning (trains only 0.1-10% of parameters)
    - Automatic base parameter freezing
    - Support for both attention and convolutional layers
    - Flexible target module selection
"""

import logging
from typing import Optional, Union, List
import torch
import torch.nn as nn
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.architecture.computational.devices import default_device
from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.utils.trainer_factory import create_trainer_from_input_data
from fedcore.models.network_modules.layers.lora import (
    Linear as LoRALinear,
    Embedding as LoRAEmbedding,
    Conv2d as LoRAConv2d,
)


logger = logging.getLogger(__name__)


class BaseLoRA(BaseCompressionModel):
    """
    Base class for LoRA (Low-Rank Adaptation) operations.
    
    LoRA adds trainable low-rank decomposition matrices to existing weights,
    allowing parameter-efficient fine-tuning of large models.
    
    Parameters:
        lora_r (int): Rank of LoRA decomposition. Default: 8
        lora_alpha (int): Scaling factor for LoRA. Default: 16
        lora_dropout (float): Dropout probability for LoRA layers. Default: 0.1
        lora_target_modules (List[str]): Names of modules to apply LoRA to.
            Default: ["q_proj", "v_proj"] for attention layers
        use_peft (bool): Whether to use PEFT library for HuggingFace models. Default: False
        lora_bias (str): Bias strategy - "none", "all", or "lora_only". Default: "none"
        
    Example:
        >>> from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
        >>> lora = BaseLoRA(params={'lora_r': 8, 'lora_alpha': 16})
        >>> lora.fit(input_data)
        >>> compressed_model = lora.predict(input_data)
    """
    
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params if params else {})
        
        # LoRA hyperparameters
        self.lora_r = self.params.get("lora_r", 8)
        self.lora_alpha = self.params.get("lora_alpha", 16)
        self.lora_dropout = self.params.get("lora_dropout", 0.1)
        self.lora_target_modules = self.params.get(
            "lora_target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"]  # Default attention modules
        )
        self.use_peft = self.params.get("use_peft", False)
        self.lora_bias = self.params.get("lora_bias", "none")
        
        # Training parameters
        self.epochs = self.params.get("epochs", 3)
        self.learning_rate = self.params.get("lr", 1e-4)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"BaseLoRA initialized: r={self.lora_r}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, target_modules={self.lora_target_modules}"
        )
        
        self._lora_modules = []  # Track applied LoRA modules
        
    def __repr__(self):
        return f"LoRA(r={self.lora_r}, alpha={self.lora_alpha})"
    
    def _is_huggingface_model(self, model: torch.nn.Module) -> bool:
        """Check if model is from HuggingFace transformers library."""
        return (
            hasattr(model, 'config') and
            hasattr(model.config, 'model_type') and
            hasattr(model, 'base_model')
        )
    
    def _should_apply_lora(self, module_name: str) -> bool:
        """Determine if LoRA should be applied to a module based on its name."""
        if not self.lora_target_modules:
            # If no target modules specified, apply to ALL supported layer types
            # (Conv2d, Linear, Embedding) - we check type, not name
            return True
        
        # Check if module name contains any of the target patterns
        return any(
            target in module_name
            for target in self.lora_target_modules
        )
    
    def _apply_lora_to_model(self, model: torch.nn.Module, params: dict) -> torch.nn.Module:
        """
        Universal interface for applying LoRA to any model.
        
        Determines model type (HuggingFace or custom) and applies appropriate LoRA method.
        
        For HuggingFace models:
            - If use_peft=True: uses peft.LoraConfig and get_peft_model()
            - Otherwise: uses fedcore implementation from lora.py
        
        For custom models:
            - Uses classes from fedcore/models/network_modules/layers/lora.py
            - Supports nn.Linear, nn.Conv2d, nn.Embedding
            - Wraps layers with LoRALayer.Linear, LoRALayer.Conv2d, LoRALayer.Embedding
        
        Args:
            model: PyTorch model to apply LoRA to
            params: LoRA configuration parameters
            
        Returns:
            Model with LoRA applied
        """
        self.logger.info('Applying LoRA to model'.center(80, '='))
        
        # Determine model type
        is_hf = self._is_huggingface_model(model)
        model_type = 'HuggingFace' if is_hf else 'Custom'
        
        self.logger.info(f"Model type detected: {model_type}")
        self.logger.info(
            f"LoRA config: r={params.get('lora_r', self.lora_r)}, "
            f"alpha={params.get('lora_alpha', self.lora_alpha)}, "
            f"dropout={params.get('lora_dropout', self.lora_dropout)}"
        )
        
        # Apply LoRA based on model type
        if is_hf:
            model_with_lora = self._apply_lora_huggingface(model, params)
        else:
            model_with_lora = self._apply_lora_custom(model, params)
        
        self.logger.info('LoRA application complete'.center(80, '='))
        
        return model_with_lora
    
    def _apply_lora_to_layer(
        self,
        layer: nn.Module,
        layer_name: str,
        adapter_name: str = "default"
    ) -> nn.Module:
        """
        Apply LoRA to a specific layer.
        
        Args:
            layer: The layer to apply LoRA to
            layer_name: Name of the layer
            adapter_name: Name of the LoRA adapter
            
        Returns:
            Layer with LoRA applied
        """
        if isinstance(layer, nn.Linear):
            self.logger.debug(f"Applying LoRA to Linear layer: {layer_name}")
            return LoRALinear(
                base_layer=layer,
                adapter_name=adapter_name,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                init_lora_weights=True
            )
        
        elif isinstance(layer, nn.Embedding):
            self.logger.debug(f"Applying LoRA to Embedding layer: {layer_name}")
            return LoRAEmbedding(
                base_layer=layer,
                adapter_name=adapter_name,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                init_lora_weights=True
            )
        
        elif isinstance(layer, nn.Conv2d):
            # Support for convolutional layers (e.g., EfficientNet, ResNet)
            self.logger.debug(f"Applying LoRA to Conv2d layer: {layer_name}")
            return LoRAConv2d(
                base_layer=layer,
                adapter_name=adapter_name,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                init_lora_weights=True
            )
        
        else:
            self.logger.debug(
                f"Layer type {type(layer).__name__} not supported for LoRA, skipping"
            )
            return layer
    
    def _apply_lora_custom(self, model: torch.nn.Module, params: dict) -> torch.nn.Module:
        """
        Apply LoRA to custom (non-HuggingFace) models using fedcore implementation.
        
        Supports nn.Linear, nn.Conv2d, and nn.Embedding layers.
        Wraps them with LoRALayer.Linear, LoRALayer.Conv2d, LoRALayer.Embedding.
        
        Args:
            model: Model to apply LoRA to
            params: LoRA parameters (r, alpha, dropout, etc.)
            
        Returns:
            Model with LoRA applied
        """
        self.logger.info("Applying LoRA to custom model using fedcore implementation")
        
        lora_applied_count = 0
        modules_to_replace = []
        
        # First pass: identify modules to replace
        for name, module in model.named_modules():
            # Check if module is a supported layer type
            is_supported = isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding))
            
            if is_supported and self._should_apply_lora(name):
                modules_to_replace.append((name, module))
                self.logger.debug(
                    f"Identified {type(module).__name__} layer for LoRA: {name}"
                )
        
        # Second pass: replace modules
        for name, module in modules_to_replace:
            # Get parent module and attribute name
            name_parts = name.split('.')
            attr_name = name_parts[-1]
            
            if len(name_parts) > 1:
                parent_name = '.'.join(name_parts[:-1])
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            # Apply LoRA and replace the layer
            try:
                lora_layer = self._apply_lora_to_layer(module, name)
                if lora_layer is not module:  # If layer was actually modified
                    setattr(parent, attr_name, lora_layer)
                    self._lora_modules.append((name, lora_layer))
                    lora_applied_count += 1
                    self.logger.debug(f"✓ LoRA applied to {name}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to apply LoRA to {name}: {e}"
                )
        
        self.logger.info(
            f"Applied LoRA to {lora_applied_count} layers "
            f"(Linear, Conv2d, Embedding)"
        )
        return model
    
    def _apply_lora_huggingface(self, model: torch.nn.Module, params: dict) -> torch.nn.Module:
        """
        Apply LoRA to HuggingFace models.
        
        If use_peft=True, uses PEFT library (peft.LoraConfig + get_peft_model).
        Otherwise, uses fedcore implementation from lora.py.
        
        Args:
            model: HuggingFace model to apply LoRA to
            params: LoRA parameters (r, alpha, dropout, etc.)
            
        Returns:
            Model with LoRA applied
        """
        if params.get('use_peft', self.use_peft):
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                
                self.logger.info("Applying LoRA using PEFT library")
                
                # Determine task type
                task_type = params.get('task_type', TaskType.CAUSAL_LM)
                if isinstance(task_type, str):
                    task_type_map = {
                        'causal_lm': TaskType.CAUSAL_LM,
                        'seq_cls': TaskType.SEQ_CLS,
                        'seq_2_seq_lm': TaskType.SEQ_2_SEQ_LM,
                        'token_cls': TaskType.TOKEN_CLS,
                    }
                    task_type = task_type_map.get(task_type.lower(), TaskType.CAUSAL_LM)
                
                peft_config = LoraConfig(
                    task_type=task_type,
                    r=params.get('lora_r', self.lora_r),
                    lora_alpha=params.get('lora_alpha', self.lora_alpha),
                    lora_dropout=params.get('lora_dropout', self.lora_dropout),
                    target_modules=params.get('lora_target_modules', self.lora_target_modules),
                    bias=params.get('lora_bias', self.lora_bias),
                )
                
                model = get_peft_model(model, peft_config)
                self.logger.info("PEFT LoRA applied successfully")
                
                # Print trainable parameters info
                if hasattr(model, 'print_trainable_parameters'):
                    model.print_trainable_parameters()
                
                return model
                
            except ImportError as e:
                self.logger.warning(
                    f"PEFT library not available: {e}. "
                    "Install with: pip install peft. "
                    "Falling back to fedcore LoRA implementation."
                )
                self.use_peft = False
            except Exception as e:
                self.logger.error(
                    f"Error applying PEFT LoRA: {e}. "
                    "This may occur if model is not compatible with PEFT. "
                    "Falling back to fedcore LoRA implementation."
                )
                self.use_peft = False
        
        # Fallback to custom implementation for HuggingFace models
        self.logger.info("Using fedcore LoRA implementation for HuggingFace model")
        return self._apply_lora_custom(model, params)
    
    def _setup_optimizer_for_lora_params(self):
        """
        Override optimizer generation to use only trainable (LoRA) parameters.
        
        This is critical because by default OptimizerGen uses model.parameters()
        which includes frozen parameters. We need to pass only parameters with
        requires_grad=True to avoid the "element 0 of tensors does not require grad" error.
        """
        from functools import partial
        import torch.optim as optim
        
        # Get only trainable parameters
        trainable_params = [p for p in self.trainer.model.parameters() if p.requires_grad]
        
        trainable_count = len(trainable_params)
        total_params = sum(p.numel() for p in trainable_params)
        
        self.logger.info(
            f"Optimizer will be created for {trainable_count} trainable parameter groups "
            f"({total_params:,} parameters)"
        )
        
        if trainable_count == 0:
            raise ValueError(
                "No trainable parameters found! Cannot create optimizer. "
                "LoRA may not have been applied correctly."
            )
        
        # Get optimizer type and learning rate from params
        opt_type = self.params.get('optimizer', 'adam')
        learning_rate = self.params.get('learning_rate', self.learning_rate)
        
        # Create optimizer generator that uses only trainable parameters
        if isinstance(opt_type, str):
            if opt_type.lower() == 'adam':
                opt_constructor = optim.Adam
            elif opt_type.lower() == 'sgd':
                opt_constructor = optim.SGD
            elif opt_type.lower() == 'adamw':
                opt_constructor = optim.AdamW
            else:
                opt_constructor = optim.Adam  # default
        else:
            opt_constructor = opt_type
        
        # Create optimizer immediately with trainable parameters
        optimizer = opt_constructor(trainable_params, lr=learning_rate)
        
        # Store optimizer in trainer
        self.trainer.trainer_objects['optimizer'] = optimizer
        
        # CRITICAL: Disable OptimizerGen hook to prevent it from overwriting our optimizer
        # OptimizerGen would create optimizer with ALL parameters (including frozen ones)
        # We need to remove it from hooks to keep our LoRA-specific optimizer
        if hasattr(self.trainer, '_hooks'):
            # Remove OptimizerGen from hooks list
            from fedcore.models.network_impl.utils.hooks import OptimizerGen
            self.trainer._hooks = [
                hook for hook in self.trainer._hooks 
                if not isinstance(hook, OptimizerGen)
            ]
            self.logger.info("Disabled OptimizerGen hook to preserve LoRA optimizer")
        
        self.logger.info(f"Optimizer created: {type(optimizer).__name__} with lr={learning_rate}")
    
    def _freeze_non_lora_parameters(self, model: torch.nn.Module) -> None:
        """
        Freeze all base model parameters, keeping only LoRA parameters trainable.
        
        This ensures that only LoRA adapters (lora_A, lora_B) are updated during training,
        while the base model weights remain frozen.
        
        Args:
            model: Model with LoRA layers applied
        """
        self.logger.info("Freezing non-LoRA parameters")
        
        # First, freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze only LoRA parameters (lora_A, lora_B)
        trainable_params = 0
        total_params = 0
        lora_params_count = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            # Check if this is a LoRA parameter (lora_A, lora_B, or lora_embedding_*)
            is_lora_param = any(
                lora_name in name.lower()
                for lora_name in ['lora_a', 'lora_b', 'lora_embedding_a', 'lora_embedding_b']
            )
            
            if is_lora_param:
                param.requires_grad = True
                trainable_params += param.numel()
                lora_params_count += 1
                self.logger.debug(f"✓ LoRA parameter trainable: {name}")
        
        # Handle bias strategy
        if self.lora_bias == "all":
            self.logger.info("Unfreezing all bias parameters (lora_bias='all')")
            for name, param in model.named_parameters():
                if 'bias' in name.lower() and not param.requires_grad:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    self.logger.debug(f"✓ Bias parameter trainable: {name}")
        elif self.lora_bias == "lora_only":
            self.logger.info("Unfreezing only LoRA bias parameters (lora_bias='lora_only')")
            for name, param in model.named_parameters():
                if 'bias' in name.lower() and 'lora' in name.lower() and not param.requires_grad:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    self.logger.debug(f"✓ LoRA bias parameter trainable: {name}")
        
        trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
        
        self.logger.info(
            f"Parameter freeze complete: {lora_params_count} LoRA parameter groups enabled"
        )
        self.logger.info(
            f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({trainable_percent:.2f}%)"
        )
        
        # Diagnostic check - verify that we actually have trainable parameters
        if trainable_params == 0:
            self.logger.error("WARNING: No trainable parameters found! LoRA may not be applied correctly.")
            # List all parameters to debug
            for name, param in model.named_parameters():
                self.logger.error(f"  {name}: requires_grad={param.requires_grad}")
    
    def _init_model(self, input_data: CompressionInputData):
        """
        Initialize model with LoRA adaptation.
        
        Steps:
        1. Get model from input_data.target
        2. Determine model type (HuggingFace or custom)
        3. Apply LoRA to appropriate layers via _apply_lora_to_model()
        4. Freeze base parameters via _freeze_non_lora_parameters()
        5. Create trainer via factory
        """
        self.logger.info('Initializing LoRA model'.center(80, '='))
        
        # Get model from input_data
        model = input_data.target
        if isinstance(model, str):
            device = default_device()
            loaded = torch.load(model, map_location=device)
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
            else:
                model = loaded
        
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                f"Expected model to be torch.nn.Module, got {type(model)}"
            )
        
        # Store original model
        self.model_before = model
        
        # Move to device BEFORE applying LoRA to ensure all parameters are on correct device
        device = default_device()
        model.to(device)
        
        # Prepare LoRA parameters
        lora_params = {
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'lora_target_modules': self.lora_target_modules,
            'use_peft': self.use_peft,
            'lora_bias': self.lora_bias,
        }
        
        # Apply LoRA using universal interface
        model_with_lora = self._apply_lora_to_model(model, lora_params)
        
        # Freeze non-LoRA parameters
        self._freeze_non_lora_parameters(model_with_lora)
        
        # Store model with LoRA applied
        self.model_after = model_with_lora
        
        # Create trainer using factory
        self.logger.info("Creating trainer via factory")
        self.trainer = create_trainer_from_input_data(input_data, self.params)
        self.trainer.model = model_with_lora
        
        # CRITICAL: Override optimizer creation to use only trainable parameters
        # This ensures the optimizer only receives LoRA parameters with requires_grad=True
        self._setup_optimizer_for_lora_params()
        
        self.logger.info('LoRA initialization complete'.center(80, '='))
        
        return model_with_lora
    
    def _fit_model(self, input_data: CompressionInputData, split_data: bool = False):
        """
        Fit the LoRA-adapted model.
        
        Args:
            input_data: Training data
            split_data: Whether to split data (unused, for compatibility)
        """
        self.logger.info('Starting LoRA training'.center(80, '='))
        
        # Initialize model if not already done
        if self.model_after is None:
            self._init_model(input_data)
        
        # Train using the trainer
        self.logger.info(f"Training for {self.epochs} epochs")
        self.trainer.fit(input_data)
        
        # Update model_after with trained weights
        self.model_after = self.trainer.model
        
        self.logger.info('LoRA training complete'.center(80, '='))
    
    def fit(self, input_data: CompressionInputData):
        """
        Main fit method - applies LoRA and trains the model.
        
        Args:
            input_data: Training data with model in target field
            
        Returns:
            Self
        """
        self.num_classes = input_data.num_classes
        self.target = input_data.target
        self.task_type = input_data.task
        
        # Fit the model
        self._fit_model(input_data)
        
        # Set self.model for compatibility with base class
        self.model = self.model_after
        
        # Save and clear cache
        self._save_and_clear_cache()
        
        return self
    
    def predict(
        self,
        input_data: CompressionInputData,
        output_mode: str = "fedcore"
    ) -> torch.nn.Module:
        """
        Get the LoRA-adapted model for inference.
        
        Args:
            input_data: Input data for prediction
            output_mode: "fedcore" returns adapted model, "default" returns original
            
        Returns:
            Model (either with or without LoRA)
        """
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        
        return self.trainer.predict(input_data, output_mode)
    
    def predict_for_fit(
        self,
        input_data: CompressionInputData,
        output_mode: str = 'fedcore'
    ) -> torch.nn.Module:
        """
        Get model for fit phase.
        
        Args:
            input_data: Input data
            output_mode: "fedcore" returns adapted model, "default" returns original
            
        Returns:
            Model (either with or without LoRA)
        """
        return self.model_after if output_mode == 'fedcore' else self.model_before
    
    def get_lora_state_dict(self) -> dict:
        """
        Get only the LoRA parameters for saving/loading.
        
        Returns:
            Dictionary containing only LoRA parameters
        """
        if self.model_after is None:
            raise ValueError("Model not initialized. Call fit() first.")
        
        lora_state_dict = {}
        for name, param in self.model_after.named_parameters():
            if any(
                lora_name in name.lower()
                for lora_name in ['lora_a', 'lora_b', 'lora_embedding']
            ):
                lora_state_dict[name] = param.cpu().detach()
        
        return lora_state_dict
    
    def load_lora_state_dict(self, state_dict: dict) -> None:
        """
        Load LoRA parameters from state dict.
        
        Args:
            state_dict: Dictionary containing LoRA parameters
        """
        if self.model_after is None:
            raise ValueError("Model not initialized. Call _init_model() first.")
        
        model_state = self.model_after.state_dict()
        model_state.update(state_dict)
        self.model_after.load_state_dict(model_state, strict=False)
        
        self.logger.info(f"Loaded {len(state_dict)} LoRA parameters")
    
    def _save_and_clear_cache(self):
        """
        Override base class method for LoRA-specific saving.
        
        LoRA models don't need the complex pruning-specific save logic.
        Just ensure gradients are cleared and memory is freed.
        """
        if self.model_after is not None:
            self.model_after.zero_grad()
            
            # Set self.model for compatibility with base class expectations
            if self.model is None:
                self.model = self.model_after
            
            # Clear CUDA cache
            import torch
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()