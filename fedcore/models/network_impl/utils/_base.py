from abc import ABC, abstractmethod
from typing import Iterable, Any, Dict, Optional
from enum import Enum
import torch
import os
from pathlib import Path
from typing import Literal, Union
from transformers import PreTrainedModel
from torch.nn import Module
from functools import reduce
import numpy as np

from fedcore.models.network_impl.utils.interfaces import ITrainer, IHookable
from fedcore.architecture.computational.devices import default_device


HookType = Literal['start', 'end', 'batch_start', 'batch_end', 'validation']

class BaseTrainer(ITrainer, IHookable):
    
    def __init__(self, model: Union['PreTrainedModel', 'Module', None], params: Optional[Dict] = None):
        self.params = params or {}
        self.learning_params = self.params.get('custom_learning_params', {})
        
        self._hooks = []
        self._additional_hooks = []
        self.hooks_collection = {
            'start': [],
            'end': [],
            'batch_start': [],
            'batch_end': [],
            'validation': []
        }
        
        self.trainer_objects = {
            'optimizer': None,
            'scheduler': None,
            'trainer': None
        }
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        self.model = model
        self.device = default_device()
    
    def _init_hooks(self) -> None:
        raise NotImplementedError("Subclasses must implement _init_hooks")

    def execute_hooks(self, hook_type: HookType, epoch: int, **kwargs) -> None:
        for hook in self.hooks_collection[hook_type]:
            hook(epoch=epoch, trainer_objects=self.trainer_objects, 
                     history=self.history, **kwargs)
    
    @abstractmethod
    def fit(self, input_data: Any, supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        pass
    
    @abstractmethod
    def predict(self, input_data: Any, output_mode: str = "default") -> Any:
        pass

    def _clear_cache(self):
        """Clear GPU cache using ModelRegistry cleanup methods."""
        try:
            from fedcore.tools.registry.model_registry import ModelRegistry
            registry = ModelRegistry()
            registry.force_cleanup()
        except Exception:
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _compute_loss(self, criterion, model_output, target, stage='train', epoch=None):
        from fedcore.repository.constant_repository import (
            StructureCriterions,
            TorchLossesConstant,
        )
        if hasattr(model_output, 'loss'):
            quality_loss = model_output.loss
        else:
            quality_loss = criterion(model_output, target)
        if isinstance(model_output, torch.Tensor):
            additional_losses = {name: custom_criterion_weight * criterion(model_output, target)
                                 for name, (criterion, custom_criterion_weight) in self.custom_criterions.items()
                                 if hasattr(TorchLossesConstant, name)}
            additional_losses.update({name: custom_criterion_weight * criterion(self.model)
                                      for name, (criterion, custom_criterion_weight) in self.custom_criterions.items()
                                      if hasattr(StructureCriterions, name)})
            for name, val in additional_losses.items():
                self.history[f'{stage}_{name}_loss'].append((epoch, val))
        final_loss = reduce(torch.add, additional_losses.values(), quality_loss)
        return final_loss
    
    @property
    def is_quantised(self) -> bool:
        return getattr(self.model, '_is_quantised', False)
    
    @property
    def optimizer(self) -> Any:
        return self.trainer_objects.get('optimizer')
    
    @optimizer.setter
    def optimizer(self, value: Any) -> None:
        self.trainer_objects['optimizer'] = value
    
    @property
    def scheduler(self) -> Any:
        return self.trainer_objects.get('scheduler')
    
    @scheduler.setter
    def scheduler(self, value: Any) -> None:
        self.trainer_objects['scheduler'] = value
    
    def _normalize_kwargs(self, kwargs: Dict[str, Any], allowed_keys: set) -> Dict[str, Any]:
        normalized = {}
        synonym_mapping = {
            'num_epochs': 'num_train_epochs',
            'epochs': 'num_train_epochs',
            'batch_size': 'per_device_train_batch_size',
            'train_batch_size': 'per_device_train_batch_size',
            'eval_batch_size': 'per_device_eval_batch_size',
            'learning_rate': 'learning_rate',
            'lr': 'learning_rate',
        }
        
        for key, value in kwargs.items():
            if key in allowed_keys:
                normalized[key] = value
            elif key in synonym_mapping and synonym_mapping[key] in allowed_keys:
                normalized[synonym_mapping[key]] = value
        
        return normalized
    
    def _extract_output_fields(self, input_data: Any) -> Dict[str, Any]:
        """Extract common fields from input_data for CompressionOutputData creation.
        
        Uses a mapping approach similar to _normalize_kwargs for consistent field extraction.
        
        Args:
            input_data: InputData or CompressionInputData instance
            
        Returns:
            Dictionary with train_dataloader, val_dataloader, num_classes, and features
        """
        if input_data is None:
            return {
                'train_dataloader': None,
                'val_dataloader': None,
                'num_classes': None,
                'features': None
            }
        
        field_path_mapping = {
            'train_dataloader': ['train_dataloader', ('features', 'train_dataloader')],
            'val_dataloader': ['val_dataloader', ('features', 'val_dataloader')],
            'num_classes': ['num_classes', ('features', 'num_classes')],
            'features': ['features', ('features', 'features')]
        }
        
        def _get_value_by_path(obj: Any, path) -> Any:
            """Extract value by attribute path, similar to normalization logic."""
            if isinstance(path, str):
                path = (path,)
            
            try:
                result = obj
                for attr in path:
                    result = getattr(result, attr) if hasattr(result, attr) else None
                    if result is None:
                        return None
                return result
            except (AttributeError, TypeError):
                return None
        
        extracted = {}
        
        for field_name, paths in field_path_mapping.items():
            value = None
            for path in paths:
                value = _get_value_by_path(input_data, path)
                
                if value is not None:
                    if field_name == 'features':
                        if isinstance(value, np.ndarray):
                            break
                        elif hasattr(value, 'features'):
                            value = value.features
                            break
                    else:
                        break
            
            extracted[field_name] = value
        
        return extracted
    
    def _register_model_checkpoint(self, model: Any, fedcore_id: str = None, 
                                    stage: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Register model in ModelRegistry and return checkpoint information.
        
        Args:
            model: Model to register
            fedcore_id: FedCore instance identifier (optional)
            stage: Stage name (e.g., 'before', 'after') (optional)
            
        Returns:
            Dictionary with 'model_id', 'checkpoint_path', and 'fedcore_id'
        """
        try:
            from fedcore.tools.registry.model_registry import ModelRegistry
            
            registry = ModelRegistry()
            
            if fedcore_id is None:
                fedcore_id = getattr(self, '_fedcore_id', None)
                if fedcore_id is None:
                    from fedcore.tools.registry.model_registry import _registry_context
                    fedcore_id = getattr(_registry_context, 'fedcore_id', None)
            
            if fedcore_id is None or model is None:
                return {
                    'model_id': None,
                    'checkpoint_path': None,
                    'fedcore_id': fedcore_id
                }
            
            model_id = registry.register_model(
                fedcore_id=fedcore_id,
                model=model,
                stage=stage,
                delete_model_after_save=False  
            )
            
            checkpoint_path = registry.get_checkpoint_path(fedcore_id, model_id)
            
            return {
                'model_id': model_id,
                'checkpoint_path': checkpoint_path,
                'fedcore_id': fedcore_id
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"ModelRegistry registration failed: {e}")
            return {
                'model_id': None,
                'checkpoint_path': None,
                'fedcore_id': fedcore_id
            }