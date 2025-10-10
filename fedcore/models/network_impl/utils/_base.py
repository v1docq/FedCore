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

from fedcore.models.network_impl.utils.interfaces import ITrainer, IHookable
from fedcore.architecture.computational.devices import default_device
from fedcore.repository.constant_repository import (
    ModelLearningHooks,
    LoggingHooks,
    StructureCriterions,
    TorchLossesConstant,
)

HookType = Literal['start', 'end', 'batch_start', 'batch_end', 'validation']

class BaseTrainer(ITrainer, IHookable):
    
    def __init__(self, model=None, params: Optional[Dict] = None):
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
        
        self.model: Union['PreTrainedModel', 'Module', None] = model
        self.device = default_device()
            
    def register_additional_hooks(self, hooks: Iterable[Enum]) -> None:
        self._additional_hooks.extend(hooks)
    
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
    
    def save_model(self, path: str) -> None:
        if self.model is not None:
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(path)
            else:
                torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def load_model(self, path: str) -> None:
        """Load the model - default implementation"""
        if os.path.exists(path):
            if self.model is None:
                print("Model not initialized, cannot load weights")
                return
                
            if hasattr(self.model, 'from_pretrained'):
                self.model = self.model.from_pretrained(path)
            else:
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            print(f"Model loaded from {path}")
        else:
            print(f"Model path {path} does not exist")
    
    def _clear_cache(self):
        """Clear CUDA cache - shared by BaseNeuralModel and LLMTrainer"""
        with torch.no_grad():
            torch.cuda.empty_cache()

    def _compute_loss(self, criterion, model_output, target, stage='train', epoch=None):
        if hasattr(model_output, 'loss'):
            quality_loss = model_output.loss
        else:
            quality_loss = criterion(model_output, target)
        if isinstance(model_output, torch.Tensor):
            additional_losses = {name: coef * criterion(model_output, target)
                                 for name, (criterion, coef) in self.custom_criterions.items()
                                 if hasattr(TorchLossesConstant, name)}
            additional_losses.update({name: coef * criterion(self.model)
                                      for name, (criterion, coef) in self.custom_criterions.items()
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