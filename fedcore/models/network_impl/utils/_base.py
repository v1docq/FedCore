from abc import ABC, abstractmethod
from typing import Protocol, Iterable, Any, Dict, Optional, runtime_checkable, List
from enum import Enum
import torch
import os
from pathlib import Path
from typing import Literal, Union
from transformers import PreTrainedModel
from torch.nn import Module

from fedcore.models.network_impl.utils.interfaces import ITrainer, IHookable
from fedcore.architecture.comptutaional.devices import default_device

HookType = Literal['start', 'end', 'batch_start', 'batch_end', 'validation']

class BaseTrainer(ITrainer, IHookable):
    
    def __init__(self, model=None, params: Optional[Dict] = None):
        self.params = params or {}
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
            try:
                hook(epoch=epoch, trainer_objects=self.trainer_objects, 
                     history=self.history, **kwargs)
            except Exception as e:
                print(f"Error executing hook {hook.__class__.__name__}: {e}")
    
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
    
    @property
    def is_quantised(self) -> bool:
        return getattr(self.model, '_is_quantised', False)
    
    @property
    def optimizer(self) -> Any:
        return self.trainer_objects.get('optimizer')
    
    @property
    def scheduler(self) -> Any:
        return self.trainer_objects.get('scheduler')
    
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